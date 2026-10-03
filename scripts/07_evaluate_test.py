from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import pandas as pd
import torch
from jiwer import cer, process_words, wer
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from common import ensure_columns, normalize_text

SAMPLE_RATE = 16_000
OFFICIAL_TEST_UTTERANCES = 2_829
DOCUMENTED_REFERENCE_WORDS = 14_133


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned XLS-R model on the official NCHLT test set."
    )
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    ensure_columns(df, ["filename", "audio_path", "reference"])
    if len(df) != OFFICIAL_TEST_UTTERANCES:
        raise ValueError(
            f"Expected the official {OFFICIAL_TEST_UTTERANCES:,}-utterance test set, "
            f"got {len(df):,}."
        )

    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Official-test evaluation"):
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            logits = model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(pred_ids)[0]
        reference_norm = normalize_text(row["reference"])
        prediction_norm = normalize_text(prediction)

        rows.append(
            {
                "filename": row["filename"],
                "duration": float(row["duration"]) if "duration" in row and pd.notna(row["duration"]) else None,
                "reference": row["reference"],
                "prediction": prediction,
                "reference_norm": reference_norm,
                "prediction_norm": prediction_norm,
                "exact_match": reference_norm == prediction_norm,
            }
        )

    predictions_df = pd.DataFrame(rows)
    refs = predictions_df["reference_norm"].tolist()
    hyps = predictions_df["prediction_norm"].tolist()
    word_result = process_words(refs, hyps)

    reference_words = int(predictions_df["reference_norm"].str.split().str.len().sum())
    if reference_words != DOCUMENTED_REFERENCE_WORDS:
        print(
            "WARNING: normalized reference word count is "
            f"{reference_words:,}; the final report records "
            f"{DOCUMENTED_REFERENCE_WORDS:,}."
        )

    substitutions = int(word_result.substitutions)
    deletions = int(word_result.deletions)
    insertions = int(word_result.insertions)
    total_errors = substitutions + deletions + insertions
    exact_matches = int(predictions_df["exact_match"].sum())

    metrics = {
        "dataset_partition": "official_test",
        "utterances": int(len(predictions_df)),
        "reference_word_tokens": reference_words,
        "wer_percent": float(100.0 * wer(refs, hyps)),
        "cer_percent": float(100.0 * cer(refs, hyps)),
        "exact_matches": exact_matches,
        "exact_match_percent": float(100.0 * exact_matches / len(predictions_df)),
        "total_word_errors": total_errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "device": str(device),
        "decoding": "greedy CTC",
        "external_language_model": False,
        "beam_search_rescoring": False,
        "pronunciation_lexicon": False,
    }

    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
