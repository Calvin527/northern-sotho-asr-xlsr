from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import pandas as pd
import torch
from jiwer import cer, process_words, wer
from tqdm.auto import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC

from common import ensure_columns, normalize_text

MODEL_ID = "facebook/mms-1b-fl102"
TARGET_LANG = "nso"
SAMPLE_RATE = 16_000
OFFICIAL_TEST_UTTERANCES = 2_829
DOCUMENTED_REFERENCE_WORDS = 14_133


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the off-the-shelf MMS-1B-FL102 Northern Sotho (nso) adapter "
            "on the same official NCHLT test set used for XLS-R."
        )
    )
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    ensure_columns(df, ["filename", "audio_path", "reference"])
    if len(df) != OFFICIAL_TEST_UTTERANCES:
        raise ValueError(
            f"Expected {OFFICIAL_TEST_UTTERANCES:,} official-test utterances, "
            f"got {len(df):,}."
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    processor.tokenizer.set_target_lang(TARGET_LANG)
    model.load_adapter(TARGET_LANG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MMS Northern Sotho baseline"):
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        with torch.inference_mode():
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(pred_ids)[0]
        reference_norm = normalize_text(row["reference"])
        prediction_norm = normalize_text(prediction)

        rows.append(
            {
                "filename": row["filename"],
                "reference": row["reference"],
                "mms_prediction": prediction,
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

    exact_matches = int(predictions_df["exact_match"].sum())
    substitutions = int(word_result.substitutions)
    deletions = int(word_result.deletions)
    insertions = int(word_result.insertions)

    metrics = {
        "model": MODEL_ID,
        "target_language": TARGET_LANG,
        "additional_nchlt_finetuning": False,
        "dataset_partition": "official_test",
        "utterances": int(len(predictions_df)),
        "reference_word_tokens": reference_words,
        "wer_percent": float(100.0 * wer(refs, hyps)),
        "cer_percent": float(100.0 * cer(refs, hyps)),
        "exact_matches": exact_matches,
        "exact_match_percent": float(100.0 * exact_matches / len(predictions_df)),
        "total_word_errors": substitutions + deletions + insertions,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "decoding": "greedy CTC",
        "device": str(device),
        "runtime_comparison_with_xlsr": False,
        "comparison_scope": "transcription accuracy only",
    }

    predictions_df.to_csv(output_dir / "mms_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "mms_baseline_results.csv", index=False)
    (output_dir / "mms_baseline_results.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
