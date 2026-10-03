from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import pandas as pd
import torch
from jiwer import cer, wer
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from common import ensure_columns, normalize_text

SAMPLE_RATE = 16_000
DOCUMENTED_VALIDATION_UTTERANCES = 5_629
DOCUMENTED_VALIDATION_WER = 21.69
DOCUMENTED_VALIDATION_CER = 5.60


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an XLS-R checkpoint on the Northern Sotho validation split."
    )
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.validation_csv)
    ensure_columns(df, ["filename", "audio_path", "reference"])
    if len(df) != DOCUMENTED_VALIDATION_UTTERANCES:
        print(
            "WARNING: validation manifest contains "
            f"{len(df):,} utterances; the final report documents "
            f"{DOCUMENTED_VALIDATION_UTTERANCES:,}."
        )

    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validation evaluation"):
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
    validation_wer = 100.0 * wer(refs, hyps)
    validation_cer = 100.0 * cer(refs, hyps)
    exact_matches = int(predictions_df["exact_match"].sum())

    metrics = {
        "dataset_partition": "validation",
        "utterances": int(len(predictions_df)),
        "wer_percent": float(validation_wer),
        "cer_percent": float(validation_cer),
        "exact_matches": exact_matches,
        "exact_match_percent": float(100.0 * exact_matches / len(predictions_df)),
        "device": str(device),
        "documented_final_report_wer_percent": DOCUMENTED_VALIDATION_WER,
        "documented_final_report_cer_percent": DOCUMENTED_VALIDATION_CER,
        "note": (
            "Recomputed metrics depend on the supplied checkpoint and validation "
            "manifest. A reconstructed split is not claimed to reproduce the exact "
            "historical validation membership."
        ),
    }

    predictions_df.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "validation_metrics.csv", index=False)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
