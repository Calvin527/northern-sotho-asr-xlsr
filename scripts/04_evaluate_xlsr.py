from __future__ import annotations

import argparse, json, time
from pathlib import Path
import librosa
import pandas as pd
import torch
from jiwer import cer, process_words, wer
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from common import ensure_columns, normalize_text

SAMPLE_RATE = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    ensure_columns(df, ["filename", "audio_path", "duration", "reference"])
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="XLS-R evaluation"):
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None: attention_mask = attention_mask.to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_values, attention_mask=attention_mask)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        prediction = processor.batch_decode(pred_ids)[0]
        rows.append({
            "filename": row["filename"],
            "duration": float(row["duration"]),
            "reference": row["reference"],
            "prediction": prediction,
            "reference_norm": normalize_text(row["reference"]),
            "prediction_norm": normalize_text(prediction),
            "forward_time_seconds": elapsed,
            "rtf": elapsed / float(row["duration"]),
        })

    out = pd.DataFrame(rows)
    refs, hyps = out["reference_norm"].tolist(), out["prediction_norm"].tolist()
    word = process_words(refs, hyps)
    exact = int((out["reference_norm"] == out["prediction_norm"]).sum())
    total_errors = int(word.substitutions + word.deletions + word.insertions)
    metrics = {
        "utterances": len(out),
        "reference_word_tokens": int(out["reference_norm"].str.split().str.len().sum()),
        "wer_percent": 100 * wer(refs, hyps),
        "cer_percent": 100 * cer(refs, hyps),
        "exact_matches": exact,
        "exact_match_percent": 100 * exact / len(out),
        "total_word_errors": total_errors,
        "substitutions": int(word.substitutions),
        "deletions": int(word.deletions),
        "insertions": int(word.insertions),
        "mean_forward_time_seconds": float(out["forward_time_seconds"].mean()),
        "mean_rtf": float(out["rtf"].mean()),
        "median_rtf": float(out["rtf"].median()),
        "min_rtf": float(out["rtf"].min()),
        "max_rtf": float(out["rtf"].max()),
        "rtf_below_1_percent": float(100 * (out["rtf"] < 1.0).mean()),
        "device": str(device),
    }
    out.to_csv(args.output_dir / "xlsr_predictions.csv", index=False)
    (args.output_dir / "xlsr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
