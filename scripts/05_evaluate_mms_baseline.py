from __future__ import annotations

import argparse, json
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
SAMPLE_RATE = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    ensure_columns(df, ["filename", "audio_path", "duration", "reference"])
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    processor.tokenizer.set_target_lang(TARGET_LANG)
    model.load_adapter(TARGET_LANG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MMS Northern Sotho"):
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.inference_mode():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(pred_ids)[0]
        rows.append({
            "filename": row["filename"],
            "duration": float(row["duration"]),
            "reference": row["reference"],
            "mms_prediction": prediction,
            "reference_norm": normalize_text(row["reference"]),
            "prediction_norm": normalize_text(prediction),
        })

    out = pd.DataFrame(rows)
    refs, hyps = out["reference_norm"].tolist(), out["prediction_norm"].tolist()
    word = process_words(refs, hyps)
    exact = int((out["reference_norm"] == out["prediction_norm"]).sum())
    metrics = {
        "model": MODEL_ID,
        "target_language": TARGET_LANG,
        "utterances": len(out),
        "reference_word_tokens": int(out["reference_norm"].str.split().str.len().sum()),
        "wer_percent": 100 * wer(refs, hyps),
        "cer_percent": 100 * cer(refs, hyps),
        "exact_matches": exact,
        "exact_match_percent": 100 * exact / len(out),
        "total_word_errors": int(word.substitutions + word.deletions + word.insertions),
        "substitutions": int(word.substitutions),
        "deletions": int(word.deletions),
        "insertions": int(word.insertions),
        "device": str(device),
        "runtime_comparison_with_xlsr": False,
    }
    out.to_csv(args.output_dir / "mms_predictions.csv", index=False)
    (args.output_dir / "mms_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
