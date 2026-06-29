"""
Evaluate a fine-tuned XLS-R CTC model on the Northern Sotho test set.

Outputs:
- full_predictions_xlsr_nso_19000_steps.csv
- evaluation_summary_xlsr_nso_19000_steps.csv
- error_summary_xlsr_nso_19000_steps.csv
"""

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import cer, process_words, wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def normalise_text(text: str) -> str:
    return str(text).lower().strip()


def transcribe_audio(audio_path: str, model, processor, device: str, sampling_rate: int = 16000):
    speech, _ = librosa.load(audio_path, sr=sampling_rate)
    audio_duration = librosa.get_duration(y=speech, sr=sampling_rate)

    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    start = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    processing_time = time.time() - start

    predicted_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(predicted_ids)[0]
    rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

    return normalise_text(prediction), audio_duration, processing_time, rtf


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Northern Sotho XLS-R ASR model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model/processor folder.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test.csv with audio_path and sentence columns.")
    parser.add_argument("--output_dir", type=str, default="results", help="Folder for evaluation outputs.")
    parser.add_argument("--training_steps", type=int, default=19000)
    parser.add_argument("--approx_epochs", type=float, default=3.0)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_csv = Path(args.test_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()

    test_df = pd.read_csv(test_csv)
    rows = []

    for idx, row in test_df.iterrows():
        reference = normalise_text(row["sentence"])
        prediction, duration, processing_time, rtf = transcribe_audio(
            row["audio_path"], model, processor, device, sampling_rate=args.sampling_rate
        )

        sent_wer = wer([reference], [prediction])
        sent_cer = cer([reference], [prediction])

        rows.append({
            "audio_path": row["audio_path"],
            "reference": reference,
            "prediction": prediction,
            "audio_duration_seconds": duration,
            "processing_time_seconds": processing_time,
            "real_time_factor": rtf,
            "sentence_wer": sent_wer,
            "sentence_cer": sent_cer,
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(test_df)}")

    predictions_df = pd.DataFrame(rows)
    references = predictions_df["reference"].tolist()
    predictions = predictions_df["prediction"].tolist()

    overall_wer = wer(references, predictions)
    overall_cer = cer(references, predictions)
    word_results = process_words(references, predictions)

    predictions_path = output_dir / f"full_predictions_xlsr_nso_{args.training_steps}_steps.csv"
    summary_path = output_dir / f"evaluation_summary_xlsr_nso_{args.training_steps}_steps.csv"
    error_path = output_dir / f"error_summary_xlsr_nso_{args.training_steps}_steps.csv"

    predictions_df.to_csv(predictions_path, index=False)

    summary_df = pd.DataFrame([{
        "model": "facebook/wav2vec2-xls-r-300m",
        "language": "Northern Sotho / Sepedi",
        "training_steps": args.training_steps,
        "approx_epochs": args.approx_epochs,
        "test_samples": len(predictions_df),
        "WER": overall_wer,
        "WER_percent": overall_wer * 100,
        "CER": overall_cer,
        "CER_percent": overall_cer * 100,
        "substitutions": word_results.substitutions,
        "deletions": word_results.deletions,
        "insertions": word_results.insertions,
        "hits": word_results.hits,
        "average_audio_duration_seconds": predictions_df["audio_duration_seconds"].mean(),
        "average_processing_time_seconds": predictions_df["processing_time_seconds"].mean(),
        "average_real_time_factor": predictions_df["real_time_factor"].mean(),
    }])
    summary_df.to_csv(summary_path, index=False)

    error_df = pd.DataFrame({
        "error_type": ["Substitutions", "Deletions", "Insertions", "Hits"],
        "count": [word_results.substitutions, word_results.deletions, word_results.insertions, word_results.hits],
    })
    error_df.to_csv(error_path, index=False)

    print("Final evaluation results")
    print(summary_df)
    print(f"Predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Error summary saved to: {error_path}")


if __name__ == "__main__":
    main()
