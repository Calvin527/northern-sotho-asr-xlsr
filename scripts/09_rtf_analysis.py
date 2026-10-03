from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from common import ensure_columns

SAMPLE_RATE = 16_000
OFFICIAL_TEST_UTTERANCES = 2_829


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure XLS-R forward-pass inference time and real-time factor (RTF) "
            "with batch size 1. Model loading and audio loading/preprocessing are "
            "excluded from the timed region."
        )
    )
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Optional untimed warm-up forward passes before corpus timing.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    ensure_columns(df, ["filename", "audio_path"])
    if len(df) != OFFICIAL_TEST_UTTERANCES:
        raise ValueError(
            f"Expected {OFFICIAL_TEST_UTTERANCES:,} official-test utterances, "
            f"got {len(df):,}."
        )

    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if args.warmup_runs > 0:
        first_audio, _ = librosa.load(df.iloc[0]["audio_path"], sr=SAMPLE_RATE, mono=True)
        first_inputs = processor(
            first_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        warm_values = first_inputs.input_values.to(device)
        warm_mask = getattr(first_inputs, "attention_mask", None)
        if warm_mask is not None:
            warm_mask = warm_mask.to(device)
        for _ in range(args.warmup_runs):
            with torch.inference_mode():
                model(warm_values, attention_mask=warm_mask)
            _sync_if_cuda(device)

    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="RTF benchmark"):
        # Audio loading + processor work happen outside the timer by design.
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        audio_duration = float(len(audio) / SAMPLE_RATE)
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

        _sync_if_cuda(device)
        start = time.perf_counter()
        with torch.inference_mode():
            model(input_values, attention_mask=attention_mask)
        _sync_if_cuda(device)
        processing_time = time.perf_counter() - start

        rows.append(
            {
                "filename": row["filename"],
                "audio_duration_seconds": audio_duration,
                "processing_time_seconds": processing_time,
                "rtf": processing_time / audio_duration,
            }
        )

    rtf_df = pd.DataFrame(rows)
    summary = {
        "utterances": int(len(rtf_df)),
        "batch_size": 1,
        "timing_scope": "model forward pass only",
        "model_loading_included": False,
        "audio_loading_preprocessing_included": False,
        "warmup_runs": int(args.warmup_runs),
        "mean_audio_duration_seconds": float(rtf_df["audio_duration_seconds"].mean()),
        "mean_processing_time_seconds": float(rtf_df["processing_time_seconds"].mean()),
        "mean_rtf": float(rtf_df["rtf"].mean()),
        "median_rtf": float(rtf_df["rtf"].median()),
        "min_rtf": float(rtf_df["rtf"].min()),
        "max_rtf": float(rtf_df["rtf"].max()),
        "rtf_below_1_percent": float(100.0 * (rtf_df["rtf"] < 1.0).mean()),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    rtf_df.to_csv(output_dir / "rtf_per_utterance.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "rtf_summary.csv", index=False)
    (output_dir / "rtf_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
