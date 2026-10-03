from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import audit_manifest, ensure_columns

# Counts documented in the final report.
DEVELOPMENT_UTTERANCES = 56_284
TRAIN_UTTERANCES = 50_655
VALIDATION_UTTERANCES = 5_629
SEED = 42


def _select_validation_speakers(
    development_df: pd.DataFrame,
    target_utterances: int,
    seed: int,
) -> list[str]:
    """
    Construct a deterministic speaker-disjoint validation subset whose utterance
    count exactly matches the final report.

    This is a clean report-aligned reconstruction. It is NOT presented as the
    exact historical split executed in the original notebook.
    """
    speaker_counts = (
        development_df.groupby("speaker", sort=True).size().astype(int).to_dict()
    )

    speaker_ids = np.array(sorted(speaker_counts), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(speaker_ids)

    # Subset-sum dynamic programming over speaker utterance counts.
    reachable = {0}
    parent: dict[int, tuple[int, str]] = {}

    for speaker_id in speaker_ids.tolist():
        count = speaker_counts[speaker_id]
        additions: list[tuple[int, int]] = []
        for subtotal in sorted(reachable, reverse=True):
            new_total = subtotal + count
            if new_total > target_utterances or new_total in reachable:
                continue
            additions.append((new_total, subtotal))

        for new_total, subtotal in additions:
            if new_total not in reachable:
                reachable.add(new_total)
                parent[new_total] = (subtotal, speaker_id)

        if target_utterances in reachable:
            break

    if target_utterances not in reachable:
        nearest = max(reachable)
        raise RuntimeError(
            "Could not construct an exact speaker-disjoint validation subset with "
            f"{target_utterances:,} utterances. Closest reachable total without "
            f"exceeding the target was {nearest:,}. Do not silently switch to an "
            "utterance-level split."
        )

    selected: list[str] = []
    subtotal = target_utterances
    while subtotal:
        previous, speaker_id = parent[subtotal]
        selected.append(str(speaker_id))
        subtotal = previous

    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic speaker-disjoint internal train/validation "
            "reconstruction aligned with the final report counts."
        )
    )
    parser.add_argument("--development-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/split"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    development_df = pd.read_csv(args.development_csv)
    ensure_columns(
        development_df,
        ["speaker", "audio_relpath", "audio_path", "reference"],
    )

    if len(development_df) != DEVELOPMENT_UTTERANCES:
        raise ValueError(
            f"Expected {DEVELOPMENT_UTTERANCES:,} development utterances, "
            f"got {len(development_df):,}."
        )

    audit = audit_manifest(development_df)
    if audit["missing_audio_files"] or audit["empty_transcripts"] or audit["duplicate_audio_paths"]:
        raise RuntimeError(f"Development manifest failed integrity checks: {audit}")

    validation_speakers = set(
        _select_validation_speakers(
            development_df=development_df,
            target_utterances=VALIDATION_UTTERANCES,
            seed=args.seed,
        )
    )

    validation_df = development_df.loc[
        development_df["speaker"].astype(str).isin(validation_speakers)
    ].copy()
    train_df = development_df.loc[
        ~development_df["speaker"].astype(str).isin(validation_speakers)
    ].copy()

    # Deterministic row order for the clean reconstruction.
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    validation_df = validation_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if len(train_df) != TRAIN_UTTERANCES or len(validation_df) != VALIDATION_UTTERANCES:
        raise RuntimeError(
            "Reconstructed split does not match the final report counts: "
            f"train={len(train_df):,}, validation={len(validation_df):,}."
        )

    train_speakers = set(train_df["speaker"].astype(str))
    validation_speakers_actual = set(validation_df["speaker"].astype(str))
    if train_speakers & validation_speakers_actual:
        raise RuntimeError("Speaker overlap detected between train and validation.")

    train_audio = set(train_df["audio_relpath"].astype(str))
    validation_audio = set(validation_df["audio_relpath"].astype(str))
    if train_audio & validation_audio:
        raise RuntimeError("Audio overlap detected between train and validation.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    validation_df.to_csv(output_dir / "validation.csv", index=False)

    metadata = {
        "provenance": (
            "Clean report-aligned speaker-disjoint reconstruction. The exact "
            "historical train/validation membership from the original notebook is "
            "not claimed to be reproduced."
        ),
        "seed": args.seed,
        "speaker_disjoint": True,
        "development_utterances": int(len(development_df)),
        "train_utterances": int(len(train_df)),
        "validation_utterances": int(len(validation_df)),
        "train_speakers": int(len(train_speakers)),
        "validation_speakers": int(len(validation_speakers_actual)),
        "speaker_overlap": 0,
        "audio_overlap": 0,
        "validation_speaker_ids": sorted(validation_speakers_actual),
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("INTERNAL SPLIT RECONSTRUCTION")
    print("Training utterances:", len(train_df))
    print("Validation utterances:", len(validation_df))
    print("Training speakers:", len(train_speakers))
    print("Validation speakers:", len(validation_speakers_actual))
    print("Speaker overlap: 0")
    print("NOTE: this is a clean reconstruction, not the archived historical split.")


if __name__ == "__main__":
    main()
