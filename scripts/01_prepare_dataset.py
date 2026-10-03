from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import audit_manifest, parse_nchlt_xml

# Corpus counts documented in the final honours report / SATNAC paper.
DEVELOPMENT_UTTERANCES = 56_284
OFFICIAL_TEST_UTTERANCES = 2_829
TOTAL_UTTERANCES = 59_113
OFFICIAL_TEST_SPEAKERS = 8


def _fail_on_manifest_problems(name: str, audit: dict) -> None:
    problems = {
        key: audit.get(key, 0)
        for key in ("missing_audio_files", "empty_transcripts", "duplicate_audio_paths")
        if audit.get(key, 0)
    }
    if problems:
        raise RuntimeError(f"{name} manifest failed integrity checks: {problems}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and audit the NCHLT Northern Sotho development and official-test "
            "partitions. This stage does not create the internal train/validation split."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing the top-level nchlt_nso folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests/raw"),
        help="Directory for development_full.csv, test.csv, and dataset_audit.json.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_xml = dataset_root / "nchlt_nso" / "transcriptions" / "nchlt_nso.trn.xml"
    test_xml = dataset_root / "nchlt_nso" / "transcriptions" / "nchlt_nso.tst.xml"

    development_df = parse_nchlt_xml(train_xml, dataset_root)
    test_df = parse_nchlt_xml(test_xml, dataset_root)

    development_audit = audit_manifest(development_df)
    test_audit = audit_manifest(test_df)

    _fail_on_manifest_problems("Development", development_audit)
    _fail_on_manifest_problems("Official test", test_audit)

    if len(development_df) != DEVELOPMENT_UTTERANCES:
        raise ValueError(
            f"Expected {DEVELOPMENT_UTTERANCES:,} development utterances, "
            f"got {len(development_df):,}."
        )
    if len(test_df) != OFFICIAL_TEST_UTTERANCES:
        raise ValueError(
            f"Expected {OFFICIAL_TEST_UTTERANCES:,} official-test utterances, "
            f"got {len(test_df):,}."
        )
    if len(development_df) + len(test_df) != TOTAL_UTTERANCES:
        raise ValueError("Combined corpus size does not match the documented total.")

    if int(test_df["speaker"].nunique()) != OFFICIAL_TEST_SPEAKERS:
        raise ValueError(
            f"Expected {OFFICIAL_TEST_SPEAKERS} official-test speakers, "
            f"got {test_df['speaker'].nunique()}."
        )

    development_speakers = set(development_df["speaker"].astype(str))
    test_speakers = set(test_df["speaker"].astype(str))
    speaker_overlap = sorted(development_speakers & test_speakers)
    if speaker_overlap:
        raise RuntimeError(
            "Official-test speaker leakage detected: " + ", ".join(speaker_overlap)
        )

    development_audio = set(development_df["audio_relpath"].astype(str))
    test_audio = set(test_df["audio_relpath"].astype(str))
    audio_overlap = development_audio & test_audio
    if audio_overlap:
        raise RuntimeError(
            f"Audio overlap detected between development and official test: "
            f"{len(audio_overlap)} file(s)."
        )

    development_path = output_dir / "development_full.csv"
    test_path = output_dir / "test.csv"
    development_df.to_csv(development_path, index=False)
    test_df.to_csv(test_path, index=False)

    audit_record = {
        "stage": "dataset_extraction_and_integrity_audit",
        "dataset": "NCHLT Northern Sotho / Sepedi Speech Corpus",
        "development": development_audit,
        "official_test": test_audit,
        "documented_counts": {
            "development_utterances": DEVELOPMENT_UTTERANCES,
            "official_test_utterances": OFFICIAL_TEST_UTTERANCES,
            "total_utterances": TOTAL_UTTERANCES,
            "official_test_speakers": OFFICIAL_TEST_SPEAKERS,
        },
        "development_test_speaker_overlap": 0,
        "development_test_audio_overlap": 0,
        "note": (
            "The NCHLT corpus is not redistributed by this repository. Generated "
            "manifests contain local paths and are excluded from Git tracking."
        ),
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("NCHLT NORTHERN SOTHO DATASET AUDIT")
    print("Development utterances:", len(development_df))
    print("Official-test utterances:", len(test_df))
    print("Official-test speakers:", test_df["speaker"].nunique())
    print("Speaker overlap between development and official test: 0")
    print("Audio overlap between development and official test: 0")
    print("Saved:", development_path)
    print("Saved:", test_path)


if __name__ == "__main__":
    main()
