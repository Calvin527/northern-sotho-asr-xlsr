from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ensure_columns, normalize_text

EXPECTED_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzš")


def preprocess_manifest(input_csv: Path, output_csv: Path, split_name: str) -> dict:
    df = pd.read_csv(input_csv)
    ensure_columns(df, ["reference"])
    df = df.copy()

    df["reference_norm"] = df["reference"].fillna("").map(normalize_text)
    empty_after = int(df["reference_norm"].eq("").sum())
    if empty_after:
        raise RuntimeError(
            f"{split_name} contains {empty_after} empty transcript(s) after normalisation."
        )

    characters = {
        char
        for text in df["reference_norm"].astype(str)
        for char in text
        if char != " "
    }
    unexpected = sorted(characters - EXPECTED_CHARACTERS)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "split": split_name,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "utterances": int(len(df)),
        "empty_after_normalisation": empty_after,
        "unique_non_space_characters": sorted(characters),
        "unexpected_characters_relative_to_verified_inventory": unexpected,
        "normalisation": {
            "unicode": "NFC",
            "lowercase": True,
            "punctuation_digits_symbols_controls": "replaced with spaces",
            "whitespace": "collapsed",
            "northern_sotho_diacritics": "preserved (including š)",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise Northern Sotho transcripts for CTC ASR training/evaluation."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests/processed"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        preprocess_manifest(args.train_csv, output_dir / "train.csv", "train"),
        preprocess_manifest(
            args.validation_csv,
            output_dir / "validation.csv",
            "validation",
        ),
        preprocess_manifest(args.test_csv, output_dir / "test.csv", "official_test"),
    ]

    train_chars = set(records[0]["unique_non_space_characters"])
    if train_chars != EXPECTED_CHARACTERS:
        missing = sorted(EXPECTED_CHARACTERS - train_chars)
        unexpected = sorted(train_chars - EXPECTED_CHARACTERS)
        raise RuntimeError(
            "Training character inventory does not match the verified 27-character "
            f"inventory. Missing={missing}; unexpected={unexpected}."
        )

    for record in records[1:]:
        oov = sorted(set(record["unique_non_space_characters"]) - train_chars)
        record["oov_characters_relative_to_training"] = oov
        if oov:
            raise RuntimeError(
                f"{record['split']} contains OOV characters relative to training: {oov}"
            )

    (output_dir / "text_preprocessing_audit.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TEXT PREPROCESSING COMPLETE")
    for record in records:
        print(f"{record['split']}: {record['utterances']} utterances")
    print("Verified training character inventory: a-z plus š")
    print("Validation/test OOV characters: 0")


if __name__ == "__main__":
    main()
