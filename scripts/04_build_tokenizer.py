from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import Wav2Vec2CTCTokenizer

from common import ensure_columns, normalize_text

# Exact 32-token inventory verified in the executed experiment and documented
# by the final research methodology.
EXPECTED_CHARACTERS = list("abcdefghijklmnopqrstuvwxyz") + ["š"]
EXPECTED_VOCAB = {
    "|": 0,
    **{character: index for index, character in enumerate(EXPECTED_CHARACTERS, start=1)},
    "[UNK]": 28,
    "[PAD]": 29,
    "<s>": 30,
    "</s>": 31,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_normalized(csv_path: Path, split_name: str) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{split_name} CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ensure_columns(df, ["reference"])
    df = df.copy()
    df["reference_norm"] = df["reference"].fillna("").map(normalize_text)

    empty_count = int(df["reference_norm"].eq("").sum())
    if empty_count:
        raise RuntimeError(
            f"{split_name} contains {empty_count} empty transcript(s) after normalisation."
        )

    return df


def _character_set(series: pd.Series) -> set[str]:
    return {
        character
        for transcript in series.astype(str)
        for character in transcript
        if character != " "
    }


def _round_trip_audit(tokenizer: Wav2Vec2CTCTokenizer, texts: pd.Series) -> int:
    samples = texts.dropna().astype(str).str.strip()
    samples = samples[samples.ne("")].head(5).tolist()
    if not samples:
        raise RuntimeError("No non-empty training transcripts are available for tokenizer audit.")

    for transcript in samples:
        encoded = tokenizer(transcript, add_special_tokens=False).input_ids
        decoded = tokenizer.decode(
            encoded,
            group_tokens=False,
            skip_special_tokens=True,
        )
        decoded = " ".join(decoded.strip().split())
        if transcript != decoded:
            raise RuntimeError(
                "Tokenizer round-trip verification failed: "
                f"reference={transcript!r}, decoded={decoded!r}"
            )

        for token in ("<s>", "</s>", "[PAD]"):
            if EXPECTED_VOCAB[token] in encoded:
                raise RuntimeError(
                    f"Special token {token} was unexpectedly added to CTC target labels."
                )

    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit the 32-token Northern Sotho character-level CTC tokenizer. "
            "The vocabulary is derived from training text only; validation and test "
            "manifests are used only for OOV auditing."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_df = _load_normalized(args.train_csv, "Training")
    validation_df = _load_normalized(args.validation_csv, "Validation")
    test_df = _load_normalized(args.test_csv, "Official test")

    # Critical methodological constraint: vocabulary construction uses TRAINING TEXT ONLY.
    training_characters = sorted(_character_set(train_df["reference_norm"]))
    expected_characters = sorted(EXPECTED_CHARACTERS)

    if training_characters != expected_characters:
        missing = sorted(set(EXPECTED_CHARACTERS) - set(training_characters))
        unexpected = sorted(set(training_characters) - set(EXPECTED_CHARACTERS))
        raise RuntimeError(
            "Training character inventory does not match the verified 27-symbol "
            "Northern Sotho inventory. "
            f"Missing={missing}; unexpected={unexpected}. "
            "Check the manifest and text-normalisation stage rather than silently "
            "changing the published vocabulary."
        )

    validation_oov = sorted(
        _character_set(validation_df["reference_norm"]) - set(training_characters)
    )
    test_oov = sorted(_character_set(test_df["reference_norm"]) - set(training_characters))

    if validation_oov:
        raise RuntimeError(f"Validation contains OOV characters: {validation_oov}")
    if test_oov:
        raise RuntimeError(f"Official test contains OOV characters: {test_oov}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / "vocab.json"
    manifest_path = args.output_dir / "tokenizer_manifest.json"

    # Fixed IDs reproduce the verified vocabulary exactly:
    # | -> 0, a-z -> 1..26, š -> 27, then the four control tokens.
    vocab_path.write_text(
        json.dumps(EXPECTED_VOCAB, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token="<s>",
        eos_token="</s>",
        do_lower_case=False,
    )

    if len(tokenizer) != 32:
        raise RuntimeError(f"Expected tokenizer size 32, got {len(tokenizer)}")
    if tokenizer.word_delimiter_token != "|":
        raise RuntimeError("Tokenizer word delimiter is not '|'.")
    if tokenizer.unk_token_id != EXPECTED_VOCAB["[UNK]"]:
        raise RuntimeError("Unexpected [UNK] token ID.")
    if tokenizer.pad_token_id != EXPECTED_VOCAB["[PAD]"]:
        raise RuntimeError("Unexpected [PAD] token ID.")
    if tokenizer.bos_token_id != EXPECTED_VOCAB["<s>"]:
        raise RuntimeError("Unexpected <s> token ID.")
    if tokenizer.eos_token_id != EXPECTED_VOCAB["</s>"]:
        raise RuntimeError("Unexpected </s> token ID.")

    round_trip_samples = _round_trip_audit(tokenizer, train_df["reference_norm"])
    tokenizer.save_pretrained(args.output_dir)

    # Reload locally so publication does not depend on an in-memory tokenizer object.
    reloaded = Wav2Vec2CTCTokenizer.from_pretrained(
        args.output_dir,
        local_files_only=True,
    )
    if len(reloaded) != 32:
        raise RuntimeError("Saved tokenizer failed reload validation.")

    saved_vocab = json.loads((args.output_dir / "vocab.json").read_text(encoding="utf-8"))
    if saved_vocab != EXPECTED_VOCAB:
        raise RuntimeError("Saved vocabulary differs from the verified 32-token mapping.")

    manifest = {
        "provenance": (
            "Clean reproducibility implementation aligned with the verified experimental "
            "tokenizer design; not an archival copy of the original notebook."
        ),
        "vocabulary_source": "training text only",
        "training_utterances": int(len(train_df)),
        "validation_utterances": int(len(validation_df)),
        "official_test_utterances": int(len(test_df)),
        "training_character_symbols": len(training_characters),
        "validation_oov_characters": validation_oov,
        "official_test_oov_characters": test_oov,
        "vocabulary_size": len(EXPECTED_VOCAB),
        "word_delimiter_token": "|",
        "unknown_token": "[UNK]",
        "padding_token": "[PAD]",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "ctc_labels_add_special_tokens": False,
        "normalisation": {
            "unicode": "NFC",
            "lowercase": True,
            "preserve_unicode_letters_and_combining_marks": True,
            "replace_punctuation_digits_symbols_and_controls_with_spaces": True,
            "collapse_whitespace": True,
        },
        "round_trip_samples_tested": round_trip_samples,
        "vocab_sha256": _sha256(args.output_dir / "vocab.json"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NORTHERN SOTHO TOKENIZER AUDIT")
    print("Vocabulary source: training text only")
    print("Training character symbols:", len(training_characters))
    print("Validation OOV characters: 0")
    print("Official-test OOV characters: 0")
    print("Vocabulary size:", len(reloaded))
    print("Word delimiter token: '|' (ID 0)")
    print("Special token IDs: [UNK]=28, [PAD]=29, <s>=30, </s>=31")
    print("Round-trip samples passed:", round_trip_samples)
    print("Saved tokenizer:", args.output_dir)


if __name__ == "__main__":
    main()
