"""
Prepare the NCHLT Northern Sotho/Sepedi dataset for XLS-R ASR fine-tuning.

This script:
1. Parses NCHLT XML transcription files.
2. Matches transcriptions to existing audio files.
3. Cleans transcript text.
4. Creates train/validation/test CSV files.
5. Builds a CTC character-level vocabulary file.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


NORTHERN_SOTHO_CHARS_PATTERN = r"[^\w\sšêôáéíóúàèìòùäëïöü]"


def clean_text(text: str) -> str:
    """Clean transcription text for character-level CTC training."""
    text = str(text).lower().strip()
    text = re.sub(NORTHERN_SOTHO_CHARS_PATTERN, "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_nchlt_xml(xml_path: Path, dataset_dir: Path) -> pd.DataFrame:
    """Parse one NCHLT XML file and return valid audio-text pairs."""
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    content = xml_path.read_text(encoding="utf-8", errors="ignore")
    pattern = r'<recording audio="([^"]+)".*?>\s*<orth>(.*?)</orth>\s*</recording>'
    matches = re.findall(pattern, content, flags=re.DOTALL)

    rows = []
    for audio_rel, transcript in matches:
        audio_rel = audio_rel.replace("\\", "/")
        if audio_rel.startswith("nchlt_nso/"):
            audio_rel = audio_rel.replace("nchlt_nso/", "", 1)

        audio_path = dataset_dir / audio_rel
        sentence = clean_text(transcript)

        if audio_path.exists() and sentence:
            rows.append({"audio_path": str(audio_path), "sentence": sentence})

    return pd.DataFrame(rows)


def create_vocabulary(train_df: pd.DataFrame, output_path: Path) -> dict:
    """Create a character vocabulary from training transcripts."""
    all_text = " ".join(train_df["sentence"].astype(str).tolist())
    vocab_chars = sorted(set(all_text))

    vocab = {char: idx for idx, char in enumerate(vocab_chars)}

    # CTC convention: represent spaces using vertical bar.
    if " " in vocab:
        space_id = vocab.pop(" ")
        vocab["|"] = space_id

    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NCHLT Northern Sotho ASR dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to nchlt_nso dataset folder.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output folder for processed CSVs and vocab.")
    parser.add_argument("--validation_size", type=float, default=0.10, help="Validation split from training XML.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"
    processed_dir = output_dir / "processed"
    split_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    trn_xml = dataset_dir / "transcriptions" / "nchlt_nso.trn.xml"
    tst_xml = dataset_dir / "transcriptions" / "nchlt_nso.tst.xml"

    print(f"Dataset directory: {dataset_dir}")
    print(f"Training XML: {trn_xml}")
    print(f"Testing XML: {tst_xml}")

    train_metadata = parse_nchlt_xml(trn_xml, dataset_dir)
    test_df = parse_nchlt_xml(tst_xml, dataset_dir)

    train_df, val_df = train_test_split(
        train_metadata,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(processed_dir / "all_metadata.csv", index=False)

    vocab = create_vocabulary(train_df, processed_dir / "vocab.json")

    summary = pd.DataFrame([
        {"split": "train", "samples": len(train_df)},
        {"split": "validation", "samples": len(val_df)},
        {"split": "test", "samples": len(test_df)},
        {"split": "total", "samples": len(full_df)},
    ])
    summary.to_csv(processed_dir / "dataset_summary.csv", index=False)

    print("Dataset preparation complete.")
    print(summary)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
