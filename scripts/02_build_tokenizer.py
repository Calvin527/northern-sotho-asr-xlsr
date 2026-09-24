from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
from transformers import Wav2Vec2CTCTokenizer
from common import normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(args.train_csv), pd.read_csv(args.validation_csv)], ignore_index=True)
    texts = df["reference"].fillna("").map(normalize_text)
    chars = sorted(set("".join(texts.tolist())))
    if " " in chars:
        chars.remove(" ")
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    tokenizer.save_pretrained(args.output_dir)

    print("Vocabulary size:", len(vocab))
    if len(vocab) != 32:
        print("WARNING: final report documents a 32-token vocabulary; check manifests/normalization.")


if __name__ == "__main__":
    main()
