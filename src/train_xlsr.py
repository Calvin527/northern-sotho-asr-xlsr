"""
Fine-tune facebook/wav2vec2-xls-r-300m for Northern Sotho/Sepedi ASR.

This script expects CSV files produced by prepare_dataset.py:
- data/splits/train.csv
- data/splits/validation.csv
- data/processed/vocab.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import librosa
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def load_audio(path: str, sampling_rate: int = 16000) -> np.ndarray:
    speech, _ = librosa.load(path, sr=sampling_rate)
    return speech


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XLS-R 300M for Northern Sotho ASR.")
    parser.add_argument("--data_dir", type=str, default="data", help="Folder created by prepare_dataset.py.")
    parser.add_argument("--output_dir", type=str, default="asr_model_xlsr_nso", help="Model output directory.")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=19000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 if GPU supports it.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    train_csv = data_dir / "splits" / "train.csv"
    val_csv = data_dir / "splits" / "validation.csv"
    vocab_path = data_dir / "processed" / "vocab.json"

    if not train_csv.exists() or not val_csv.exists() or not vocab_path.exists():
        raise FileNotFoundError("Missing train/validation CSV or vocab.json. Run prepare_dataset.py first.")

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_vocab_path = tokenizer_dir / "vocab.json"
    tokenizer_vocab_path.write_text(vocab_path.read_text(encoding="utf-8"), encoding="utf-8")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(tokenizer_vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=args.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    def prepare_batch(batch):
        speech = load_audio(batch["audio_path"], sampling_rate=args.sampling_rate)
        batch["input_values"] = processor(speech, sampling_rate=args.sampling_rate).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(prepare_batch, remove_columns=val_ds.column_names)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=30,
        max_steps=args.max_steps,
        fp16=args.fp16,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    (output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Training complete. Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
