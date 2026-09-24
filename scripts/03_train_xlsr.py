from __future__ import annotations

import argparse, inspect, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import cer, wer
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
from common import normalize_text

MODEL_ID = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class SpeechDataset(Dataset):
    def __init__(self, csv_path: Path, processor: Wav2Vec2Processor):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.processor = processor
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        text = normalize_text(row["reference"])
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_attention_mask=True)
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids
        return {"input_values": inputs.input_values[0], "attention_mask": inputs.attention_mask[0], "labels": labels}


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict]):
        input_features = [{"input_values": f["input_values"], "attention_mask": f["attention_mask"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--validation-csv", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    set_seed()

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.tokenizer_dir)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, ctc_loss_reduction="mean", ctc_zero_infinity=True, ignore_mismatched_sizes=True)

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()

    train_ds = SpeechDataset(args.train_csv, processor)
    val_ds = SpeechDataset(args.validation_csv, processor)
    collator = DataCollatorCTCWithPadding(processor)

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy(); label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = [normalize_text(x) for x in processor.batch_decode(pred_ids)]
        references = [normalize_text(x) for x in processor.batch_decode(label_ids, group_tokens=False)]
        return {"wer": wer(references, predictions), "cer": cer(references, predictions)}

    kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        max_steps=19000,
        eval_steps=900,
        save_steps=900,
        logging_steps=100,
        fp16=True,
        seed=SEED,
        data_seed=SEED,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"] = "steps"
    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.output_dir / "final_model")
    processor.save_pretrained(args.output_dir / "final_model")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation metric:", trainer.state.best_metric)
    print("Final report selected checkpoint-8100; a different checkpoint may indicate split/environment differences.")


if __name__ == "__main__":
    main()
