#!/usr/bin/env python3
"""
train binary classification head lang1 - lang2 
python detect_trainer2.py --output_dir="detecthead0" --data_dir="data/en_zh" --lang1="en" --lang2="zh"
This script now accepts hyperparameters from the command line:
    --learning_rate
    --num_train_epochs
    --save_total_limit
    --output_dir
    --per_device_train_batch_size
    --per_device_eval_batch_size
    --data_dir (path to ingested DatasetDict)
    --lang1 (first language code, label 0)
    --lang2 (second language code, label 1)

=== Dataset Size Summary ===
Assumes ingested data with 'train', 'valid', 'test' splits.
Labels: 0 for lang1, 1 for lang2.
Uses 'train' for training, 'valid' for evaluation.
Subsamples larger class to balance with smaller if unbalanced.
"""

import os
import sys
import argparse
import torchaudio
import torch
import numpy as np
import torch.nn as nn

from datasets import load_from_disk, Audio, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperModel,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny Whisper-based language detector head (binary or multiclass)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the ingested DatasetDict directory.",
    )
    parser.add_argument(
        "--lang1",
        type=str,
        required=False,
        help="Optional: first language code for binary mode (label 0).",
    )
    parser.add_argument(
        "--lang2",
        type=str,
        required=False,
        help="Optional: second language code for binary mode (label 1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the model and checkpoints.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep. Older ones will be deleted.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients before updating.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed-precision (FP16).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "none"],
        help="Where to report logs (wandb or none).",
    )
    return parser.parse_args()


def subsample_larger_class(dataset, lang1, lang2, label_col="labels"):
    """Subsample the larger class to match the size of the smaller class."""
    lang1_indices = [i for i, ex in enumerate(dataset) if ex[label_col] == 0]
    lang2_indices = [i for i, ex in enumerate(dataset) if ex[label_col] == 1]
    n_lang1 = len(lang1_indices)
    n_lang2 = len(lang2_indices)
    min_size = min(n_lang1, n_lang2)
    if n_lang1 > min_size:
        # Subsample lang1
        lang1_indices = np.random.choice(lang1_indices, min_size, replace=False)
    if n_lang2 > min_size:
        # Subsample lang2
        lang2_indices = np.random.choice(lang2_indices, min_size, replace=False)
    selected_indices = sorted(lang1_indices + lang2_indices)
    return dataset.select(selected_indices)


def main():
    args = parse_args()

    # Validate binary vs multiclass mode
    if (args.lang1 is None) ^ (args.lang2 is None):
        print("[Error] Provide both --lang1 and --lang2 for binary mode, or neither for multiclass.")
        sys.exit(1)
    binary_mode = (args.lang1 is not None and args.lang2 is not None)

    os.makedirs(args.output_dir, exist_ok=True)

    TARGET_SR = 16000
    # Defining the consistent schema (ingested data should already be 16kHz mono float32)
    audio_feature = Audio(sampling_rate=16000, mono=True, decode=True)

    # 1) Load processor (tokenizer + feature_extractor)
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # 2) Load the ingested DatasetDict
    raw_datasets = load_from_disk(args.data_dir)
    print(f"Loaded DatasetDict from {args.data_dir}: {raw_datasets}")

    # If binary mode, filter to only lang1 and lang2
    if binary_mode:
        def filter_lang(example):
            return example["lang"] in [args.lang1, args.lang2]
        for split_name in ["train", "valid", "test"]:
            if split_name in raw_datasets:
                raw_datasets[split_name] = raw_datasets[split_name].filter(filter_lang)

    # Print dataset sizes per language
    print("\n=== Dataset Size Summary ===")
    for split_name, split in raw_datasets.items():
        counts = {}
        for ex in split:
            lang = ex.get("lang")
            counts[lang] = counts.get(lang, 0) + 1
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"{split_name} size: {len(split)} | {counts_str}")

    # Use 'train' for training, 'valid' for evaluation, 'test' for final eval if available
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["valid"] if "valid" in raw_datasets else raw_datasets["test"]
    test_dataset = raw_datasets["test"] if "test" in raw_datasets else eval_dataset

    # Cast audio column if needed (ingested should already match)
    train_dataset = train_dataset.cast_column("audio", audio_feature)
    eval_dataset = eval_dataset.cast_column("audio", audio_feature)
    test_dataset = test_dataset.cast_column("audio", audio_feature)

    # 3) Build classes and mapping
    if binary_mode:
        class_names = [args.lang1, args.lang2]  # preserve order for backward compatibility
    else:
        # Infer classes from train split
        classes = set(ex["lang"] for ex in train_dataset)
        class_names = sorted(classes)
    class_to_id = {lang: i for i, lang in enumerate(class_names)}
    num_classes = len(class_to_id)

    # If multiclass, ensure eval/test only contain known classes from train
    if not binary_mode:
        known = set(class_to_id.keys())
        eval_dataset = eval_dataset.filter(lambda ex: ex["lang"] in known)
        test_dataset = test_dataset.filter(lambda ex: ex["lang"] in known)

    # Add labels using mapping
    def tag_batch(example):
        example["labels"] = class_to_id[example["lang"]]
        return example

    train_labeled = train_dataset.map(tag_batch)
    eval_labeled = eval_dataset.map(tag_batch)
    test_labeled = test_dataset.map(tag_batch)

    # Shuffle train
    train_labeled = train_labeled.shuffle(seed=1024)

    # Class distribution in train
    label_counts = {}
    for ex in train_labeled:
        label_counts[ex["labels"]] = label_counts.get(ex["labels"], 0) + 1
    dist_str = ", ".join(
        f"{class_names[i]}:{label_counts.get(i, 0)}" for i in range(num_classes)
    )
    print(f"Train class distribution: {dist_str}")

    # Create DatasetDict for Trainer
    raw_datasets = DatasetDict({
        "train": train_labeled,
        "evaluation": eval_labeled,
    })

    print(f"Combined dataset: train (shuffled)={len(raw_datasets['train'])}, eval={len(raw_datasets['evaluation'])}")

    # 5) Preprocess: convert each audio→mel features, store in "input_features"
    def preprocess(batch):
        audio_array = batch["audio"]["array"]
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array)
        feats = processor.feature_extractor(
            audio_array,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).input_features[0]
        batch["input_features"] = feats
        return batch

    print("Preprocessing audio features...")
    raw_datasets = raw_datasets.map(
        preprocess,
        remove_columns=["audio", "text", "lang", "accent"],  # Remove unused columns
        batched=False,
        desc="Extracting features",
    )
    test_labeled = test_labeled.map(
        preprocess,
        remove_columns=["audio", "text", "lang", "accent"],  # Remove unused columns
        batched=False,
        desc="Extracting features",
    )

    # 6) Build the LangDetectHead on top of Whisper's encoder
    class LangDetectHead(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            whisper_model = WhisperModel.from_pretrained("openai/whisper-small")
            # Freeze all Whisper weights
            for p in whisper_model.parameters():
                p.requires_grad = False
            self.encoder = whisper_model.encoder
            self.classifier = nn.Sequential(
                nn.Linear(whisper_model.config.d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes),
            )
            print(f"[Info] Whisper encoder d_model = {whisper_model.config.d_model}")

        def forward(self, input_features, labels=None):
            # Pass through frozen encoder
            with torch.no_grad():
                hidden = self.encoder(input_features).last_hidden_state  # (B, T, D)
            pooled = hidden.mean(dim=1)  # (B, D)
            logits = self.classifier(pooled)  # (B, C)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}

    print("Initializing LangDetectHead model...")
    model = LangDetectHead(num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Total parameters: {total_params:,}")
    print(f"[Info] Trainable parameters: {trainable_params:,}")

    # 7) Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # 8) Construct TrainingArguments using command-line hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        fp16=args.fp16,
        report_to=args.report_to if args.report_to != "none" else None,
    )

    print(f"[Info] TrainingArguments:\n  output_dir = {args.output_dir}\n" +
          f"  learning_rate = {args.learning_rate}\n" +
          f"  num_train_epochs = {args.num_train_epochs}\n" +
          f"  save_total_limit = {args.save_total_limit}\n" +
          f"  per_device_train_batch_size = {args.per_device_train_batch_size}\n" +
          f"  per_device_eval_batch_size = {args.per_device_eval_batch_size}\n" +
          f"  fp16 = {args.fp16}\n" +
          f"  report_to = {args.report_to}\n")

    # 9) Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["evaluation"],
        compute_metrics=compute_metrics,
    )

    print("[Info] Starting training …")
    trainer.train()

    # Final evaluation on test set if available
    print("\n=== Final Test Evaluation ===")
    test_results = trainer.evaluate(test_labeled)
    for key, value in test_results.items():
        print(f"test_{key}: {value}")

    print("\n=== Final Evaluation ===")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value}")


    print(f"\n[Info] Saving model to {args.output_dir} …")
    trainer.save_model()

if __name__ == "__main__":
    main()