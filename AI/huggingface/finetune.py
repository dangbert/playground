#!/usr/bin/env python3

import os
import argparse
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import datasets
import evaluate  # pip install evaluate scikit-learn
import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT = "bert-base-uncased"


def main():
    parser = argparse.ArgumentParser(
        description="fine tune BERT on the glue mrpc task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="path of folder to output markdown and csv files (e.g. './output')",
        default=os.path.join(SCRIPT_DIR, "output"),
    )
    METHODS = {
        "trainer": finetune_trainer,
        "pytorch": finetune_pytorch,
    }
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help=f"Method to fine tune with (one of {','.join(list(METHODS.keys()))})",
        default=METHODS["trainer"],
    )
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help=f"Enable wandb logging",
    )

    ops = 0  # num operations performed
    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"  # don't sync
    else:
        os.environ["WANDB_MODE"] = "online"
    # os.environ["WANDB_ENTITY"] = "dan-thesis"
    # os.environ["WANDB_PROJECT"] = "playground"

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    print(f"starting finetuning with method '{args.method}'")
    start_time = time.perf_counter()
    METHODS[args.method(model, tokenizer, args.output_dir)]
    end_time = time.perf_counter()
    print(f"finetuning completed in {((end_time - start_time) / 60):.2f} minutes")


def finetune_trainer(model, tokenizer, output_dir: str):
    """Finetune using huggingface.Trainer class."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_datasets(tokenizer)

    training_args = TrainingArguments(
        output_dir,
        # evaluate every epoch
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    print(f"\nfinetuning with Trainer...", flush=True)
    trainer.train()


def finetune_pytorch(model, tokenizer, output_dir: str):
    """Finetune using pytorch directly."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_datasets(tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=3e-5)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def get_datasets(tokenizer):
    raw_datasets = datasets.load_dataset("glue", "mrpc")

    def tokenize_function(example):
        # intentionally not padding here yet (each batch should have the minimum padding necessary)
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    return tokenized_datasets


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    main()
