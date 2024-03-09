#!/usr/bin/env python3

import os
import argparse
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_scheduler,
)
import torch
from torch.utils.data import DataLoader
import datasets
import evaluate  # pip install evaluate scikit-learn
import numpy as np
import time
from tqdm.auto import tqdm
import wandb

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
        default="pytorch",
    )
    parser.add_argument(
        "--num-epochs",
        "-n",
        type=int,
        help="number of epochs to train for",
        default=4,
    )
    parser.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help=f"Enable wandb logging",
    )

    args = parser.parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"  # don't sync
    else:
        os.environ["WANDB_MODE"] = "online"

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    print(
        f"\nstarting finetuning with method '{args.method}' for {args.num_epochs} epochs"
    )
    start_time = time.perf_counter()
    METHODS[args.method](model, tokenizer, args)
    end_time = time.perf_counter()
    print(f"finetuning complete in {((end_time - start_time) / 60):.2f} minutes!")


def finetune_trainer(model, tokenizer, args):
    """
    Finetune using huggingface.Trainer class.
    Automatically uses wandb if os.environ["WANDB_MODE"] != "offline"
    https://huggingface.co/learn/nlp-course/chapter3/3
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_datasets(tokenizer, for_pytorch=False)

    training_args = TrainingArguments(
        args.output_dir,
        # evaluate every epoch
        evaluation_strategy="epoch",
        num_train_epochs=args.num_epochs,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_trainer,
    )
    print(f"\nfinetuning with Trainer...", flush=True)
    trainer.train()

    model.save_pretrained(args.output_dir)
    print(f"model saved to '{args.output_dir}'")


def finetune_pytorch(model, tokenizer, args):
    """
    Finetune using pytorch directly.
    https://huggingface.co/learn/nlp-course/chapter3/4
    Note: this could be tweaked to use accelerate to enable training on multiple GPUs:
    https://huggingface.co/learn/nlp-course/chapter3/4#supercharge-your-training-loop-with-accelerate
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_datasets(tokenizer, for_pytorch=True)

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

    device = get_device()
    model.to(device)

    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    run = wandb.init(config=args)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        eval_metrics = compute_metrics_pytorch(model, eval_dataloader)
        train_metrics = {}
        if epoch % 4 == 0 or epoch + 1 == args.num_epochs:
            # expensive computation:
            train_metrics = compute_metrics_pytorch(model, train_dataloader)

        wandb_metrics = {
            "train_loss": total_loss,
            **{f"{name}/eval": value for name, value in eval_metrics.items()},
            **{f"{name}/train": value for name, value in train_metrics.items()},
        }
        wandb.log(
            wandb_metrics,
            step=epoch,
        )

    print("\nfinal metrics:\n", wandb_metrics)
    wandb.finish()


def get_datasets(tokenizer, for_pytorch: bool):
    raw_datasets = datasets.load_dataset("glue", "mrpc")

    def tokenize_function(example):
        # intentionally not padding here yet (each batch should have the minimum padding necessary)
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # https://huggingface.co/learn/nlp-course/chapter3/4#prepare-for-training
    if for_pytorch:
        # ensure we only have columns the pytorch model can accept
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    return tokenized_datasets


def compute_metrics_trainer(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_pytorch(model, dataloader):
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()


def get_device() -> str:
    """Returns the device for PyTorch to use."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # mac MPS support: https://pytorch.org/docs/stable/notes/mps.html
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            device = "mps"
    return device


if __name__ == "__main__":
    main()
