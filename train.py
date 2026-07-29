import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import (
    SlidingWindowPhishingDataset,
    packed_sliding_window_collate_fn,
)
from src.models.deberta_model import DebertaSlidingWindowClassifier
from src.utils.metrics import (
    compute_evaluation_metrics,
    print_metrics_summary,
    save_experiment_artifacts,
)


def main():
    is_dry_run = "--dry-run" in sys.argv

    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(
        f"🚀 Initializing Training Pipeline for Experiment 1 (DeBERTa)... {'[DRY RUN]' if is_dry_run else ''}"
    )

    if os.path.exists(config["data"]["kaggle_parquet"]):
        data_path = config["data"]["kaggle_parquet"]
    elif os.path.exists(config["data"]["local_parquet"]):
        data_path = config["data"]["local_parquet"]
    else:
        raise FileNotFoundError(
            "❌ Parquet file not found at local or Kaggle paths!"
        )

    df = pd.read_parquet(data_path)
    if is_dry_run:
        df = df.head(100)

    train_df, val_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df["label"],
    )

    cfg_deb = config["deberta"]
    batch_size = 4 if is_dry_run else cfg_deb["batch_size"]
    epochs = 1 if is_dry_run else cfg_deb["epochs"]

    train_dataset = SlidingWindowPhishingDataset(
        texts=train_df["full_text"].values,
        labels=train_df["label"].values,
        tokenizer_name=cfg_deb["model_name"],
        max_length=cfg_deb["max_length"],
        stride=cfg_deb["stride"],
    )

    val_dataset = SlidingWindowPhishingDataset(
        texts=val_df["full_text"].values,
        labels=val_df["label"].values,
        tokenizer_name=cfg_deb["model_name"],
        max_length=cfg_deb["max_length"],
        stride=cfg_deb["stride"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=packed_sliding_window_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=packed_sliding_window_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaSlidingWindowClassifier(
        model_name=cfg_deb["model_name"], num_classes=2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_deb["learning_rate"]),
        weight_decay=cfg_deb["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    # -------------------------------------------------------------------------
    # Auto-Resume Checkpoint Engine
    # -------------------------------------------------------------------------
    ckpt_dir = config["outputs"]["model_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint_exp1.pt")

    if os.path.exists(latest_ckpt_path) and not is_dry_run:
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_f1 = checkpoint["best_f1"]
        history = checkpoint["history"]
        print(
            f"🔄 Resuming Experiment 1 from Epoch {start_epoch + 1}/{epochs} (Best F1 so far: {best_f1:.4f})"
        )

    latest_val_targets, latest_val_preds, latest_val_probs = [], [], []
    latest_metrics = {}

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Exp 1 Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_indices = batch["batch_indices"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_indices=batch_indices,
                batch_size=batch["batch_size"],
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if is_dry_run:
                break

        history["train_loss"].append(running_loss / len(train_loader))

        model.eval()
        val_running_loss = 0.0
        val_preds, val_probs, val_targets = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                batch_indices = batch["batch_indices"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch_indices=batch_indices,
                    batch_size=batch["batch_size"],
                )
                loss = criterion(logits, labels)
                val_running_loss += loss.item()

                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)

                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                if is_dry_run:
                    break

        history["val_loss"].append(val_running_loss / len(val_loader))
        metrics = compute_evaluation_metrics(
            y_true=np.array(val_targets),
            y_pred=np.array(val_preds),
            y_prob=np.array(val_probs),
        )
        history["val_f1"].append(metrics["f1"])
        print_metrics_summary(f"Exp 1: DeBERTa (Epoch {epoch + 1})", metrics)

        latest_val_targets, latest_val_preds, latest_val_probs = (
            np.array(val_targets),
            np.array(val_preds),
            np.array(val_probs),
        )
        latest_metrics = metrics

        # Save latest epoch checkpoint for auto-resuming
        if not is_dry_run:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": max(best_f1, metrics["f1"]),
                    "history": history,
                },
                latest_ckpt_path,
            )

        # Save best model checkpoint
        if metrics["f1"] >= best_f1 and not is_dry_run:
            best_f1 = metrics["f1"]
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, "best_deberta_exp1.pt"),
            )

    if not is_dry_run:
        save_experiment_artifacts(
            output_dir=os.path.join(config["outputs"]["results_dir"], "exp1"),
            history=history,
            final_metrics=latest_metrics,
            y_true=latest_val_targets,
            y_pred=latest_val_preds,
            y_prob=latest_val_probs,
            exp_name="exp1",
        )


if __name__ == "__main__":
    main()