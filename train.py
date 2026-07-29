import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SlidingWindowPhishingDataset
from src.models.deberta_model import DebertaSlidingWindowClassifier
from src.utils.metrics import compute_evaluation_metrics, print_metrics_summary


def main():
    # 1. Load Central Config
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("🚀 Initializing Training Pipeline for Experiment 1...")

    # 2. Dynamic Dataset Path Detection (Kaggle GPU vs. Local CPU)
    if os.path.exists(config["data"]["kaggle_parquet"]):
        data_path = config["data"]["kaggle_parquet"]
        print(f"📦 Kaggle Environment Detected! Path: {data_path}")
    elif os.path.exists(config["data"]["local_parquet"]):
        data_path = config["data"]["local_parquet"]
        print(f"💻 Local Environment Detected! Path: {data_path}")
    else:
        raise FileNotFoundError(
            f"❌ Parquet file not found at local or Kaggle paths!"
        )

    # 3. Load Dataset & Train/Val Split
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} total email records.")

    train_df, val_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df["label"],
    )
    print(f"  ├─ Training samples:   {len(train_df)}")
    print(f"  └─ Validation samples: {len(val_df)}")

    # 4. Instantiate Datasets & DataLoaders
    cfg_deb = config["deberta"]

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
        batch_size=cfg_deb["batch_size"],
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_deb["batch_size"],
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 5. Device Setup & Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Compute Device: {device}")

    model = DebertaSlidingWindowClassifier(
        model_name=cfg_deb["model_name"], num_classes=2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_deb["learning_rate"]),
        weight_decay=cfg_deb["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision Scaler for fast CUDA forward/backward passes
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 6. Training Loop
    epochs = cfg_deb["epochs"]
    best_f1 = 0.0

    for epoch in range(epochs):
        print(f"\n================ Epoch {epoch + 1}/{epochs} ================")
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"  └─ Average Training Loss: {epoch_loss:.4f}")

        # 7. Validation Evaluation Loop
        model.eval()
        val_preds, val_probs, val_targets = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    preds = torch.argmax(logits, dim=-1)

                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(labels.numpy())

        # 8. Compute and Display Metrics
        metrics = compute_evaluation_metrics(
            y_true=np.array(val_targets),
            y_pred=np.array(val_preds),
            y_prob=np.array(val_probs),
        )

        print_metrics_summary(
            f"Exp 1: DeBERTa Text-Only (Epoch {epoch + 1})", metrics
        )

        # 9. Save Best Model Checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs("models", exist_ok=True)
            checkpoint_path = "models/best_deberta_exp1.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved new best model checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()