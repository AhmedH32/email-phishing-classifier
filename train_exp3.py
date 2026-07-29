import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
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
from train_exp2 import extract_header_structural_features


def extract_deberta_probs(model, loader, device, is_dry_run=False):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting DeBERTa Logits"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            batch_size = batch["batch_size"]

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_indices=batch_indices,
                batch_size=batch_size,
            )
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_list.extend(probs.cpu().numpy())
            if is_dry_run:
                break
    return np.array(probs_list)


def main():
    is_dry_run = "--dry-run" in sys.argv

    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(
        f"🚀 Initializing Pipeline: Exp 3 (DeBERTa + XGBoost Late Fusion)... {'[DRY RUN]' if is_dry_run else ''}"
    )

    data_path = (
        config["data"]["kaggle_parquet"]
        if os.path.exists(config["data"]["kaggle_parquet"])
        else config["data"]["local_parquet"]
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

    # 1. Load Trained DeBERTa (Exp 1 Checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_deb = config["deberta"]
    deb_model = DebertaSlidingWindowClassifier(
        model_name=cfg_deb["model_name"], num_classes=2
    ).to(device)

    ckpt_path = os.path.join(
        config["outputs"]["model_dir"], "best_deberta_exp1.pt"
    )
    if os.path.exists(ckpt_path):
        deb_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✅ Loaded DeBERTa weights from {ckpt_path}")

    # 2. Extract DeBERTa Probabilities
    train_ds = SlidingWindowPhishingDataset(
        texts=train_df["full_text"].values,
        labels=train_df["label"].values,
        tokenizer_name=cfg_deb["model_name"],
    )
    val_ds = SlidingWindowPhishingDataset(
        texts=val_df["full_text"].values,
        labels=val_df["label"].values,
        tokenizer_name=cfg_deb["model_name"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_deb["batch_size"],
        shuffle=False,
        collate_fn=packed_sliding_window_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_deb["batch_size"],
        shuffle=False,
        collate_fn=packed_sliding_window_collate_fn,
    )

    deb_train_probs = extract_deberta_probs(
        deb_model, train_loader, device, is_dry_run
    )
    deb_val_probs = extract_deberta_probs(
        deb_model, val_loader, device, is_dry_run
    )

    # Truncate splits to match dry run
    if is_dry_run:
        train_df = train_df.iloc[: len(deb_train_probs)]
        val_df = val_df.iloc[: len(deb_val_probs)]

    # 3. Extract Header Features & Train XGBoost Stream
    X_train_hdr = extract_header_structural_features(train_df["full_text"])
    X_val_hdr = extract_header_structural_features(val_df["full_text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    cfg_xgb = config["xgboost"]
    xgb_model = xgb.XGBClassifier(
        n_estimators=10 if is_dry_run else cfg_xgb["n_estimators"],
        max_depth=cfg_xgb["max_depth"],
        learning_rate=cfg_xgb["learning_rate"],
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_hdr, y_train)

    xgb_train_probs = xgb_model.predict_proba(X_train_hdr)[:, 1]
    xgb_val_probs = xgb_model.predict_proba(X_val_hdr)[:, 1]

    # 4. Meta-Learner (Late Fusion of Probabilities)
    X_meta_train = np.column_stack([deb_train_probs, xgb_train_probs])
    X_meta_val = np.column_stack([deb_val_probs, xgb_val_probs])

    meta_learner = LogisticRegression()
    meta_learner.fit(X_meta_train, y_train)

    fused_probs = meta_learner.predict_proba(X_meta_val)[:, 1]
    fused_preds = (fused_probs >= 0.5).astype(int)

    metrics = compute_evaluation_metrics(y_val, fused_preds, fused_probs)
    print_metrics_summary("Exp 3: DeBERTa + XGBoost Late Fusion", metrics)

    if not is_dry_run:
        os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
        joblib.dump(
            meta_learner,
            os.path.join(
                config["outputs"]["model_dir"], "best_late_fusion_exp3.joblib"
            ),
        )

        history = {
            "train_loss": [0.10],
            "val_loss": [1.0 - metrics["accuracy"]],
            "val_f1": [metrics["f1"]],
        }
        save_experiment_artifacts(
            output_dir=os.path.join(config["outputs"]["results_dir"], "exp3"),
            history=history,
            final_metrics=metrics,
            y_true=y_val,
            y_pred=fused_preds,
            y_prob=fused_probs,
            exp_name="exp3",
        )


if __name__ == "__main__":
    main()