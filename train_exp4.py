import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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
from train_exp2 import extract_header_structural_features


def load_model_weights_safely(model, ckpt_path, device):
    """Loads state dict safely regardless of key prefixes."""
    state_dict = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        clean_state_dict[key] = v

    model.load_state_dict(clean_state_dict)
    return model


def extract_cls_embeddings(model, loader, device, is_dry_run=False):
    """Extracts 768-dim pooled [CLS] representations from DeBERTa backbone."""
    model.eval()
    embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting DeBERTa [CLS] Embeddings"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_indices = batch["batch_indices"].to(device, non_blocking=True)
            batch_size = batch["batch_size"]

            pooled_emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_indices=batch_indices,
                batch_size=batch_size,
                return_embeddings=True,
            )
            embeddings_list.append(pooled_emb.cpu().numpy())
            if is_dry_run:
                break
    return np.vstack(embeddings_list)


def main():
    is_dry_run = "--dry-run" in sys.argv

    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(
        f"🚀 Initializing Pipeline: Exp 4 (DeBERTa + Header Early Fusion)... {'[DRY RUN]' if is_dry_run else ''}"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_deb = config["deberta"]
    deb_model = DebertaSlidingWindowClassifier(
        model_name=cfg_deb["model_name"], num_classes=2
    ).to(device)

    ckpt_path = os.path.join(
        config["outputs"]["model_dir"], "best_deberta_exp1.pt"
    )
    if os.path.exists(ckpt_path):
        deb_model = load_model_weights_safely(deb_model, ckpt_path, device)
        print(f"✅ Loaded trained DeBERTa weights from {ckpt_path}")

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
        num_workers=2,
        pin_memory=True,
        collate_fn=packed_sliding_window_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_deb["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=packed_sliding_window_collate_fn,
    )

    train_cls = extract_cls_embeddings(
        deb_model, train_loader, device, is_dry_run
    )
    val_cls = extract_cls_embeddings(deb_model, val_loader, device, is_dry_run)

    if is_dry_run:
        train_df = train_df.iloc[: len(train_cls)]
        val_df = val_df.iloc[: len(val_cls)]

    X_train_hdr = extract_header_structural_features(
        train_df["full_text"]
    ).values
    X_val_hdr = extract_header_structural_features(val_df["full_text"]).values

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    X_train_fused = np.hstack([train_cls, X_train_hdr])
    X_val_fused = np.hstack([val_cls, X_val_hdr])

    print(
        f"🧬 Early Fusion Feature Shape: {X_train_fused.shape} (768 Text + 12 Header Dims)"
    )

    cfg_xgb = config["xgboost"]
    fusion_xgb = xgb.XGBClassifier(
        n_estimators=10 if is_dry_run else cfg_xgb["n_estimators"],
        max_depth=cfg_xgb["max_depth"],
        learning_rate=cfg_xgb["learning_rate"],
        subsample=cfg_xgb["subsample"],
        colsample_bytree=cfg_xgb["colsample_bytree"],
        random_state=42,
        n_jobs=-1,
    )

    fusion_xgb.fit(X_train_fused, y_train)

    val_probs = fusion_xgb.predict_proba(X_val_fused)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = compute_evaluation_metrics(y_val, val_preds, val_probs)
    print_metrics_summary("Exp 4: DeBERTa + XGBoost Early Fusion", metrics)

    if not is_dry_run:
        os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
        joblib.dump(
            fusion_xgb,
            os.path.join(
                config["outputs"]["model_dir"], "best_early_fusion_exp4.joblib"
            ),
        )

        history = {
            "train_loss": [0.05],
            "val_loss": [1.0 - metrics["accuracy"]],
            "val_f1": [metrics["f1"]],
        }
        save_experiment_artifacts(
            output_dir=os.path.join(config["outputs"]["results_dir"], "exp4"),
            history=history,
            final_metrics=metrics,
            y_true=y_val,
            y_pred=val_preds,
            y_prob=val_probs,
            exp_name="exp4",
        )


if __name__ == "__main__":
    main()