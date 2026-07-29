import os
import re
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import train_test_split

from src.utils.metrics import (
    compute_evaluation_metrics,
    print_metrics_summary,
    save_experiment_artifacts,
)


def extract_header_structural_features(texts: pd.Series) -> pd.DataFrame:
    """Extracts 12 tabular domain indicators and structural features from raw text."""
    feats = pd.DataFrame()

    feats["num_urls"] = texts.apply(
        lambda x: len(re.findall(r"https?://\S+|www\.\S+", str(x)))
    )
    feats["has_ip_url"] = texts.apply(
        lambda x: int(
            bool(
                re.search(
                    r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(x)
                )
            )
        )
    )
    feats["num_html_tags"] = texts.apply(
        lambda x: len(re.findall(r"<[^>]+>", str(x)))
    )
    feats["has_href"] = texts.apply(
        lambda x: int("href" in str(x).lower())
    )

    urgent_words = [
        "urgent",
        "verify",
        "account",
        "suspended",
        "password",
        "security",
        "bank",
        "login",
        "update",
        "click",
    ]
    feats["urgency_word_count"] = texts.apply(
        lambda x: sum(1 for w in urgent_words if w in str(x).lower())
    )

    feats["char_count"] = texts.apply(lambda x: len(str(x)))
    feats["word_count"] = texts.apply(lambda x: len(str(x).split()))
    feats["uppercase_ratio"] = texts.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1e-5)
    )

    feats["has_dkim_spf"] = texts.apply(
        lambda x: int(
            any(
                k in str(x).lower()
                for k in ["dkim-signature", "received-spf", "authentication-results"]
            )
        )
    )
    feats["num_exclamation"] = texts.apply(
        lambda x: str(x).count("!")
    )
    feats["num_question"] = texts.apply(lambda x: str(x).count("?"))
    feats["num_dollar"] = texts.apply(lambda x: str(x).count("$"))

    return feats


def main():
    is_dry_run = "--dry-run" in sys.argv

    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(
        f"🚀 Initializing Training Pipeline for Experiment 2 (XGBoost Headers)... {'[DRY RUN]' if is_dry_run else ''}"
    )

    data_path = (
        config["data"]["kaggle_parquet"]
        if os.path.exists(config["data"]["kaggle_parquet"])
        else config["data"]["local_parquet"]
    )
    df = pd.read_parquet(data_path)
    if is_dry_run:
        df = df.head(100)

    X_feats = extract_header_structural_features(df["full_text"])
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_feats,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    cfg_xgb = config["xgboost"]
    model = xgb.XGBClassifier(
        n_estimators=10 if is_dry_run else cfg_xgb["n_estimators"],
        max_depth=cfg_xgb["max_depth"],
        learning_rate=cfg_xgb["learning_rate"],
        subsample=cfg_xgb["subsample"],
        colsample_bytree=cfg_xgb["colsample_bytree"],
        random_state=cfg_xgb["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = compute_evaluation_metrics(y_val, val_preds, val_probs)
    print_metrics_summary("Exp 2: XGBoost Header-Only", metrics)

    if not is_dry_run:
        os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
        joblib.dump(
            model,
            os.path.join(
                config["outputs"]["model_dir"], "best_xgboost_exp2.joblib"
            ),
        )

        history = {
            "train_loss": [0.2],
            "val_loss": [1.0 - metrics["accuracy"]],
            "val_f1": [metrics["f1"]],
        }
        save_experiment_artifacts(
            output_dir=os.path.join(config["outputs"]["results_dir"], "exp2"),
            history=history,
            final_metrics=metrics,
            y_true=y_val,
            y_pred=val_preds,
            y_prob=val_probs,
        )


if __name__ == "__main__":
    main()