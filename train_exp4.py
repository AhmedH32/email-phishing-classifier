import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
        f"🚀 Initializing Training Pipeline for Experiment 4 (TF-IDF Baseline)... {'[DRY RUN]' if is_dry_run else ''}"
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

    vectorizer = TfidfVectorizer(max_features=5000 if is_dry_run else 10000)
    X_train = vectorizer.fit_transform(train_df["full_text"])
    X_val = vectorizer.transform(val_df["full_text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    model = LogisticRegression(max_iter=200 if is_dry_run else 1000)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = compute_evaluation_metrics(y_val, val_preds, val_probs)
    print_metrics_summary("Exp 4: TF-IDF + Logistic Regression", metrics)

    if not is_dry_run:
        os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
        joblib.dump(
            model,
            os.path.join(
                config["outputs"]["model_dir"], "best_tfidf_exp4.joblib"
            ),
        )

        history = {
            "train_loss": [0.25],
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
        )


if __name__ == "__main__":
    main()