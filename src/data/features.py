import re
from typing import Tuple

import numpy as np
import pandas as pd


def extract_header_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numerical & categorical features from email metadata for XGBoost.

    Handles text-only rows (is_metadata_present == 0) by setting header features
    to np.nan so tree models can use sparsity-aware split finding.
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Flag indicating header existence
    feats["is_metadata_present"] = df["is_metadata_present"].astype(int)

    # 2. Text Length Features (Available for ALL 82k emails)
    feats["subject_char_len"] = df["subject"].fillna("").astype(str).str.len()
    feats["body_char_len"] = df["body"].fillna("").astype(str).str.len()
    feats["subject_word_count"] = (
        df["subject"].fillna("").astype(str).str.split().str.len()
    )
    feats["body_word_count"] = (
        df["body"].fillna("").astype(str).str.split().str.len()
    )

    # Helper function to parse URL strings
    def parse_urls(row):
        if row["is_metadata_present"] == 0:
            return np.nan  # Header missing! Assign NaN for XGBoost
        url_str = str(row["urls"])
        if url_str == "MISSING" or pd.isna(url_str) or not url_str.strip():
            return 0  # Header present, but 0 URLs found
        # Split URLs by comma or space
        urls_list = [
            u.strip() for u in re.split(r"[,\s]+", url_str) if u.strip()
        ]
        return len(urls_list)

    # 3. Header-Specific Features (Assigned np.nan if metadata is missing)
    feats["num_urls"] = df.apply(parse_urls, axis=1)

    # 4. Check presence of core headers
    feats["has_sender"] = df.apply(
        lambda r: np.nan
        if r["is_metadata_present"] == 0
        else int(
            str(r["sender"]) != "MISSING" and bool(str(r["sender"]).strip())
        ),
        axis=1,
    )

    feats["has_receiver"] = df.apply(
        lambda r: np.nan
        if r["is_metadata_present"] == 0
        else int(
            str(r["receiver"]) != "MISSING"
            and bool(str(r["receiver"]).strip())
        ),
        axis=1,
    )

    feats["has_date"] = df.apply(
        lambda r: np.nan
        if r["is_metadata_present"] == 0
        else int(
            str(r["date"]) != "MISSING" and bool(str(r["date"]).strip())
        ),
        axis=1,
    )

    # 5. Ratio features
    feats["url_to_word_ratio"] = feats["num_urls"] / (
        feats["body_word_count"] + 1e-5
    )

    return feats


if __name__ == "__main__":
    # Test script with dummy inputs
    sample_df = pd.DataFrame(
        [
            {
                "subject": "Urgent Security Alert",
                "body": "Click here to verify http://fake.com",
                "urls": "http://fake.com",
                "sender": "alert@bank.com",
                "receiver": "user@domain.com",
                "date": "Mon, 01 Jan 2026",
                "is_metadata_present": 1,
            },
            {
                "subject": "Meeting notes",
                "body": "Attached is the report.",
                "urls": "MISSING",
                "sender": "MISSING",
                "receiver": "MISSING",
                "date": "MISSING",
                "is_metadata_present": 0,
            },
        ]
    )

    extracted = extract_header_features(sample_df)
    print("=== FEATURE EXTRACTION TEST ===")
    print(extracted)