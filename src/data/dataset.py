from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SlidingWindowPhishingDataset(Dataset):
    """PyTorch Dataset that splits long email bodies into overlapping sliding windows

    (max_length=512, stride=256) for DeBERTa-v3 processing.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int] = None,
        tokenizer_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        stride: int = 256,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.stride = stride

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        # Tokenize with sliding window chunking
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
            return_tensors="pt",
        )

        # encoding['input_ids'] shape: [num_chunks, max_length]
        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "num_chunks": torch.tensor(
                encoding["input_ids"].size(0), dtype=torch.long
            ),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


if __name__ == "__main__":
    print("Dataset module loaded cleanly!")