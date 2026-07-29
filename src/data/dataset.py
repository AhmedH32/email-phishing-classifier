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

        item = {
            "input_ids": encoding[
                "input_ids"
            ],  # Shape: [num_chunks_for_email, 512]
            "attention_mask": encoding["attention_mask"],
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def packed_sliding_window_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Packs ONLY valid chunks across the batch without padding zero chunks.

    Eliminates wasted compute and prevents CUDA memory spikes on long-tail inputs.
    """
    all_input_ids = []
    all_attention_masks = []
    batch_indices = []
    labels = []

    for sample_idx, item in enumerate(batch):
        input_ids = item["input_ids"]  # [num_chunks, seq_len]
        attention_mask = item["attention_mask"]  # [num_chunks, seq_len]
        num_chunks = input_ids.size(0)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        # Track which sample index each chunk belongs to
        batch_indices.extend([sample_idx] * num_chunks)

        if "label" in item:
            labels.append(item["label"])

    collated = {
        "input_ids": torch.cat(
            all_input_ids, dim=0
        ),  # [Total_Real_Chunks, seq_len]
        "attention_mask": torch.cat(
            all_attention_masks, dim=0
        ),  # [Total_Real_Chunks, seq_len]
        "batch_indices": torch.tensor(
            batch_indices, dtype=torch.long
        ),  # [Total_Real_Chunks]
        "batch_size": len(batch),
    }

    if labels:
        collated["label"] = torch.stack(labels, dim=0)

    return collated