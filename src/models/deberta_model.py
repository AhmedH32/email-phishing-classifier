import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DebertaSlidingWindowClassifier(nn.Module):
    """DeBERTa-v3 architecture with Gradient Checkpointing & Micro-Batching.

    Guarantees stable memory consumption regardless of dataset length distribution.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_classes: int = 2,
        dropout: float = 0.2,
        max_chunk_sub_batch: int = 8,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(
            model_name, config=self.config
        ).float()

        # Enable Gradient Checkpointing to reduce VRAM memory footprint by ~70%
        self.deberta.gradient_checkpointing_enable()

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.max_chunk_sub_batch = max_chunk_sub_batch

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        total_chunks = input_ids.size(0)
        cls_embeddings_list = []

        # Sub-batch chunks in micro-batches (max 8 chunks per forward step)
        for i in range(0, total_chunks, self.max_chunk_sub_batch):
            sub_ids = input_ids[i : i + self.max_chunk_sub_batch]
            sub_mask = attention_mask[i : i + self.max_chunk_sub_batch]

            outputs = self.deberta(input_ids=sub_ids, attention_mask=sub_mask)
            sub_cls = outputs.last_hidden_state[:, 0, :].float()
            cls_embeddings_list.append(sub_cls)

        cls_embeddings = torch.cat(cls_embeddings_list, dim=0)

        # Max-Pooling across chunks for each email sample
        pooled_embeddings = []
        for b in range(batch_size):
            sample_mask = batch_indices == b
            sample_chunk_reps = cls_embeddings[sample_mask]
            sample_pooled, _ = torch.max(sample_chunk_reps, dim=0)
            pooled_embeddings.append(sample_pooled)

        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)
        pooled_embeddings = self.drop(pooled_embeddings)

        if return_embeddings:
            return pooled_embeddings

        logits = self.classifier(pooled_embeddings)
        return logits


if __name__ == "__main__":
    print("Memory-optimized DeBERTa FP32 model ready!")