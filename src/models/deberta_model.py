import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DebertaSlidingWindowClassifier(nn.Module):
    """DeBERTa-v3 architecture with Packed Chunk Processing & Sample-wise Max-Pooling.

    Executes a single parallel GPU forward pass on ONLY valid chunks across the batch,
    and aggregates chunk representations per sample via Max-Pooling.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(
            model_name, config=self.config
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        # input_ids shape: [Total_Real_Chunks_In_Batch, seq_len]
        # attention_mask shape: [Total_Real_Chunks_In_Batch, seq_len]

        # 1. Single parallel forward pass over ONLY real chunks
        outputs = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # 2. Extract [CLS] token representations: [Total_Real_Chunks_In_Batch, hidden_size]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # 3. Max-Pooling across chunks for each email sample in the batch
        pooled_embeddings = []
        for b in range(batch_size):
            sample_mask = batch_indices == b
            sample_chunk_reps = cls_embeddings[
                sample_mask
            ]  # [num_chunks_for_b, hidden_size]
            sample_pooled, _ = torch.max(sample_chunk_reps, dim=0)
            pooled_embeddings.append(sample_pooled)

        pooled_embeddings = torch.stack(
            pooled_embeddings, dim=0
        )  # [batch_size, hidden_size]
        pooled_embeddings = self.drop(pooled_embeddings)

        if return_embeddings:
            return pooled_embeddings

        logits = self.classifier(pooled_embeddings)
        return logits


if __name__ == "__main__":
    print("Packed Chunk DeBERTa model ready!")