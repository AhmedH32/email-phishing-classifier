import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DebertaSlidingWindowClassifier(nn.Module):
    """DeBERTa-v3 architecture with Packed Chunk Processing & Sample-wise Max-Pooling.

    Enforces FP32 precision across backbone and classification head to prevent
    dtype mismatches on Kaggle GPU environments.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # Force backbone weights to FP32 float precision
        self.deberta = AutoModel.from_pretrained(
            model_name, config=self.config
        ).float()
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
        # Single parallel forward pass over valid chunks
        outputs = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Extract [CLS] embeddings and cast explicitly to float32
        cls_embeddings = outputs.last_hidden_state[:, 0, :].float()

        # Max-Pooling across chunks per email sample
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
    print("Packed Chunk DeBERTa FP32 model ready!")