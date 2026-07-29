import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DebertaSlidingWindowClassifier(nn.Module):
    """DeBERTa-v3 Small architecture with Flattened Parallel Chunk Pass & Max-Pooling.

    Flattens batch and chunk dimensions into [batch_size * num_chunks, seq_len]
    to execute a single parallel forward pass on GPU for maximum throughput.
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
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        # Shapes: [batch_size, num_chunks, seq_len]
        batch_size, num_chunks, seq_len = input_ids.shape

        # Flatten [batch_size, num_chunks] into single dimension for parallel GPU computation
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        # Single Parallel Forward Pass
        outputs = self.deberta(
            input_ids=flat_input_ids, attention_mask=flat_attention_mask
        )

        # Extract [CLS] token embeddings: [batch_size * num_chunks, hidden_size]
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]

        # Reshape back to [batch_size, num_chunks, hidden_size]
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)

        # Max-Pooling across chunk dimension (dim 1)
        pooled_embeddings, _ = torch.max(chunk_embeddings, dim=1)
        pooled_embeddings = self.drop(pooled_embeddings)

        # Return pooled vector for Early Fusion (Exp 4) or logits for Exp 1/3
        if return_embeddings:
            return pooled_embeddings

        logits = self.classifier(pooled_embeddings)
        return logits


if __name__ == "__main__":
    print("Flattened DeBERTa-v3 Small model ready!")