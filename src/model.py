"""
src/model.py - the multi-task transformer (Phase 3).

THE MODEL IN ONE PICTURE:

    text -> tokenizer -> pretrained multilingual transformer
                              |
                        mean pooling over real tokens (not padding)
                              |
                           dropout 0.1
                          /           \\
                 verdict head       tag head
                 (2 outputs)       (13 outputs)

WHY MEAN POOLING: we average the token vectors but multiply padding positions
by 0 first, so short messages are not diluted by <PAD> tokens.

WHY TWO HEADS: the verdict is the main decision (scam/safe); the 13 tags
explain WHY. The shared encoder learns both at once.

LOSS = weighted cross-entropy(verdict) + lambda_tags * BCE(tags, pos_weight)
If lambda_tags == 0 the tag loss is skipped - that is experiment E2, the
verdict-only ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MultiTaskClassifier(nn.Module):
    """Pretrained encoder + mean pooling + binary verdict head + tag head."""

    def __init__(self, model_name: str, num_verdicts: int = 2,
                 num_tags: int = 13, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.verdict_head = nn.Linear(hidden, num_verdicts)
        self.tag_head = nn.Linear(hidden, num_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return (verdict_logits [B,2], tag_logits [B,13])."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_vectors = out.last_hidden_state                 # [B, T, H]

        # Mean pooling that ignores padding. mask is 1 for real tokens.
        mask = attention_mask.unsqueeze(-1).float()           # [B, T, 1]
        summed = (token_vectors * mask).sum(dim=1)            # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)              # [B, 1]
        pooled = summed / counts

        pooled = self.dropout(pooled)
        return self.verdict_head(pooled), self.tag_head(pooled)


def multi_task_loss(verdict_logits: torch.Tensor,
                    tag_logits: torch.Tensor,
                    verdict_labels: torch.Tensor,
                    tag_labels: torch.Tensor,
                    verdict_weights: torch.Tensor | None = None,
                    tag_pos_weight: torch.Tensor | None = None,
                    lambda_tags: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Weighted CE for the verdict + (optionally) BCE for the tags.

    Returns (total_loss, {"verdict_loss": ..., "tag_loss": ...}) so the
    training loop can print both parts.
    """
    verdict_loss = F.cross_entropy(verdict_logits, verdict_labels,
                                   weight=verdict_weights)
    if lambda_tags <= 0:
        tag_loss = torch.zeros((), device=verdict_logits.device)
        total = verdict_loss
    else:
        tag_loss = F.binary_cross_entropy_with_logits(
            tag_logits, tag_labels, pos_weight=tag_pos_weight)
        total = verdict_loss + float(lambda_tags) * tag_loss
    return total, {"verdict_loss": float(verdict_loss.detach()),
                   "tag_loss": float(tag_loss.detach())}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters (for the benchmark report)."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
