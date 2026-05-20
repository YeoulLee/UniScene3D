"""Projector that maps pooled 3D scene tokens into the LLM embedding space."""

import torch.nn as nn


class Qwen3DProjector(nn.Module):
    """2-layer MLP projecting voxel-token features to the LLM hidden size.

    LLaVA-style connector: Linear -> GELU -> Linear. This is one of the two
    trainable components (the other being the Qwen LoRA adapters).
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """x (..., in_dim) -> (..., out_dim)."""
        return self.net(x)
