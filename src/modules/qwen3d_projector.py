"""Projector that maps pooled 3D scene tokens into the LLM embedding space."""

import torch.nn as nn


class Qwen3DProjector(nn.Module):
    """2-layer MLP projecting voxel-token features to the LLM hidden size.

    LLaVA-style connector: Linear -> GELU -> Linear -> LayerNorm, then scaled to
    the LLM's text-embedding std. The final LayerNorm + scale is essential: the
    raw MLP output can be ~1000x larger than Qwen's input embeddings (std ~0.01),
    which makes the LLM treat the visual tokens as out-of-distribution outliers
    and ignore them entirely. Matching the embedding scale keeps the visual
    tokens usable from initialization so gradient can flow into the vision path.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, output_std=1.0):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.output_std = float(output_std)

    def forward(self, x):
        """x (..., in_dim) -> (..., out_dim), scaled to the LLM embed std."""
        x = self.net(x)
        x = self.norm(x)
        return x * self.output_std
