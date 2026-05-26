"""Per-patch 3D coordinate extraction and sinusoidal 3D position encoding."""

import math

import torch
import torch.nn as nn


def extract_patch_coords(point_map, patch_size=16):
    """Average each ViT patch's valid world coordinates from a dense point map.

    A pixel is valid when all 3 coordinates are finite and not all-zero (the
    point map stores 0 for missing depth). Patches with no valid pixel get a
    zero coordinate and valid=False.

    Args:
        point_map: (B, V, 3, H, W) per-pixel XYZ world coordinates.
        patch_size: ViT patch size in pixels (H and W must be divisible by it).

    Returns:
        coords: (B, V, P, 3) per-patch mean world coordinate, P = (H/ps)*(W/ps).
        valid:  (B, V, P) bool mask, True where the patch had >=1 valid pixel.
    """
    B, V, C, H, W = point_map.shape
    if C != 3:
        raise ValueError(f"point_map must have 3 channels (XYZ), got {C}.")
    if H % patch_size or W % patch_size:
        raise ValueError(f"H,W ({H},{W}) must be divisible by patch_size {patch_size}.")

    gh, gw = H // patch_size, W // patch_size
    P = gh * gw

    pm = point_map.float().reshape(B, V, 3, gh, patch_size, gw, patch_size)
    # -> (B, V, P, patch_size*patch_size, 3)
    pm = pm.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, V, P, patch_size * patch_size, 3)

    pix_valid = torch.isfinite(pm).all(dim=-1) & (pm.abs().sum(dim=-1) > 0)  # (B,V,P,ps^2)
    pm = torch.where(pix_valid.unsqueeze(-1), pm, torch.zeros_like(pm))

    count = pix_valid.sum(dim=-1)                              # (B,V,P)
    coords = pm.sum(dim=3) / count.clamp_min(1).unsqueeze(-1)  # (B,V,P,3)
    valid = count > 0                                          # (B,V,P)
    coords = torch.where(valid.unsqueeze(-1), coords, torch.zeros_like(coords))
    return coords, valid


class Sinusoidal3DPositionEncoding(nn.Module):
    """Additive sinusoidal position encoding of (x, y, z) world coordinates.

    Each axis is encoded with log-spaced spatial frequencies and the three axes
    are concatenated, matching Video-3D LLM's additive 3D position encoding.
    Parameter-free; the output is added to visual token features.
    """

    def __init__(self, dim, min_wavelength=0.1, max_wavelength=20.0):
        """Args:
            dim: output dimension, must be divisible by 6 (3 axes x sin/cos).
            min_wavelength / max_wavelength: spatial period range in metres.
        """
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"dim must be divisible by 6, got {dim}.")
        self.dim = dim
        half = dim // 6  # frequencies per axis
        log_lam = torch.linspace(
            math.log(min_wavelength), math.log(max_wavelength), half
        )
        freqs = 2.0 * math.pi / torch.exp(log_lam)  # (half,)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, coords):
        """coords (..., 3) -> (..., dim) additive position encoding."""
        ang = coords.float().unsqueeze(-1) * self.freqs  # (..., 3, half)
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)   # (..., 3, 2*half)
        return pe.flatten(-2)                            # (..., dim)


def quaternion_to_rotation_matrix(quat):
    """Convert (x, y, z, w) quaternions to 3x3 rotation matrices.

    Args:
        quat: (..., 4) tensor in (x, y, z, w) order (SQA3D convention).

    Returns:
        (..., 3, 3) rotation matrix R such that p_world = R @ p_local.
    """
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dim=-1)
    return R.reshape(*quat.shape[:-1], 3, 3)


def world_to_agent_frame(coords, agent_pos, agent_quat):
    """Transform world-frame coords into the agent's local frame.

    Args:
        coords:      (B, N, 3) world-frame points.
        agent_pos:   (B, 3) agent position in world frame.
        agent_quat:  (B, 4) agent rotation in world frame, (x, y, z, w).

    Returns:
        (B, N, 3) coords expressed relative to the agent (translation removed,
        rotation inverted), so a point in front of the agent has positive y
        (or whatever convention the SQA3D pose follows).
    """
    R = quaternion_to_rotation_matrix(agent_quat)         # (B, 3, 3) world<-agent
    R_inv = R.transpose(-1, -2)                           # (B, 3, 3) agent<-world
    rel = coords - agent_pos.unsqueeze(1)                 # (B, N, 3)
    return torch.einsum("bij,bnj->bni", R_inv, rel)
