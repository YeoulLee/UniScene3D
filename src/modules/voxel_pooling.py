"""Voxel pooling: collapse multi-view patch tokens into bounded 3D voxel tokens."""

import torch
import torch.nn as nn


class VoxelPooling(nn.Module):
    """Pool per-patch tokens into a fixed-size set of 3D voxel tokens.

    Patches from many views observe overlapping 3D regions; voxelizing by world
    coordinate deduplicates them and yields a spatially-organized, bounded token
    set for the language model. Output is padded to a fixed budget with a mask.
    """

    def __init__(self, voxel_size=0.2, max_tokens=512):
        super().__init__()
        self.voxel_size = float(voxel_size)
        self.max_tokens = int(max_tokens)

    def _voxelize_one(self, feats, coords):
        """Voxelize one sample. feats (M,D), coords (M,3) -> (U,D),(U,3),(U,)."""
        vidx = torch.floor(coords / self.voxel_size).long()             # (M,3)
        uniq, inverse = torch.unique(vidx, dim=0, return_inverse=True)   # (U,3),(M,)
        U, M, D = uniq.shape[0], feats.shape[0], feats.shape[-1]

        cnt = torch.zeros(U, device=feats.device, dtype=torch.float32)
        cnt.index_add_(0, inverse, torch.ones(M, device=feats.device))

        sum_feat = torch.zeros(U, D, device=feats.device, dtype=feats.dtype)
        sum_feat.index_add_(0, inverse, feats)
        sum_coord = torch.zeros(U, 3, device=coords.device, dtype=coords.dtype)
        sum_coord.index_add_(0, inverse, coords)

        denom = cnt.clamp_min(1.0).unsqueeze(-1)
        return sum_feat / denom, sum_coord / denom, cnt

    def forward(self, feats, coords, valid):
        """Args:
            feats:  (B, V, P, D) patch token features.
            coords: (B, V, P, 3) per-patch world coordinates.
            valid:  (B, V, P) bool, True for usable patches.

        Returns:
            voxel_feats:  (B, N, D)
            voxel_coords: (B, N, 3)
            voxel_mask:   (B, N) bool, True for real voxels (rest is padding).
        """
        B, D = feats.shape[0], feats.shape[-1]
        feats = feats.float().reshape(B, -1, D)    # (B, V*P, D)
        coords = coords.float().reshape(B, -1, 3)  # (B, V*P, 3)
        valid = valid.reshape(B, -1)               # (B, V*P)

        N = self.max_tokens
        out_feat = torch.zeros(B, N, D, device=feats.device, dtype=feats.dtype)
        out_coord = torch.zeros(B, N, 3, device=feats.device, dtype=coords.dtype)
        out_mask = torch.zeros(B, N, dtype=torch.bool, device=feats.device)

        for b in range(B):
            m = valid[b]
            if not bool(m.any()):
                continue
            vf, vc, cnt = self._voxelize_one(feats[b][m], coords[b][m])
            U = vf.shape[0]
            if U > N:
                # Keep the most-observed voxels (seen by the most patches).
                keep = torch.topk(cnt, N).indices
                vf, vc, U = vf[keep], vc[keep], N
            out_feat[b, :U] = vf
            out_coord[b, :U] = vc
            out_mask[b, :U] = True
        return out_feat, out_coord, out_mask
