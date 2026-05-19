"""
Shared compute utilities for wide-path cross features.

All overlap-style cross features share the same shape — a weighted user-side pool
over a per-item binary feature matrix, dotted against the candidate's binary row,
optionally normalized by the candidate's feature count. The functions here are
parameterized on the **pool indices** and **pool weights** so the same compute path
serves both:

  - Bucket 1 (full history):   pool = (X_hist_full, X_hist_playtime_weights)
  - Bucket 3 (last-3-liked):   pool = (last 3 non-pad of X_hist_liked,
                                       playtime weights re-normalized over those 3)

Add Bucket 3 by extracting the last-N pool with a helper (TBD) and calling the
exact same functions below.

All functions expect tensors already on the same device, and are batched over (B).
"""
from __future__ import annotations

import torch


def overlap_pool(item_binary:   torch.Tensor,   # (n_items+1, n_feat) float32 — binary one-hot per item
                 item_count:    torch.Tensor,   # (n_items+1,)        float32 — per-item feature count, clamp(min=1)
                 pool_indices:  torch.Tensor,   # (B, P)              int64   — history indices (pad allowed; pad row of item_binary is zero → harmless)
                 pool_weights:  torch.Tensor,   # (B, P)              float32 — per-position weights (should sum to 1 over non-pad)
                 cand_idx:      torch.Tensor,   # (B, n_cand)         int64
                 ) -> torch.Tensor:             # (B, n_cand)         float32
    """
    Weighted overlap cross feature.

    user_w[b, f]   = Σ_i pool_weights[b, i] · item_binary[pool_indices[b, i], f]
    overlap[b, c]  = (user_w[b] · item_binary[cand_idx[b, c]]) / item_count[cand_idx[b, c]]
                   = mean over the candidate's features of "user's pool weight on that feature."

    Strategy: build the (B, n_items+1) full overlap matrix via a single dense matmul
    (cheap on MPS), divide elementwise by item_count, then gather per (b, c). Avoids
    materializing (B, n_cand, n_feat) at any point. Pad row of item_binary is all-zero
    so pad positions in pool_indices contribute nothing; pool_weights should also be
    zero at pad positions for the same reason.

    Used by:
      - B-1 Genre Overlap  (item_binary = game_genre_binary, count = game_genre_count)
      - B-1b Tag Overlap   (item_binary = game_tag_binary,   count = game_tag_count)
      - B-8a/b Recent versions of the above (Bucket 3) — same matrices, different pool.
    """
    user_w           = (item_binary[pool_indices] * pool_weights.unsqueeze(-1)).sum(dim=1)  # (B, n_feat)
    full_overlap_num = user_w @ item_binary.t()                                              # (B, n_items+1)
    full_overlap     = full_overlap_num / item_count                                         # (B, n_items+1)
    return full_overlap.gather(1, cand_idx)                                                  # (B, n_cand)


def dev_affinity_pool(game_dev_idx:  torch.Tensor,    # (n_items+1,)  int64  — per-item dev index (pad row = dev_pad_idx)
                      dev_pad_idx:   int,             #                       — n_developers (matches Embedding padding_idx)
                      pool_indices:  torch.Tensor,    # (B, P)        int64
                      pool_weights:  torch.Tensor,    # (B, P)        float32
                      cand_idx:      torch.Tensor,    # (B, n_cand)   int64
                      ) -> torch.Tensor:              # (B, n_cand)   float32
    """
    Playtime-weighted developer affinity:

      affinity[b, c] = Σ_i pool_weights[b, i] · 1[game_dev_idx[pool_indices[b, i]] == game_dev_idx[cand_idx[b, c]]]
                     = fraction of user's pool playtime on games by the candidate's developer.

    Strategy: build a per-user dev profile via scatter_add into a (B, n_developers+1)
    tensor, then gather per (b, c) using the candidate's dev index. Pad positions in
    pool_indices map to dev_pad_idx (via game_dev_idx[pad_idx]); pool_weights is 0 at
    pad positions, so the scatter contributes 0 weight at the dev-pad slot — harmless.

    Used by:
      - B-2 Developer Affinity (full history)
      - B-8c Recent Developer Affinity (Bucket 3) — same util, different pool.
    """
    B          = pool_indices.shape[0]
    device     = pool_indices.device
    hist_devs  = game_dev_idx[pool_indices]                                                  # (B, P) int64
    user_dev_w = torch.zeros(B, dev_pad_idx + 1, device=device, dtype=pool_weights.dtype)
    user_dev_w.scatter_add_(1, hist_devs, pool_weights)                                       # (B, n_devs+1)
    cand_devs  = game_dev_idx[cand_idx]                                                       # (B, n_cand) int64
    return user_dev_w.gather(1, cand_devs)                                                    # (B, n_cand)
