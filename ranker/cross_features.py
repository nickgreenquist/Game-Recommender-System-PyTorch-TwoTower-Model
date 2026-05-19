"""
Shared compute utilities for wide-path cross features.

All overlap-style cross features share the same shape — a weighted reduction over
a slice of the user's interaction history against a per-item categorical buffer
(binary one-hot, or dev-index lookup), evaluated for each candidate. The functions
here do NOT pool embeddings — they reduce per-item categorical signals weighted by
playtime, producing one scalar per (history slice, candidate) pair.

The functions are parameterized on **history_indices** and **history_weights** so
the same compute path serves all four planned history variants (plan §9):

  - Bucket 1 ✓ (full history):  (X_hist_full,      X_hist_playtime_weights)
  - Bucket 2  (liked history):  (X_hist_liked,     X_hist_liked_playtime_weights)
  - Bucket 3  (last-3 liked):   (last 3 non-pad of X_hist_liked, re-normalized)
  - Bucket 4  (disliked):       (X_hist_disliked,  X_hist_disliked_playtime_weights)

Buckets 2-4 require new precompute columns; the compute path here doesn't change.

All functions expect tensors already on the same device, and are batched over (B).
"""
from __future__ import annotations

import torch


def weighted_overlap(item_binary:      torch.Tensor,   # (n_items+1, n_feat) float32 — binary one-hot per item
                     item_count:       torch.Tensor,   # (n_items+1,)        float32 — per-item feature count, clamp(min=1)
                     history_indices:  torch.Tensor,   # (B, H)              int64   — history indices (pad allowed; pad row of item_binary is zero → harmless)
                     history_weights:  torch.Tensor,   # (B, H)              float32 — per-position weights (should sum to 1 over non-pad)
                     cand_idx:         torch.Tensor,   # (B, n_cand)         int64
                     ) -> torch.Tensor:                # (B, n_cand)         float32
    """
    Weighted overlap cross feature.

    user_w[b, f]   = Σ_i history_weights[b, i] · item_binary[history_indices[b, i], f]
    overlap[b, c]  = (user_w[b] · item_binary[cand_idx[b, c]]) / item_count[cand_idx[b, c]]
                   = mean over the candidate's features of "user's history weight on that feature."

    Strategy: build the (B, n_items+1) full overlap matrix via a single dense matmul
    (cheap on MPS), divide elementwise by item_count, then gather per (b, c). Avoids
    materializing (B, n_cand, n_feat) at any point. Pad row of item_binary is all-zero
    so pad positions in history_indices contribute nothing; history_weights should also
    be zero at pad positions for the same reason.

    Used by:
      - B-1  Genre Overlap   (item_binary = game_genre_binary, count = game_genre_count)
      - B-1b Tag Overlap     (item_binary = game_tag_binary,   count = game_tag_count)
      - Buckets 2-4 variants — same matrices, different history slice.
    """
    user_w           = (item_binary[history_indices] * history_weights.unsqueeze(-1)).sum(dim=1)  # (B, n_feat)
    full_overlap_num = user_w @ item_binary.t()                                                    # (B, n_items+1)
    full_overlap     = full_overlap_num / item_count                                               # (B, n_items+1)
    return full_overlap.gather(1, cand_idx)                                                        # (B, n_cand)


def dev_affinity(game_dev_idx:     torch.Tensor,    # (n_items+1,)  int64  — per-item dev index (pad row = dev_pad_idx)
                 dev_pad_idx:      int,             #                       — n_developers (matches Embedding padding_idx)
                 history_indices:  torch.Tensor,    # (B, H)        int64
                 history_weights:  torch.Tensor,    # (B, H)        float32
                 cand_idx:         torch.Tensor,    # (B, n_cand)   int64
                 ) -> torch.Tensor:                 # (B, n_cand)   float32
    """
    Playtime-weighted developer affinity:

      affinity[b, c] = Σ_i history_weights[b, i] · 1[game_dev_idx[history_indices[b, i]] == game_dev_idx[cand_idx[b, c]]]
                     = fraction of user's history playtime on games by the candidate's developer.

    Strategy: build a per-user dev profile via scatter_add into a (B, n_developers+1)
    tensor, then gather per (b, c) using the candidate's dev index. Pad positions in
    history_indices map to dev_pad_idx (via game_dev_idx[pad_idx]); history_weights is
    0 at pad positions, so the scatter contributes 0 weight at the dev-pad slot — harmless.

    Used by:
      - B-2 Developer Affinity (full history)
      - Buckets 2-4 variants  — same util, different history slice.
    """
    B          = history_indices.shape[0]
    device     = history_indices.device
    hist_devs  = game_dev_idx[history_indices]                                                  # (B, H) int64
    user_dev_w = torch.zeros(B, dev_pad_idx + 1, device=device, dtype=history_weights.dtype)
    user_dev_w.scatter_add_(1, hist_devs, history_weights)                                       # (B, n_devs+1)
    cand_devs  = game_dev_idx[cand_idx]                                                          # (B, n_cand) int64
    return user_dev_w.gather(1, cand_devs)                                                       # (B, n_cand)
