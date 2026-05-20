"""
Shared compute utilities for wide-path cross features.

Two families of utilities live here:

1. **Categorical-overlap reductions** (`weighted_overlap`, `dev_affinity`,
   `categorical_overlap_triple`, `last_n_history`). Cover Buckets 1+2 cross
   features — a weighted reduction over a slice of the user's interaction history
   against a per-item categorical buffer (binary one-hot or dev-index lookup),
   evaluated for each candidate. Parameterized on `(history_indices,
   history_weights)` so the same compute path serves all planned history
   variants:

       - Bucket 1 ✓ (full history):  (X_hist_full,  X_hist_playtime_weights)
       - Bucket 2 ✓ (liked):         (X_hist_liked, X_hist_liked_playtime_weights)
       - Bucket 2 ✓ (last-3 liked):  (last 3 non-pad of X_hist_liked, re-normalized)

   (Buckets 3 and 4 were tried with this family and dropped — see plan §8/§9.)

2. **Scalar numeric-match quintuple** (`numeric_match_quintuple`,
   `Bucket5GameBuffers`, the `B5_*` field indices). Covers Bucket 5 — five
   per-(user, candidate) differences on numeric stats (price, year, median log
   playtime, popularity, sentiment). No history reduction here; the user-side
   values are full-history aggregates precomputed once per user and passed in as
   a `(B, 5)` tensor.

All functions expect tensors already on the same device, and are batched over (B).
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class OverlapBuffers(NamedTuple):
    """Bundle of per-item static buffers needed by `categorical_overlap_triple`.

    Lives on the model (`ranker/model.py` registers all six as non-persistent
    buffers built from FeatureStore) or as device-resident tensors on the caller
    (precompute builds them locally). Either way, pass the bundle through and the
    triple-overlap util gets all the inputs it needs to score a single history
    slice against the candidate set.
    """
    genre_binary: torch.Tensor      # (n_items+1, n_genres) float32 — binary one-hot
    genre_count:  torch.Tensor      # (n_items+1,)          float32 — per-item genre count, clamp(min=1)
    tag_binary:   torch.Tensor      # (n_items+1, n_tags)   float32 — binary one-hot
    tag_count:    torch.Tensor      # (n_items+1,)          float32 — per-item tag count, clamp(min=1)
    game_dev_idx: torch.Tensor      # (n_items+1,)          int64   — per-item dev index (pad row = dev_pad_idx)
    dev_pad_idx:  int                                                # n_developers


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


def last_n_history(history_indices:  torch.Tensor,   # (B, H) int64 — pad_idx fills trailing positions
                   history_weights:  torch.Tensor,   # (B, H) float32
                   n:                int,            # window size (3 for Recent-3 Liked)
                   pad_idx:          int,            # n_items (matches game_pad_idx on the model)
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the **last n non-pad positions** of each row, with weights re-normalized
    to sum to 1 over the selected window. If fewer than n non-pad positions exist,
    the leading slots in the (B, n) output are filled with `pad_idx` (index) and 0
    (weight) so the cross-feature utils see them as harmless pad. If a row has zero
    non-pad positions, all n slots return pad_idx / 0 and the downstream features
    fall through to 0.

    Convention: pad_idx fills the TRAILING positions of `history_indices` (matches
    precompute's `X_hist_liked[ex, :len(liked_ids)] = liked_ids` write — real games
    populate [0, count), pad fills [count, H)). The non-pad count per row is just
    `(history_indices != pad_idx).sum(dim=1)`.

    Semantic correctness depends on what you feed in: passing `X_hist_liked` returns
    the last n LIKED games (since the array only contains liked games to begin with);
    passing `X_hist_full` would return the last n games of any kind.

    Returns (last_indices (B, n) int64, last_weights (B, n) float32).
    """
    B, H        = history_indices.shape
    device      = history_indices.device
    mask        = (history_indices != pad_idx)                                  # (B, H)
    counts      = mask.sum(dim=1)                                               # (B,)

    # For each row, the last n non-pad positions live at H-array offsets
    # [counts-n, counts-n+1, ..., counts-1]. When counts < n the leading slots
    # become negative — clamp to 0 for the gather and mask them out afterwards
    # so they end up as pad_idx / 0 explicitly.
    pos_offsets = torch.arange(-n, 0, device=device)                            # (n,)  [-n, ..., -1]
    raw_pos     = counts.unsqueeze(1) + pos_offsets.unsqueeze(0)                # (B, n)
    valid_mask  = raw_pos >= 0                                                  # (B, n)
    safe_pos    = raw_pos.clamp(min=0)                                          # (B, n) — safe for gather

    last_indices = history_indices.gather(1, safe_pos)                          # (B, n)
    last_weights = history_weights.gather(1, safe_pos)                          # (B, n)

    pad_fill = torch.full_like(last_indices, pad_idx)
    last_indices = torch.where(valid_mask, last_indices, pad_fill)
    last_weights = torch.where(valid_mask, last_weights, torch.zeros_like(last_weights))

    # Re-normalize over the selected window. If all weights are 0 (e.g. counts==0,
    # or the row's weights happen to sum to 0), keep zeros — downstream features
    # then evaluate to 0 (no signal), matching the empty-history fallback.
    sums = last_weights.sum(dim=1, keepdim=True)                                # (B, 1)
    last_weights = torch.where(sums > 0, last_weights / sums, last_weights)

    return last_indices, last_weights


def categorical_overlap_triple(buffers:          OverlapBuffers,
                               history_indices:  torch.Tensor,    # (B, H) int64
                               history_weights:  torch.Tensor,    # (B, H) float32
                               cand_idx:         torch.Tensor,    # (B, n_cand) int64
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The three categorical-overlap cross features (genre_overlap, tag_overlap,
    dev_affinity) for a single user-side history slice. All three reuse the
    primitives (`weighted_overlap` and `dev_affinity`) — this wrapper just bundles
    the buffers + call sites so every history slice (Full / Liked / Recent-3 /
    Disliked / future) computes its 3 features in one line.

    Returns (genre_overlap, tag_overlap, dev_affinity), each shaped (B, n_cand).

    Used by:
      - ranker/precompute.py — once per history slice, results written to parquet
      - ranker/train.py      — once per history slice, results stacked into cross
      - ranker/canary.py     — once per history slice for the synthetic user
    """
    genre_ov = weighted_overlap(buffers.genre_binary, buffers.genre_count,
                                history_indices=history_indices,
                                history_weights=history_weights,
                                cand_idx=cand_idx)
    tag_ov   = weighted_overlap(buffers.tag_binary, buffers.tag_count,
                                history_indices=history_indices,
                                history_weights=history_weights,
                                cand_idx=cand_idx)
    dev_aff  = dev_affinity(buffers.game_dev_idx,
                            dev_pad_idx=buffers.dev_pad_idx,
                            history_indices=history_indices,
                            history_weights=history_weights,
                            cand_idx=cand_idx)
    return genre_ov, tag_ov, dev_aff


# ── Bucket 5 — Numeric Matching ──────────────────────────────────────────────
#
# Scalar-arithmetic cross features. Each feature is one subtract on two scalars
# (one user-side aggregate, one per-candidate static), with abs or signed reduction.
# No history slicing — user-side values are precomputed once per user (full-history
# aggregates living in FeatureStore as user_to_mean_* / user_to_median_log_playtime)
# and passed in here as a (B, 5) tensor.
#
# Indices into the (B, 5) user-side tensor. Order MUST stay stable — used by
# ranker/precompute.py, ranker/dataset.py, ranker/train.py, ranker/canary.py to
# pack/unpack consistently. The matching FeatureStore field names below must
# track these in lockstep.

B5_USER_MEAN_PRICE        = 0
B5_USER_MEAN_YEAR         = 1
B5_USER_MEDIAN_LOG_PT     = 2
B5_USER_MEAN_LOG_COUNT    = 3
B5_USER_MEAN_SENTIMENT    = 4
B5_USER_FIELDS = (                        # FeatureStore dict keys, ordered to match B5_USER_*
    'user_to_mean_price_bucket',
    'user_to_mean_year_numeric',
    'user_to_median_log_playtime',
    'user_to_mean_log_count',
    'user_to_mean_sentiment',
)


class Bucket5GameBuffers(NamedTuple):
    """Per-item static scalars for Bucket 5 numeric-match cross features.

    Each shaped (n_items+1,) float32 with pad row appended (pad value = 0; never
    indexed since cand_idx stays in [0, n_items)). Registered as non-persistent
    buffers on the ranker (rebuilt from FeatureStore on every load), see
    ranker/train.py:_buffers_from_fs / ranker/model.py constructor.

    `price_bucket` is float-cast from `game_price_idx` (int64 on the model for
    the Embedding lookup) at use time — no separate buffer.
    """
    price_bucket:     torch.Tensor     # (n_items+1,) float32 — 0-8 price bucket as float
    year_numeric:     torch.Tensor     # (n_items+1,) float32 — release year (e.g. 2018.0)
    median_log_hours: torch.Tensor     # (n_items+1,) float32 — log1p(median_hours)
    log_count:        torch.Tensor     # (n_items+1,) float32 — log1p(interaction_count)
    sentiment:        torch.Tensor     # (n_items+1,) float32 — sentiment ordinal 0-7


def numeric_match_quintuple(buffers:   Bucket5GameBuffers,
                            user_b5:   torch.Tensor,    # (B, 5) float32 — user-side aggregates
                            cand_idx:  torch.Tensor,    # (B, n_cand) int64
                            ) -> tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The five Bucket 5 numeric-match cross features for a batch. Returns:

      (price_match, era_gap, playtime_calibration_median,
       popularity_match, sentiment_match)

    each shaped (B, n_cand) float32.

    Match semantics:
      - price_match, era_gap, popularity_match, sentiment_match : abs(user - item)
      - playtime_calibration_median                            : signed (user - item)

    Pre-Z-score values (raw scalars). The model applies Z-score inside score_pairs /
    score_pairs_batched before head concat — keep this util free of normalization
    so precompute writes the raw values to parquet (Z-score buffers are populated
    AFTER precompute, from train-split parquet stats).

    Used by:
      - ranker/precompute.py — once per row, raw values written to parquet
      - ranker/train.py      — once per sampled batch, on-the-fly for random negs
      - ranker/canary.py     — once per synthetic user
    """
    # Per-cand item scalars: gather (B, n_cand)
    cand_price   = buffers.price_bucket[cand_idx]
    cand_year    = buffers.year_numeric[cand_idx]
    cand_median  = buffers.median_log_hours[cand_idx]
    cand_count   = buffers.log_count[cand_idx]
    cand_sent    = buffers.sentiment[cand_idx]

    # User-side scalars: (B, 1) — broadcasts to (B, n_cand)
    u_price   = user_b5[:, B5_USER_MEAN_PRICE     ].unsqueeze(-1)
    u_year    = user_b5[:, B5_USER_MEAN_YEAR      ].unsqueeze(-1)
    u_median  = user_b5[:, B5_USER_MEDIAN_LOG_PT  ].unsqueeze(-1)
    u_count   = user_b5[:, B5_USER_MEAN_LOG_COUNT ].unsqueeze(-1)
    u_sent    = user_b5[:, B5_USER_MEAN_SENTIMENT ].unsqueeze(-1)

    price_match           = (u_price  - cand_price ).abs()
    era_gap               = (u_year   - cand_year  ).abs()
    playtime_cal_median   = (u_median - cand_median)            # signed
    popularity_match      = (u_count  - cand_count ).abs()
    sentiment_match       = (u_sent   - cand_sent  ).abs()

    return price_match, era_gap, playtime_cal_median, popularity_match, sentiment_match
