"""
Stage 1+ — Ranker dataset (CG-parity feature set + Phase A/B cross features).

Loads ranker_candidates_{train,val}.parquet → RankerDataset with pre-cached tensors.
The model itself does not consume per-game item-feature matrices — those live as
registered buffers on the WideDeepRanker (built from FeatureStore in train.build_ranker).

sample_batch returns the raw inputs (user-tower inputs + sampled candidate matrix +
target). train.py runs user_forward + item_embedding + computes all cross features on
the fly via ranker.cross_features utils (same code path as precompute → bit-exact
parquet identity), then calls compute_cross_features to stack them for the wide head.

Active cross features (10 total — **Bucket 2** roster, plan §9). Bucket 3 (disliked
slice) was tried and dropped — the Steam "disliked" partition is too noisy (most users
don't review games; the hours-based heuristic flags below-average games for heavy
players that they actually liked) and adding it slightly regressed vs Bucket 2 across
every headline metric. See plan §9 / git log for the comparison.
  Phase A:
    1. tag_cosine               (B-0)  — raw TF-IDF cosine on user's full tag profile.
  Bucket 1 (full-history slice):
    2. genre_overlap            (B-1)  — weighted binary genre overlap / item genre count.
    3. tag_overlap              (B-1b) — weighted binary TAG overlap, magnitude-aware.
    4. dev_affinity             (B-2)  — playtime-weighted developer match, full hist.
  Bucket 2A (liked-only history slice):
    5. genre_overlap_liked      (B-3a) — same as B-1, restricted to liked games.
    6. tag_overlap_liked        (B-3b) — same as B-1b, restricted to liked games.
    7. dev_affinity_liked       (B-3c) — same as B-2, restricted to liked games.
  Bucket 2B (last-3-liked window):
    8. genre_overlap_recent3    (B-4a) — same as B-1, on last 3 non-pad of X_hist_liked.
    9. tag_overlap_recent3      (B-4b) — same as B-1b, on last 3 non-pad of X_hist_liked.
   10. dev_affinity_recent3     (B-4c) — same as B-2, on last 3 non-pad of X_hist_liked.

TRAIN: computed on the fly via ranker/cross_features.py utils (categorical_overlap_triple
+ last_n_history). EVAL / CANARY: read from parquet (or rebuilt via the same utils
for synthetic canary users — see ranker/canary.py).

Plan §9 Buckets 4-6 land here in future phases — see Wide-feature normalization
section in plan §9 for fixed-stats persistent buffers and gain=0.1 init.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import load_features


CANDIDATES_PER_ROW = 100   # 1 label + 99 hard negatives per row (matches precompute.TOP_K_CANDIDATES)


# ── Dataset class ────────────────────────────────────────────────────────────

class RankerDataset:
    """
    Holds compact per-row arrays from one ranker_candidates_{split}.parquet.

    Cross-feature scalars (currently just tag_cosine) are loaded from the parquet
    and exposed via sample_batch / evaluate.compute_label_ranks. The model has its
    own registered buffers (built in train.build_ranker from FeatureStore) — it does
    NOT consume any item-feature matrix from this class.
    """

    def __init__(self, parquet_path: str, n_items: int, pad_idx: int):
        df = pd.read_parquet(parquet_path)

        # Corpus-index columns. Keep int32 — n_items ~5.4k fits easily.
        self.label_idx     = df['label_item_idx'].values.astype(np.int32)              # (N,)
        self.neg_idx       = np.stack(df['neg_item_idxs'].values).astype(np.int32)     # (N, 99)
        self.cg_label_rank = df['cg_label_rank'].values.astype(np.int32)               # (N,)

        # Cross features (precomputed by precompute.py — Bucket 2 roster, 10 features total).
        # Phase A:
        #   B-0  tag_cosine             : raw TF-IDF cosine of user tag pool vs candidate
        # Bucket 1 (full-history slice):
        #   B-1  genre_overlap          : weighted binary genre overlap / item genre count
        #   B-1b tag_overlap            : weighted binary tag overlap / item tag count
        #   B-2  dev_affinity           : playtime-weighted fraction of user history under candidate's developer
        # Bucket 2A (liked-only history slice):
        #   B-3a genre_overlap_liked    : same as B-1, restricted to liked games
        #   B-3b tag_overlap_liked      : same as B-1b, restricted to liked games
        #   B-3c dev_affinity_liked     : same as B-2, restricted to liked games
        # Bucket 2B (last-3-liked window):
        #   B-4a genre_overlap_recent3  : same as B-1, on last 3 non-pad of X_hist_liked
        #   B-4b tag_overlap_recent3    : same as B-1b, on last 3 non-pad of X_hist_liked
        #   B-4c dev_affinity_recent3   : same as B-2, on last 3 non-pad of X_hist_liked
        self.tag_cosine_label             = df['tag_cosine_label'].values.astype(np.float32)              # (N,)
        self.tag_cosine_negs              = np.stack(df['tag_cosine_negs'].values).astype(np.float32)     # (N, 99)
        # Bucket 1
        self.genre_overlap_label          = df['genre_overlap_label'].values.astype(np.float32)
        self.genre_overlap_negs           = np.stack(df['genre_overlap_negs'].values).astype(np.float32)
        self.tag_overlap_label            = df['tag_overlap_label'].values.astype(np.float32)
        self.tag_overlap_negs             = np.stack(df['tag_overlap_negs'].values).astype(np.float32)
        self.dev_affinity_label           = df['dev_affinity_label'].values.astype(np.float32)
        self.dev_affinity_negs            = np.stack(df['dev_affinity_negs'].values).astype(np.float32)
        # Bucket 2A — liked slice
        self.genre_overlap_liked_label    = df['genre_overlap_liked_label'].values.astype(np.float32)
        self.genre_overlap_liked_negs     = np.stack(df['genre_overlap_liked_negs'].values).astype(np.float32)
        self.tag_overlap_liked_label      = df['tag_overlap_liked_label'].values.astype(np.float32)
        self.tag_overlap_liked_negs       = np.stack(df['tag_overlap_liked_negs'].values).astype(np.float32)
        self.dev_affinity_liked_label     = df['dev_affinity_liked_label'].values.astype(np.float32)
        self.dev_affinity_liked_negs      = np.stack(df['dev_affinity_liked_negs'].values).astype(np.float32)
        # Bucket 2B — recent-3 window
        self.genre_overlap_recent3_label  = df['genre_overlap_recent3_label'].values.astype(np.float32)
        self.genre_overlap_recent3_negs   = np.stack(df['genre_overlap_recent3_negs'].values).astype(np.float32)
        self.tag_overlap_recent3_label    = df['tag_overlap_recent3_label'].values.astype(np.float32)
        self.tag_overlap_recent3_negs     = np.stack(df['tag_overlap_recent3_negs'].values).astype(np.float32)
        self.dev_affinity_recent3_label   = df['dev_affinity_recent3_label'].values.astype(np.float32)
        self.dev_affinity_recent3_negs    = np.stack(df['dev_affinity_recent3_negs'].values).astype(np.float32)

        # User-tower inputs (pre-padded to MAX_HISTORY_LEN by precompute).
        self.X_user_avg_log     = df['user_avg_log_playtime'].values.astype(np.float32)
        self.X_hist_liked       = np.stack(df['X_hist_liked'].values).astype(np.int64)
        self.X_hist_disliked    = np.stack(df['X_hist_disliked'].values).astype(np.int64)
        self.X_hist_full        = np.stack(df['X_hist_full'].values).astype(np.int64)
        self.X_hist_pw          = np.stack(df['X_hist_playtime_weights'].values).astype(np.float32)
        # Bucket 2 — playtime weights for the LIKED slice (parallel to X_hist_liked,
        # normalized to sum 1 over non-pad liked entries; 0 at pad). Distinct from
        # X_hist_pw (which is normalized over the FULL slice).
        self.X_hist_liked_pw    = np.stack(df['X_hist_liked_playtime_weights'].values).astype(np.float32)

        self.N        = len(self.label_idx)
        self.n_neg    = self.neg_idx.shape[1]
        self.max_hist = self.X_hist_full.shape[1]
        self.pad_idx  = pad_idx
        self.n_items  = n_items

        # User-tower inputs are big (N × 50 ints/floats × 5 arrays ≈ 2.7 GB at train scale)
        # and only ever fetched as device tensors → cache as tensors. Lookups (label_idx,
        # neg_idx, cg_label_rank) and cross-feature scalars (tag_cosine_*) stay numpy:
        # sample_batch builds the cand matrix via numpy slicing/np.random; evaluate.py
        # reads them directly for the 100-cand eval pool. Both are awkward on-device.
        self._X_user_avg_log_t     = torch.from_numpy(self.X_user_avg_log).unsqueeze(-1)  # (N, 1)
        self._X_hist_liked_t       = torch.from_numpy(self.X_hist_liked)
        self._X_hist_liked_pw_t    = torch.from_numpy(self.X_hist_liked_pw)
        self._X_hist_disliked_t    = torch.from_numpy(self.X_hist_disliked)
        self._X_hist_full_t        = torch.from_numpy(self.X_hist_full)
        self._X_hist_pw_t          = torch.from_numpy(self.X_hist_pw)

    def to(self, device: torch.device) -> 'RankerDataset':
        """Move user-tower tensors to device, then drop the numpy originals to free RAM.
        Numpy lookup/scalar arrays (label_idx, neg_idx, cg_label_rank, *_label/*_negs)
        stay on host — sample_batch and evaluate.py read them via numpy slicing."""
        self._X_user_avg_log_t     = self._X_user_avg_log_t.to(device)
        self._X_hist_liked_t       = self._X_hist_liked_t.to(device)
        self._X_hist_liked_pw_t    = self._X_hist_liked_pw_t.to(device)
        self._X_hist_disliked_t    = self._X_hist_disliked_t.to(device)
        self._X_hist_full_t        = self._X_hist_full_t.to(device)
        self._X_hist_pw_t          = self._X_hist_pw_t.to(device)
        del self.X_user_avg_log, self.X_hist_liked, self.X_hist_liked_pw
        del self.X_hist_disliked, self.X_hist_full, self.X_hist_pw
        return self


# ── Cross-feature computation (shared by sample_batch + evaluate + canary) ──

def compute_cross_features(tag_cosine:             torch.Tensor,
                           genre_overlap:          torch.Tensor,
                           tag_overlap:            torch.Tensor,
                           dev_affinity:           torch.Tensor,
                           genre_overlap_liked:    torch.Tensor,
                           tag_overlap_liked:      torch.Tensor,
                           dev_affinity_liked:     torch.Tensor,
                           genre_overlap_recent3:  torch.Tensor,
                           tag_overlap_recent3:    torch.Tensor,
                           dev_affinity_recent3:   torch.Tensor) -> torch.Tensor:
    """
    Bucket 2 wide-path stacker. Returns (B, 10).

    Inputs are 1-D (B,) tensors of per-(row, candidate) scalars. Column order in
    the output tensor must match `n_cross_features=10` and the head weight slot
    interpretation — checkpoints rely on this stable ordering:

        column 0  : tag_cosine              (B-0,  Phase A)
        column 1  : genre_overlap           (B-1,  Bucket 1)
        column 2  : tag_overlap             (B-1b, Bucket 1)
        column 3  : dev_affinity            (B-2,  Bucket 1)
        column 4  : genre_overlap_liked     (B-3a, Bucket 2A)
        column 5  : tag_overlap_liked       (B-3b, Bucket 2A)
        column 6  : dev_affinity_liked      (B-3c, Bucket 2A)
        column 7  : genre_overlap_recent3   (B-4a, Bucket 2B)
        column 8  : tag_overlap_recent3     (B-4b, Bucket 2B)
        column 9  : dev_affinity_recent3    (B-4c, Bucket 2B)

    Future buckets append further columns at indices ≥ 10; do not reorder existing
    columns or older checkpoints become silently mis-aligned at load time.
    """
    return torch.stack([tag_cosine, genre_overlap, tag_overlap, dev_affinity,
                        genre_overlap_liked, tag_overlap_liked, dev_affinity_liked,
                        genre_overlap_recent3, tag_overlap_recent3, dev_affinity_recent3],
                       dim=-1)


# ── Sampled softmax batch sampler ──────────────────────────────────────────

def sample_batch(dataset: RankerDataset, batch_size: int, device: torch.device,
                 rng: np.random.Generator,
                 n_random_negs: int = 999,
                 n_hard_negs:   int = 99) -> tuple:
    """
    Sampled softmax. Each row's softmax pool is:
        1 label  +  min(n_hard_negs, dataset.n_neg) hard negs (top-N from parquet)
                 +  n_random_negs sampled corpus items

    Defaults (999, 99) match `train.get_config()`. The training loop passes both
    explicitly, so these defaults are only used by ad-hoc callers.

    `n_hard_negs` semantics:
      - 0                       → no hard negs (Phase A ablation A-N2: pure random)
      - 1..dataset.n_neg        → top-N hard negs from parquet (CG-ranked confusables)
      - > dataset.n_neg         → silently clipped to dataset.n_neg

    The standard expensive-softmax workaround (Bengio 2003 / word2vec NEG):
      - Hard negs preserved → fine discrimination signal (label vs CG-confusables)
      - Random negs anchor the broad landscape (proxy for full-softmax easy items)
      - Cost scales with (1 + n_hard + n_random) per row instead of full corpus (5,437)
      - Near-unbiased estimator of full softmax gradient; bias affects calibration
        but not rank order (which is what NDCG/MRR care about)

    Cross features (tag_cosine, etc.) are computed on-the-fly in train.py for the
    sampled candidates — mathematically identical to precompute values.

    Returns 8-tuple (all on `device`):
      X_user_avg_log     (B, 1)
      X_hist_liked       (B, H)
      X_hist_liked_pw    (B, H)   ← Bucket 2: weights normalized over liked slice
      X_hist_disliked    (B, H)
      X_hist_full        (B, H)
      X_hist_pw          (B, H)   ← weights normalized over full slice
      cand_idx           (B, 1 + n_hard_take + n_random)   ← label at col 0
      target             (B,)     ← all zeros (label is at column 0)
    """
    rows = rng.integers(0, dataset.N, size=batch_size)
    rows_t = torch.from_numpy(rows).long()

    n_hard_take = max(0, min(int(n_hard_negs), dataset.n_neg))
    n_total     = 1 + n_hard_take + n_random_negs

    cand = np.empty((batch_size, n_total), dtype=np.int64)
    cand[:, 0] = dataset.label_idx[rows]
    if n_hard_take > 0:
        cand[:, 1:1 + n_hard_take] = dataset.neg_idx[rows][:, :n_hard_take]
    # Random negs sampled fresh per step from the corpus. Occasional collision with
    # label (~few % per row) is harmless — counts label twice in denominator,
    # negligible gradient effect.
    cand[:, 1 + n_hard_take:] = rng.integers(0, dataset.n_items,
                                              size=(batch_size, n_random_negs))

    cand_t   = torch.from_numpy(cand).to(device)
    target_t = torch.zeros(batch_size, dtype=torch.long, device=device)

    return (
        dataset._X_user_avg_log_t[rows_t].to(device),
        dataset._X_hist_liked_t[rows_t].to(device),
        dataset._X_hist_liked_pw_t[rows_t].to(device),
        dataset._X_hist_disliked_t[rows_t].to(device),
        dataset._X_hist_full_t[rows_t].to(device),
        dataset._X_hist_pw_t[rows_t].to(device),
        cand_t,
        target_t,
    )


# ── Public loader ───────────────────────────────────────────────────────────

def load_splits(data_dir: str = 'data') -> tuple:
    """Returns (train_dataset, val_dataset, FeatureStore)."""
    fs       = load_features(data_dir=data_dir)
    n_items  = fs['n_items']
    pad_idx  = n_items                                    # matches src/model.py game_pad_idx

    train_ds = RankerDataset(os.path.join(data_dir, 'ranker_candidates_train.parquet'),
                              n_items=n_items, pad_idx=pad_idx)
    val_ds   = RankerDataset(os.path.join(data_dir, 'ranker_candidates_val.parquet'),
                              n_items=n_items, pad_idx=pad_idx)
    print(f"Train: {train_ds.N:,} rollback rows  |  Val: {val_ds.N:,} rollback rows")
    print(f"  candidates_per_row={CANDIDATES_PER_ROW}  max_hist={train_ds.max_hist}")
    return train_ds, val_ds, fs
