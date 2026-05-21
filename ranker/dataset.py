"""
Stage 1+ — Ranker dataset (CG-parity feature set + Phase A/B cross features).

Loads ranker_candidates_{train,val}.parquet → RankerDataset with pre-cached tensors.
The model itself does not consume per-game item-feature matrices — those live as
registered buffers on the WideDeepRanker (built from FeatureStore in train.build_ranker).

sample_batch returns the raw inputs (user-tower inputs + per-user Bucket 5 scalars
+ sampled candidate matrix + target). train.py runs user_forward + item_embedding +
computes all cross features on the fly via ranker.cross_features utils (same code
path as precompute → bit-exact parquet identity), then calls compute_cross_features
to stack them for the wide head.

Active cross features (23 total — **Bucket 6** roster, plan §9). Bucket 3 (disliked
slice), Bucket 4 (dev-catalog), and Bucket 7 (item-intrinsic priors) were tried and
dropped / dropped in planning — see plan §9 / git log.

  Phase A:
    1. tag_cosine                  (B-0)  — raw TF-IDF cosine on user's full tag profile.
  Bucket 1 (full-history slice):
    2. genre_overlap               (B-1)  — weighted binary genre overlap / item genre count.
    3. tag_overlap                 (B-1b) — weighted binary TAG overlap, magnitude-aware.
    4. dev_affinity                (B-2)  — playtime-weighted developer match, full hist.
  Bucket 2A (liked-only history slice):
    5. genre_overlap_liked         (B-3a) — same as B-1, restricted to liked games.
    6. tag_overlap_liked           (B-3b) — same as B-1b, restricted to liked games.
    7. dev_affinity_liked          (B-3c) — same as B-2, restricted to liked games.
  Bucket 2B (last-3-liked window):
    8. genre_overlap_recent3       (B-4a) — same as B-1, on last 3 non-pad of X_hist_liked.
    9. tag_overlap_recent3         (B-4b) — same as B-1b, on last 3 non-pad of X_hist_liked.
   10. dev_affinity_recent3        (B-4c) — same as B-2, on last 3 non-pad of X_hist_liked.
  Bucket 5 (numeric matching — scalar arithmetic on per-user/per-item stats):
   11. price_match                 (B-7a) — abs price-bucket diff.
   12. era_gap                     (B-7b) — abs release-year diff.
   13. playtime_cal_median         (B-7c) — SIGNED median-log-playtime diff (user - item).
   14. popularity_match            (B-7d) — abs log-count diff.
   15. sentiment_match             (B-7e) — abs sentiment-ordinal diff.
  Bucket 6 (niche feature crosses — 4 concepts × {full, liked} = 8 features):
   16. tag_overlap_idf_full        (B-9a) — IDF-reweighted tag overlap, full slice.
   17. tag_overlap_idf_liked       (B-9b) — IDF-reweighted tag overlap, liked slice.
   18. niche_tag_match_full        (B-9c) — abs mean-tag-IDF diff, full slice.
   19. niche_tag_match_liked       (B-9d) — abs mean-tag-IDF diff, liked slice.
   20. max_tag_idf_match_full      (B-9e) — abs max-tag-IDF diff (weighted max), full.
   21. max_tag_idf_match_liked     (B-9f) — abs max-tag-IDF diff (weighted max), liked.
   22. niche_dev_match_full        (B-9g) — abs log-dev-catalog-size diff, full slice.
   23. niche_dev_match_liked       (B-9h) — abs log-dev-catalog-size diff, liked slice.

  Bucket 5 + 6 features are RAW values in the parquet — the model Z-scores them
  (cols 10-22, n_wide_normalized=13) inside score_pairs / score_pairs_batched using
  `wide_norm_mean` / `wide_norm_std` persistent buffers, populated once on training
  start via ranker.train.populate_wide_norm_buffers.

TRAIN: computed on the fly via ranker/cross_features.py utils (categorical_overlap_triple
+ last_n_history + numeric_match_quintuple + weighted_overlap + niche_scalar_triple).
EVAL / CANARY: read from parquet (or rebuilt via the same utils for synthetic canary
users — see ranker/canary.py).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import load_features


CANDIDATES_PER_ROW = 100   # 1 label + 99 hard negatives per row (matches precompute.TOP_K_CANDIDATES)


# Column manifests for selective parquet loading. Splitting train vs eval columns
# saves ~40 GB of RAM on the train dataset (where the eval columns are never
# touched — see RankerDataset docstring for details).
#
# USER_AND_LABEL_COLS — needed by sample_batch (training + train-time val_loss):
#   label_item_idx, user history arrays, X_user_b5.
# EVAL_ONLY_COLS — needed by evaluate.compute_label_ranks only:
#   neg_item_idxs, cg_label_rank, and all 23 × {label, negs} = 46 cross-feature columns.
_USER_AND_LABEL_COLS = [
    'label_item_idx',
    'user_avg_log_playtime',
    'X_hist_liked',
    'X_hist_disliked',
    'X_hist_full',
    'X_hist_playtime_weights',
    'X_hist_liked_playtime_weights',
    'X_user_b5',
]
# Cross-feature column manifest — order matches ranker.dataset.compute_cross_features.
# Used both to select what to read AND to set RankerDataset attribute names. Each
# entry is (label_attr / col, negs_attr / col).
_CROSS_FEATURE_COLS = [
    ('tag_cosine_label',                 'tag_cosine_negs'),
    ('genre_overlap_label',              'genre_overlap_negs'),
    ('tag_overlap_label',                'tag_overlap_negs'),
    ('dev_affinity_label',               'dev_affinity_negs'),
    ('genre_overlap_liked_label',        'genre_overlap_liked_negs'),
    ('tag_overlap_liked_label',          'tag_overlap_liked_negs'),
    ('dev_affinity_liked_label',         'dev_affinity_liked_negs'),
    ('genre_overlap_recent3_label',      'genre_overlap_recent3_negs'),
    ('tag_overlap_recent3_label',        'tag_overlap_recent3_negs'),
    ('dev_affinity_recent3_label',       'dev_affinity_recent3_negs'),
    ('price_match_label',                'price_match_negs'),
    ('era_gap_label',                    'era_gap_negs'),
    ('playtime_cal_median_label',        'playtime_cal_median_negs'),
    ('popularity_match_label',           'popularity_match_negs'),
    ('sentiment_match_label',            'sentiment_match_negs'),
    ('tag_overlap_idf_full_label',       'tag_overlap_idf_full_negs'),
    ('tag_overlap_idf_liked_label',      'tag_overlap_idf_liked_negs'),
    ('niche_tag_match_full_label',       'niche_tag_match_full_negs'),
    ('niche_tag_match_liked_label',      'niche_tag_match_liked_negs'),
    ('max_tag_idf_match_full_label',     'max_tag_idf_match_full_negs'),
    ('max_tag_idf_match_liked_label',    'max_tag_idf_match_liked_negs'),
    ('niche_dev_match_full_label',       'niche_dev_match_full_negs'),
    ('niche_dev_match_liked_label',      'niche_dev_match_liked_negs'),
]
_EVAL_ONLY_COLS = ['neg_item_idxs', 'cg_label_rank']
for _l, _n in _CROSS_FEATURE_COLS:
    _EVAL_ONLY_COLS += [_l, _n]


def _scalar_to_numpy(table, name, dtype):
    """pa.Table column → 1-D numpy of the requested dtype.

    Forces a copy (`np.array(..., copy=True)`) so the result is writable —
    pyarrow buffers are immutable, and torch.from_numpy on a read-only buffer
    emits a UserWarning at startup. The copy is cheap (4.3M × 4 bytes ≈ 17 MB)
    and only fires per-column at dataset construction.
    """
    arr = table[name].combine_chunks()
    return np.array(arr.to_numpy(zero_copy_only=False), dtype=dtype, copy=True)


def _fixed_list_to_numpy(table, name, list_size, dtype):
    """pa.Table FixedSizeList column → 2-D numpy (N, list_size). Forces a writable
    copy on the final array — same reason as `_scalar_to_numpy`. The list buffers
    are bigger (e.g. (4.3M, 99) float32 ≈ 1.7 GB) so the copy is more noticeable,
    but is still bounded and transient per column.

    Avoids the `np.stack(df[col].values)` Python-loop allocation pattern that
    the pandas code path forced (which paid Python-side object-array overhead on
    top of the buffer cost).
    """
    arr = table[name].combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    return np.array(flat.reshape(-1, list_size), dtype=dtype, copy=True)


# ── Dataset class ────────────────────────────────────────────────────────────

class RankerDataset:
    """
    Holds compact per-row arrays from one ranker_candidates_{split}.parquet.

    Two load modes — pick based on caller:
      mode='full'       (default) — load EVERY parquet column. Required for
                                    `evaluate.compute_label_ranks` (E2E ceiling
                                    metrics) which reads neg_idx + cg_label_rank
                                    + all 23 × {label, negs} cross features.
      mode='train_only'           — load ONLY the columns `sample_batch` needs
                                    (label_idx + user history arrays + X_user_b5).
                                    Skips ~40 GB of eval-only data — saves the train
                                    dataset from pulling cross feature negs that
                                    training never consumes (train computes cross
                                    features on-the-fly per sampled candidate, see
                                    train._forward_batch). Use this for train_ds
                                    when `n_hard_negs == 0` (current default config),
                                    which means `neg_idx` is also unused.

    Implementation: pyarrow.parquet.read_table with explicit `columns=` list (no
    pandas intermediate). FixedSizeList → numpy via zero-copy flatten + reshape
    (no `np.stack` Python-loop copy that the old pandas path forced). The model has
    its own registered buffers (built in train.build_ranker from FeatureStore) — it
    does NOT consume any item-feature matrix from this class.
    """

    def __init__(self, parquet_path: str, n_items: int, pad_idx: int, mode: str = 'full'):
        if mode not in ('full', 'train_only'):
            raise ValueError(f"mode must be 'full' or 'train_only', got {mode!r}")
        self.mode = mode

        # Select column manifest for this mode.
        columns = list(_USER_AND_LABEL_COLS)
        if mode == 'full':
            columns += _EVAL_ONLY_COLS

        # pyarrow direct read (no pandas). Reading only the selected columns means
        # parquet's per-column compression + projection pushdown also halves IO time
        # for train_only loads.
        table = pq.read_table(parquet_path, columns=columns)

        # ── User-tower inputs (loaded in both modes) ────────────────────────
        # int64 for history indices because torch.gather() / torch.scatter_add_()
        # require int64 indices downstream; .astype with copy=False reuses the
        # buffer if the source is already int64 (parquet stores int32 here so one
        # copy is needed). Float arrays load zero-copy.
        # Inferred from the first 2-D column — same value for every history column.
        h_table_arr = table['X_hist_liked'].combine_chunks()
        max_hist = h_table_arr.type.list_size

        self.label_idx         = _scalar_to_numpy(table, 'label_item_idx',        np.int32)
        self.X_user_avg_log    = _scalar_to_numpy(table, 'user_avg_log_playtime', np.float32)
        self.X_hist_liked      = _fixed_list_to_numpy(table, 'X_hist_liked',                  max_hist, np.int64)
        self.X_hist_disliked   = _fixed_list_to_numpy(table, 'X_hist_disliked',               max_hist, np.int64)
        self.X_hist_full       = _fixed_list_to_numpy(table, 'X_hist_full',                   max_hist, np.int64)
        self.X_hist_pw         = _fixed_list_to_numpy(table, 'X_hist_playtime_weights',       max_hist, np.float32)
        # Bucket 2 — playtime weights for the LIKED slice (parallel to X_hist_liked,
        # normalized to sum 1 over non-pad liked entries; 0 at pad). Distinct from
        # X_hist_pw (which is normalized over the FULL slice).
        self.X_hist_liked_pw   = _fixed_list_to_numpy(table, 'X_hist_liked_playtime_weights', max_hist, np.float32)
        # Bucket 5 — per-row user-side numeric aggregates (5 scalars). Order matches
        # B5_USER_* indices in ranker/cross_features.py. Same value repeated for all
        # rollback positions of the same user (full-history aggregates).
        self.X_user_b5         = _fixed_list_to_numpy(table, 'X_user_b5',                     5,        np.float32)

        # ── Eval-only columns ──────────────────────────────────────────────
        # neg_idx + cg_label_rank + all 23 cross-feature pairs (Bucket 1 → 6).
        # See `_CROSS_FEATURE_COLS` at module level for the full list / order.
        # Used by evaluate.compute_label_ranks; NOT referenced by sample_batch.
        if mode == 'full':
            n_neg = table['neg_item_idxs'].combine_chunks().type.list_size
            self.neg_idx       = _fixed_list_to_numpy(table, 'neg_item_idxs', n_neg, np.int32)
            self.cg_label_rank = _scalar_to_numpy(table, 'cg_label_rank', np.int32)
            for label_col, negs_col in _CROSS_FEATURE_COLS:
                setattr(self, label_col, _scalar_to_numpy(table, label_col, np.float32))
                setattr(self, negs_col,  _fixed_list_to_numpy(table, negs_col, n_neg, np.float32))
            self.n_neg = n_neg
        else:
            # train_only: no neg pool stored (sample_batch only uses dataset.label_idx
            # for col 0 + sampled random negs from rng). n_neg matches the corpus
            # default so callers that introspect it (e.g. config-time `n_hard_take`
            # clamp) see the right ceiling even though the array isn't materialized.
            self.n_neg = CANDIDATES_PER_ROW - 1

        # ── Drop the pa.Table reference so its buffers can be reclaimed once
        # the numpy arrays above are detached. The numpy arrays may share buffers
        # with the table (zero-copy) or be detached copies (dtype mismatch on
        # astype); either way the table itself is no longer needed.
        del table

        self.N        = len(self.label_idx)
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
        self._X_user_b5_t          = torch.from_numpy(self.X_user_b5)                     # (N, 5)

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
        self._X_user_b5_t          = self._X_user_b5_t.to(device)
        del self.X_user_avg_log, self.X_hist_liked, self.X_hist_liked_pw
        del self.X_hist_disliked, self.X_hist_full, self.X_hist_pw, self.X_user_b5
        return self


# ── Cross-feature computation (shared by sample_batch + evaluate + canary) ──

def compute_cross_features(tag_cosine:                 torch.Tensor,
                           genre_overlap:              torch.Tensor,
                           tag_overlap:                torch.Tensor,
                           dev_affinity:               torch.Tensor,
                           genre_overlap_liked:        torch.Tensor,
                           tag_overlap_liked:          torch.Tensor,
                           dev_affinity_liked:         torch.Tensor,
                           genre_overlap_recent3:      torch.Tensor,
                           tag_overlap_recent3:        torch.Tensor,
                           dev_affinity_recent3:       torch.Tensor,
                           price_match:                torch.Tensor,
                           era_gap:                    torch.Tensor,
                           playtime_cal_median:        torch.Tensor,
                           popularity_match:           torch.Tensor,
                           sentiment_match:            torch.Tensor,
                           tag_overlap_idf_full:       torch.Tensor,
                           tag_overlap_idf_liked:      torch.Tensor,
                           niche_tag_match_full:       torch.Tensor,
                           niche_tag_match_liked:      torch.Tensor,
                           max_tag_idf_match_full:     torch.Tensor,
                           max_tag_idf_match_liked:    torch.Tensor,
                           niche_dev_match_full:       torch.Tensor,
                           niche_dev_match_liked:      torch.Tensor) -> torch.Tensor:
    """
    Bucket 6 wide-path stacker. Returns (B, 23).

    Inputs are 1-D (B,) tensors of per-(row, candidate) scalars. Column order in
    the output tensor must match `n_cross_features=23` and the head weight slot
    interpretation — checkpoints rely on this stable ordering:

        column  0 : tag_cosine                   (B-0,  Phase A)
        column  1 : genre_overlap                (B-1,  Bucket 1)
        column  2 : tag_overlap                  (B-1b, Bucket 1)
        column  3 : dev_affinity                 (B-2,  Bucket 1)
        column  4 : genre_overlap_liked          (B-3a, Bucket 2A)
        column  5 : tag_overlap_liked            (B-3b, Bucket 2A)
        column  6 : dev_affinity_liked           (B-3c, Bucket 2A)
        column  7 : genre_overlap_recent3        (B-4a, Bucket 2B)
        column  8 : tag_overlap_recent3          (B-4b, Bucket 2B)
        column  9 : dev_affinity_recent3         (B-4c, Bucket 2B)
        column 10 : price_match                  (B-7a, Bucket 5)
        column 11 : era_gap                      (B-7b, Bucket 5)
        column 12 : playtime_cal_median          (B-7c, Bucket 5)
        column 13 : popularity_match             (B-7d, Bucket 5)
        column 14 : sentiment_match              (B-7e, Bucket 5)
        column 15 : tag_overlap_idf_full         (B-9a, Bucket 6)
        column 16 : tag_overlap_idf_liked        (B-9b, Bucket 6)
        column 17 : niche_tag_match_full         (B-9c, Bucket 6)
        column 18 : niche_tag_match_liked        (B-9d, Bucket 6)
        column 19 : max_tag_idf_match_full       (B-9e, Bucket 6)
        column 20 : max_tag_idf_match_liked      (B-9f, Bucket 6)
        column 21 : niche_dev_match_full         (B-9g, Bucket 6)
        column 22 : niche_dev_match_liked        (B-9h, Bucket 6)

    Cols 10-22 are RAW values — the model Z-scores them inside score_pairs /
    score_pairs_batched (model.wide_norm_mean / wide_norm_std, populated by
    train.populate_wide_norm_buffers from train-parquet stats). Cols 0-9 are
    bounded in [-1, 1] or [0, 1] and pass through unchanged.

    Future buckets append further columns at indices ≥ 23; do not reorder existing
    columns or older checkpoints become silently mis-aligned at load time.
    """
    return torch.stack([tag_cosine, genre_overlap, tag_overlap, dev_affinity,
                        genre_overlap_liked, tag_overlap_liked, dev_affinity_liked,
                        genre_overlap_recent3, tag_overlap_recent3, dev_affinity_recent3,
                        price_match, era_gap, playtime_cal_median,
                        popularity_match, sentiment_match,
                        tag_overlap_idf_full, tag_overlap_idf_liked,
                        niche_tag_match_full, niche_tag_match_liked,
                        max_tag_idf_match_full, max_tag_idf_match_liked,
                        niche_dev_match_full, niche_dev_match_liked],
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

    Returns 9-tuple (all on `device`):
      X_user_avg_log     (B, 1)
      X_hist_liked       (B, H)
      X_hist_liked_pw    (B, H)   ← Bucket 2: weights normalized over liked slice
      X_hist_disliked    (B, H)
      X_hist_full        (B, H)
      X_hist_pw          (B, H)   ← weights normalized over full slice
      X_user_b5          (B, 5)   ← Bucket 5: 5 per-user numeric aggregates
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
        if dataset.mode != 'full':
            raise RuntimeError(
                f"sample_batch needs hard negs (n_hard_take={n_hard_take}) but dataset "
                f"was loaded with mode='{dataset.mode}' (no neg_idx column). Either "
                f"load with mode='full' or set n_hard_negs=0 in the training config."
            )
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
        dataset._X_user_b5_t[rows_t].to(device),
        cand_t,
        target_t,
    )


# ── Public loader ───────────────────────────────────────────────────────────

def load_splits(data_dir: str = 'data', train_mode: str = 'full') -> tuple:
    """Returns (train_dataset, val_dataset, FeatureStore).

    train_mode='train_only' skips ~40 GB of eval-only columns on the train dataset
    (cross features, neg_idx, cg_label_rank — none of which sample_batch uses when
    n_hard_negs=0). Default 'full' keeps backward compat. val_ds always loads
    mode='full' since evaluation runs on it.
    """
    fs       = load_features(data_dir=data_dir)
    n_items  = fs['n_items']
    pad_idx  = n_items                                    # matches src/model.py game_pad_idx

    train_ds = RankerDataset(os.path.join(data_dir, 'ranker_candidates_train.parquet'),
                              n_items=n_items, pad_idx=pad_idx, mode=train_mode)
    val_ds   = RankerDataset(os.path.join(data_dir, 'ranker_candidates_val.parquet'),
                              n_items=n_items, pad_idx=pad_idx, mode='full')
    print(f"Train: {train_ds.N:,} rollback rows ({train_ds.mode})  |  "
          f"Val: {val_ds.N:,} rollback rows ({val_ds.mode})")
    print(f"  candidates_per_row={CANDIDATES_PER_ROW}  max_hist={train_ds.max_hist}")
    return train_ds, val_ds, fs
