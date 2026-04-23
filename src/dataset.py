"""
Stage 3 — Dataset Building
Builds rollback training examples for in-batch negatives softmax training.

Usage (from train.py or main.py):
    from src.dataset import load_features, make_softmax_splits, save_softmax_splits, load_softmax_splits
"""
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from src.features import load_features, FEATURES_VERSION

MAX_ROLLBACK_EXAMPLES_PER_USER = 50


# ── Rollback dataset builder ──────────────────────────────────────────────────

def _build_rollback_dataset(users: list, fs: dict,
                             max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
                             seed: int = 42) -> tuple:
    """
    Build rollback examples for a list of users.

    For user with play history [g0, g1, ..., gN]:
      - Sample up to max_per_user target positions from {1, ..., N}
      - For target at position i: context = [g0, ..., g_{i-1}]

    No timestamps — items are in file order from base_interactions.parquet.

    Genre context is computed from the rollback context slice using the same
    formula as features.py: [debiased_avg_log_playtime_per_genre | play_frac].
    Debiasing uses the user's full-history avg_log_playtime (stored in fs) as
    the reference — same known approximation as the book model uses for avg_rating.

    Returns 8-tuple:
        [0] X_genre           — (N, 2*n_genres) float32  user genre context from rollback slice
        [1] X_history         — list[list[int]]           item indices in context
        [2] X_history_weights — list[list[float]]         re-normalized log(1+h) weights for context
        [3] target_item_idx   — (N,) int64
        [4] target_genre      — (N, n_genres) float32
        [5] target_dev_idx    — (N,) int64
        [6] target_year_idx   — (N,) int64
        [7] target_price      — (N,) int64
    """
    rng      = random.Random(seed)
    n_genres = fs['n_genres']

    game_genre_matrix = fs['game_genre_matrix']   # (n_items, n_genres) float32
    game_dev_idx      = fs['game_developer_idx']
    game_year_idx     = fs['game_year_idx']
    game_price        = fs['game_price_bucket']

    # Precompute genre index lists per item (avoids np.where in inner loop)
    item_genre_idxs = [
        np.where(game_genre_matrix[i] > 0)[0].tolist()
        for i in range(fs['n_items'])
    ]

    X_genre           = []
    X_history         = []
    X_history_weights = []
    target_item_idx   = []
    target_genre      = []
    target_dev_idx    = []
    target_year_idx   = []
    target_price      = []

    for uid in tqdm(users, desc="Building rollback examples"):
        history = fs['user_to_play_history'].get(uid, [])
        weights = fs['user_to_play_weights'].get(uid, [])
        n = len(history)
        if n < 2:
            continue

        # Recover approximate raw log(1+h) values from normalized weights.
        # weights[i] = log(1+h_i) / sum_j(log(1+h_j))
        # raw[i] = weights[i] * n * avg_log  (since sum_j/n = avg_log)
        avg_log   = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = n * avg_log
        raw_logs  = [w * total_log for w in weights]

        # Sample target positions
        k       = min(max_per_user, n - 1)
        sampled = sorted(rng.sample(range(1, n), k))
        sampled_set = set(sampled)

        # Single left-to-right scan with genre accumulators
        running_count = np.zeros(n_genres, dtype=np.float32)
        running_sum   = np.zeros(n_genres, dtype=np.float32)
        ctx_ids      = []
        ctx_raw_logs = []

        for pos, (item_idx, raw_log) in enumerate(zip(history, raw_logs)):
            if pos in sampled_set:
                # Genre context from context slice [0, pos-1]
                total_assign = running_count.sum()
                genre_ctx = np.zeros(2 * n_genres, dtype=np.float32)
                if total_assign > 0:
                    mask = running_count > 0
                    genre_ctx[:n_genres][mask] = (
                        running_sum[mask] / running_count[mask]
                    ) - avg_log                                # debiased avg log-playtime per genre
                    genre_ctx[n_genres:] = running_count / total_assign  # play fraction

                # Re-normalize context log(1+h) weights over the slice
                total_ctx = sum(ctx_raw_logs) or 1.0
                norm_weights = [rl / total_ctx for rl in ctx_raw_logs]

                X_genre.append(genre_ctx.tolist())
                X_history.append(list(ctx_ids))
                X_history_weights.append(norm_weights)
                target_item_idx.append(item_idx)
                target_genre.append(game_genre_matrix[item_idx].tolist())
                target_dev_idx.append(int(game_dev_idx[item_idx]))
                target_year_idx.append(int(game_year_idx[item_idx]))
                target_price.append(int(game_price[item_idx]))

            # Update context and accumulators
            ctx_ids.append(item_idx)
            ctx_raw_logs.append(raw_log)
            for g_idx in item_genre_idxs[item_idx]:
                running_count[g_idx] += 1
                running_sum[g_idx]   += raw_log

    n = len(target_item_idx)
    print(f"  {n:,} rollback examples — building tensors ...")

    X_genre_t         = torch.from_numpy(np.array(X_genre,          dtype=np.float32))
    target_item_idx_t = torch.from_numpy(np.array(target_item_idx,  dtype=np.int64))
    target_genre_t    = torch.from_numpy(np.array(target_genre,     dtype=np.float32))
    target_dev_idx_t  = torch.from_numpy(np.array(target_dev_idx,   dtype=np.int64))
    target_year_idx_t = torch.from_numpy(np.array(target_year_idx,  dtype=np.int64))
    target_price_t    = torch.from_numpy(np.array(target_price,     dtype=np.int64))

    return (X_genre_t, X_history, X_history_weights, target_item_idx_t,
            target_genre_t, target_dev_idx_t, target_year_idx_t, target_price_t)


# ── Padding helpers ───────────────────────────────────────────────────────────

def pad_history_batch(histories: list, pad_idx: int) -> torch.Tensor:
    """Pad a list of variable-length index lists to a (B, max_len) tensor."""
    max_len = max((len(h) for h in histories), default=1)
    padded  = torch.full((len(histories), max_len), pad_idx, dtype=torch.long)
    for i, hist in enumerate(histories):
        if hist:
            padded[i, :len(hist)] = torch.tensor(hist, dtype=torch.long)
    return padded


def pad_weights_batch(weight_lists: list) -> torch.Tensor:
    """Pad a list of variable-length weight lists to a (B, max_len) tensor."""
    max_len = max((len(w) for w in weight_lists), default=1)
    padded  = torch.zeros(len(weight_lists), max_len, dtype=torch.float)
    for i, weights in enumerate(weight_lists):
        if weights:
            padded[i, :len(weights)] = torch.tensor(weights, dtype=torch.float)
    return padded


# ── Orchestrator ──────────────────────────────────────────────────────────────

def make_softmax_splits(fs: dict, data_dir: str = 'data',
                        max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
                        seed: int = 42) -> tuple:
    """
    Build rollback train and val datasets from the feature store.
    Train/val split is already set in fs['train_users'] / fs['val_users'].
    Returns (train_data, val_data), each an 8-tuple of tensors/lists.
    """
    train_users = fs['train_users']
    val_users   = fs['val_users']

    print(f"Building softmax train dataset ({len(train_users):,} users) ...")
    train_data = _build_rollback_dataset(train_users, fs, max_per_user, seed)
    print(f"  train examples: {train_data[0].shape[0]:,}")

    print(f"\nBuilding softmax val dataset ({len(val_users):,} users) ...")
    val_data = _build_rollback_dataset(val_users, fs, max_per_user, seed + 1)
    print(f"  val examples:   {val_data[0].shape[0]:,}")

    return train_data, val_data


def save_softmax_splits(train_data: tuple, val_data: tuple,
                        data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_softmax_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_softmax_val_{version}.pt'))
    print(f"✓ Saved dataset_softmax_train_{version}.pt and dataset_softmax_val_{version}.pt → {data_dir}/")


def load_softmax_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_softmax_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_softmax_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data
