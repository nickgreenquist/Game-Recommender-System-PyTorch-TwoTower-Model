"""
Stage 3 — Dataset Building
Builds rollback training examples for in-batch negatives softmax training.

Usage (from train.py or main.py):
    from src.dataset import load_features, make_softmax_splits, save_softmax_splits, load_softmax_splits
"""
import os
import random
import math

import numpy as np
import torch
from tqdm import tqdm

from src.features import load_features, FEATURES_VERSION

MAX_ROLLBACK_EXAMPLES_PER_USER = 50
MAX_HISTORY_LEN                = 50   # cap context fed to the user tower (most recent N games)
N_SHUFFLES                     = 3    # independent shuffles per user; multiplies training examples


# ── Rollback dataset builder ──────────────────────────────────────────────────

def _build_rollback_dataset(users: list, fs: dict,
                             max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
                             n_shuffles: int = 1,
                             seed: int = 42) -> tuple:
    """
    Build rollback examples for a list of users.

    For user with play history [g0, g1, ..., gN]:
      - Sample up to max_per_user target positions from {1, ..., N}
      - For target at position i: context = [g0, ..., g_{i-1}]

    Returns tuple:
        [0] X_genre           — (N, 2*n_genres) float32
        [1] X_tag             — (N, n_tags) float32
        [2] X_history_liked   — list[list[int]]
        [3] X_history_disliked — list[list[int]]
        [4] X_history_full    — list[list[int]]
        [5] target_item_idx   — (N,) int64
        [6] target_genre      — (N, n_genres) float32
        [7] target_dev_idx    — (N,) int64
        [8] target_year_idx   — (N,) int64
        [9] target_price      — (N,) int64
    """
    rng      = random.Random(seed)
    n_genres = fs['n_genres']
    n_tags   = fs['n_tags']

    game_genre_matrix  = fs['game_genre_matrix']   # (n_items, n_genres) float32
    game_tag_matrix    = fs['game_tag_matrix']     # (n_items, n_tags) float32
    game_median_hours  = fs['game_median_hours']   # (n_items,) float32
    game_dev_idx       = fs['game_developer_idx']
    game_year_idx      = fs['game_year_idx']
    game_price         = fs['game_price_bucket']

    # Precompute genre index lists per item
    item_genre_idxs = [
        np.where(game_genre_matrix[i] > 0)[0].tolist()
        for i in range(fs['n_items'])
    ]

    X_genre            = []
    X_tag              = []
    X_history_liked    = []
    X_history_disliked = []
    X_history_full     = []
    target_item_idx    = []
    target_genre       = []
    target_dev_idx     = []
    target_year_idx    = []
    target_price       = []

    for uid in tqdm(users, desc="Building rollback examples"):
        history_orig = fs['user_to_play_history'].get(uid, [])
        weights_orig = fs['user_to_play_weights'].get(uid, [])
        recs_orig    = fs['user_to_recommend_history'].get(uid, [])
        n = len(history_orig)
        if n < 2:
            continue

        # avg_log is a per-user scalar — compute once, order-independent
        avg_log   = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = n * avg_log

        for _ in range(n_shuffles):
            # Fresh independent shuffle for each pass
            indices = list(range(n))
            rng.shuffle(indices)
            history = [history_orig[i] for i in indices]
            weights = [weights_orig[i] for i in indices]
            recs    = [recs_orig[i]    for i in indices]

            raw_logs  = [w * total_log for w in weights]
            raw_hours = [math.exp(rl) - 1.0 for rl in raw_logs]

            # Sample target positions where hours > 0.5
            all_potential = [i for i in range(1, n) if raw_hours[i] > 0.5]
            if not all_potential:
                continue
            k       = min(max_per_user, len(all_potential))
            sampled = sorted(rng.sample(all_potential, k))
            sampled_set = set(sampled)

            # Single left-to-right scan with context accumulators (reset per shuffle)
            running_genre_count = np.zeros(n_genres, dtype=np.float32)
            running_genre_sum   = np.zeros(n_genres, dtype=np.float32)
            running_tag_sum     = np.zeros(n_tags, dtype=np.float32)

            for pos, (item_idx, raw_h, raw_rl, rec) in enumerate(zip(history, raw_hours, raw_logs, recs)):
                if pos in sampled_set:
                    # 1. Rolling Genre Context (Full History)
                    total_assign = running_genre_count.sum()
                    genre_ctx = np.zeros(2 * n_genres, dtype=np.float32)
                    if total_assign > 0:
                        mask = running_genre_count > 0
                        genre_ctx[:n_genres][mask] = (
                            running_genre_sum[mask] / running_genre_count[mask]
                        ) - avg_log
                        genre_ctx[n_genres:] = running_genre_count / total_assign

                    # 2. Rolling Tag Context (Full History)
                    tag_ctx = running_tag_sum.copy()

                    # 3. Triple Pool Partitioning
                    user_median  = np.median(raw_hours[:pos])
                    liked_ids    = []
                    disliked_ids = []

                    for i in range(pos):
                        ctx_iid = history[i]
                        ctx_h   = raw_hours[i]
                        ctx_rec = recs[i]

                        is_liked    = (ctx_rec is True) or (ctx_h >= game_median_hours[ctx_iid]) or (ctx_h >= user_median * 2)
                        is_disliked = (ctx_rec is False) or (0.1 < ctx_h < 1.0) or (ctx_h <= user_median / 2)

                        if is_liked:
                            liked_ids.append(ctx_iid)
                        if is_disliked:
                            disliked_ids.append(ctx_iid)

                    full_ids     = history[max(0, pos - MAX_HISTORY_LEN):pos]
                    liked_ids    = liked_ids[-MAX_HISTORY_LEN:]
                    disliked_ids = disliked_ids[-MAX_HISTORY_LEN:]

                    X_genre.append(genre_ctx.tolist())
                    X_tag.append(tag_ctx.tolist())
                    X_history_liked.append(liked_ids)
                    X_history_disliked.append(disliked_ids)
                    X_history_full.append(full_ids)

                    target_item_idx.append(item_idx)
                    target_genre.append(game_genre_matrix[item_idx].tolist())
                    target_dev_idx.append(int(game_dev_idx[item_idx]))
                    target_year_idx.append(int(game_year_idx[item_idx]))
                    target_price.append(int(game_price[item_idx]))

                # Update context accumulators
                for g_idx in item_genre_idxs[item_idx]:
                    running_genre_count[g_idx] += 1
                    running_genre_sum[g_idx]   += raw_rl
                running_tag_sum += game_tag_matrix[item_idx]

    n_examples = len(target_item_idx)
    print(f"  {n_examples:,} rollback examples — building tensors ...")

    X_genre_t         = torch.from_numpy(np.array(X_genre,          dtype=np.float32))
    X_tag_t           = torch.from_numpy(np.array(X_tag,            dtype=np.float32))
    target_item_idx_t = torch.from_numpy(np.array(target_item_idx,  dtype=np.int64))
    target_genre_t    = torch.from_numpy(np.array(target_genre,     dtype=np.float32))
    target_dev_idx_t  = torch.from_numpy(np.array(target_dev_idx,   dtype=np.int64))
    target_year_idx_t = torch.from_numpy(np.array(target_year_idx,  dtype=np.int64))
    target_price_t    = torch.from_numpy(np.array(target_price,     dtype=np.int64))

    return (X_genre_t, X_tag_t, X_history_liked, X_history_disliked, X_history_full, 
            target_item_idx_t, target_genre_t, target_dev_idx_t, target_year_idx_t, target_price_t)


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
    Returns (train_data, val_data), each a 10-tuple of tensors/lists.
    """
    train_users = fs['train_users']
    val_users   = fs['val_users']

    print(f"Building softmax train dataset ({len(train_users):,} users, {N_SHUFFLES} shuffles) ...")
    train_data = _build_rollback_dataset(train_users, fs, max_per_user, n_shuffles=N_SHUFFLES, seed=seed)
    print(f"  train examples: {train_data[0].shape[0]:,}")

    print(f"\nBuilding softmax val dataset ({len(val_users):,} users, 1 shuffle) ...")
    val_data = _build_rollback_dataset(val_users, fs, max_per_user, n_shuffles=1, seed=seed + 1)
    print(f"  val examples:   {val_data[0].shape[0]:,}")

    return train_data, val_data


def save_softmax_splits(train_data: tuple, val_data: tuple,
                        data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_softmax_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_softmax_val_{version}.pt'))
    print(f"✓ Saved dataset_softmax_train_{version}.pt and dataset_softmax_val_{version}.pt → {data_dir}/")


def _dataset_info(data: tuple) -> tuple[int, float]:
    """Return (n_examples, estimated_gb). Tensors measured exactly; lists estimated."""
    n = data[5].shape[0]
    total = 0
    for x in data:
        if isinstance(x, torch.Tensor):
            total += x.numel() * x.element_size()
        elif isinstance(x, list):
            total += sum(len(h) for h in x) * 28 + len(x) * 56  # CPython int + list overhead
    return n, total / 1e9


def load_softmax_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_softmax_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_softmax_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)

    n_train, gb_train = _dataset_info(train_data)
    n_val,   gb_val   = _dataset_info(val_data)
    print(f"  train: {n_train:>10,} examples  (~{gb_train:.1f} GB)")
    print(f"  val:   {n_val:>10,} examples  (~{gb_val:.1f} GB)")

    return train_data, val_data
