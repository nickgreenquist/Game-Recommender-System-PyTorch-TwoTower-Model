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
    Build rollback examples for a list of users using pre-allocated NumPy arrays
    to minimize memory overhead (prevents 24GB+ RAM explosion).

    Returns tuple of TENSORS:
        [0]  X_user_avg_log             — (N, 1) float32
        [1]  X_history_liked            — (N, MAX_HISTORY_LEN) int32
        [2]  X_history_disliked         — (N, MAX_HISTORY_LEN) int32
        [3]  X_history_full             — (N, MAX_HISTORY_LEN) int32
        [4]  X_history_playtime_weights — (N, MAX_HISTORY_LEN) float32
        [5]  target_item_idx            — (N,) int64
        [6]  target_dev_idx             — (N,) int64
        [7]  target_year_idx            — (N,) int64
        [8]  target_price               — (N,) int64
    """
    rng          = random.Random(seed)
    pad_idx      = fs['n_items']
    game_median  = fs['game_median_hours']
    game_dev     = fs['game_developer_idx']
    game_year    = fs['game_year_idx']
    game_price   = fs['game_price_bucket']

    # ── Step 1: Count total examples (fast first pass) ────────────────────────
    total_examples = 0
    for uid in users:
        history = fs['user_to_play_history'].get(uid, [])
        weights = fs['user_to_play_weights'].get(uid, [])
        n = len(history)
        if n < 2: continue
        
        avg_log   = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = n * avg_log
        
        # We don't shuffle in the count pass, just estimate potential targets
        # based on the hour filter (> 0.5h). This matches sampling logic.
        potential_count = 0
        for w in weights:
            if (w * total_log) > 0.405: # log(1+0.5) approx 0.405
                potential_count += 1
        
        # history[0] can never be a target (needs context)
        if (weights[0] * total_log) > 0.405:
            potential_count -= 1
            
        total_examples += min(max_per_user, max(0, potential_count)) * n_shuffles

    print(f"  Allocating memory for {total_examples:,} examples ...")

    # ── Step 2: Pre-allocate NumPy arrays ─────────────────────────────────────
    # Using float32 and int32 where possible to keep memory tight.
    X_user_avg_log   = np.zeros((total_examples, 1), dtype=np.float32)
    X_hist_liked     = np.full((total_examples, MAX_HISTORY_LEN), pad_idx, dtype=np.int32)
    X_hist_disliked  = np.full((total_examples, MAX_HISTORY_LEN), pad_idx, dtype=np.int32)
    X_hist_full      = np.full((total_examples, MAX_HISTORY_LEN), pad_idx, dtype=np.int32)
    X_hist_weights   = np.zeros((total_examples, MAX_HISTORY_LEN), dtype=np.float32)
    
    target_idx   = np.zeros(total_examples, dtype=np.int64)
    target_dev   = np.zeros(total_examples, dtype=np.int64)
    target_year  = np.zeros(total_examples, dtype=np.int64)
    target_price = np.zeros(total_examples, dtype=np.int64)

    # ── Step 3: Fill Arrays ───────────────────────────────────────────────────
    ex_idx = 0
    for uid in tqdm(users, desc="Building rollback examples"):
        history_orig = fs['user_to_play_history'].get(uid, [])
        weights_orig = fs['user_to_play_weights'].get(uid, [])
        recs_orig    = fs['user_to_recommend_history'].get(uid, [])
        n = len(history_orig)
        if n < 2: continue

        avg_log   = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = n * avg_log

        for _ in range(n_shuffles):
            indices = list(range(n))
            rng.shuffle(indices)
            history = [history_orig[i] for i in indices]
            weights = [weights_orig[i] for i in indices]
            recs    = [recs_orig[i]    for i in indices]

            raw_logs  = [w * total_log for w in weights]
            raw_hours = [math.exp(rl) - 1.0 for rl in raw_logs]

            all_potential = [i for i in range(1, n) if raw_hours[i] > 0.5]
            if not all_potential: continue
            
            k = min(max_per_user, len(all_potential))
            sampled = sorted(rng.sample(all_potential, k))

            for pos in sampled:
                if ex_idx >= total_examples: break # safety

                # 1. Triple Pool Partitioning
                user_median  = np.median(raw_hours[:pos])
                liked_ids    = []
                disliked_ids = []

                for i in range(pos):
                    ctx_iid = history[i]
                    ctx_h   = raw_hours[i]
                    ctx_rec = recs[i]
                    is_liked    = (ctx_rec is True) or (ctx_h >= game_median[ctx_iid]) or (ctx_h >= user_median * 2)
                    is_disliked = (ctx_rec is False) or (0.1 < ctx_h < 1.0) or (ctx_h <= user_median / 2)
                    if is_liked: liked_ids.append(ctx_iid)
                    if is_disliked: disliked_ids.append(ctx_iid)

                full_start = max(0, pos - MAX_HISTORY_LEN)
                full_ids   = history[full_start:pos]
                liked_ids  = liked_ids[-MAX_HISTORY_LEN:]
                disliked_ids = disliked_ids[-MAX_HISTORY_LEN:]

                # 2. Assign to pre-allocated arrays
                X_user_avg_log[ex_idx] = avg_log
                
                L_liked = len(liked_ids)
                if L_liked > 0:
                    X_hist_liked[ex_idx, :L_liked] = liked_ids
                
                L_dis = len(disliked_ids)
                if L_dis > 0:
                    X_hist_disliked[ex_idx, :L_dis] = disliked_ids
                
                L_full = len(full_ids)
                if L_full > 0:
                    X_hist_full[ex_idx, :L_full] = full_ids
                    # Playtime weights: normalized log(1+hours)
                    full_logs = raw_logs[full_start:pos]
                    total_pw  = sum(full_logs) or 1.0
                    X_hist_weights[ex_idx, :L_full] = [rl / total_pw for rl in full_logs]

                target_idx[ex_idx]   = history[pos]
                target_dev[ex_idx]   = int(game_dev[history[pos]])
                target_year[ex_idx]  = int(game_year[history[pos]])
                target_price[ex_idx] = int(game_price[history[pos]])
                
                ex_idx += 1

    # Truncate if we over-estimated slightly (unlikely, but safe)
    if ex_idx < total_examples:
        X_user_avg_log = X_user_avg_log[:ex_idx]
        X_hist_liked   = X_hist_liked[:ex_idx]
        X_hist_disliked = X_hist_disliked[:ex_idx]
        X_hist_full    = X_hist_full[:ex_idx]
        X_hist_weights = X_hist_weights[:ex_idx]
        target_idx     = target_idx[:ex_idx]
        target_dev     = target_dev[:ex_idx]
        target_year    = target_year[:ex_idx]
        target_price   = target_price[:ex_idx]

    print(f"  {ex_idx:,} rollback examples — converting to tensors ...")

    return (
        torch.from_numpy(X_user_avg_log),
        torch.from_numpy(X_hist_liked).long(),
        torch.from_numpy(X_hist_disliked).long(),
        torch.from_numpy(X_hist_full).long(),
        torch.from_numpy(X_hist_weights),
        torch.from_numpy(target_idx),
        torch.from_numpy(target_dev),
        torch.from_numpy(target_year),
        torch.from_numpy(target_price)
    )


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
    Returns (train_data, val_data), each a 9-tuple of tensors/lists.
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
    n = data[5].shape[0]  # index 5 = target_item_idx
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
