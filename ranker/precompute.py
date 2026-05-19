"""
Stage 0 — Ranker candidate precompute.

For each rollback example (user, position), build the V5 CG user context from the
shuffled prefix history[:pos], score against all corpus items via the PROD CG model
(α=0.4 by default), and save the top-(K-1) hard negatives + label as a single row.

Cross features (precomputed; columns written per (label, neg) per row):
  - tag_cosine    (B-0) — raw TF-IDF cosine of user tag pool vs candidate tag vector.
                    Computed off game_tag_matrix (the registered buffer CG uses),
                    *not* off the tag tower outputs — model-independent so the
                    parquet stays valid if tower hidden dims change.
  - genre_overlap (B-1) — playtime-weighted binary genre overlap, normalized by
                    candidate's genre count. Range [0, 1].
  - tag_overlap   (B-1b) — playtime-weighted binary TAG overlap (set-membership),
                    normalized by candidate's tag count. Magnitude-aware complement
                    to tag_cosine (which is direction-only). Range [0, 1].
  - dev_affinity  (B-2) — fraction of user's playtime on games by candidate's
                    developer. Range [0, 1].

Output:
    data/ranker_candidates_train.parquet
    data/ranker_candidates_val.parquet
    data/ranker_game_stats.parquet     ← per-game avg_log_playtime + log1p(interaction_count)
"""
import glob
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import MAX_HISTORY_LEN, MAX_ROLLBACK_EXAMPLES_PER_USER
from src.features import load_features
from src.train import (build_model, get_config,
                        load_config_for_checkpoint)
# Auto-resolve CG checkpoint by α so the parquet's cg_label_rank / cg_label_score
# encode the SAME CG that the ranker will compete against in evaluate.py.
# Mismatched precompute (e.g. α=0.4 CG retrieval + α=0 ranker training) means the
# E2E-ceiling "CG baseline" measured at eval time isn't the CG the ranker is fighting.
from ranker.cross_features import overlap_pool, dev_affinity_pool
from ranker.train import _ALPHA_TO_CG_GLOB


TOP_K_CANDIDATES         = 100   # 1 label + 99 hard negatives per row
SCORING_BATCH_SIZE       = 512
SPLIT_SEED               = 42    # mirror CG val builder seed (seed + 1 used by offline_eval)
DEFAULT_PRECOMPUTE_ALPHA = 0.0   # Phase 1 default — matches ranker α=0 in train.get_config()


# ── Device helper ────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── CG checkpoint loader (defaults to α matching ranker's Phase 1 default) ──

def _resolve_cg_checkpoint(checkpoint_arg: str | None) -> str:
    """Explicit path wins. Otherwise resolve by DEFAULT_PRECOMPUTE_ALPHA via the same
    α→glob mapping the ranker uses for warm-start, so precompute and warm-start CGs
    are guaranteed α-aligned for the default Phase 1 run.
    """
    if checkpoint_arg:
        return checkpoint_arg
    glob_pattern = _ALPHA_TO_CG_GLOB[DEFAULT_PRECOMPUTE_ALPHA]
    matches = sorted(glob.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(
            f"No CG checkpoint matched α={DEFAULT_PRECOMPUTE_ALPHA} "
            f"glob '{glob_pattern}'. Pass an explicit path: "
            f"python ranker/main.py precompute <checkpoint>"
        )
    return matches[-1]


def load_cg(checkpoint_path: str, fs: dict, device: torch.device):
    config = load_config_for_checkpoint(checkpoint_path)
    model  = build_model(config, fs)
    state  = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    model.load_state_dict(state)
    model.eval().to(device)
    return model, config


# ── Rollback builder (mirrors src/dataset._build_rollback_dataset, tracks user_id) ─

def _build_rollback_arrays(users: list, fs: dict, max_per_user: int, seed: int,
                            n_shuffles: int = 1) -> dict:
    """
    Mirrors src/dataset._build_rollback_dataset (Liked/Disliked/Full pool partitioning,
    playtime weights, MAX_HISTORY_LEN cap, seeded shuffle) but additionally tracks
    (user_id, rollback_n) per example. n_shuffles=3 for train (matches CG's data
    augmentation: N_SHUFFLES=3 in src/dataset.py); n_shuffles=1 for val.

    Returns dict of pre-allocated arrays:
        user_id, rollback_n, label_item_idx, user_avg_log_playtime,
        user_interaction_count,
        X_hist_liked, X_hist_disliked, X_hist_full, X_hist_playtime_weights
    """
    rng         = random.Random(seed)
    pad_idx     = fs['n_items']
    game_median = fs['game_median_hours']

    # ── Step 1: count total examples (matches CG count pass × n_shuffles) ───
    total_examples = 0
    for uid in users:
        history = fs['user_to_play_history'].get(uid, [])
        weights = fs['user_to_play_weights'].get(uid, [])
        n = len(history)
        if n < 2:
            continue
        avg_log   = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = n * avg_log
        potential = sum(1 for w in weights if (w * total_log) > 0.405)
        if (weights[0] * total_log) > 0.405:
            potential -= 1
        total_examples += min(max_per_user, max(0, potential)) * n_shuffles

    print(f"  Allocating buffers for {total_examples:,} examples ...")

    # Steam user_ids are usernames (strings), not ints — use object dtype.
    user_id_arr     = np.empty(total_examples, dtype=object)
    rollback_n_arr  = np.zeros(total_examples, dtype=np.int32)
    label_idx_arr   = np.zeros(total_examples, dtype=np.int64)
    user_avg_log    = np.zeros(total_examples, dtype=np.float32)
    user_int_count  = np.zeros(total_examples, dtype=np.int32)

    X_hist_liked    = np.full((total_examples, MAX_HISTORY_LEN), pad_idx, dtype=np.int32)
    X_hist_disliked = np.full((total_examples, MAX_HISTORY_LEN), pad_idx, dtype=np.int32)
    X_hist_full     = np.full((total_examples, MAX_HISTORY_LEN), pad_idx, dtype=np.int32)
    X_hist_pw       = np.zeros((total_examples, MAX_HISTORY_LEN), dtype=np.float32)

    # ── Step 2: fill arrays ─────────────────────────────────────────────────
    ex = 0
    for uid in tqdm(users, desc="Building rollback examples"):
        history_orig = fs['user_to_play_history'].get(uid, [])
        weights_orig = fs['user_to_play_weights'].get(uid, [])
        recs_orig    = fs['user_to_recommend_history'].get(uid, [])
        n = len(history_orig)
        if n < 2:
            continue

        avg_log_u = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = n * avg_log_u

        # Multiple independent shuffles per user (matches CG's N_SHUFFLES=3 for train).
        # Each shuffle produces genuinely different (context, target) pairs while
        # preserving varied context lengths.
        for _ in range(n_shuffles):
            indices = list(range(n))
            rng.shuffle(indices)
            history = [history_orig[i] for i in indices]
            weights = [weights_orig[i] for i in indices]
            recs    = [recs_orig[i]    for i in indices]

            raw_logs  = [w * total_log for w in weights]
            raw_hours = [math.exp(rl) - 1.0 for rl in raw_logs]

            # Steam play history has occasional (user_id, item_id) duplicates (~0.2% of
            # rollback examples). After shuffling, the label can land in its own prefix —
            # CG ignores this but for the ranker it's a leak (cross features like Recent
            # CF Sim could trivially recover the label). Skip those positions entirely.
            all_potential = []
            seen = {history[0]}
            for i in range(1, n):
                if raw_hours[i] > 0.5 and history[i] not in seen:
                    all_potential.append(i)
                seen.add(history[i])
            if not all_potential:
                continue

            k = min(max_per_user, len(all_potential))
            sampled = sorted(rng.sample(all_potential, k))

            for pos in sampled:
                if ex >= total_examples:
                    break

                user_median = float(np.median(raw_hours[:pos]))
                liked_ids, disliked_ids = [], []
                for i in range(pos):
                    ctx_iid = history[i]
                    ctx_h   = raw_hours[i]
                    ctx_rec = recs[i]
                    is_liked    = (ctx_rec is True) or (ctx_h >= game_median[ctx_iid]) \
                                  or (ctx_h >= user_median * 2)
                    is_disliked = (ctx_rec is False) or (0.1 < ctx_h < 1.0) \
                                  or (ctx_h <= user_median / 2)
                    if is_liked:
                        liked_ids.append(ctx_iid)
                    if is_disliked:
                        disliked_ids.append(ctx_iid)

                full_start    = max(0, pos - MAX_HISTORY_LEN)
                full_ids      = history[full_start:pos]
                liked_ids     = liked_ids[-MAX_HISTORY_LEN:]
                disliked_ids  = disliked_ids[-MAX_HISTORY_LEN:]

                if liked_ids:
                    X_hist_liked[ex, :len(liked_ids)] = liked_ids
                if disliked_ids:
                    X_hist_disliked[ex, :len(disliked_ids)] = disliked_ids
                if full_ids:
                    X_hist_full[ex, :len(full_ids)] = full_ids
                    full_logs = raw_logs[full_start:pos]
                    total_pw  = sum(full_logs) or 1.0
                    X_hist_pw[ex, :len(full_ids)] = [rl / total_pw for rl in full_logs]

                user_id_arr[ex]    = uid
                rollback_n_arr[ex] = pos
                label_idx_arr[ex]  = history[pos]
                user_avg_log[ex]   = avg_log_u
                user_int_count[ex] = n

                ex += 1

    # Truncate over-allocation
    return {
        'user_id':                 user_id_arr[:ex],
        'rollback_n':              rollback_n_arr[:ex],
        'label_item_idx':          label_idx_arr[:ex],
        'user_avg_log_playtime':   user_avg_log[:ex],
        'user_interaction_count':  user_int_count[:ex],
        'X_hist_liked':            X_hist_liked[:ex],
        'X_hist_disliked':         X_hist_disliked[:ex],
        'X_hist_full':             X_hist_full[:ex],
        'X_hist_playtime_weights': X_hist_pw[:ex],
    }


# ── CG scoring (batched) ─────────────────────────────────────────────────────

@torch.no_grad()
def _score_candidates(model, V_all: torch.Tensor, arrays: dict, fs: dict,
                      device: torch.device, batch_size: int, top_k: int) -> dict:
    """
    Score each rollback example against the full corpus with the V5 CG model.

    Returns:
      neg_item_idxs : (n, top_k - 1) int32 — top hard negatives by raw CG dot, label masked
      cg_label_rank : (n,)          int32 — label's rank in full corpus, 1-indexed, capped at top_k
      cg_label_score: (n,)          float32 — raw CG dot for the label (pre-mask)
      cg_neg_scores : (n, top_k - 1) float32 — raw CG dot per negative
      tag_cosine_label,    tag_cosine_negs    — raw-TFIDF cosine cross feature (B-0)
      genre_overlap_label, genre_overlap_negs — weighted genre overlap cross feature (B-1)
      tag_overlap_label,   tag_overlap_negs   — weighted tag overlap cross feature  (B-1b)
      dev_affinity_label,  dev_affinity_negs  — playtime-weighted dev affinity      (B-2)
    """
    n     = arrays['X_hist_full'].shape[0]
    n_neg = top_k - 1
    pad_idx = fs['n_items']

    X_avg_log    = torch.from_numpy(arrays['user_avg_log_playtime']).reshape(-1, 1)
    X_hist_liked = torch.from_numpy(arrays['X_hist_liked']).long()
    X_hist_dis   = torch.from_numpy(arrays['X_hist_disliked']).long()
    X_hist_full  = torch.from_numpy(arrays['X_hist_full']).long()
    X_hist_pw    = torch.from_numpy(arrays['X_hist_playtime_weights'])
    label_idx    = torch.from_numpy(arrays['label_item_idx']).long()

    neg_item_idxs    = np.zeros((n, n_neg), dtype=np.int32)
    cg_label_rank    = np.zeros(n,          dtype=np.int32)
    cg_label_score   = np.zeros(n,          dtype=np.float32)
    cg_neg_scores    = np.zeros((n, n_neg), dtype=np.float32)
    tag_cos_label    = np.zeros(n,          dtype=np.float32)
    tag_cos_negs     = np.zeros((n, n_neg), dtype=np.float32)
    genre_ov_label   = np.zeros(n,          dtype=np.float32)
    genre_ov_negs    = np.zeros((n, n_neg), dtype=np.float32)
    tag_ov_label     = np.zeros(n,          dtype=np.float32)
    tag_ov_negs      = np.zeros((n, n_neg), dtype=np.float32)
    dev_aff_label    = np.zeros(n,          dtype=np.float32)
    dev_aff_negs     = np.zeros((n, n_neg), dtype=np.float32)

    # Tag matrix on device once. Shape: (n_items+1, n_tags) — pad row appended for safe indexing.
    n_tags = fs['n_tags']
    tag_matrix_dev = torch.from_numpy(np.vstack([
        fs['game_tag_matrix'],
        np.zeros((1, n_tags), dtype=np.float32),
    ])).to(device)

    # Genre BINARY matrix + per-item genre count on device once. fs['game_genre_matrix']
    # is L1-row-normalized (each entry = 1/k per genre, sums to 1); for the cross feature
    # we want true set-membership semantics, so binarize. Pad row appended (count=0,
    # clamped to 1 for safe divide — never gathered since cand_b stays in [0, n_items)).
    n_genres = fs['n_genres']
    genre_binary_dev = torch.from_numpy(np.vstack([
        (fs['game_genre_matrix'] > 0).astype(np.float32),
        np.zeros((1, n_genres), dtype=np.float32),
    ])).to(device)
    genre_count_dev = genre_binary_dev.sum(dim=1).clamp(min=1.0)                       # (n_items+1,)

    # Tag BINARY matrix + per-item tag count on device once (B-1b). fs['game_tag_matrix']
    # is raw TF-IDF (varying float values); for the overlap cross feature we want true
    # set-membership semantics on which tags appear, so binarize (>0). Same pad/divide
    # convention as the genre buffers.
    tag_binary_dev = torch.from_numpy(np.vstack([
        (fs['game_tag_matrix'] > 0).astype(np.float32),
        np.zeros((1, n_tags), dtype=np.float32),
    ])).to(device)
    tag_count_dev  = tag_binary_dev.sum(dim=1).clamp(min=1.0)                          # (n_items+1,)

    # Developer index per item, on device (B-2). Pad row = n_developers (matches
    # CG's developer Embedding padding_idx).
    n_devs       = fs['n_developers']
    game_dev_dev = torch.from_numpy(np.concatenate([
        fs['game_developer_idx'].astype(np.int64),
        np.array([n_devs], dtype=np.int64),
    ])).to(device)

    for s in tqdm(range(0, n, batch_size), desc="Scoring candidates"):
        e = min(s + batch_size, n)
        B = e - s

        x_avg     = X_avg_log[s:e].to(device)
        h_liked   = X_hist_liked[s:e].to(device)
        h_dis     = X_hist_dis[s:e].to(device)
        h_full    = X_hist_full[s:e].to(device)
        h_pw      = X_hist_pw[s:e].to(device)
        label_b   = label_idx[s:e].to(device)

        U = model.user_embedding(x_avg, h_liked, h_dis, h_full, h_pw)
        scores = U @ V_all.T                                         # (B, n_items) raw cosine
        scores = scores.nan_to_num(nan=float('-inf'))

        b_idx     = torch.arange(B, device=device)
        label_sc  = scores[b_idx, label_b]
        cg_label_score[s:e] = label_sc.cpu().numpy()

        # Label rank in full corpus, capped at top_k.
        full_rank = (scores > label_sc.unsqueeze(1)).sum(dim=1) + 1
        cg_label_rank[s:e] = full_rank.clamp(max=top_k).cpu().numpy()

        scores[b_idx, label_b] = float('-inf')
        topk = scores.topk(n_neg, dim=1)
        neg_item_idxs[s:e]  = topk.indices.cpu().numpy()
        cg_neg_scores[s:e]  = topk.values.cpu().numpy()

        # ── Tag cosine over raw TF-IDF ──────────────────────────────────────
        # user_tag_pool: playtime-weighted sum of game_tag_matrix[X_hist_full] rows.
        # h_pw is already normalized per row (sum=1 over real positions; pads weight=0).
        # tag_matrix[pad_idx] is a zero row, so padding contributes nothing.
        hist_tags = tag_matrix_dev[h_full]                           # (B, H, n_tags)
        user_tag_pool = (hist_tags * h_pw.unsqueeze(-1)).sum(dim=1)  # (B, n_tags)
        user_tag_norm = F.normalize(user_tag_pool, p=2, dim=1)

        label_tag_norm = F.normalize(tag_matrix_dev[label_b], p=2, dim=1)
        tag_cos_label[s:e] = (user_tag_norm * label_tag_norm).sum(dim=1).cpu().numpy()

        neg_tag_norm = F.normalize(tag_matrix_dev[topk.indices], p=2, dim=-1)  # (B, n_neg, n_tags)
        tag_cos_negs[s:e] = (user_tag_norm.unsqueeze(1) * neg_tag_norm).sum(dim=-1).cpu().numpy()

        # ── Bucket 1 cross features (shared compute path with train) ────────
        # Build a unified (B, 1 + n_neg) cand matrix with label at column 0, then call
        # the same overlap / dev-affinity utils the training loop uses. This guarantees
        # bit-exact identity between parquet (eval-time) values and on-the-fly (train-
        # time) values — they're literally the same function on the same tensors.
        cand_b = torch.cat([label_b.unsqueeze(1), topk.indices], dim=1)                   # (B, 1 + n_neg)

        # B-1 Genre Overlap
        genre_ov = overlap_pool(genre_binary_dev, genre_count_dev,
                                pool_indices=h_full, pool_weights=h_pw, cand_idx=cand_b)
        genre_ov_np = genre_ov.cpu().numpy()
        genre_ov_label[s:e] = genre_ov_np[:, 0]
        genre_ov_negs[s:e]  = genre_ov_np[:, 1:]

        # B-1b Tag Overlap (same util, tag binary buffers)
        tag_ov = overlap_pool(tag_binary_dev, tag_count_dev,
                              pool_indices=h_full, pool_weights=h_pw, cand_idx=cand_b)
        tag_ov_np = tag_ov.cpu().numpy()
        tag_ov_label[s:e] = tag_ov_np[:, 0]
        tag_ov_negs[s:e]  = tag_ov_np[:, 1:]

        # B-2 Developer Affinity
        dev_aff = dev_affinity_pool(game_dev_dev, dev_pad_idx=n_devs,
                                    pool_indices=h_full, pool_weights=h_pw, cand_idx=cand_b)
        dev_aff_np = dev_aff.cpu().numpy()
        dev_aff_label[s:e] = dev_aff_np[:, 0]
        dev_aff_negs[s:e]  = dev_aff_np[:, 1:]

    return {
        'neg_item_idxs':       neg_item_idxs,
        'cg_label_rank':       cg_label_rank,
        'cg_label_score':      cg_label_score,
        'cg_neg_scores':       cg_neg_scores,
        'tag_cosine_label':    tag_cos_label,
        'tag_cosine_negs':     tag_cos_negs,
        'genre_overlap_label': genre_ov_label,
        'genre_overlap_negs':  genre_ov_negs,
        'tag_overlap_label':   tag_ov_label,
        'tag_overlap_negs':    tag_ov_negs,
        'dev_affinity_label':  dev_aff_label,
        'dev_affinity_negs':   dev_aff_negs,
    }


# ── Per-game stats (used by ranker dataset for content cross features later) ─

def compute_game_stats(fs: dict, output_dir: str) -> None:
    """Per-game global avg_log_playtime + log1p(interaction_count). Written to parquet."""
    n_items = fs['n_items']
    item_ids = fs['item_ids']

    # Per-game avg log(1+hours) across all observed user interactions.
    avg_log = np.zeros(n_items, dtype=np.float32)
    cnt     = np.zeros(n_items, dtype=np.int64)
    for uid, history in fs['user_to_play_history'].items():
        weights = fs['user_to_play_weights'].get(uid, [])
        avg_log_u = fs['user_to_avg_log_playtime'].get(uid, 1.0) or 1.0
        total_log = len(history) * avg_log_u
        for idx, w in zip(history, weights):
            if 0 <= idx < n_items:
                avg_log[idx] += w * total_log   # raw_log = w * total_log
                cnt[idx]     += 1

    safe_cnt = np.where(cnt > 0, cnt, 1)
    avg_log_per_game = (avg_log / safe_cnt).astype(np.float32)

    log_count = np.log1p(fs['game_interaction_counts']).astype(np.float32)

    df = pd.DataFrame({
        'item_idx':                np.arange(n_items, dtype=np.int32),
        'item_id':                 item_ids,
        'avg_log_playtime':        avg_log_per_game,
        'log1p_interaction_count': log_count,
    })
    out_path = os.path.join(output_dir, 'ranker_game_stats.parquet')
    df.to_parquet(out_path, index=False)
    print(f"  Game stats: {len(df):,} rows → {out_path}")
    print(f"    avg_log_playtime:        mean={avg_log_per_game.mean():.3f}  "
          f"min={avg_log_per_game.min():.3f}  max={avg_log_per_game.max():.3f}")
    print(f"    log1p_interaction_count: mean={log_count.mean():.3f}  "
          f"min={log_count.min():.3f}  max={log_count.max():.3f}")


# ── Verification ─────────────────────────────────────────────────────────────

def _verify_split(arrays: dict, neg: np.ndarray, cg_rank: np.ndarray,
                  fs: dict, label: str, top_k: int) -> None:
    n = arrays['X_hist_full'].shape[0]
    label_idx = arrays['label_item_idx']
    pad_idx   = fs['n_items']
    n_items   = fs['n_items']

    in_neg = (neg == label_idx[:, None]).any(axis=1).sum()
    assert in_neg == 0, f"[{label}] label appears in negatives in {in_neg} rows"
    assert neg.min() >= 0,        f"[{label}] negative idx < 0"
    assert neg.max() < n_items,   f"[{label}] negative idx >= n_items"

    n_neg = top_k - 1
    sample_n   = min(1000, n)
    sample_idx = np.random.default_rng(0).choice(n, sample_n, replace=False)
    uniq = np.array([len(np.unique(neg[i])) for i in sample_idx])
    assert (uniq == n_neg).all(), \
        f"[{label}] non-unique negatives in {(uniq != n_neg).sum()}/{sample_n} rows"

    # Label must not appear in any history pool for the same example.
    for col in ('X_hist_liked', 'X_hist_disliked', 'X_hist_full'):
        h = arrays[col][sample_idx]
        ls = label_idx[sample_idx]
        leak = ((h == ls[:, None]) & (h != pad_idx)).any(axis=1).sum()
        assert leak == 0, f"[{label}] label leaked into {col} for {leak}/{sample_n} sampled rows"

    assert cg_rank.min() >= 1,     f"[{label}] cg_label_rank < 1"
    assert cg_rank.max() <= top_k, f"[{label}] cg_label_rank > {top_k}"

    # CG quality summary — what does CG do unaided on this split?
    cg_mrr     = float((1.0 / cg_rank).mean())
    cg_ndcg10  = float(np.where(cg_rank <= 10, 1.0 / np.log2(cg_rank + 1), 0.0).mean())
    cg_hit1    = float((cg_rank == 1).mean())
    cg_hit10   = float((cg_rank <= 10).mean())
    cg_recall  = float((cg_rank < top_k).mean())   # label inside top-(top_k-1) negs
    print(f"  [{label}] verification passed: n={n:,}  unique_users={len(np.unique(arrays['user_id'])):,}")
    print(f"  [{label}] CG baseline: MRR={cg_mrr:.4f}  NDCG@10={cg_ndcg10:.4f}  "
          f"Hit@1={cg_hit1:.4f}  Hit@10={cg_hit10:.4f}  Recall@{top_k}={cg_recall:.4f}")


def _build_dataframe(arrays: dict, scoring: dict) -> pd.DataFrame:
    return pd.DataFrame({
        'user_id':                 arrays['user_id'],
        'rollback_n':              arrays['rollback_n'].astype(np.int32),
        'label_item_idx':          arrays['label_item_idx'].astype(np.int32),
        'neg_item_idxs':           list(scoring['neg_item_idxs'].astype(np.int32)),
        'cg_label_rank':           scoring['cg_label_rank'].astype(np.int32),
        'cg_label_score':          scoring['cg_label_score'],
        'cg_neg_scores':           list(scoring['cg_neg_scores']),
        'tag_cosine_label':        scoring['tag_cosine_label'],
        'tag_cosine_negs':         list(scoring['tag_cosine_negs']),
        'genre_overlap_label':     scoring['genre_overlap_label'],
        'genre_overlap_negs':      list(scoring['genre_overlap_negs']),
        'tag_overlap_label':       scoring['tag_overlap_label'],
        'tag_overlap_negs':        list(scoring['tag_overlap_negs']),
        'dev_affinity_label':      scoring['dev_affinity_label'],
        'dev_affinity_negs':       list(scoring['dev_affinity_negs']),
        'user_avg_log_playtime':   arrays['user_avg_log_playtime'],
        'user_interaction_count':  arrays['user_interaction_count'].astype(np.int32),
        'X_hist_liked':            list(arrays['X_hist_liked'].astype(np.int32)),
        'X_hist_disliked':         list(arrays['X_hist_disliked'].astype(np.int32)),
        'X_hist_full':             list(arrays['X_hist_full'].astype(np.int32)),
        'X_hist_playtime_weights': list(arrays['X_hist_playtime_weights']),
    })


# ── Entry point ──────────────────────────────────────────────────────────────

def precompute(checkpoint_path: str | None = None,
               data_dir: str = 'data',
               output_dir: str = 'data',
               max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
               batch_size: int = SCORING_BATCH_SIZE,
               top_k: int = TOP_K_CANDIDATES) -> None:
    cg_path = _resolve_cg_checkpoint(checkpoint_path)
    print(f"CG checkpoint: {cg_path}")

    print("\nLoading FeatureStore ...")
    fs = load_features(data_dir=data_dir)
    n_items = fs['n_items']
    print(f"  {n_items:,} corpus games")

    device = get_device()
    print(f"\nDevice: {device}")
    model, cg_config = load_cg(cg_path, fs, device)
    print(f"  CG popularity_alpha = {cg_config.get('popularity_alpha', 'unknown')}")

    # Pre-compute V_all (n_items, output_dim) — full corpus item embeddings.
    print("\nBuilding full corpus item embeddings (V_all) ...")
    all_game_idxs = torch.arange(n_items, device=device)
    all_years     = torch.from_numpy(fs['game_year_idx']).to(device)
    all_devs      = torch.from_numpy(fs['game_developer_idx']).to(device)
    all_prices    = torch.from_numpy(fs['game_price_bucket']).to(device)
    with torch.no_grad():
        V_all = model.item_embedding(all_years, all_game_idxs, all_devs, all_prices)
    print(f"  V_all: shape={tuple(V_all.shape)}  device={V_all.device}")

    train_users = fs['train_users']
    val_users   = fs['val_users']
    print(f"\n  Train users: {len(train_users):,}   Val users: {len(val_users):,}")

    for split_name, users in [('train', train_users), ('val', val_users)]:
        print(f"\n── {split_name.upper()} split ──")
        # Mirror src/dataset seed convention: train uses SPLIT_SEED, val uses SPLIT_SEED + 1.
        split_seed     = SPLIT_SEED if split_name == 'train' else SPLIT_SEED + 1
        # Match CG (src/dataset.py): N_SHUFFLES=3 for train data augmentation, 1 for val.
        split_shuffles = 3 if split_name == 'train' else 1
        arrays = _build_rollback_arrays(users, fs, max_per_user, seed=split_seed,
                                          n_shuffles=split_shuffles)
        scoring = _score_candidates(model, V_all, arrays, fs, device, batch_size, top_k)
        _verify_split(arrays, scoring['neg_item_idxs'], scoring['cg_label_rank'],
                      fs, split_name, top_k)
        df = _build_dataframe(arrays, scoring)
        out_path = os.path.join(output_dir, f'ranker_candidates_{split_name}.parquet')
        df.to_parquet(out_path, index=False)
        print(f"  → {out_path}  ({len(df):,} rows)")

    print("\n── Game stats ──")
    compute_game_stats(fs, output_dir)

    print("\nPrecompute complete.")
