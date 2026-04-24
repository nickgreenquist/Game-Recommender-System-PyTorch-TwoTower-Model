"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Protocol: user-level hold-out.
  Val users are held out entirely from training — none of their interactions
  are seen during training.  All their rollback examples are used for eval.
  For each (context, target) pair: rank all corpus games, measure whether
  the target appears in top K.

  Because each rollback example has exactly one target, Recall@K == Hit Rate@K.

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import random

import torch
from tqdm import tqdm

from src.dataset import (
    _build_rollback_dataset,
    pad_history_batch,
    pad_weights_batch,
    MAX_ROLLBACK_EXAMPLES_PER_USER,
)
from src.evaluate import build_game_embeddings
from src.model import GameRecommender


def run_offline_eval(model: GameRecommender, fs: dict,
                     checkpoint_path: str = '',
                     n_users: int = 2_000,
                     ks: tuple = (1, 5, 10, 20, 50),
                     seed: int = 42) -> None:
    model.eval()

    # ── Pre-compute item embedding matrix ─────────────────────────────────────
    print("Building game embeddings ...")
    _, all_ids, all_embs = build_game_embeddings(model, fs)
    # all_embs: (n_items, 105)

    # ── Sample val users ──────────────────────────────────────────────────────
    val_users = fs['val_users']
    rng = random.Random(seed)
    eval_users = rng.sample(val_users, min(n_users, len(val_users)))
    print(f"Evaluating on {len(eval_users):,} val users ...")

    # ── Generate rollback examples ────────────────────────────────────────────
    print("Building rollback examples for val users ...")
    (X_genre, X_history, X_history_weights, target_item_idx,
     *_) = _build_rollback_dataset(
        eval_users, fs,
        max_per_user=MAX_ROLLBACK_EXAMPLES_PER_USER,
        seed=seed + 1,
    )

    n_examples = target_item_idx.shape[0]
    pad_idx    = fs['n_items']   # model.game_pad_idx

    # ── Accumulators ──────────────────────────────────────────────────────────
    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    batch_size = 512

    with torch.no_grad():
        for s in tqdm(range(0, n_examples, batch_size), desc="Scoring"):
            e = min(s + batch_size, n_examples)
            B = e - s

            hist_t = pad_history_batch(X_history[s:e], pad_idx)   # (B, max_len)
            wt_t   = pad_weights_batch(X_history_weights[s:e])     # (B, max_len)

            user_emb = model.user_embedding(
                X_genre[s:e], hist_t, wt_t
            )  # (B, 105)

            # scores: (B, n_items)
            scores = (all_embs @ user_emb.T).T

            target_idxs = target_item_idx[s:e]  # (B,)

            for i in range(B):
                t_pos        = target_idxs[i].item()
                target_score = scores[i, t_pos]
                rank         = int((scores[i] > target_score).sum().item()) + 1

                mrr_sum += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        recall[k]   += 1.0
                        hit_rate[k] += 1
                        ndcg[k]     += 1.0 / math.log2(rank + 1)
                n_eval += 1

    if n_eval == 0:
        print("No examples evaluated.")
        return

    # ── Random baselines (single-target, uniform ranker) ─────────────────────
    # Recall@K = Hit Rate@K = K/N  (target equally likely at any rank)
    # NDCG@K   = (1/N) * sum_{r=1}^{K} 1/log2(r+1)
    # MRR      = (1/N) * H_N  (harmonic number / N)
    n_corpus = len(all_ids)
    rand_recall = {k: k / n_corpus for k in ks}
    rand_ndcg   = {k: sum(1.0 / math.log2(r + 1) for r in range(1, k + 1)) / n_corpus
                   for k in ks}
    rand_mrr    = sum(1.0 / r for r in range(1, n_corpus + 1)) / n_corpus

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n── Offline Evaluation  ({n_eval:,} rollback examples, "
          f"{len(eval_users):,} val users) " + "─" * 20)
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus: {n_corpus:,} games  |  1 target per example  "
          f"(Recall@K = Hit Rate@K for single-target eval)\n")

    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    sep    = "─" * len(header)
    thin   = "·" * len(header)

    print(header)
    print(sep)
    for k in ks:
        print(f"{k:>6}  "
              f"{rand_recall[k]:>10.4f}  "
              f"{rand_recall[k]:>11.4f}  "
              f"{rand_ndcg[k]:>8.4f}  ← random")
    print(thin)
    for k in ks:
        print(f"{k:>6}  "
              f"{recall[k]/n_eval:>10.4f}  "
              f"{hit_rate[k]/n_eval:>11.4f}  "
              f"{ndcg[k]/n_eval:>8.4f}  ← model")
    print(sep)
    print(f"MRR  random: {rand_mrr:.4f}   model: {mrr_sum/n_eval:.4f}  "
          f"(+{mrr_sum/n_eval - rand_mrr:.4f})")
