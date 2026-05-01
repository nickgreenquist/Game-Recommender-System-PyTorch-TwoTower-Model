"""
Offline retrieval evaluation V2 — Recall@K, NDCG@K, Hit Rate@K, MRR.
Results are written to eval_results/<checkpoint_stem>.txt
"""
import math
import os
import random

import torch
from tqdm import tqdm

from src.dataset import (
    _build_rollback_dataset,
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
    device = next(model.parameters()).device

    # ── Pre-compute item embedding matrix ─────────────────────────────────────
    print("Building game embeddings ...")
    _, all_ids, all_embs = build_game_embeddings(model, fs)

    # ── Popularity bias (match training objective) ────────────────────────────
    from src.train import load_config_for_checkpoint
    cp_config   = load_config_for_checkpoint(checkpoint_path)
    alpha       = cp_config.get('popularity_alpha', 0.0)
    temperature = cp_config.get('temperature', 0.1)
    if alpha > 0 and 'game_interaction_counts' in fs:
        counts = torch.from_numpy(fs['game_interaction_counts']).to(device)
        # Scale to dot-product space: training used (u·v)/temp - bias → inference: u·v - temp*bias
        pop_bias = (temperature * alpha * torch.log1p(counts)).unsqueeze(0)  # (1, n_items)
    else:
        pop_bias = None

    # ── Sample val users ──────────────────────────────────────────────────────
    val_users = fs['val_users']
    rng = random.Random(seed)
    eval_users = rng.sample(val_users, min(n_users, len(val_users)))
    print(f"Evaluating on {len(eval_users):,} val users ...")

    # ── Generate rollback examples ────────────────────────────────────────────
    print("Building rollback examples for val users ...")
    (X_avg_log, X_hist_liked, X_hist_disliked, X_hist_full,
     X_hist_playtime_weights, target_item_idx,
     *_) = _build_rollback_dataset(
        eval_users, fs,
        max_per_user=MAX_ROLLBACK_EXAMPLES_PER_USER,
        seed=seed + 1,
    )

    n_examples = target_item_idx.shape[0]

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

            h_liked    = X_hist_liked[s:e].to(device)
            h_disliked = X_hist_disliked[s:e].to(device)
            h_full     = X_hist_full[s:e].to(device)
            h_pw       = X_hist_playtime_weights[s:e].to(device)
            x_avg_log  = X_avg_log[s:e].to(device)

            user_emb = model.user_embedding(x_avg_log, h_liked, h_disliked, h_full, h_pw)

            # scores: (B, n_items)
            scores = (user_emb @ all_embs.T)
            if pop_bias is not None:
                scores = scores - pop_bias

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

    n_corpus = len(all_ids)
    rand_recall = {k: k / n_corpus for k in ks}
    rand_ndcg   = {k: sum(1.0 / math.log2(r + 1) for r in range(1, k + 1)) / n_corpus
                   for k in ks}
    rand_mrr    = sum(1.0 / r for r in range(1, n_corpus + 1)) / n_corpus

    lines = []
    lines.append(f"── Offline Evaluation  ({n_eval:,} rollback examples, "
                 f"{len(eval_users):,} val users) " + "─" * 20)
    if checkpoint_path:
        lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Corpus: {n_corpus:,} games\n")

    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    sep    = "─" * len(header)
    thin   = "·" * len(header)

    lines.append(header)
    lines.append(sep)
    for k in ks:
        lines.append(f"{k:>6}  "
                     f"{rand_recall[k]:>10.4f}  "
                     f"{rand_recall[k]:>11.4f}  "
                     f"{rand_ndcg[k]:>8.4f}  ← random")
    lines.append(thin)
    for k in ks:
        lines.append(f"{k:>6}  "
                     f"{recall[k]/n_eval:>10.4f}  "
                     f"{hit_rate[k]/n_eval:>11.4f}  "
                     f"{ndcg[k]/n_eval:>8.4f}  ← model")
    lines.append(sep)
    lines.append(f"MRR  random: {rand_mrr:.4f}   model: {mrr_sum/n_eval:.4f}  "
                 f"(+{mrr_sum/n_eval - rand_mrr:.4f})")

    output = "\n".join(lines)
    print(f"\n{output}")

    os.makedirs('eval_results', exist_ok=True)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0] if checkpoint_path else 'unknown'
    out_path = os.path.join('eval_results', f'{stem}.txt')
    with open(out_path, 'w') as f:
        f.write(output + "\n")
    print(f"\n✓ Saved → {out_path}")
