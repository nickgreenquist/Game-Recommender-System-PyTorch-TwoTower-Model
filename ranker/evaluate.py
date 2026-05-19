"""
Ranker evaluation: NDCG@K, MRR, Hit@K, plus the CG baseline (read from precompute parquet).

For each rollback group (1 label + 99 hard negs = 100 candidates):
  - Score all candidates with the ranker
  - Compute label's rank within the group (1-indexed)
  - NDCG@K = 1/log2(rank+1) if rank <= K else 0
  - MRR    = 1/rank

E2E ceiling (plan §4): if CG didn't organically retrieve the label
(cg_label_rank >= n_cand), the ranker never sees it in production → rank = n_cand + 1
(score = 0 for all metrics). Both CG baseline and ranker numbers use the same ceiling
so the comparison is apples-to-apples.
"""
import numpy as np
import torch

from ranker.dataset import RankerDataset, compute_cross_features


KS = (1, 5, 10, 20, 50, 100)


@torch.no_grad()
def compute_label_ranks(model, dataset: RankerDataset, device: torch.device,
                        batch_size: int = 32,
                        eval_indices: np.ndarray | None = None) -> np.ndarray:
    """
    Score rollback groups with `model`. Return label ranks (1-indexed, E2E-adjusted).

    eval_indices: optional np.int64 array of row indices to evaluate. If None, evaluates
                  the full val set (slow). For training-time logging, pass a deterministic
                  sample (fixed seed) so successive evals are directly comparable.
    """
    model.eval()
    n_cand = 1 + dataset.n_neg
    if eval_indices is None:
        eval_indices = np.arange(dataset.N, dtype=np.int64)
    n_eval      = len(eval_indices)
    label_ranks = np.zeros(n_eval, dtype=np.int32)

    # Precompute item embeddings for the entire corpus once per eval call.
    # Same math-identity optimization as train._forward_batch — at 100-cand pools
    # over 30+ batches it still amortizes well, especially on MPS where dispatch
    # overhead per Embedding/Linear call dominates small-batch latency.
    n_items_int  = int(model.game_pad_idx)
    all_item_ids = torch.arange(n_items_int, device=device)
    all_item_concat = model.item_embedding(all_item_ids)                       # (n_items, I)

    for s in range(0, n_eval, batch_size):
        e = min(s + batch_size, n_eval)
        B = e - s
        rows   = eval_indices[s:e]
        rows_t = torch.from_numpy(rows).long()

        # ── User side: compute ONCE per row (not per candidate) ─────────────
        x_avg  = dataset._X_user_avg_log_t[rows_t].to(device)
        h_lkd  = dataset._X_hist_liked_t[rows_t].to(device)
        h_dis  = dataset._X_hist_disliked_t[rows_t].to(device)
        h_full = dataset._X_hist_full_t[rows_t].to(device)
        h_pw   = dataset._X_hist_pw_t[rows_t].to(device)
        us     = model.user_forward(x_avg, h_lkd, h_dis, h_full, h_pw)
        user_concat = us.user_concat

        # ── Item side: build (B, n_cand) candidate matrix → gather from corpus ──
        cand = np.empty((B, n_cand), dtype=np.int64)
        cand[:, 0]  = dataset.label_idx[rows]
        cand[:, 1:] = dataset.neg_idx[rows]
        cand_flat   = torch.from_numpy(cand.reshape(-1)).to(device)            # (B*n_cand,)
        item_concat = all_item_concat[cand_flat]                               # (B*n_cand, item_dim)

        # ── Cross features ──────────────────────────────────────────────────
        # Phase A/B (Bucket 1): read precomputed values from parquet (eval-time is the
        # only place these are persisted; matches the precompute write). On-the-fly
        # compute in train._forward_batch is mathematically identical to these values
        # (same weighted_overlap / dev_affinity utils on both sides).
        def _gather(label_col: np.ndarray, neg_col: np.ndarray) -> torch.Tensor:
            buf = np.empty((B, n_cand), dtype=np.float32)
            buf[:, 0]  = label_col[rows]
            buf[:, 1:] = neg_col[rows]
            return torch.from_numpy(buf.reshape(-1)).to(device)

        tag_cos_flat  = _gather(dataset.tag_cosine_label,    dataset.tag_cosine_negs)
        genre_ov_flat = _gather(dataset.genre_overlap_label, dataset.genre_overlap_negs)
        tag_ov_flat   = _gather(dataset.tag_overlap_label,   dataset.tag_overlap_negs)
        dev_aff_flat  = _gather(dataset.dev_affinity_label,  dataset.dev_affinity_negs)
        cross = compute_cross_features(tag_cos_flat, genre_ov_flat, tag_ov_flat, dev_aff_flat)

        # ── Score: factorized MLP layer-1 over the (B, n_cand) layout ───────
        # See model.score_pairs_batched — math identity, runs user-side projection
        # on B rows (not B*n_cand), skips a (B*n_cand, U) user-replica materialization.
        scores = model.score_pairs_batched(user_concat, item_concat, cross, n_cand)

        label_score = scores[:, 0].unsqueeze(1)
        rank_np     = ((scores > label_score).sum(dim=1) + 1).cpu().numpy()

        cg_found         = dataset.cg_label_rank[rows] < n_cand
        label_ranks[s:e] = np.where(cg_found, rank_np, n_cand + 1)

    return label_ranks


def _ndcg_mrr_from_ranks(ranks: np.ndarray) -> tuple[float, float]:
    mrr  = float((1.0 / ranks).mean())
    ndcg = float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return ndcg, mrr


def evaluate_ndcg_mrr(model, dataset: RankerDataset, device: torch.device,
                      batch_size: int = 32,
                      eval_indices: np.ndarray | None = None) -> tuple[float, float]:
    return _ndcg_mrr_from_ranks(
        compute_label_ranks(model, dataset, device, batch_size, eval_indices=eval_indices))


def cg_baseline(dataset: RankerDataset,
                eval_indices: np.ndarray | None = None) -> tuple[float, float]:
    """CG baseline NDCG@10 and MRR, E2E-consistent with compute_label_ranks."""
    n_cand = 1 + dataset.n_neg
    raw    = dataset.cg_label_rank if eval_indices is None else dataset.cg_label_rank[eval_indices]
    ranks  = np.where(raw < n_cand, raw, n_cand + 1)
    return _ndcg_mrr_from_ranks(ranks)


def hit_rates_from_ranks(ranks: np.ndarray, ks: tuple = KS) -> dict:
    return {f'Hit@{k}': float((ranks <= k).mean()) for k in ks}


def ndcg_at_k_from_ranks(ranks: np.ndarray, ks: tuple = KS) -> dict:
    return {f'NDCG@{k}': float(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean())
            for k in ks}
