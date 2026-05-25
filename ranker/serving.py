"""
Shared ranker rerank pipeline — used by BOTH canary (synthetic users) and the
Streamlit app (real users). Factoring this out keeps a single copy of the fragile
cross-feature compute path so the two callers can never drift.

The flow, given a user's history (as corpus indices + per-item playtime weights) and
a list of candidate corpus indices (the CG's top-K):

  1. build_user_inputs_from_indices → the 6 pre-padded user-side tensors + the
     full-history index/weight lists the cross features need.
  2. rerank_candidates → item_concat + all 23 cross features (Bucket 0/1/2/5/6) for
     the candidates, then ranker.score_pairs → a raw logit per candidate.

Both functions are index-driven (no titles, no synthetic-user assumptions) so the
exact same code serves a canary's seed list and a real user's clicked games. The
canary's title→index resolution + display metadata stays in ranker/canary.py.

Parity contract: the cross-feature compute below is a verbatim move of the block
that previously lived inline in ranker/canary.run_canary. Canary output is byte-
identical before/after the extraction.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ranker.cross_features import (B5_USER_MEAN_LOG_COUNT, B5_USER_MEAN_PRICE,
                                     B5_USER_MEAN_SENTIMENT, B5_USER_MEAN_YEAR,
                                     B5_USER_MEDIAN_LOG_PT, Bucket5GameBuffers,
                                     NicheBuffers, OverlapBuffers,
                                     categorical_overlap_triple, last_n_history,
                                     niche_scalar_triple, numeric_match_quintuple,
                                     weighted_overlap)
from ranker.dataset import compute_cross_features
from src.dataset import MAX_HISTORY_LEN
from src.features import SENTIMENT_UNKNOWN_FILL

# Width of the cross-feature tensor compute_cross_features produces here. Bump in
# lockstep with the compute path when a new bucket lands. Callers compare this to the
# checkpoint's n_cross_features to decide slice-vs-zero-pad alignment.
COMPUTED_CROSS_FEATURES = 24   # tag_cosine + 9 overlaps + 5 numeric + 8 niche + text_cosine


# ── User-side inputs (index-driven; mirrors the old _build_synthetic_user_inputs) ──

def build_user_inputs_from_indices(liked_idxs: list, anchor_idxs: list, disliked_idxs: list,
                                   pad_idx: int,
                                   fav_weight: float, anchor_weight: float, dis_weight: float):
    """
    Build the ranker's user-side inputs from corpus indices + per-class playtime weights.

    Pool construction (matches src/evaluate._build_user_embedding + the canary):
      liked    = favorites only
      disliked = disliked only
      full     = favorites + anchors + disliked   (equal-membership, weighted by playtime)
    The liked-slice playtime weights (Bucket 2) normalize over the favorites subset only.

    Returns (user_inputs dict, full_ids list, full_pw list, full_raw_pw list):
      user_inputs — 6 numpy tensors shaped (1, ...) ready to move to device
      full_ids    — unpadded full-history indices (tag_cosine + B5 scalars need them)
      full_pw     — normalized full-history playtime weights (parallel to full_ids)
      full_raw_pw — un-normalized per-position weights (B5 median needs the raw values)
    """
    liked_ids    = list(liked_idxs)
    disliked_ids = list(disliked_idxs)
    full_ids     = list(liked_idxs) + list(anchor_idxs) + list(disliked_idxs)

    raw_pw = ([fav_weight]    * len(liked_idxs) +
              [anchor_weight] * len(anchor_idxs) +
              [dis_weight]    * len(disliked_idxs))
    total_pw = sum(raw_pw) or 1.0
    full_pw  = [w / total_pw for w in raw_pw]

    # Bucket 2 — liked-slice playtime weights (favorites only; normalize over that subset).
    liked_raw   = [fav_weight] * len(liked_idxs)
    total_liked = sum(liked_raw) or 1.0
    liked_pw    = [w / total_liked for w in liked_raw]

    total_items = max(len(full_ids), 1)
    avg_log = (len(liked_idxs)    * fav_weight +
               len(anchor_idxs)   * anchor_weight +
               len(disliked_idxs) * dis_weight) / total_items

    def pad_list(ids, length=MAX_HISTORY_LEN):
        out  = np.full(length, pad_idx, dtype=np.int64)
        take = min(len(ids), length)
        if take:
            out[:take] = ids[:take]
        return out

    def pad_weights(ws, length=MAX_HISTORY_LEN):
        out  = np.zeros(length, dtype=np.float32)
        take = min(len(ws), length)
        if take:
            out[:take] = ws[:take]
        return out

    user_inputs = {
        'X_user_avg_log':                np.array([[avg_log]], dtype=np.float32),    # (1, 1)
        'X_hist_liked':                  pad_list(liked_ids).reshape(1, -1),          # (1, H)
        'X_hist_disliked':               pad_list(disliked_ids).reshape(1, -1),
        'X_hist_full':                   pad_list(full_ids).reshape(1, -1),
        'X_hist_playtime_weights':       pad_weights(full_pw).reshape(1, -1),
        'X_hist_liked_playtime_weights': pad_weights(liked_pw).reshape(1, -1),        # Bucket 2
    }
    return user_inputs, full_ids, full_pw, raw_pw


# ── Tag cosine (B-0) for an arbitrary user against candidate items ──────────────

def tag_cosine_for_user(ranker, full_ids: list, full_pw: list, cand_idx: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """Recreate precompute's tag_cosine (B-0). User side is sum-weighted RAW TF-IDF then
    L2-normalized; candidate side reads the pre-normalized buffer — identical to
    train._forward_batch."""
    if not full_ids:
        return torch.zeros(cand_idx.shape[0], device=device)
    full_t = torch.tensor(full_ids, dtype=torch.long, device=device)
    pw_t   = torch.tensor(full_pw,  dtype=torch.float32, device=device).unsqueeze(-1)   # (L, 1)
    user_tag_pool = (ranker.game_tag_matrix[full_t] * pw_t).sum(dim=0, keepdim=True)    # (1, n_tags)
    user_tag_norm = F.normalize(user_tag_pool, p=2, dim=1)
    cand_tag_norm = ranker.game_tag_matrix_l2[cand_idx]                                 # (n_cand, n_tags)
    return (user_tag_norm * cand_tag_norm).sum(dim=1)                                   # (n_cand,)


def text_cosine_for_user(ranker, full_ids: list, full_pw: list, cand_idx: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """Recreate precompute's text_cosine (B-10). Same shape as tag_cosine_for_user but
    over the frozen 768-d description embeddings: user side is sum-weighted RAW text rows
    then L2-normalized; candidate side reads the pre-normalized buffer — identical to
    train._forward_batch."""
    if not full_ids:
        return torch.zeros(cand_idx.shape[0], device=device)
    full_t = torch.tensor(full_ids, dtype=torch.long, device=device)
    pw_t   = torch.tensor(full_pw,  dtype=torch.float32, device=device).unsqueeze(-1)   # (L, 1)
    user_text_pool = (ranker.game_text_matrix[full_t] * pw_t).sum(dim=0, keepdim=True)  # (1, 768)
    user_text_norm = F.normalize(user_text_pool, p=2, dim=1)
    cand_text_norm = ranker.game_text_matrix_l2[cand_idx]                               # (n_cand, 768)
    return (user_text_norm * cand_text_norm).sum(dim=1)                                 # (n_cand,)


# ── User-side Bucket 5 scalars (price / year / sentiment / log_count / median log pt) ──

def build_user_b5_scalars(ranker, full_ids: list, full_raw_pw: list,
                          device: torch.device) -> torch.Tensor:
    """Build the (1, 5) Bucket 5 user-side scalar tensor. Plain mean over the user's full
    history for price / year / sentiment / log_count; median of raw log-hours for playtime.
    Reads per-game scalars from the ranker's buffers so values match training exactly."""
    n_items = int(ranker.game_pad_idx)
    valid   = [i for i in full_ids if 0 <= i < n_items]
    user_b5 = torch.zeros(1, 5, device=device)
    if not valid:
        user_b5[0, B5_USER_MEAN_SENTIMENT] = SENTIMENT_UNKNOWN_FILL
        return user_b5

    v_t = torch.tensor(valid, dtype=torch.long, device=device)
    user_b5[0, B5_USER_MEAN_PRICE]     = ranker.game_price_idx[v_t].float().mean()
    user_b5[0, B5_USER_MEAN_YEAR]      = ranker.game_year_numeric[v_t].mean()
    user_b5[0, B5_USER_MEAN_LOG_COUNT] = ranker.game_log_count[v_t].mean()
    user_b5[0, B5_USER_MEAN_SENTIMENT] = ranker.game_sentiment[v_t].mean()
    if full_raw_pw:
        user_b5[0, B5_USER_MEDIAN_LOG_PT] = float(np.median(full_raw_pw))
    return user_b5


# ── Rerank: cross features for the candidate pool → ranker logits ───────────────

def rerank_candidates(ranker, device, user_inputs: dict,
                      full_ids: list, full_pw: list, full_raw_pw: list,
                      cand_indices: list) -> torch.Tensor:
    """
    Score a candidate pool for one user. Returns (n_cand,) raw ranker logits in the
    same order as cand_indices. Builds all 23 cross features (Bucket 0/1/2/5/6) exactly
    as the training loop / precompute does, then calls ranker.score_pairs.

    Aligns the computed cross tensor (COMPUTED_CROSS_FEATURES wide) to the loaded
    checkpoint's ranker.n_cross_features: slice for shorter (stable leading-N column
    order), zero-pad for longer (historical dropped-bucket checkpoints — BIASED, the
    caller should warn).
    """
    n_cand = len(cand_indices)
    cand_t = torch.tensor(cand_indices, dtype=torch.long, device=device)

    x_avg    = torch.from_numpy(user_inputs['X_user_avg_log']).to(device)
    h_lkd    = torch.from_numpy(user_inputs['X_hist_liked']).to(device)
    h_lkd_pw = torch.from_numpy(user_inputs['X_hist_liked_playtime_weights']).to(device)
    h_dis    = torch.from_numpy(user_inputs['X_hist_disliked']).to(device)
    h_full   = torch.from_numpy(user_inputs['X_hist_full']).to(device)
    h_pw     = torch.from_numpy(user_inputs['X_hist_playtime_weights']).to(device)
    pad_idx_int = int(ranker.game_pad_idx)

    # Per-item buffer bundles — same bundles the training loop builds (guarantees parity).
    buffers = OverlapBuffers(
        genre_binary=ranker.game_genre_binary,
        genre_count =ranker.game_genre_count,
        tag_binary  =ranker.game_tag_binary,
        tag_count   =ranker.game_tag_count,
        game_dev_idx=ranker.game_dev_idx,
        dev_pad_idx =int(ranker.dev_pad_idx),
    )
    b5_buffers = Bucket5GameBuffers(
        price_bucket    =ranker.game_price_idx.float(),
        year_numeric    =ranker.game_year_numeric,
        median_log_hours=ranker.game_median_log_hours,
        log_count       =ranker.game_log_count,
        sentiment       =ranker.game_sentiment,
    )
    niche_buffers = NicheBuffers(
        tag_binary_idf       =ranker.game_tag_binary_idf,
        tag_count_idf        =ranker.game_tag_count_idf,
        tag_mean_idf         =ranker.game_tag_mean_idf,
        tag_max_idf          =ranker.game_tag_max_idf,
        dev_log_catalog_size =ranker.game_dev_log_catalog_size,
    )

    with torch.no_grad():
        user_concat = ranker.user_forward(x_avg, h_lkd, h_dis, h_full, h_pw)
        user_concat_exp = user_concat.expand(n_cand, -1)
        item_concat     = ranker.item_embedding(cand_t)
        cand_t1         = cand_t.unsqueeze(0)                                # (1, n_cand)

        def _slice_triple(slice_indices, slice_weights):
            """Categorical overlap triple for one history slice, squeezed to (n_cand,).
            Zero-tensors if the slice has no non-pad entries (matches precompute's
            graceful-degrade path)."""
            if not bool((slice_indices != pad_idx_int).any().item()):
                z = torch.zeros(n_cand, device=device)
                return z, z, z
            g, t, d = categorical_overlap_triple(buffers,
                                                  history_indices=slice_indices,
                                                  history_weights=slice_weights,
                                                  cand_idx=cand_t1)
            return g.squeeze(0), t.squeeze(0), d.squeeze(0)

        tag_cos  = tag_cosine_for_user(ranker, full_ids, full_pw, cand_t, device)
        text_cos = text_cosine_for_user(ranker, full_ids, full_pw, cand_t, device)

        # Bucket 1 — full-history slice; Bucket 2A — liked slice; Bucket 2B — last-3-liked.
        genre_ov,    tag_ov,    dev_aff    = _slice_triple(h_full, h_pw)
        genre_ov_l,  tag_ov_l,  dev_aff_l  = _slice_triple(h_lkd,  h_lkd_pw)
        h_recent3, h_recent3_pw = last_n_history(h_lkd, h_lkd_pw, n=3, pad_idx=pad_idx_int)
        genre_ov_r3, tag_ov_r3, dev_aff_r3 = _slice_triple(h_recent3, h_recent3_pw)

        # Bucket 5 — numeric-match quintuple.
        user_b5 = build_user_b5_scalars(ranker, full_ids, full_raw_pw, device)
        price_m, era_g, ptcal_m, pop_m, sent_m = numeric_match_quintuple(
            b5_buffers, user_b5, cand_t1)
        price_m, era_g, ptcal_m, pop_m, sent_m = (
            price_m.squeeze(0), era_g.squeeze(0), ptcal_m.squeeze(0),
            pop_m.squeeze(0),   sent_m.squeeze(0))

        # Bucket 6 — 8 niche feature crosses (4 concepts × {full, liked}).
        tag_ov_idf_f = weighted_overlap(niche_buffers.tag_binary_idf,
                                         niche_buffers.tag_count_idf,
                                         history_indices=h_full, history_weights=h_pw,
                                         cand_idx=cand_t1).squeeze(0)
        tag_ov_idf_l = weighted_overlap(niche_buffers.tag_binary_idf,
                                         niche_buffers.tag_count_idf,
                                         history_indices=h_lkd, history_weights=h_lkd_pw,
                                         cand_idx=cand_t1).squeeze(0)
        niche_tag_f, max_tag_f, niche_dev_f = niche_scalar_triple(niche_buffers,
                                         history_indices=h_full, history_weights=h_pw,
                                         cand_idx=cand_t1)
        niche_tag_l, max_tag_l, niche_dev_l = niche_scalar_triple(niche_buffers,
                                         history_indices=h_lkd, history_weights=h_lkd_pw,
                                         cand_idx=cand_t1)
        niche_tag_f, max_tag_f, niche_dev_f = (
            niche_tag_f.squeeze(0), max_tag_f.squeeze(0), niche_dev_f.squeeze(0))
        niche_tag_l, max_tag_l, niche_dev_l = (
            niche_tag_l.squeeze(0), max_tag_l.squeeze(0), niche_dev_l.squeeze(0))

        cross = compute_cross_features(tag_cos, genre_ov, tag_ov, dev_aff,
                                       genre_ov_l, tag_ov_l, dev_aff_l,
                                       genre_ov_r3, tag_ov_r3, dev_aff_r3,
                                       price_m, era_g, ptcal_m, pop_m, sent_m,
                                       tag_ov_idf_f, tag_ov_idf_l,
                                       niche_tag_f,  niche_tag_l,
                                       max_tag_f,    max_tag_l,
                                       niche_dev_f,  niche_dev_l,
                                       text_cos)

        model_expected = ranker.n_cross_features
        if model_expected != COMPUTED_CROSS_FEATURES:
            if model_expected < COMPUTED_CROSS_FEATURES:
                cross = cross[..., :model_expected]
            else:
                pad = torch.zeros(*cross.shape[:-1],
                                   model_expected - COMPUTED_CROSS_FEATURES,
                                   device=cross.device, dtype=cross.dtype)
                cross = torch.cat([cross, pad], dim=-1)

        return ranker.score_pairs(user_concat_exp, item_concat, cross)
