"""
Qualitative ranker inspection on canary users.

For each canary:
  1. Run CG on the synthetic user history → top-K corpus candidates (CG order)
  2. Build synthetic ranker user inputs (4 pools + playtime weights + avg_log)
  3. Run ranker on the K candidates → re-sort
  4. Print side-by-side top-N

Synthetic playtime weights mirror src/evaluate._build_user_embedding:
  fav = 10.0, anchor = 2.0, dis = 0.5  (normalized to sum=1 for the playtime pool).
"""
import glob
import json
import os
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.cross_features import (B5_USER_MEAN_LOG_COUNT, B5_USER_MEAN_PRICE,
                                     B5_USER_MEAN_SENTIMENT, B5_USER_MEAN_YEAR,
                                     B5_USER_MEDIAN_LOG_PT, Bucket5GameBuffers,
                                     OverlapBuffers, categorical_overlap_triple,
                                     last_n_history, numeric_match_quintuple)
from ranker.dataset import compute_cross_features
from ranker.train import _ALPHA_TO_CG_GLOB, build_ranker, get_config, get_device
from src.features import SENTIMENT_UNKNOWN_FILL
from src.dataset import MAX_HISTORY_LEN
from src.evaluate import (USER_TYPE_TO_DISLIKED_GAMES,
                           USER_TYPE_TO_FAVORITE_GAMES,
                           USER_TYPE_TO_TAGS,
                           SIMULATED_FAV_LOG_HOURS,
                           SIMULATED_ANCHOR_LOG_HOURS,
                           SIMULATED_DISLIKE_LOG_HOURS,
                           _get_anchor_titles,
                           _build_user_embedding,
                           build_game_embeddings)
from src.features import load_features
from src.train import (build_model as build_cg_model,
                        load_config_for_checkpoint)


TOP_K_CG              = 100      # candidates from CG to rerank (matches precompute)
TOP_N_DISPLAY_DEFAULT = 10
DEFAULT_CANARIES      = list(USER_TYPE_TO_FAVORITE_GAMES.keys())
# Width of the cross_features tensor the canary code below computes. Bump this in
# lockstep with the cross-feature compute path when a new bucket lands — the
# alignment helper in run_canary uses it to decide slice vs zero-pad when a
# historical checkpoint expects a different n_cross_features.
CANARY_COMPUTED_CROSS_FEATURES = 15      # Bucket 5 roster (tag_cosine + 9 overlaps + 5 numeric)


# ── Resolve checkpoints ──────────────────────────────────────────────────────

def _resolve_cg_checkpoint(alpha: float = 0.0) -> str:
    """Resolve the CG checkpoint whose training α matches the ranker's α.

    Defaults to α=0 (the offline-metrics throwaway CG). Pass `alpha` to match the
    ranker being canary'd — α=0 ranker should compare against α=0 CG, α=0.4 ranker
    against the α=0.4 PROD CG. Mismatched α makes the side-by-side useless: the
    α=0 ranker is trained without popularity suppression, and putting it next to
    an α=0.4 CG measures "what popularity tax does the CG impose" rather than
    "did the ranker's reranking add value."

    Uses the same α→glob map (ranker.train._ALPHA_TO_CG_GLOB) that precompute and
    warm-start use, so all three call sites agree on which CG file is the right
    α match.
    """
    pattern = _ALPHA_TO_CG_GLOB.get(alpha)
    if pattern is None:
        supported = sorted(_ALPHA_TO_CG_GLOB.keys())
        raise ValueError(
            f"No CG checkpoint convention for ranker α={alpha}. "
            f"Supported: {supported}. Either train a matching CG and add it to "
            f"ranker.train._ALPHA_TO_CG_GLOB, or pass an explicit cg_checkpoint."
        )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No CG checkpoint matched α={alpha} glob '{pattern}'. "
            "Either train one, or pass an explicit cg_checkpoint to canary."
        )
    return matches[-1]


def _resolve_ranker_checkpoint() -> str:
    matches = glob.glob('saved_models/ranker/ranker_wd_*.pth')
    if not matches:
        raise FileNotFoundError("No ranker checkpoint found in saved_models/ranker/")
    return max(matches, key=os.path.getmtime)


def _read_ranker_alpha(ranker_checkpoint: str) -> float:
    """Read `popularity_alpha` from a ranker checkpoint's _config.json sidecar.
    Returns 0.0 if the sidecar is missing or the key is absent — same default the
    ranker's get_config() uses."""
    cfg_path = os.path.splitext(ranker_checkpoint)[0] + '_config.json'
    if not os.path.exists(cfg_path):
        return 0.0
    with open(cfg_path) as f:
        return float(json.load(f).get('popularity_alpha', 0.0))


# ── Synthetic user-side inputs (mirrors src/evaluate._build_user_embedding) ─

def _build_synthetic_user_inputs(fs: dict, user_type: str, pad_idx: int):
    """
    Build the user-side inputs for the ranker:
      X_user_avg_log (1, 1), X_hist_liked / disliked / full (1, MAX_HISTORY_LEN),
      X_hist_playtime_weights (1, MAX_HISTORY_LEN),
      X_hist_liked_playtime_weights (1, MAX_HISTORY_LEN)   ← Bucket 2

    Also returns metadata for display + cross-feature computation:
      fav_titles, anchor_titles, dis_titles, full_idxs (unpadded list).
    """
    fav_titles = USER_TYPE_TO_FAVORITE_GAMES[user_type]
    dis_titles = USER_TYPE_TO_DISLIKED_GAMES.get(user_type, [])
    tag_names  = USER_TYPE_TO_TAGS.get(user_type, [])
    anchor_titles = _get_anchor_titles(fs, tag_names,
                                        exclude=set(fav_titles) | set(dis_titles))

    title_to_iid = {v: k for k, v in fs['item_id_to_title'].items()}
    item_to_idx  = fs['item_to_idx']

    def titles_to_idxs(titles):
        idxs = []
        for t in titles:
            iid = title_to_iid.get(t)
            if iid and iid in item_to_idx:
                idxs.append(item_to_idx[iid])
        return idxs

    fav_idxs    = titles_to_idxs(fav_titles)
    dis_idxs    = titles_to_idxs(dis_titles)
    anchor_idxs = titles_to_idxs(anchor_titles)

    # Mirror _build_user_embedding triple pool construction.
    liked_ids    = list(fav_idxs)
    disliked_ids = list(dis_idxs)
    full_ids     = list(fav_idxs) + list(anchor_idxs) + list(dis_idxs)

    raw_pw = ([SIMULATED_FAV_LOG_HOURS]     * len(fav_idxs) +
              [SIMULATED_ANCHOR_LOG_HOURS]  * len(anchor_idxs) +
              [SIMULATED_DISLIKE_LOG_HOURS] * len(dis_idxs))
    total_pw = sum(raw_pw) or 1.0
    full_pw  = [w / total_pw for w in raw_pw]

    # Bucket 2 — playtime weights for the LIKED slice (favorites only, since
    # the synthetic user's "liked" history is exactly the favorite-games list).
    # Mirrors precompute's per-slice normalization (sum to 1 over the liked subset).
    liked_raw = [SIMULATED_FAV_LOG_HOURS] * len(fav_idxs)
    total_liked = sum(liked_raw) or 1.0
    liked_pw = [w / total_liked for w in liked_raw]

    total_items = max(len(full_ids), 1)
    avg_log = (len(fav_idxs)    * SIMULATED_FAV_LOG_HOURS +
               len(anchor_idxs) * SIMULATED_ANCHOR_LOG_HOURS +
               len(dis_idxs)    * SIMULATED_DISLIKE_LOG_HOURS) / total_items

    def pad_list(ids, length=MAX_HISTORY_LEN):
        out = np.full(length, pad_idx, dtype=np.int64)
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
        'X_user_avg_log':                   np.array([[avg_log]], dtype=np.float32),       # (1, 1)
        'X_hist_liked':                     pad_list(liked_ids).reshape(1, -1),             # (1, H)
        'X_hist_disliked':                  pad_list(disliked_ids).reshape(1, -1),
        'X_hist_full':                      pad_list(full_ids).reshape(1, -1),
        'X_hist_playtime_weights':          pad_weights(full_pw).reshape(1, -1),
        'X_hist_liked_playtime_weights':    pad_weights(liked_pw).reshape(1, -1),           # (1, H) ← Bucket 2
    }
    # raw_pw (un-normalized SIMULATED_* log-hours per full-history position) is the
    # source for the synthetic user_median_log_playtime — Bucket 5 needs the raw
    # values, not the normalized weights.
    return user_inputs, fav_titles, anchor_titles, dis_titles, full_ids, full_pw, raw_pw


# ── Tag cosine for a synthetic canary against candidate items ──────────────

def _synthetic_tag_cosine(ranker, full_ids: list, full_pw: list, cand_idx: torch.Tensor,
                          device: torch.device) -> torch.Tensor:
    """
    Recreate precompute's tag_cosine (B-0) for a synthetic user. Uses the ranker's own
    game_tag_matrix buffer (raw TF-IDF rows).
    """
    if not full_ids:
        return torch.zeros(cand_idx.shape[0], device=device)
    full_t = torch.tensor(full_ids, dtype=torch.long, device=device)
    pw_t   = torch.tensor(full_pw,  dtype=torch.float32, device=device).unsqueeze(-1)  # (L, 1)
    # User-side stays on RAW (sum-weighted raw TF-IDF, then normalize the pooled vector).
    # Cand-side reads from the pre-normalized buffer — identical to train._forward_batch.
    user_tag_pool = (ranker.game_tag_matrix[full_t] * pw_t).sum(dim=0, keepdim=True)   # (1, n_tags)
    user_tag_norm = F.normalize(user_tag_pool, p=2, dim=1)
    cand_tag_norm = ranker.game_tag_matrix_l2[cand_idx]                                # (n_cand, n_tags)
    return (user_tag_norm * cand_tag_norm).sum(dim=1)                                  # (n_cand,)


# ── Synthetic user-side Bucket 5 scalars ────────────────────────────────────

def _build_synthetic_b5_scalars(ranker, full_ids: list, full_raw_pw: list,
                                device: torch.device) -> torch.Tensor:
    """
    Build the (1, 5) Bucket 5 user-side scalar tensor for a synthetic canary user.

    Mirrors features.load_features per-user aggregation logic: plain mean over the
    user's full history for price / year / sentiment / log_count; median of raw
    log-hours for playtime. Reads per-game scalars from the ranker's registered
    buffers (game_price_idx / game_year_numeric / game_log_count / game_sentiment),
    so synthetic canary values use exactly the same per-game numerics the model
    saw during training.
    """
    n_items = int(ranker.game_pad_idx)
    valid   = [i for i in full_ids if 0 <= i < n_items]
    user_b5 = torch.zeros(1, 5, device=device)
    if not valid:
        # Empty history → 0s except sentiment fallback (parallel to features.load_features).
        user_b5[0, B5_USER_MEAN_SENTIMENT] = SENTIMENT_UNKNOWN_FILL
        return user_b5

    v_t = torch.tensor(valid, dtype=torch.long, device=device)
    user_b5[0, B5_USER_MEAN_PRICE]      = ranker.game_price_idx[v_t].float().mean()
    user_b5[0, B5_USER_MEAN_YEAR]       = ranker.game_year_numeric[v_t].mean()
    user_b5[0, B5_USER_MEAN_LOG_COUNT]  = ranker.game_log_count[v_t].mean()
    user_b5[0, B5_USER_MEAN_SENTIMENT]  = ranker.game_sentiment[v_t].mean()
    # full_raw_pw holds SIMULATED_*_LOG_HOURS per history position (FAV=10 / ANCHOR=2 /
    # DISLIKE=0.5). For canary the median of these simulated log values stands in for
    # `median(log1p(hours))` — Z-scored at the model so the unit isn't load-bearing.
    if full_raw_pw:
        user_b5[0, B5_USER_MEDIAN_LOG_PT] = float(np.median(full_raw_pw))
    return user_b5


# ── Main canary loop ────────────────────────────────────────────────────────

def _format_title(title: str, max_w: int) -> str:
    return title if len(title) <= max_w else title[:max_w - 1] + '…'


def run_canary(cg_checkpoint: str | None = None,
               ranker_checkpoint: str | None = None,
               canaries: list = None,
               data_dir: str = 'data',
               top_n: int = TOP_N_DISPLAY_DEFAULT,
               output_file: str | None = None) -> str:
    import io
    canaries = canaries or DEFAULT_CANARIES

    ranker_checkpoint = ranker_checkpoint or _resolve_ranker_checkpoint()
    # Resolve CG by the ranker's training α — otherwise the side-by-side compares
    # an α=0 ranker against an α=0.4 CG (or vice versa), which mostly measures the
    # CG's popularity-suppression tax rather than the ranker's reranking lift.
    # An explicit cg_checkpoint argument still wins for one-off comparisons.
    ranker_alpha      = _read_ranker_alpha(ranker_checkpoint)
    cg_checkpoint     = cg_checkpoint     or _resolve_cg_checkpoint(ranker_alpha)
    if output_file is None:
        base = os.path.splitext(os.path.basename(ranker_checkpoint))[0]
        output_file = f"ranker/canary_results/{base}.txt"

    out = io.StringIO()

    def emit(line: str = ''):
        print(line)
        out.write(line + '\n')

    emit(f"CG checkpoint:     {cg_checkpoint}")
    emit(f"Ranker checkpoint: {ranker_checkpoint}")
    emit(f"α match:           ranker α={ranker_alpha} → CG α={ranker_alpha} (parity)")
    emit(f"Top-N per canary:  {top_n}    Top-K from CG: {TOP_K_CG}")
    emit(f"Canaries:          {len(canaries)}")
    emit('')

    fs = load_features(data_dir=data_dir)
    pad_idx = fs['n_items']

    device = get_device()
    emit(f"Device: {device}")
    emit('')

    # ── CG model (read-only retrieval) ───────────────────────────────────────
    cg_config = load_config_for_checkpoint(cg_checkpoint)
    cg = build_cg_model(cg_config, fs)
    cg.load_state_dict(torch.load(cg_checkpoint, weights_only=True, map_location='cpu'))
    cg.eval().to(device)
    # Pre-compute V_all (n_items, output_dim) for retrieval scoring.
    _, all_ids, V_all = build_game_embeddings(cg, fs)
    emit(f"CG: popularity_alpha={cg_config.get('popularity_alpha', 'unknown')}  "
         f"V_all shape={tuple(V_all.shape)}")

    # ── Ranker (load config from sidecar; disable warm-start during construction) ──
    cfg = get_config()
    cfg_path = os.path.splitext(ranker_checkpoint)[0] + '_config.json'
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k in ('hidden_dims', 'dropout', 'n_cross_features', 'n_wide_normalized',
                  'item_id_emb_dim', 'item_genre_emb_dim', 'item_tag_emb_dim',
                  'developer_emb_dim', 'year_emb_dim', 'price_emb_dim',
                  'user_genre_emb_dim', 'user_tag_emb_dim',
                  'item_tag_hidden', 'user_tag_hidden', 'user_genre_hidden',
                  'popularity_alpha'):
            if k in saved:
                cfg[k] = saved[k]
    cfg['warm_start_cg_checkpoint'] = None    # we're loading the trained ranker checkpoint
    ranker = build_ranker(cfg, fs).to(device)
    ranker.load_state_dict(torch.load(ranker_checkpoint, weights_only=True, map_location=device))
    ranker.eval()
    emit(f"Ranker: popularity_alpha={cfg.get('popularity_alpha', 0.0)}  "
         f"n_cross_features={cfg.get('n_cross_features', 0)}")

    # Canary's cross-feature compute path produces a fixed-width tensor matching the
    # canary code's CURRENT bucket era (Bucket 5 → 15 features). Historical checkpoints
    # trained with a different n_cross_features (Phase A: 1, Bucket 1: 4, Bucket 2: 10,
    # dropped trials at 13 / 16) won't match without alignment. Slice for shorter
    # (column ordering is stable — keep the first N). Pad with zeros for longer (we
    # can't reproduce dropped buckets' missing features here; the head's weights for
    # those slots get fed zero, which BIASES the output — flag loudly).
    #
    # Derived from CANARY_COMPUTED_CROSS_FEATURES at module level so the constant is
    # one edit away from the actual compute-path width and can't drift silently when a
    # new bucket lands.
    canary_computed_cross = CANARY_COMPUTED_CROSS_FEATURES
    model_expected_cross  = int(cfg.get('n_cross_features', 0))
    if model_expected_cross > canary_computed_cross:
        emit(f"⚠ ALIGNMENT WARNING: this canary code computes {canary_computed_cross} cross "
             f"features but the checkpoint was trained with {model_expected_cross}. The extra "
             f"{model_expected_cross - canary_computed_cross} columns will be zero-padded — "
             f"the model's head weights for those slots see zero instead of trained-real values, "
             f"so the ranker scores below are BIASED relative to a true reproduction. The "
             f"top-N ordering is still indicative of the model's behavior on the in-canary "
             f"features but should not be treated as a faithful re-run of the original canary.")
    elif model_expected_cross < canary_computed_cross:
        emit(f"Note: canary computes {canary_computed_cross} cross features; checkpoint expects "
             f"{model_expected_cross}. Slicing to first {model_expected_cross} (stable column order).")
    emit('')

    for user_type in canaries:
        if user_type not in USER_TYPE_TO_FAVORITE_GAMES:
            print(f"[skip] unknown canary: {user_type}")
            continue

        # ── 1. CG retrieval ──────────────────────────────────────────────────
        with torch.no_grad():
            user_emb  = _build_user_embedding(cg, fs, user_type)               # (1, output_dim)
            cg_scores = (V_all @ user_emb.T).squeeze(-1)                       # (n_items,)

        ui, fav_titles, anchor_titles, dis_titles, full_ids, full_pw, full_raw_pw = \
            _build_synthetic_user_inputs(fs, user_type, pad_idx)
        exclude_titles = set(fav_titles) | set(anchor_titles) | set(dis_titles)

        # Take top-K by CG score, excluding seed titles.
        sorted_cg = torch.argsort(cg_scores, descending=True).tolist()
        cg_top_corpus = []
        for idx in sorted_cg:
            iid   = all_ids[idx]
            title = fs['item_id_to_title'].get(iid, '')
            if title in exclude_titles:
                continue
            cg_top_corpus.append(idx)
            if len(cg_top_corpus) >= TOP_K_CG:
                break

        # ── 2. Ranker rerank ─────────────────────────────────────────────────
        n_cand = len(cg_top_corpus)
        cand_t = torch.tensor(cg_top_corpus, dtype=torch.long, device=device)

        x_avg     = torch.from_numpy(ui['X_user_avg_log']).to(device)
        h_lkd     = torch.from_numpy(ui['X_hist_liked']).to(device)
        h_lkd_pw  = torch.from_numpy(ui['X_hist_liked_playtime_weights']).to(device)
        h_dis     = torch.from_numpy(ui['X_hist_disliked']).to(device)
        h_full    = torch.from_numpy(ui['X_hist_full']).to(device)
        h_pw      = torch.from_numpy(ui['X_hist_playtime_weights']).to(device)
        pad_idx_int = int(ranker.game_pad_idx)

        # Bundle the 6 per-item buffers used by every history-slice triple-overlap
        # compute. Same bundle the training loop uses — guarantees parity.
        buffers = OverlapBuffers(
            genre_binary=ranker.game_genre_binary,
            genre_count =ranker.game_genre_count,
            tag_binary  =ranker.game_tag_binary,
            tag_count   =ranker.game_tag_count,
            game_dev_idx=ranker.game_dev_idx,
            dev_pad_idx =int(ranker.dev_pad_idx),
        )
        # Bucket 5 — same bundle the training loop builds. price_bucket cast to float
        # once per canary; cheap.
        b5_buffers = Bucket5GameBuffers(
            price_bucket    =ranker.game_price_idx.float(),
            year_numeric    =ranker.game_year_numeric,
            median_log_hours=ranker.game_median_log_hours,
            log_count       =ranker.game_log_count,
            sentiment       =ranker.game_sentiment,
        )

        def _slice_triple(slice_indices, slice_weights):
            """Categorical overlap triple for a single history slice, squeezed back to (n_cand,)
            for the B=1 synthetic-user case. Returns (genre, tag, dev_affinity) or three
            zero-tensors if the slice has no non-pad entries (the precompute graceful-degrade
            path produces 0 in that case anyway; we short-circuit for clarity)."""
            if not bool((slice_indices != pad_idx_int).any().item()):
                z = torch.zeros(n_cand, device=device)
                return z, z, z
            g, t, d = categorical_overlap_triple(buffers,
                                                  history_indices=slice_indices,
                                                  history_weights=slice_weights,
                                                  cand_idx=cand_t1)
            return g.squeeze(0), t.squeeze(0), d.squeeze(0)

        with torch.no_grad():
            us = ranker.user_forward(x_avg, h_lkd, h_dis, h_full, h_pw)
            user_concat_exp = us.user_concat.expand(n_cand, -1)
            item_concat = ranker.item_embedding(cand_t)

            # Wide-path cross features for the synthetic user.
            # The padded (1, MAX_HISTORY_LEN) user-side tensors are the same shape the
            # training loop uses, so the cross-feature utils take the same inputs and
            # produce bit-exactly identical outputs (no special unpadded code path).
            cand_t1 = cand_t.unsqueeze(0)                                            # (1, n_cand)
            tag_cos = _synthetic_tag_cosine(ranker, full_ids, full_pw, cand_t, device)

            # Bucket 1 — full-history slice (h_full, h_pw)
            genre_ov,    tag_ov,    dev_aff    = _slice_triple(h_full, h_pw)
            # Bucket 2A — liked-only slice (h_lkd, h_lkd_pw)
            genre_ov_l,  tag_ov_l,  dev_aff_l  = _slice_triple(h_lkd,  h_lkd_pw)
            # Bucket 2B — last-3-liked window (derived via last_n_history)
            h_recent3, h_recent3_pw = last_n_history(h_lkd, h_lkd_pw, n=3, pad_idx=pad_idx_int)
            genre_ov_r3, tag_ov_r3, dev_aff_r3 = _slice_triple(h_recent3, h_recent3_pw)

            # Bucket 5 — numeric-match quintuple. user_b5 is (1, 5); call returns
            # tensors shaped (1, n_cand) each, squeezed to (n_cand,) for the head.
            user_b5 = _build_synthetic_b5_scalars(ranker, full_ids, full_raw_pw, device)
            price_m, era_g, ptcal_m, pop_m, sent_m = numeric_match_quintuple(
                b5_buffers, user_b5, cand_t1)
            price_m, era_g, ptcal_m, pop_m, sent_m = (
                price_m.squeeze(0), era_g.squeeze(0), ptcal_m.squeeze(0),
                pop_m.squeeze(0),   sent_m.squeeze(0))

            cross = compute_cross_features(tag_cos, genre_ov, tag_ov, dev_aff,
                                           genre_ov_l, tag_ov_l, dev_aff_l,
                                           genre_ov_r3, tag_ov_r3, dev_aff_r3,
                                           price_m, era_g, ptcal_m, pop_m, sent_m)

            # Align cross to the checkpoint's n_cross_features — see ALIGNMENT WARNING
            # above for the trade-offs. Stable column ordering means leading-N slice
            # gives the right subset for older (1, 4) checkpoints; zero-pad fills missing
            # slots for newer (13, 16) dropped-bucket checkpoints.
            if model_expected_cross != canary_computed_cross:
                if model_expected_cross < canary_computed_cross:
                    cross = cross[..., :model_expected_cross]
                else:
                    pad = torch.zeros(*cross.shape[:-1],
                                       model_expected_cross - canary_computed_cross,
                                       device=cross.device, dtype=cross.dtype)
                    cross = torch.cat([cross, pad], dim=-1)

            ranker_scores = ranker.score_pairs(user_concat_exp, item_concat, cross)

        rk_order = torch.argsort(ranker_scores, descending=True).tolist()
        rk_top   = [cg_top_corpus[i] for i in rk_order[:top_n]]
        cg_top   = cg_top_corpus[:top_n]

        # ── 3. Render side-by-side ───────────────────────────────────────────
        col_w = 50
        bar_w = col_w * 2 + 8
        title_line = user_type
        tags = USER_TYPE_TO_TAGS.get(user_type, [])
        if tags:
            title_line += f"  |  Tags: {', '.join(tags)}"
        emit('')
        emit('═' * bar_w)
        emit(title_line)
        emit('═' * bar_w)
        emit(f"Liked: {', '.join(fav_titles)}")
        if dis_titles:
            emit(f"Disliked: {', '.join(dis_titles)}")
        if anchor_titles:
            emit(f"Anchors: {', '.join(anchor_titles[:5])}")
        emit('')
        emit(f"{'#':<3}  {f'CG top-{top_n}':<{col_w}}  {f'Ranker top-{top_n}':<{col_w}}")
        emit('─' * bar_w)
        for i in range(top_n):
            cg_iid = all_ids[cg_top[i]]
            rk_iid = all_ids[rk_top[i]] if i < len(rk_top) else None
            cg_t_str = _format_title(fs['item_id_to_title'].get(cg_iid, ''), col_w) if i < len(cg_top) else ''
            rk_t_str = _format_title(fs['item_id_to_title'].get(rk_iid, ''), col_w) if rk_iid else ''
            emit(f"{i+1:<3}  {cg_t_str:<{col_w}}  {rk_t_str:<{col_w}}")

    emit('')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(out.getvalue())
    print(f"Wrote canary results → {output_file}  ({out.getvalue().count(chr(10)):,} lines)")
    return output_file


def dump_canary(top_n: int = 20, output_file: str | None = None,
                cg_checkpoint: str | None = None,
                ranker_checkpoint: str | None = None) -> str:
    """Top-N side-by-side CG vs Ranker for ALL canaries."""
    return run_canary(
        cg_checkpoint=cg_checkpoint,
        ranker_checkpoint=ranker_checkpoint,
        canaries=DEFAULT_CANARIES,
        top_n=top_n,
        output_file=output_file,
    )
