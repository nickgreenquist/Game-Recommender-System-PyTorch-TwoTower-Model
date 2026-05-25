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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.serving import (COMPUTED_CROSS_FEATURES,
                            build_user_inputs_from_indices,
                            rerank_candidates)
from ranker.train import _ALPHA_TO_CG_GLOB, build_ranker, get_config, get_device
from src.evaluate import (USER_TYPE_TO_DISLIKED_GAMES,
                           USER_TYPE_TO_FAVORITE_GAMES,
                           USER_TYPE_TO_TAGS,
                           SIMULATED_FAV_LOG_HOURS,
                           SIMULATED_ANCHOR_LOG_HOURS,
                           SIMULATED_DISLIKE_LOG_HOURS,
                           NICK_PLAYTIME,
                           _get_anchor_titles,
                           _build_user_embedding,
                           _build_nick_embedding,
                           build_game_embeddings)
from src.features import load_features
from src.train import (build_model as build_cg_model,
                        load_config_for_checkpoint)


TOP_K_CG              = 100      # candidates from CG to rerank (matches precompute)
TOP_N_DISPLAY_DEFAULT = 10
# Nick — the real-user canary (real Steam playtime), same profile Streamlit's Examples
# tab surfaces. Leads the list, like the Streamlit dropdown. Not a USER_TYPE_TO_* key;
# handled by a dedicated branch in run_canary.
NICK_CANARY           = "Nick (real Steam playtime)"
DEFAULT_CANARIES      = [NICK_CANARY] + list(USER_TYPE_TO_FAVORITE_GAMES.keys())
# Width of the cross-feature tensor the shared rerank path computes. Imported from
# ranker.serving (single source of truth) — the alignment warning below uses it to
# decide slice vs zero-pad when a historical checkpoint expects a different width.
CANARY_COMPUTED_CROSS_FEATURES = COMPUTED_CROSS_FEATURES


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
    Resolve a canary user_type's seed/anchor/disliked titles to corpus indices, then
    delegate to the shared ranker.serving.build_user_inputs_from_indices for the actual
    tensor construction (so canary and Streamlit build user inputs identically).

    Returns (user_inputs, fav_titles, anchor_titles, dis_titles, full_ids, full_pw, raw_pw)
    — the title lists are canary-only (for display); the rest is the shared payload.
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

    user_inputs, full_ids, full_pw, raw_pw = build_user_inputs_from_indices(
        liked_idxs=titles_to_idxs(fav_titles),
        anchor_idxs=titles_to_idxs(anchor_titles),
        disliked_idxs=titles_to_idxs(dis_titles),
        pad_idx=pad_idx,
        fav_weight=SIMULATED_FAV_LOG_HOURS,
        anchor_weight=SIMULATED_ANCHOR_LOG_HOURS,
        dis_weight=SIMULATED_DISLIKE_LOG_HOURS,
    )
    return user_inputs, fav_titles, anchor_titles, dis_titles, full_ids, full_pw, raw_pw


def _build_nick_user_inputs(fs: dict, pad_idx: int):
    """
    Nick (real Steam user) ranker inputs — mirrors Streamlit's Examples tab exactly.

    CG retrieval (handled by the caller via src.evaluate._build_nick_embedding) uses his
    real playtime-weighted pools. The ranker side here puts every played corpus game into
    the liked pool at the synthetic fav weight — the shared build_user_inputs_from_indices
    takes per-CLASS weights, not per-item playtime, same as Streamlit's `_ranker_rerank`.
    No anchors, no disliked (unplayed library games are unknown, not disliked).

    Returns (user_inputs, played_titles_sorted, full_ids, full_pw, raw_pw); played_titles
    is sorted by hours desc (most-played first) for display + the retrieval exclude set.
    """
    item_to_idx = {str(k): v for k, v in fs['item_to_idx'].items()}
    played = [(iid, hrs) for iid, hrs in NICK_PLAYTIME.items() if str(iid) in item_to_idx]
    played.sort(key=lambda x: x[1], reverse=True)
    liked_idxs    = [item_to_idx[str(iid)] for iid, _ in played]
    played_titles = [fs['item_id_to_title'][iid] for iid, _ in played
                     if iid in fs['item_id_to_title']]

    user_inputs, full_ids, full_pw, raw_pw = build_user_inputs_from_indices(
        liked_idxs=liked_idxs, anchor_idxs=[], disliked_idxs=[],
        pad_idx=pad_idx,
        fav_weight=SIMULATED_FAV_LOG_HOURS,
        anchor_weight=SIMULATED_ANCHOR_LOG_HOURS,
        dis_weight=SIMULATED_DISLIKE_LOG_HOURS,
    )
    return user_inputs, played_titles, full_ids, full_pw, raw_pw


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
    # Retrieval CG is ALWAYS the raw α=0 CG — that is the fixed serving retrieval stage
    # (architecture decision 2026-05-23): the CG retrieves recall-maximizing candidates,
    # and the popularity penalty lives on the RANKER, not retrieval. So the side-by-side
    # is "raw α=0 CG vs ranker (α=0 or α=0.4) reranking the SAME α=0 candidate pool",
    # which isolates the ranker's effect regardless of its training α. An explicit
    # cg_checkpoint argument still wins for one-off comparisons.
    ranker_alpha      = _read_ranker_alpha(ranker_checkpoint)
    cg_checkpoint     = cg_checkpoint     or _resolve_cg_checkpoint(0.0)
    if output_file is None:
        base = os.path.splitext(os.path.basename(ranker_checkpoint))[0]
        output_file = f"ranker/canary_results/{base}.txt"

    out = io.StringIO()

    def emit(line: str = ''):
        print(line)
        out.write(line + '\n')

    emit(f"CG checkpoint:     {cg_checkpoint}")
    emit(f"Ranker checkpoint: {ranker_checkpoint}")
    emit(f"Pipeline:          raw α=0 CG retrieval → ranker α={ranker_alpha} rerank (same α=0 pool)")
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
                  'user_genre_emb_dim', 'user_tag_emb_dim', 'text_emb_dim',
                  'item_tag_hidden', 'user_tag_hidden', 'user_genre_hidden', 'item_text_hidden',
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
        is_nick = (user_type == NICK_CANARY)
        if not is_nick and user_type not in USER_TYPE_TO_FAVORITE_GAMES:
            print(f"[skip] unknown canary: {user_type}")
            continue

        # ── 1. CG retrieval user-emb + ranker user inputs ────────────────────
        if is_nick:
            # Real-user profile: CG retrieval from real playtime pools; ranker side from
            # the flat-weighted liked pool (same split Streamlit's Examples tab uses).
            with torch.no_grad():
                user_emb = _build_nick_embedding(cg, fs)                       # (1, output_dim)
            ui, played_titles, full_ids, full_pw, full_raw_pw = \
                _build_nick_user_inputs(fs, pad_idx)
            anchor_titles, dis_titles = [], []
            exclude_titles = set(played_titles)        # exclude ALL of Nick's played games
            fav_titles     = played_titles[:15]        # display cap — most-played first
        else:
            with torch.no_grad():
                user_emb = _build_user_embedding(cg, fs, user_type)            # (1, output_dim)
            ui, fav_titles, anchor_titles, dis_titles, full_ids, full_pw, full_raw_pw = \
                _build_synthetic_user_inputs(fs, user_type, pad_idx)
            exclude_titles = set(fav_titles) | set(anchor_titles) | set(dis_titles)

        with torch.no_grad():
            cg_scores = (V_all @ user_emb.T).squeeze(-1)                       # (n_items,)

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

        # ── 2. Ranker rerank (shared serving path; aligns cross to ckpt width) ──
        # The ALIGNMENT WARNING above already flagged any width mismatch; rerank_candidates
        # does the actual slice/zero-pad internally against ranker.n_cross_features.
        ranker_scores = rerank_candidates(ranker, device, ui,
                                          full_ids, full_pw, full_raw_pw, cg_top_corpus)

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
