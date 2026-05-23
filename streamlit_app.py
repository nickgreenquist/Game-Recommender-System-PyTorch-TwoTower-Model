"""
Steam Game Recommender — Streamlit app.

Run:      streamlit run streamlit_app.py
Requires: serving/model.pth
          serving/game_embeddings.pt
          serving/feature_store.pt

Generate serving/ with: python main.py export
"""
import html
import json
import os

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from src.evaluate import (
    USER_TYPE_TO_FAVORITE_GAMES,
    USER_TYPE_TO_TAGS,
    SIMULATED_FAV_LOG_HOURS,
    SIMULATED_ANCHOR_LOG_HOURS,
)

from src.model import GameRecommender
from ranker.serving import build_user_inputs_from_indices, rerank_candidates
from ranker.train import build_ranker

_FAV_WEIGHT      = 10.0   # simulated log-hours weight for explicitly liked games
_ANCHOR_WEIGHT   =  2.0   # simulated log-hours weight for tag-anchor games
_ANCHORS_PER_TAG =  5
_COVER_COLS      =  5     # columns in the Steam cover grid
_PAGE_SIZE       = 20     # games per page
_TOTAL_RESULTS   = 60     # total games to fetch (3 pages)

_NON_GAME_GENRES = {
    'Utilities', 'Design & Illustration', 'Animation & Modeling',
    'Web Publishing', 'Software Training', 'Video Production',
    'Education', 'Audio Production', 'Photo Editing',
}


# ── Startup ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    fs         = torch.load('serving/feature_store.pt', weights_only=False)
    be         = torch.load('serving/game_embeddings.pt', weights_only=False)
    state_dict = torch.load('serving/model.pth', weights_only=True)
    cfg        = fs['model_config']

    model = GameRecommender(
        n_genres=fs['n_genres'],
        n_tags=fs['n_tags'],
        n_games=fs['n_items'],
        n_years=fs['n_years'],
        n_developers=fs['n_developers'],
        n_price_buckets=fs['n_price_buckets'],
        item_id_embedding_size=cfg['item_id_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        user_tag_embedding_size=cfg['user_tag_embedding_size'],
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        tag_embedding_size=cfg['tag_embedding_size'],
        developer_embedding_size=cfg['developer_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        price_embedding_size=cfg['price_embedding_size'],
        proj_hidden=cfg.get('proj_hidden', 256),
        output_dim=cfg.get('output_dim', 128),
    )
    tag_mat = fs['game_tag_matrix']  # already (n_items+1, n_tags) tensor from export
    if not isinstance(tag_mat, torch.Tensor):
        tag_mat = torch.from_numpy(np.array(tag_mat, dtype=np.float32))
    model.game_tag_matrix.copy_(tag_mat)

    genre_mat = fs['game_genre_matrix']  # (n_items+1, n_genres) tensor from export
    if not isinstance(genre_mat, torch.Tensor):
        genre_mat = torch.from_numpy(np.array(genre_mat, dtype=np.float32))
    model.game_genre_matrix.copy_(genre_mat)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_ids   = list(be.keys())
    all_embs  = torch.cat([be[iid]['GAME_EMBEDDING_COMBINED'] for iid in all_ids], dim=0)
    all_norm  = F.normalize(all_embs, dim=1)

    all_genre_embs = torch.cat([be[iid]['GAME_GENRE_EMBEDDING'] for iid in all_ids], dim=0)
    all_norm_genre = F.normalize(all_genre_embs, dim=1)

    all_tag_embs   = torch.cat([be[iid]['GAME_TAG_EMBEDDING']   for iid in all_ids], dim=0)
    all_norm_tag   = F.normalize(all_tag_embs, dim=1)

    ranker = _load_ranker(fs)

    return model, fs, be, all_ids, all_embs, all_norm, all_norm_genre, all_norm_tag, ranker


def _load_ranker(fs: dict):
    """Reconstruct the Wide & Deep ranker from serving/ artifacts, or None if absent
    (the app then runs CG-only — no supercharge button). The non-persistent game_*
    buffers are rebuilt from feature_store.pt's source arrays by build_ranker; only
    params + the persistent wide_norm buffers come from ranker.pth. Same recipe as
    ranker/canary.py, but driven purely by serving artifacts (no saved_models/, no
    get_config glob — prod has neither)."""
    if not (os.path.exists('serving/ranker.pth') and os.path.exists('serving/ranker_config.json')):
        return None
    with open('serving/ranker_config.json') as f:
        cfg = json.load(f)
    cfg['warm_start_cg_checkpoint'] = None    # never warm-start at serving time

    # build_ranker / _buffers_from_fs expects UNPADDED numpy matrices (it re-pads and
    # derives the _l2 / _binary variants). feature_store stores them PADDED for the CG,
    # so strip the pad row here.
    n_items = fs['n_items']
    afs = dict(fs)
    afs['game_tag_matrix']   = fs['game_tag_matrix'][:n_items].numpy()
    afs['game_genre_matrix'] = fs['game_genre_matrix'][:n_items].numpy()

    ranker = build_ranker(cfg, afs)
    ranker.load_state_dict(torch.load('serving/ranker.pth', weights_only=True, map_location='cpu'))
    ranker.eval()
    # Surface the ranker's training α (popularity penalty) for the comparison header —
    # read from the config sidecar so swapping in an α=0.2 ranker updates the label.
    ranker.serving_alpha = float(cfg.get('popularity_alpha', 0.0))
    return ranker


# ── Steam cover URL ───────────────────────────────────────────────────────────

def _cover_url(item_id: str) -> str:
    return f"https://cdn.cloudflare.steamstatic.com/steam/apps/{item_id}/header.jpg"


# ── Per-game display metadata ─────────────────────────────────────────────────

def _game_meta(item_id: str, fs: dict) -> dict:
    return {
        'Cover':     _cover_url(item_id),
        'Title':     fs['item_id_to_title'].get(item_id, item_id),
        'Developer': fs['item_id_to_developer'].get(item_id, ''),
        'Year':      fs['item_id_to_year'].get(item_id, ''),
        'Genres':    ', '.join(fs['item_id_to_genres'].get(item_id, [])),
        'Top Tags':  ', '.join(fs['item_id_to_top_tags'].get(item_id, [])),
    }


# ── Paginated results grid ────────────────────────────────────────────────────

def _store_results(df, result_key: str) -> None:
    """Save a results DataFrame to session state and reset to page 0."""
    st.session_state[f'{result_key}_df']   = df
    st.session_state[f'{result_key}_page'] = 0


def _show_results(result_key: str) -> None:
    """Render the current page of stored results with Prev / Next navigation."""
    df = st.session_state.get(f'{result_key}_df')
    if df is None or df.empty:
        return

    page        = st.session_state.get(f'{result_key}_page', 0)
    total_pages = max(1, (len(df) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page_df     = df.iloc[page * _PAGE_SIZE:(page + 1) * _PAGE_SIZE]

    titles = page_df['Title'].tolist()
    covers = page_df['Cover'].tolist()
    devs   = page_df['Developer'].tolist() if 'Developer' in page_df.columns else [''] * len(titles)
    years  = page_df['Year'].tolist()      if 'Year'      in page_df.columns else [''] * len(titles)

    for row_start in range(0, len(titles), _COVER_COLS):
        s = slice(row_start, row_start + _COVER_COLS)
        cols = st.columns(_COVER_COLS)
        for col, title, cover_url, dev, year in zip(cols, titles[s], covers[s], devs[s], years[s]):
            with col:
                st.image(cover_url, use_container_width=True)
                st.caption(title)
                meta = ' · '.join(str(x) for x in [dev, year] if x)
                if meta:
                    st.caption(meta)

    if total_pages > 1:
        _, prev_col, info_col, next_col, _ = st.columns([3, 1, 1, 1, 3])
        if prev_col.button('← Prev', disabled=(page == 0), key=f'{result_key}_prev'):
            st.session_state[f'{result_key}_page'] = page - 1
            st.rerun()
        info_col.markdown(
            f"<div style='text-align:center;padding-top:0.4rem'>{page + 1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
        if next_col.button('Next →', disabled=(page >= total_pages - 1), key=f'{result_key}_next'):
            st.session_state[f'{result_key}_page'] = page + 1
            st.rerun()


# ── Tag anchor helpers ────────────────────────────────────────────────────────

def _get_tag_anchors(fs: dict, tag_names: list, exclude: set) -> list:
    """Return up to _ANCHORS_PER_TAG item_ids per tag by raw TF-IDF score."""
    tag_mat  = fs['game_tag_matrix'][:fs['n_items']]  # drop padding row
    tag_to_i = fs['tag_to_i']
    item_ids = fs['item_ids']
    id_to_title = fs['item_id_to_title']
    item_to_idx = fs['item_to_idx']

    valid_tags = [t for t in tag_names if t in tag_to_i]
    anchors = []
    seen    = set(exclude)

    for tag in valid_tags:
        tag_idx     = tag_to_i[tag]
        sorted_iids = sorted(
            item_ids,
            key=lambda iid: float(tag_mat[item_to_idx[iid], tag_idx]),
            reverse=True,
        )
        count = 0
        for iid in sorted_iids:
            if count >= _ANCHORS_PER_TAG:
                break
            title = id_to_title[iid]
            if title not in seen:
                anchors.append(iid)
                seen.add(title)
                count += 1
    return anchors


# ── User embedding builder ────────────────────────────────────────────────────

def _build_user_embedding(model: GameRecommender, fs: dict,
                          liked_iids: list, anchor_iids: list,
                          anchor_in_liked: bool = True) -> torch.Tensor:
    """
    Build a user embedding from liked game IDs + tag anchors.
    Uses V2 triple-pool signature: liked / disliked / full pools + genre + tag context.

    anchor_in_liked=True  (Recommend tab): anchors go into liked + full pools.
    anchor_in_liked=False (Examples tab):  anchors go into full pool only — matches canary.
    """
    item_to_idx = fs['item_to_idx']
    pad_idx     = model.game_pad_idx

    weighted = (
        [(iid, _FAV_WEIGHT)    for iid in liked_iids]  +
        [(iid, _ANCHOR_WEIGHT) for iid in anchor_iids]
    )
    history = [(item_to_idx[iid], w) for iid, w in weighted if iid in item_to_idx]
    avg_log = sum(w for _, w in history) / max(len(history), 1)

    # Triple pools
    liked_only_idxs  = [item_to_idx[iid] for iid in liked_iids  if iid in item_to_idx]
    anchor_only_idxs = [item_to_idx[iid] for iid in anchor_iids if iid in item_to_idx]

    def to_padded(idxs):
        return torch.tensor([idxs if idxs else [pad_idx]], dtype=torch.long)

    X_disliked = to_padded([])

    # Full pool is always liked + anchor; weights correspond positionally
    full_idxs = liked_only_idxs + anchor_only_idxs
    raw_pw    = [_FAV_WEIGHT] * len(liked_only_idxs) + [_ANCHOR_WEIGHT] * len(anchor_only_idxs)
    total_pw  = sum(raw_pw) or 1.0
    norm_pw   = [w / total_pw for w in raw_pw] if raw_pw else [1.0]
    X_pw      = torch.tensor([norm_pw], dtype=torch.float32)

    if anchor_in_liked:
        X_liked = to_padded(liked_only_idxs + anchor_only_idxs)
        X_full  = to_padded(full_idxs)
    else:
        X_liked = to_padded(liked_only_idxs)
        X_full  = to_padded(full_idxs)

    X_avg_log = torch.tensor([[avg_log]], dtype=torch.float32)
    return model.user_embedding(X_avg_log, X_liked, X_disliked, X_full, X_pw)


# ── Two-stage: CG retrieval → ranker rerank ─────────────────────────────────────

def _cg_candidate_iids(user_emb: torch.Tensor, all_ids: list, all_embs: torch.Tensor,
                       exclude_iids: set, k: int = 100) -> list:
    """Top-k corpus iids by raw CG dot product (Menon Path 2: no inference correction),
    in CG order, excluding the user's seed games. This is the candidate pool the ranker
    reranks — and (in the two-stage tabs) also the CG-only list the user browses, so the
    'before' and 'after' views operate on the exact same 100 games."""
    scores = (all_embs @ user_emb.T).squeeze(-1)
    out = []
    for idx in scores.argsort(descending=True).tolist():
        iid = all_ids[idx]
        if iid in exclude_iids:
            continue
        out.append(iid)
        if len(out) >= k:
            break
    return out


def _candidate_df(cg_iids: list, fs: dict):
    """Build the paginated CG-only display DataFrame from the candidate pool, so the
    CG-only view shows exactly the games the ranker will rerank."""
    import pandas as pd
    return pd.DataFrame([_game_meta(iid, fs) for iid in cg_iids])


def _ranker_rerank(ranker, fs: dict, liked_iids: list, anchor_iids: list,
                   cand_iids: list) -> list:
    """Rerank the CG candidate iids with the ranker. Returns the candidate iids in
    ranker (descending-score) order. Builds the ranker's user-side inputs from the same
    liked + anchor games the CG used, via the shared ranker.serving path — so the
    Streamlit user is represented exactly as a canary user is."""
    device      = torch.device('cpu')
    item_to_idx = fs['item_to_idx']
    pad_idx     = fs['n_items']

    liked_idxs  = [item_to_idx[i] for i in liked_iids  if i in item_to_idx]
    anchor_idxs = [item_to_idx[i] for i in anchor_iids if i in item_to_idx]
    # Candidate iids may include items not in item_to_idx only in pathological exports;
    # filter defensively and keep the surviving iids aligned to the score order.
    cand_pairs  = [(i, item_to_idx[i]) for i in cand_iids if i in item_to_idx]
    cand_iids_f = [i for i, _ in cand_pairs]
    cand_idxs   = [x for _, x in cand_pairs]

    ui, full_ids, full_pw, full_raw_pw = build_user_inputs_from_indices(
        liked_idxs=liked_idxs, anchor_idxs=anchor_idxs, disliked_idxs=[],
        pad_idx=pad_idx,
        fav_weight=_FAV_WEIGHT, anchor_weight=_ANCHOR_WEIGHT, dis_weight=0.5,
    )
    scores = rerank_candidates(ranker, device, ui, full_ids, full_pw, full_raw_pw, cand_idxs)
    order  = scores.argsort(descending=True).tolist()
    return [cand_iids_f[i] for i in order]


def _delta_badge(delta: int) -> str:
    """Rank-movement badge: positive delta = moved up (CG rank − ranker rank)."""
    if delta > 0:
        return f"🟢 ↑{delta}"
    if delta < 0:
        return f"🔴 ↓{-delta}"
    return "🟡 no change"


def _show_comparison(cg_iids: list, ranker_iids: list, fs: dict, page_key: str,
                     ranker_alpha: float = 0.0, per_page: int = _PAGE_SIZE) -> None:
    """Row-per-rank side-by-side: CG retrieval order vs ranker-reranked order over the
    SAME candidate pool, paginated `per_page` rows at a time. Each ranker row carries a
    rank-delta badge showing how far the ranker moved that game relative to its CG
    position (computed over the FULL pool, not the page). page_key namespaces the
    Prev/Next page state per tab; ranker_alpha labels the ranker column with the model's
    popularity penalty (CG retrieval is always α=0 by design)."""
    cg_rank     = {iid: r for r, iid in enumerate(cg_iids)}        # 0-based CG position
    ranker_rank = {iid: r for r, iid in enumerate(ranker_iids)}    # 0-based ranker position
    total       = min(len(cg_iids), len(ranker_iids))
    total_pages = max(1, (total + per_page - 1) // per_page)
    page        = max(0, min(st.session_state.get(f'{page_key}_page', 0), total_pages - 1))
    start, end  = page * per_page, min((page + 1) * per_page, total)

    # Rendered as a CSS GRID with INLINE styles (not st.columns, not CSS classes):
    # Streamlit columns stack on mobile (open issue streamlit/streamlit#11392, no native
    # fix). A 3-track grid with `minmax(0,1fr)` content columns is the canonical fix for
    # children overflowing their container — the `0` min lets each track shrink below its
    # content so long titles/headers WRAP instead of bleeding into the next column
    # (plain `flex:1 1 0; min-width:0` was unreliable here). `column-gap` gives the
    # padding between CG and Ranker. Card mirrors the CG results grid: cover on top, then
    # title, developer · year, and (ranker side) the rank-delta badge stacked beneath.
    ROW  = ("display:grid;grid-template-columns:1.6rem minmax(0,1fr) minmax(0,1fr);"
            "column-gap:22px;align-items:start;margin-bottom:16px")
    RANK = "text-align:center;font-weight:700;opacity:.55;font-size:.8rem;padding-top:2px"

    def _header_cell(line1: str, line2: str = '') -> str:
        # line2 (the α popularity-penalty caption) is omitted when α=0 — neither stage
        # applies a penalty in the shipped config, so the line would just read "α=0".
        line2_el = (f"<div style='font-size:.74rem;opacity:.65;white-space:normal;"
                    f"overflow-wrap:anywhere'>{line2}</div>" if line2 else '')
        return (f"<div style='min-width:0'>"
                f"<div style='font-weight:600;font-size:.82rem;white-space:normal;"
                f"overflow-wrap:anywhere'>{line1}</div>{line2_el}</div>")

    def _meta_line(iid: str) -> str:
        dev  = str(fs['item_id_to_developer'].get(iid, ''))
        year = str(fs['item_id_to_year'].get(iid, ''))
        return ' · '.join(x for x in (dev, year) if x)

    def _card(iid: str, badge: str = '') -> str:
        title  = html.escape(str(fs['item_id_to_title'].get(iid, iid)))
        meta   = html.escape(_meta_line(iid))
        meta_el  = (f"<div style='font-size:.72rem;opacity:.6;margin-top:2px;"
                    f"white-space:normal;overflow-wrap:anywhere'>{meta}</div>" if meta else '')
        badge_el = (f"<div style='font-size:.8rem;margin-top:4px'>{badge}</div>" if badge else '')
        # Cover rendered as a background-image div, NOT an <img>: Streamlit's markdown CSS
        # forces `margin:auto` on images (centering them and ignoring inline width). A
        # background div sidesteps that entirely. Steam headers are 460×215 → use that
        # aspect ratio so the banner fills the card width with no crop or centering.
        return (
            "<div style='min-width:0'>"
            f"<div style='width:100%;aspect-ratio:460/215;border-radius:4px;"
            f"background-color:#1b2838;background-position:center;background-size:cover;"
            f"background-image:url(\"{_cover_url(iid)}\")'></div>"
            f"<div style='font-size:.8rem;line-height:1.2;margin-top:4px;"
            f"white-space:normal;overflow-wrap:anywhere;word-break:break-word'>{title}</div>"
            f"{meta_el}{badge_el}</div>"
        )

    parts = [
        f"<div style='{ROW}'><div style='{RANK}'></div>"
        f"{_header_cell('Two-tower CG retrieval')}"
        f"{_header_cell('⚡ Reranker · Wide &amp; Deep', f'α={ranker_alpha:g} popularity penalty' if ranker_alpha else '')}</div>"
    ]
    for i in range(start, end):
        # Each game's delta is the same on both sides: CG_rank − ranker_rank (positive =
        # the ranker moved it up). The CG card shows where its row's game LANDED in the
        # ranker (top CG rows mostly ↓ — popular items the ranker demotes); the ranker
        # card shows where its row's game CAME FROM in CG (top ranker rows mostly ↑).
        cg_iid = cg_iids[i]
        rk_iid = ranker_iids[i]
        cg_delta = i - ranker_rank.get(cg_iid, i)
        rk_delta = cg_rank.get(rk_iid, i) - i
        parts.append(
            f"<div style='{ROW}'><div style='{RANK}'>{i+1}</div>"
            f"{_card(cg_iid, _delta_badge(cg_delta))}"
            f"{_card(rk_iid, _delta_badge(rk_delta))}</div>")
    # st.html (not st.markdown): st.markdown runs the HTML through a CommonMark +
    # sanitizer pass that mangles dense nested-div layouts (the source of the earlier
    # text bleed). st.html renders raw HTML faithfully — same lesson as the Book app's
    # cover rendering.
    st.html("".join(parts))

    if total_pages > 1:
        _, prev_col, info_col, next_col, _ = st.columns([3, 1, 1, 1, 3])
        if prev_col.button('← Prev', disabled=(page == 0), key=f'{page_key}_prev'):
            st.session_state[f'{page_key}_page'] = page - 1
            st.rerun()
        info_col.markdown(
            f"<div style='text-align:center;padding-top:0.4rem'>{page + 1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
        if next_col.button('Next →', disabled=(page >= total_pages - 1), key=f'{page_key}_next'):
            st.session_state[f'{page_key}_page'] = page + 1
            st.rerun()


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(model, fs, all_ids, all_embs, ranker=None):
    st.caption(
        "Select games you've enjoyed and the model will infer your taste from your "
        "play history. The more games you add, the sharper the recommendations."
    )

    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_tags'):
            st.session_state[key] = []
        for key in ('rec_df', 'rec_page', 'rec_anchor_caption',
                    'rec_cg_iids', 'rec_liked_iids', 'rec_anchor_iids',
                    'rec_ranker_iids', 'rec_ranked', 'rec_cmp_page'):
            st.session_state.pop(key, None)

    all_titles = fs['popularity_ordered_titles']
    liked_titles = st.multiselect(
        "Games you've played and enjoyed",
        all_titles, key='rec_liked', max_selections=30,
    )

    with st.expander("Refine by Tags (optional)"):
        st.caption(
            "Add Steam tags to describe what you're looking for — subgenres, moods, mechanics "
            "(e.g. 'Open World', 'Co-op', 'Rogue-like', 'Story Rich'). "
            f"The top {_ANCHORS_PER_TAG} games for each tag are added as implicit signals."
        )
        tags_sorted   = sorted(t for t in fs['tags_ordered'] if t)
        selected_tags = st.multiselect("Tags", tags_sorted, key='rec_tags', max_selections=10)

    _, btn_col, clear_col = st.columns([2, 1, 2])
    if clear_col.button("Clear All", use_container_width=False):
        st.session_state['_clear_rec'] = True
        st.rerun()

    if btn_col.button("Get Recommendations", use_container_width=True):
        title_to_iid = fs['title_to_item_id']
        liked_iids   = [title_to_iid[t] for t in liked_titles if t in title_to_iid]

        anchor_iids = []
        if selected_tags:
            anchor_iids = _get_tag_anchors(fs, selected_tags, exclude=set(liked_titles))

        if not liked_iids and not anchor_iids:
            st.warning("Select at least one game or tag.")
        else:
            with torch.no_grad():
                user_emb = _build_user_embedding(model, fs, liked_iids, anchor_iids)
            exclude_iids = set(liked_iids) | set(anchor_iids)
            # Retrieve the top-100 candidate pool once; both the CG-only browse view and
            # the ranker rerank operate on this same pool (5 pages of 20).
            cg_iids = _cg_candidate_iids(user_emb, all_ids, all_embs, exclude_iids, k=100)
            st.session_state['rec_cg_iids']     = cg_iids
            _store_results(_candidate_df(cg_iids, fs), 'rec')
            st.session_state['rec_liked_iids']  = liked_iids
            st.session_state['rec_anchor_iids'] = anchor_iids
            st.session_state['rec_ranked']      = False
            st.session_state.pop('rec_ranker_iids', None)
            if anchor_iids:
                anchor_names = [fs['item_id_to_title'][iid] for iid in anchor_iids]
                st.session_state['rec_anchor_caption'] = "Tag anchors: " + " · ".join(anchor_names[:12])
            else:
                st.session_state.pop('rec_anchor_caption', None)

    if 'rec_df' in st.session_state:
        caption = st.session_state.get('rec_anchor_caption')
        if caption:
            st.caption(caption)

        # ── Two-stage supercharge: CG retrieval → ranker rerank ──────────────
        ranked = st.session_state.get('rec_ranked', False)
        if ranker is not None and st.session_state.get('rec_cg_iids'):
            n_cand = len(st.session_state['rec_cg_iids'])
            label = ("↺  Back to CG-only recommendations" if ranked
                     else f"⚡  Apply Ranker  ·  rerank all {n_cand} candidates")
            if st.button(label, use_container_width=True, key='rec_supercharge'):
                if not ranked:
                    with st.spinner("Reranking with the Wide & Deep ranker…"):
                        st.session_state['rec_ranker_iids'] = _ranker_rerank(
                            ranker, fs,
                            st.session_state.get('rec_liked_iids', []),
                            st.session_state.get('rec_anchor_iids', []),
                            st.session_state['rec_cg_iids'])
                    st.session_state['rec_ranked']   = True
                    st.session_state['rec_cmp_page'] = 0
                else:
                    st.session_state['rec_ranked'] = False
                st.rerun()

        if st.session_state.get('rec_ranked') and st.session_state.get('rec_ranker_iids'):
            n_cand = len(st.session_state['rec_cg_iids'])
            st.caption(f"Candidate generation retrieves the top {n_cand} by using an ultra-fast "
                       f"two-tower retrieval model; the ranker reorders those top {n_cand} CG "
                       f"candidates. Badges show each game's rank movement.")
            st.divider()
            _show_comparison(st.session_state['rec_cg_iids'],
                             st.session_state['rec_ranker_iids'], fs, 'rec_cmp',
                             ranker_alpha=getattr(ranker, 'serving_alpha', 0.0))
        else:
            _show_results('rec')
    else:
        _show_results('rec')


# ── Tab: Similar ──────────────────────────────────────────────────────────────

def tab_similar(be, fs, all_ids, all_norm):
    import pandas as pd
    st.caption(
        "Each game is represented by a single combined embedding — the concatenation of "
        "its genre, tag, game-ID, developer, year, and price towers. "
        "This tab ranks all games by cosine similarity to your selected seed."
    )
    all_titles   = fs['popularity_ordered_titles']
    title_to_iid = fs['title_to_item_id']
    selections   = st.multiselect(
        "Game(s) to find similar titles for",
        all_titles, key='sim_title', max_selections=10,
    )

    if st.button("Find Similar Games"):
        if not selections:
            st.warning("Select at least one game.")
        else:
            for old_title in st.session_state.get('sim_active_titles', []):
                st.session_state.pop(f'sim_{old_title}_df', None)
                st.session_state.pop(f'sim_{old_title}_page', None)
            active_titles = []
            for title in selections:
                iid = title_to_iid.get(title)
                if iid not in be:
                    st.error(f"'{title}' not in corpus.")
                    continue
                with torch.no_grad():
                    seed_norm = F.normalize(be[iid]['GAME_EMBEDDING_COMBINED'], dim=1)
                    sims      = (all_norm @ seed_norm.T).squeeze(-1)
                rows = []
                for idx in sims.argsort(descending=True).tolist():
                    candidate = all_ids[idx]
                    if candidate == iid:
                        continue
                    rows.append(_game_meta(candidate, fs))
                    if len(rows) >= _TOTAL_RESULTS:
                        break
                _store_results(pd.DataFrame(rows), f'sim_{title}')
                active_titles.append(title)
            st.session_state['sim_active_titles'] = active_titles

    for title in selections:
        if f'sim_{title}_df' in st.session_state:
            st.subheader(f"Similar to: {title}")
            _show_results(f'sim_{title}')


# ── Tab: Explore Genres ───────────────────────────────────────────────────────

def tab_explore_genres(model, be, fs, all_ids, all_norm_genre):
    import pandas as pd
    st.caption(
        "Queries the item genre embedding space directly. "
        "Finds games whose genre embedding best matches the selected genres "
        "by cosine similarity."
    )
    game_genres = [g for g in fs['genres_ordered'] if g not in _NON_GAME_GENRES]
    selected_genres = st.multiselect(
        "Genres", game_genres, key='explore_genre'
    )

    if st.button("Explore", key='btn_genre'):
        if not selected_genres:
            st.warning("Select at least one genre.")
        else:
            ctx = [0.0] * fs['n_genres']
            for g in selected_genres:
                ctx[fs['genre_to_i'][g]] = 1.0
            with torch.no_grad():
                query = model.item_genre_tower(torch.tensor([ctx])).view(-1)
                query_norm = F.normalize(query.unsqueeze(0), dim=1)
                sims = (all_norm_genre @ query_norm.T).squeeze(-1)
            rows = []
            for idx in sims.argsort(descending=True).tolist():
                iid = all_ids[idx]
                rows.append(_game_meta(iid, fs))
                if len(rows) >= _TOTAL_RESULTS:
                    break
            _store_results(pd.DataFrame(rows), 'genres')

    _show_results('genres')


# ── Tab: Explore Tags ─────────────────────────────────────────────────────────

def tab_explore_tags(model, be, fs, all_ids, all_norm_tag):
    import pandas as pd
    st.caption(
        "Select Steam tags to describe what you're looking for — subgenres, moods, mechanics, "
        f"tropes (e.g. 'Open World', 'Rogue-like', 'Dark Souls-like', 'Cozy'). "
        f"The top {_ANCHORS_PER_TAG} games per tag by TF-IDF score are averaged into a "
        "query embedding, then all games are ranked by cosine similarity in tag space."
    )
    tags_sorted   = sorted(t for t in fs['tags_ordered'] if t)
    selected_tags = st.multiselect(
        "Tags", tags_sorted, key='explore_tag', max_selections=10
    )

    if st.button("Explore", key='btn_tag'):
        if not selected_tags:
            st.warning("Select at least one tag.")
        else:
            tag_mat  = fs['game_tag_matrix'][:fs['n_items']]
            tag_to_i = fs['tag_to_i']
            item_ids = fs['item_ids']
            item_to_idx = fs['item_to_idx']

            anchor_tag_triples = []   # (tag, iid, title)
            seen_titles = set()
            for tag in selected_tags:
                if tag not in tag_to_i:
                    continue
                tag_idx = tag_to_i[tag]
                sorted_iids = sorted(
                    item_ids,
                    key=lambda iid: float(tag_mat[item_to_idx[iid], tag_idx]),
                    reverse=True,
                )
                count = 0
                for iid in sorted_iids:
                    if count >= _ANCHORS_PER_TAG:
                        break
                    title = fs['item_id_to_title'][iid]
                    if title not in seen_titles:
                        anchor_tag_triples.append((tag, iid, title))
                        seen_titles.add(title)
                        count += 1

            if not anchor_tag_triples:
                st.warning("No tags matched the vocabulary.")
            else:
                anchor_iids = [iid for _, iid, _ in anchor_tag_triples]
                anchor_set  = set(anchor_iids)
                with torch.no_grad():
                    query_emb  = torch.stack([
                        be[iid]['GAME_TAG_EMBEDDING'].view(-1) for iid in anchor_iids
                    ]).mean(dim=0)
                    query_norm = F.normalize(query_emb.unsqueeze(0), dim=1)
                    sims       = (all_norm_tag @ query_norm.T).squeeze(-1)

                rows = []
                for idx in sims.argsort(descending=True).tolist():
                    iid = all_ids[idx]
                    row = _game_meta(iid, fs)
                    if iid in anchor_set:
                        row['Title'] = row['Title'] + '  ◀ anchor'
                    rows.append(row)
                    if len(rows) >= _TOTAL_RESULTS:
                        break
                _store_results(pd.DataFrame(rows), 'tags')
                st.session_state['tags_anchor_caption'] = (
                    "Tag anchors — "
                    + " · ".join(f"{tag}: {title}" for tag, _, title in anchor_tag_triples)
                )

    if 'tags_df' in st.session_state:
        caption = st.session_state.get('tags_anchor_caption')
        if caption:
            st.caption(caption)
    _show_results('tags')


# ── Tab: Examples (canary profiles) ──────────────────────────────────────────

def tab_examples(model, fs, all_ids, all_embs, ranker=None):
    st.caption("Select a pre-built user profile to see what the model recommends for that taste.")

    profiles = list(USER_TYPE_TO_FAVORITE_GAMES.keys())
    selected = st.selectbox(
        "Profile",
        options=[None] + profiles,
        format_func=lambda x: "Choose a profile..." if x is None else x,
        label_visibility="collapsed",
    )

    if not selected:
        for key in ('examples_profile', 'ex_cg_iids', 'ex_ranker_iids', 'ex_ranked',
                    'ex_cmp_page'):
            st.session_state.pop(key, None)
        return

    fav_titles   = USER_TYPE_TO_FAVORITE_GAMES.get(selected, [])
    tag_names    = USER_TYPE_TO_TAGS.get(selected, [])
    title_to_iid = fs['title_to_item_id']

    missing = [t for t in fav_titles if t not in title_to_iid]
    if missing:
        st.warning("Not found in corpus (check title format): " + ", ".join(missing))

    liked_iids  = [title_to_iid[t] for t in fav_titles if t in title_to_iid]
    anchor_iids = _get_tag_anchors(fs, tag_names, exclude=set(fav_titles))

    if st.session_state.get('examples_profile') != selected:
        with torch.no_grad():
            user_emb = _build_user_embedding(model, fs, liked_iids, anchor_iids, anchor_in_liked=False)
        # Retrieve the top-100 pool (exclude liked AND anchors — matches the canary's
        # exclude set so the side-by-side mirrors the canary artifact). Both the CG-only
        # browse view and the ranker rerank operate on this same pool.
        cg_iids = _cg_candidate_iids(
            user_emb, all_ids, all_embs, set(liked_iids) | set(anchor_iids), k=100)
        st.session_state['ex_cg_iids'] = cg_iids
        _store_results(_candidate_df(cg_iids, fs), 'examples')
        st.session_state['ex_ranked'] = False
        st.session_state.pop('ex_ranker_iids', None)
        st.session_state['examples_profile'] = selected

    st.subheader(f"Recommendations for: {selected}")
    if fav_titles:
        st.caption("Because you like: " + ", ".join(fav_titles))
    if anchor_iids:
        anchor_names = [fs['item_id_to_title'][iid] for iid in anchor_iids]
        st.caption("Tag anchors: " + ", ".join(anchor_names))

    # ── Two-stage supercharge: CG retrieval → ranker rerank ──────────────────
    ranked = st.session_state.get('ex_ranked', False)
    if ranker is not None and st.session_state.get('ex_cg_iids'):
        n_cand = len(st.session_state['ex_cg_iids'])
        label = ("↺  Back to CG-only recommendations" if ranked
                 else f"⚡  Apply Ranker  ·  rerank all {n_cand} candidates")
        if st.button(label, use_container_width=True, key='ex_supercharge'):
            if not ranked:
                with st.spinner("Reranking with the Wide & Deep ranker…"):
                    st.session_state['ex_ranker_iids'] = _ranker_rerank(
                        ranker, fs, liked_iids, anchor_iids, st.session_state['ex_cg_iids'])
                st.session_state['ex_ranked']   = True
                st.session_state['ex_cmp_page'] = 0
            else:
                st.session_state['ex_ranked'] = False
            st.rerun()

    if st.session_state.get('ex_ranked') and st.session_state.get('ex_ranker_iids'):
        n_cand = len(st.session_state['ex_cg_iids'])
        st.caption(f"Candidate generation retrieves the top {n_cand} by using an ultra-fast "
                   f"two-tower retrieval model; the ranker reorders those top {n_cand} CG "
                   f"candidates. Badges show each game's rank movement.")
        st.divider()
        _show_comparison(st.session_state['ex_cg_iids'],
                         st.session_state['ex_ranker_iids'], fs, 'ex_cmp',
                         ranker_alpha=getattr(ranker, 'serving_alpha', 0.0))
    else:
        _show_results('examples')


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    col, _ = st.columns([1, 1])
    with col:
        st.header("What is this?")
        st.markdown(
            "A PyTorch two-tower neural network trained on the "
            "[UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html) "
            "(~5,437 games, ~4.3M training examples)."
        )
        st.markdown(
            "Trained with full softmax cross-entropy over the entire game corpus, following the YouTube DNN retrieval "
            "approach (Covington et al., 2016)."
        )
        st.markdown(
            "At inference, a dot product of the user and item embeddings retrieves the most relevant games."
        )
        st.markdown(
            "This is a **two-stage** system: the two-tower model is the fast **retrieval** stage, and a "
            "**Wide & Deep ranker** reranks its top 100 candidates using richer user × item cross-features. "
            "Try it on the **Recommend** tab — results appear from retrieval first, then the **⚡ Apply Ranker** "
            "button reranks them side-by-side."
        )

        st.subheader("The core design choice: no user ID")
        st.markdown("Most recommender systems embed a unique ID for every user in the training set.")
        st.markdown("This works, but has a fundamental limitation: **inference is only possible for users the model has already seen.**")
        st.markdown("If a new user signs up, you have no embedding for them. Your options are:")
        st.markdown("""
- Retrain the entire model
- Partially fine-tune the new user in with a few gradient steps
- Find an existing user who seems similar and use their embedding as a proxy
""")
        st.markdown("This model takes a different approach. **There is no user ID embedding.**")
        st.markdown("Instead, every user is represented as a function of their taste signals — play history, genre affinity, and tag affinity.")
        st.markdown("The model learns to embed *features of the user*, not the user themselves.")
        st.markdown(
            "This means the model can generate recommendations for **any user** as long as you can provide "
            "even a small amount of signal: a few games they've played, some tags they like."
        )
        st.markdown("No retraining required. No cold-start problem at the user level. The same trained model works for users who never existed when the model was trained.")

    col, _ = st.columns([1, 1])
    with col:
        st.header("Stage 1: Two-Tower Retrieval")
        st.markdown(
            "The first stage is a two-tower model — a **user tower** and an **item tower** that each project into the "
            "same 128-dim space. Scoring a game is a dot product of the two embeddings, so all ~5,437 games can be "
            "ranked in a single matrix multiply. This is the fast, recall-maximizing retrieval stage."
        )

        st.subheader("User Tower")
        st.markdown(
            "Six sub-embeddings are concatenated (192-dim), then passed through a projection MLP → **128-dim**."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| liked_pool | Sum of 32-dim item ID embeddings for games you loved (recommend=True, high playtime) | Positive taste signal — what you actively seek out |
| disliked_pool | Sum of 32-dim item ID embeddings for games you bounced off (recommend=False, very low playtime) | Negative taste signal — what to avoid |
| full_pool | Sum of 32-dim item ID embeddings for all history (equal-weight) | Broad collaborative fingerprint of your overall library |
| playtime_pool | Sum of 32-dim item ID embeddings weighted by normalized log-playtime | Engagement intensity — games you sank hundreds of hours into dominate |
| user_genre_tower | Debiased avg log-playtime per genre + genre play fraction | Genre affinity — how strongly you lean toward each broad category |
| user_tag_tower | Sum of TF-IDF tag vectors from play history | Tag affinity — granular community descriptors like "Open World", "Rogue-like", "Dark Souls-like" |
""", unsafe_allow_html=True)

        st.markdown(
            "**In-model context:** Genre and tag affinity are computed inside the model's forward pass from "
            "`game_genre_matrix` and `game_tag_matrix` registered buffers — not pre-computed in the dataset. "
            "This means recommendations can be generated for any play history at inference time with no preprocessing."
        )

        st.subheader("Item Tower")
        st.markdown(
            "Six sub-embeddings are concatenated (96-dim), then passed through a projection MLP → **128-dim**."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| item_embedding_tower | Game ID (shared lookup with all four user history pools) | Collaborative identity — a learned fingerprint for each game based on who plays it together |
| item_genre_tower | Uniform-weighted genre vector | Broad genre positioning |
| item_tag_tower | TF-IDF Steam tag scores (164 tags) | Community descriptors — granular signals like "Open World", "Rogue-like", "Dark Souls-like" |
| developer_tower | Primary developer index | Developer identity — clusters games by studio |
| year_embedding_tower | Steam release year | Era — captures generational shifts in game design |
| price_embedding_tower | Price bucket (Free / <$5 / … / >$60) | Price tier — free-to-play vs. indie vs. AAA is a meaningful taste dimension |
""", unsafe_allow_html=True)

        st.subheader("Projection MLP")
        st.markdown("""
Concatenating sub-embeddings and feeding them directly into a dot product only learns **additive combinations**
of the individual signals — it cannot model interactions between them.

Each tower ends with a 2-layer projection MLP (`concat → Linear(256) → ReLU → Linear(128)`) before the dot product.
This lets the model learn cross-feature interactions, such as:
- Genre × developer (e.g. JRPGs from Japanese studios vs. Western action-RPGs)
- Liked vs. disliked signals interacting with content tags
- Tag cluster × release era (roguelikes from 2012 vs. 2020 are different products)

Both towers project to the same 128-dim output space — only this final dim needs to match.
The internal concat sizes (192 user, 96 item) are independent of each other.

Each tower's 128-dim output is L2-normalized, so the final dot product is a cosine similarity.
""")

        st.subheader("Shared Embeddings")
        st.markdown("""
**item_embedding_lookup** (32-dim) — shared between all four user history pools and the item tower.

The user pools sum raw 32-dim ID embeddings directly (shallow pooling). The item tower additionally
passes the same embedding through a small linear layer before concatenating with other item features.

This means a game you played appears in the same embedding space as the game being scored — the model
learns to align user taste with item identity through training.
""")

        st.subheader("Training")
        st.markdown("""
- **Dataset:** UCSD Steam — 88k Australian users, ~5,437 corpus games (≥10 users with ≥6 min playtime; ultra-popular Valve titles excluded)
- **Corpus filtering:** Games with fewer than 10 qualifying users excluded. Users with fewer than 5 or more than 10,000 total hours excluded. Users with fewer than 2 corpus games excluded.
- **Playtime signal:** `log(1 + hours)` — used to classify history into Liked/Disliked pools. Never a prediction target.
- **Loss:** Full softmax cross-entropy over the entire ~5,437-game corpus every step
- **Optimizer:** Adam, lr=0.001, eps=1e-6, CosineAnnealingLR (eta_min=1e-4)
- **Popularity bias:** alpha × log1p(count) **added** at training (Menon et al. 2021, Path 2); raw dot products at inference — no correction needed. The standalone CG uses alpha=0.4; the **deployed retrieval stage uses alpha=0** (recall-maximizing — the ranker does the final ranking, see Stage 2)
- **Gradient clipping:** max_norm=1.0
- **Batch size:** 512, temperature=0.000977 (= 0.5 / 512) for the full-softmax retrieval model
- **Steps:** 50,000
- **Training examples:** Rollback construction with 3× shuffle augmentation → ~4.3M examples (55k train users)
""")

        st.subheader("Offline Evaluation")
        st.markdown(
            "Evaluated on a **sample of 2,000 users** drawn from the held-out validation set (10% of all users, "
            "never seen during training). Each example has one target; Recall@K = Hit Rate@K for single-target eval. "
            "Shuffled history — no release-date ordering."
        )
        st.markdown(
            "**Which checkpoint these tables describe.** The `alpha` popularity-bias knob produces two CG variants "
            "(same architecture, different training-time bias). The tables below are the **standalone CG (alpha=0.4)** — "
            "it trades raw recall for cleaner niche-taste lists. The **deployed retrieval stage uses alpha=0**, which is "
            "recall-maximizing and therefore scores *higher* offline (NDCG@10 0.0752 vs 0.0645); that is the baseline the "
            "ranker improves on in Stage 2 below."
        )

        st.markdown("**V5 PROD** — corpus: 5,437 games (Valve titles removed, no LayerNorm, correct Menon Path 2)")
        st.markdown("""
| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0226 | 0.0226 |
| 5 | 0.0741 | 0.0481 |
| 10 | 0.1253 | 0.0645 |
| 20 | 0.2059 | 0.0848 |
| 50 | 0.3673 | 0.1166 |

MRR: **0.0611** (random: 0.0017, +36×)
""")

        st.markdown("**V4** — corpus: 5,437 games (Valve titles removed, no LayerNorm, incorrect Menon sign)")
        st.markdown("""
| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0294 | 0.0294 |
| 5 | 0.0882 | 0.0589 |
| 10 | 0.1430 | 0.0765 |
| 20 | 0.2280 | 0.0978 |
| 50 | 0.3913 | 0.1300 |

MRR: **0.0715** (random: 0.0017, +42×)
""")

        st.markdown("**V3** — corpus: 5,437 games (Valve titles removed, with LayerNorm after pools)")
        st.markdown("""
| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0278 | 0.0278 |
| 5 | 0.0882 | 0.0581 |
| 10 | 0.1428 | 0.0756 |
| 20 | 0.2287 | 0.0971 |
| 50 | 0.3944 | 0.1299 |

MRR: **0.0706** (random: 0.0017, +41×)
""")

        st.markdown("**V2** — corpus: 5,442 games (Valve titles included)")
        st.markdown("""
| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0389 | 0.0389 |
| 5 | 0.1138 | 0.0767 |
| 10 | 0.1743 | 0.0962 |
| 20 | 0.2602 | 0.1177 |
| 50 | 0.4256 | 0.1504 |

MRR: **0.0875** (random: 0.0017, +51×)
""")

        st.markdown(
            "**Why V5 metrics are lower than V4:** V4 was trained with the popularity bias *subtracted*, which caused "
            "the model to compensate by pushing popular item embeddings closer to all user embeddings — inflating Recall@K "
            "for popular targets."
        )
        st.markdown(
            "V5 uses the correct Menon Path 2 formula (bias *added* at training, raw dot products at "
            "inference), producing genuinely preference-driven rankings with cleaner per-genre quality."
        )
        st.markdown(
            "**Why V3/V4 metrics are lower than V2:** Ultra-popular Valve games (CS:GO, Garry's Mod, Left 4 Dead 2) "
            "were trivially easy prediction targets — removing them makes every target require genuine taste modeling."
        )

        st.header("Stage 2: Wide & Deep Ranker")
        st.markdown(
            "Everything above is the **retrieval** stage. It is fast — it can score all ~5,437 games in a single "
            "matrix multiply — but it has a structural limit: it only ever compares a user and an item through a "
            "**dot product of two independent embeddings**."
        )
        st.markdown(
            "It never sees the user and a specific candidate *together*, so it can't reason about signals like "
            "\"how many of this candidate's genres are in your history\" or \"is this priced like the games you "
            "usually play.\""
        )
        st.markdown(
            "A second-stage **ranker** adds exactly those signals. It takes the retrieval model's **top 100** and "
            "reorders them:"
        )
        st.markdown("""
| Stage | Model | Job |
|---|---|---|
| 1 — Retrieval | Two-tower (above) | Score all games, keep the top 100 |
| 2 — Ranking | Wide & Deep | Rerank those 100 with user × item cross-features |
""", unsafe_allow_html=True)

        st.subheader("Why Wide & Deep")
        st.markdown("""
- **Deep side** — mirrors the two-tower model exactly (same per-feature towers, same dimensions) and is
  warm-started from the trained retrieval weights. All sub-embeddings concatenate (288-dim) into an MLP, carrying
  over everything the retrieval model already learned.
- **Wide side** — hand-crafted **cross-features** that bypass the MLP and connect straight to the output, each with
  its own learned weight. A single scalar like "genre overlap" would be drowned out among ~290 deep dimensions, so
  it gets a direct path instead.
""")
        st.markdown("The cross-features are what a dot product structurally cannot compute:")
        st.markdown("""
| Group | What it measures |
|---|---|
| Categorical overlap | Genre / tag / developer overlap between the candidate and your history (full, liked, and most-recent-3 slices) |
| Numeric matching | Price gap, release-era gap, playtime calibration, popularity and sentiment match between your averages and the candidate |
| Niche / rarity | IDF-weighted tag overlap + rarest-tag / studio-scale matching — suppresses popular cross-genre titles leaking into niche-taste lists |
""", unsafe_allow_html=True)

        st.markdown(
            "The ranker is trained with the same sampled-softmax cross-entropy loss as the retrieval model "
            "(1 positive + 999 random negatives), warm-started from it, then learns the cross-feature weights on top. "
            "Because it only reranks the top 100, its hit-rate ceiling is the retrieval model's Recall@100."
        )

        st.subheader("What reranking buys")
        st.markdown(
            "Same held-out users, both capped at the retrieval ceiling. *Retrieval* here is the deployed raw "
            "alpha=0 CG — neither production stage applies a popularity penalty:"
        )
        st.markdown("""
| Metric | Retrieval (raw alpha=0 CG) | + Ranker | Lift |
|---|---|---|---|
| NDCG@10 | 0.0752 | **0.0867** | +15% |
| Hit@10 | 0.1430 | **0.1625** | +14% |
| MRR | 0.0726 | **0.0813** | +12% |
""", unsafe_allow_html=True)
        st.markdown(
            "On the pure-reranking subset (where retrieval surfaced the target), NDCG@10 rises 0.1361 → 0.1569 (+15%). "
            "Beyond the metrics, the ranker visibly cleans up niche-taste lists — for example, removing JRPGs that "
            "leak into a fighting-game query. See it live on the **Recommend** tab."
        )

        st.header("Limitations")
        st.markdown("""
- ~5,437-game corpus — games with fewer than 10 qualifying users are excluded
- No timestamps — play history is shuffled randomly (no real play sequence in source data)
- Witcher 3, Dark Souls III, Skyrim, Civilization VI and other major titles are absent from this version of the dataset's metadata
- Free-to-play games (Dota 2, TF2) are also missing from the metadata file
""")


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Steam Game Recommender", layout="wide")
st.markdown("""
    <style>
    div[data-testid="stTabs"] > div:first-child {
        overflow-x: auto;
        white-space: nowrap;
        flex-wrap: nowrap;
    }
    .main .block-container { overflow-x: hidden; max-width: 100%; }
    div[data-testid="stDataFrame"] { overflow-x: auto; max-width: 100%; }
    div[data-testid="stCaptionContainer"] p {
        word-break: break-word;
        white-space: normal;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Steam Game Recommender")
model, fs, be, all_ids, all_embs, all_norm, all_norm_genre, all_norm_tag, ranker = load_artifacts()

st.markdown(
    "<small>Two-Tower retrieval + Wide &amp; Deep ranking neural networks · Built with "
    "<a href='https://cseweb.ucsd.edu/~jmcauley/datasets.html' target='_blank'>UCSD Steam dataset</a>"
    " and <a href='https://pytorch.org' target='_blank'>PyTorch</a><br>"
    "Code: <a href='https://github.com/nickgreenquist/Game-Recommender-System-PyTorch-TwoTower-Model'"
    " target='_blank'>GitHub</a></small>",
    unsafe_allow_html=True,
)

recommend_tab, examples_tab, similar_tab, genres_tab, tags_tab, about_tab = st.tabs(
    ["Recommend", "Examples", "Similar", "Genres", "Tags", "About"]
)

with recommend_tab:
    tab_recommend(model, fs, all_ids, all_embs, ranker)

with examples_tab:
    tab_examples(model, fs, all_ids, all_embs, ranker)

with similar_tab:
    tab_similar(be, fs, all_ids, all_norm)

with genres_tab:
    tab_explore_genres(model, be, fs, all_ids, all_norm_genre)

with tags_tab:
    tab_explore_tags(model, be, fs, all_ids, all_norm_tag)

with about_tab:
    tab_about()
