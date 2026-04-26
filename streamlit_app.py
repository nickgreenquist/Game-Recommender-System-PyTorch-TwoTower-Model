"""
Steam Game Recommender — Streamlit app.

Run:      streamlit run streamlit_app.py
Requires: serving/model.pth
          serving/game_embeddings.pt
          serving/feature_store.pt

Generate serving/ with: python main.py export
"""
import importlib

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

import src.evaluate
importlib.reload(src.evaluate)
from src.evaluate import (
    USER_TYPE_TO_FAVORITE_GENRES,
    USER_TYPE_TO_FAVORITE_GAMES,
    USER_TYPE_TO_TAGS,
    SIMULATED_FAV_LOG_HOURS,
    SIMULATED_ANCHOR_LOG_HOURS,
)

from src.model import GameRecommender

_FAV_WEIGHT      = 10.0   # simulated log-hours weight for explicitly liked games
_ANCHOR_WEIGHT   =  2.0   # simulated log-hours weight for tag-anchor games
_ANCHORS_PER_TAG =  5
_COVER_ROW_HEIGHT = 160   # px — Steam header.jpg is landscape (460×215)

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

    genre_mat      = torch.from_numpy(np.array(fs['game_genre_matrix'], dtype=np.float32))
    hist_genre_buf = torch.cat([genre_mat, torch.zeros(1, genre_mat.shape[1])], dim=0)
    year_arr       = torch.from_numpy(np.array(fs['game_year_idx'], dtype=np.int64))
    hist_year_buf  = torch.cat([year_arr, torch.zeros(1, dtype=torch.long)], dim=0)
    price_arr      = torch.from_numpy(np.array(fs['game_price_bucket'], dtype=np.int64))
    hist_price_buf = torch.cat([price_arr, torch.zeros(1, dtype=torch.long)], dim=0)

    model = GameRecommender(
        n_genres=fs['n_genres'],
        n_tags=fs['n_tags'],
        n_games=fs['n_items'],
        n_years=fs['n_years'],
        n_developers=fs['n_developers'],
        n_price_buckets=fs['n_price_buckets'],
        user_context_size=2 * fs['n_genres'],
        game_tag_matrix=fs['game_tag_matrix'],
        game_dev_idx=fs['game_dev_idx'],
        hist_genre_buf=hist_genre_buf,
        hist_year_buf=hist_year_buf,
        hist_price_buf=hist_price_buf,
        item_id_embedding_size=cfg['item_id_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        tag_embedding_size=cfg['tag_embedding_size'],
        developer_embedding_size=cfg['developer_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        price_embedding_size=cfg['price_embedding_size'],
        proj_hidden=cfg.get('proj_hidden', 256),
        output_dim=cfg.get('output_dim', 128),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_ids   = list(be.keys())
    all_embs  = torch.cat([be[iid]['GAME_EMBEDDING_COMBINED'] for iid in all_ids], dim=0)
    all_norm  = F.normalize(all_embs, dim=1)

    all_genre_embs = torch.cat([be[iid]['GAME_GENRE_EMBEDDING'] for iid in all_ids], dim=0)
    all_norm_genre = F.normalize(all_genre_embs, dim=1)

    all_tag_embs   = torch.cat([be[iid]['GAME_TAG_EMBEDDING']   for iid in all_ids], dim=0)
    all_norm_tag   = F.normalize(all_tag_embs, dim=1)

    return model, fs, be, all_ids, all_embs, all_norm, all_norm_genre, all_norm_tag


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


def _render_results(df, extra_col: str = None) -> None:
    cols = ['Cover', 'Title', 'Developer', 'Year', 'Genres', 'Top Tags']
    if extra_col and extra_col in df.columns:
        cols.append(extra_col)
    st.dataframe(
        df[cols],
        use_container_width=True,
        hide_index=True,
        row_height=_COVER_ROW_HEIGHT,
        height=_COVER_ROW_HEIGHT * min(len(df), 20) + 38,
        column_config={
            'Cover': st.column_config.ImageColumn('Cover', width='large'),
        },
    )


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
                          liked_genres: list) -> torch.Tensor:
    """
    Build a user embedding from liked game IDs + tag anchors + explicit genres.
    Mirrors _build_user_embedding in evaluate.py exactly — no timestamp tower.
    """
    genre_matrix = fs['game_genre_matrix']    # numpy (n_items, n_genres)
    item_to_idx  = fs['item_to_idx']
    genre_to_i   = fs['genre_to_i']
    n_genres     = fs['n_genres']

    weighted = (
        [(iid, _FAV_WEIGHT)    for iid in liked_iids]  +
        [(iid, _ANCHOR_WEIGHT) for iid in anchor_iids]
    )
    history = [(item_to_idx[iid], w) for iid, w in weighted if iid in item_to_idx]

    # Genre context — mirrors dataset.py _build_rollback_dataset
    avg_log       = sum(w for _, w in history) / max(len(history), 1)
    running_count = np.zeros(n_genres, dtype=np.float32)
    running_sum   = np.zeros(n_genres, dtype=np.float32)
    for item_idx, raw_log in history:
        for g_idx in np.where(genre_matrix[item_idx] > 0)[0]:
            running_count[g_idx] += 1
            running_sum[g_idx]   += raw_log

    # Explicit genre boost for selected genres not covered by game history
    for g in liked_genres:
        if g in genre_to_i:
            g_idx = genre_to_i[g]
            if running_count[g_idx] == 0:
                running_count[g_idx] = 1.0
                running_sum[g_idx]   = avg_log + _FAV_WEIGHT

    genre_ctx    = np.zeros(2 * n_genres, dtype=np.float32)
    total_assign = running_count.sum()
    if total_assign > 0:
        mask = running_count > 0
        genre_ctx[:n_genres][mask] = (running_sum[mask] / running_count[mask]) - avg_log
        genre_ctx[n_genres:]       = running_count / total_assign

    X_genre = torch.tensor([genre_ctx.tolist()], dtype=torch.float32)
    if history:
        hist_idxs = torch.tensor([[h[0] for h in history]], dtype=torch.long)
        hist_wts  = torch.tensor([[h[1] for h in history]], dtype=torch.float32)
    else:
        hist_idxs = torch.tensor([[model.game_pad_idx]], dtype=torch.long)
        hist_wts  = torch.zeros(1, 1, dtype=torch.float32)
    return model.user_embedding(X_genre, hist_idxs, hist_wts)


def _score_games(user_emb: torch.Tensor, all_ids: list, all_embs: torch.Tensor,
                 fs: dict, exclude_iids: set, top_n: int = 20,
                 mark_iids: set = None):
    """Raw dot-product rank all corpus games; exclude seeds; return top-n DataFrame.
    mark_iids: item IDs to include but label '  ◀ seed' in the Title column.
    """
    import pandas as pd
    mark_iids  = mark_iids or set()
    raw_scores = (all_embs @ user_emb.T).squeeze(-1)
    rows = []
    for idx in raw_scores.argsort(descending=True).tolist():
        iid = all_ids[idx]
        if iid in exclude_iids:
            continue
        row = _game_meta(iid, fs)
        if iid in mark_iids:
            row['Title'] += '  ◀ seed'
        rows.append(row)
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(model, fs, all_ids, all_embs):
    st.caption(
        "Select games you've enjoyed and the model will infer your taste from your "
        "play history. The more games you add, the sharper the recommendations."
    )

    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_tags'):
            st.session_state[key] = []

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
            if anchor_iids:
                anchor_names = [fs['item_id_to_title'][iid] for iid in anchor_iids]
                st.caption("Tag anchors: " + " · ".join(anchor_names[:12]))

        if not liked_iids and not anchor_iids:
            st.warning("Select at least one game or tag.")
            return

        with torch.no_grad():
            user_emb = _build_user_embedding(model, fs, liked_iids, anchor_iids, [])

        exclude_iids = set(liked_iids) | set(anchor_iids)
        df = _score_games(user_emb, all_ids, all_embs, fs, exclude_iids)
        _render_results(df)


# ── Tab: Similar ──────────────────────────────────────────────────────────────

def tab_similar(be, fs, all_ids, all_norm):
    import pandas as pd
    st.caption(
        "Each game is represented by a single combined embedding — the concatenation of "
        "its genre, tag, game-ID, developer, year, and price towers. "
        "This tab ranks all games by cosine similarity to your selected seed."
    )
    all_titles  = fs['popularity_ordered_titles']
    selections  = st.multiselect(
        "Game(s) to find similar titles for",
        all_titles, key='sim_title', max_selections=10,
    )

    if st.button("Find Similar Games"):
        if not selections:
            st.warning("Select at least one game.")
            return

        title_to_iid = fs['title_to_item_id']
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
                row = _game_meta(candidate, fs)
                row['Score'] = f"{sims[idx].item():.3f}"
                rows.append(row)
                if len(rows) >= 20:
                    break

            st.subheader(f"Similar to: {title}")
            _render_results(pd.DataFrame(rows), extra_col='Score')


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
            return

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
            row = _game_meta(iid, fs)
            row['Score'] = f"{sims[idx].item():.4f}"
            rows.append(row)
            if len(rows) >= 20:
                break

        _render_results(pd.DataFrame(rows), extra_col='Score')


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
            return

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
            return

        anchor_iids = [iid for _, iid, _ in anchor_tag_triples]
        anchor_set  = set(anchor_iids)

        with torch.no_grad():
            query_emb  = torch.stack([
                be[iid]['GAME_TAG_EMBEDDING'].view(-1) for iid in anchor_iids
            ]).mean(dim=0)
            query_norm = F.normalize(query_emb.unsqueeze(0), dim=1)
            sims       = (all_norm_tag @ query_norm.T).squeeze(-1)

        st.caption(
            "Tag anchors — "
            + " · ".join(f"{tag}: {title}" for tag, _, title in anchor_tag_triples)
        )

        rows = []
        for idx in sims.argsort(descending=True).tolist():
            iid = all_ids[idx]
            row = _game_meta(iid, fs)
            if iid in anchor_set:
                row['Title'] = row['Title'] + '  ◀ anchor'
            row['Score'] = f"{sims[idx].item():.4f}"
            rows.append(row)
            if len(rows) >= 20:
                break

        _render_results(pd.DataFrame(rows), extra_col='Score')


# ── Tab: Examples (canary profiles) ──────────────────────────────────────────

def tab_examples(model, fs, all_ids, all_embs):
    st.caption("Select a pre-built user profile to see what the model recommends for that taste.")

    profiles = list(USER_TYPE_TO_FAVORITE_GENRES.keys())
    selected = st.selectbox(
        "Profile",
        options=[None] + profiles,
        format_func=lambda x: "Choose a profile..." if x is None else x,
        label_visibility="collapsed",
    )

    if not selected:
        return

    fav_titles = USER_TYPE_TO_FAVORITE_GAMES.get(selected, [])
    fav_genres = USER_TYPE_TO_FAVORITE_GENRES.get(selected, [])
    tag_names  = USER_TYPE_TO_TAGS.get(selected, [])

    title_to_iid = fs['title_to_item_id']

    missing = [t for t in fav_titles if t not in title_to_iid]
    if missing:
        st.warning("Not found in corpus (check title format): " + ", ".join(missing))

    liked_iids  = [title_to_iid[t] for t in fav_titles if t in title_to_iid]
    anchor_iids = _get_tag_anchors(fs, tag_names, exclude=set(fav_titles))

    with torch.no_grad():
        user_emb = _build_user_embedding(model, fs, liked_iids, anchor_iids, fav_genres)

    df = _score_games(user_emb, all_ids, all_embs, fs,
                      exclude_iids=set(liked_iids),
                      mark_iids=set(anchor_iids))

    st.subheader(f"Recommendations for: {selected}")
    if fav_titles:
        st.caption("Because you like: " + ", ".join(fav_titles))
    if fav_genres:
        st.caption("Favorite genres: " + ", ".join(fav_genres))
    if anchor_iids:
        anchor_names = [fs['item_id_to_title'][iid] for iid in anchor_iids]
        st.caption("Tag anchors: " + ", ".join(anchor_names))
    _render_results(df)


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    col, _ = st.columns([1, 1])
    with col:
        st.header("What is this?")
        st.markdown(
            "A PyTorch two-tower neural network trained on the "
            "[UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html) "
            "(~6,200 games, ~1.9M training examples)."
        )
        st.markdown(
            "Trained with in-batch negatives softmax loss, following the YouTube DNN retrieval "
            "approach (Covington et al., 2016)."
        )
        st.markdown(
            "At inference, a dot product of the user and item embeddings retrieves the most relevant games."
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
        st.markdown("Instead, every user is represented as a function of their taste signals — play history and genre affinity.")
        st.markdown("The model learns to embed *features of the user*, not the user themselves.")
        st.markdown(
            "This means the model can generate recommendations for **any user** as long as you can provide "
            "even a small amount of signal: a few games they've played, some tags they like."
        )
        st.markdown("No retraining required. No cold-start problem at the user level. The same trained model works for users who never existed when the model was trained.")

    col, _ = st.columns([1, 1])
    with col:
        st.header("User Tower")
        st.markdown(
            "Two sub-embeddings are concatenated (160-dim), then passed through a projection MLP → **128-dim**."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| Item-Pool History (ipool) | Play history — full 128-dim item embeddings weighted by log(1 + hours) | Collaborative taste — pools the complete item tower output (not just a raw ID lookup), capturing genre, tag, developer, era, and price signals from the user's history |
| user_genre_tower | Debiased avg log-playtime per genre + play fraction per genre | Genre affinity — how strongly you lean toward each broad genre category |
""", unsafe_allow_html=True)

        st.header("Item Tower")
        st.markdown(
            "Six sub-embeddings are concatenated (80-dim), then passed through a projection MLP → **128-dim**."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| item_embedding_tower | Game ID (shared lookup with user history pool) | Collaborative identity — a learned fingerprint for each game based on who plays it together |
| item_genre_tower | Uniform-weighted genre vector | Broad genre positioning |
| item_tag_tower | TF-IDF Steam tag scores (164 tags) | Community descriptors — granular signals like "Open World", "Rogue-like", "Dark Souls-like" |
| developer_tower | Primary developer index | Developer identity — clusters games by studio and stylistically similar developers |
| year_embedding_tower | Steam release year | Era — captures generational shifts in game design |
| price_embedding_tower | Price bucket (Free / <$5 / … / >$60) | Price tier — free-to-play vs. indie vs. AAA is a meaningful taste dimension |
""", unsafe_allow_html=True)

        st.header("Projection MLP")
        st.markdown("""
Concatenating sub-embeddings and feeding them directly into a dot product only learns **additive combinations**
of the individual signals — it cannot model interactions between them.

Each tower ends with a 2-layer projection MLP (`concat → Linear(256) → ReLU → Linear(128)`) before the dot product.
This lets the model learn cross-feature interactions, such as:
- Genre × developer (e.g. JRPGs from Japanese studios vs. Western action-RPGs)
- Price tier × history depth (price sensitivity varies by play style)
- Tag cluster × release era (roguelikes from 2012 vs. 2020 are different products)

Both towers project to the same 128-dim output space — only this final dim needs to match.
The internal concat sizes (160 user, 80 item) are independent of each other.
""")

        st.header("Shared Embeddings")
        st.markdown("""
**item_embedding_tower** — The full item tower (genre + tag + game ID + developer + year + price → projection MLP)
is shared between the target game *and* each game in the user's play history pool.

With ipool, the user history averages the **complete 128-dim item embedding** for each played game,
weighted by log(1 + hours). This gives the user tower access to all of a game's content signals —
not just its ID — when forming the user representation.

A game you played heavily pulls your user embedding directly toward that game's full embedding.
""")

        st.header("Training")
        st.markdown("""
- **Dataset:** UCSD Steam — 88k Australian users, ~5,400 corpus games (≥10 users with ≥6 min playtime)
- **Corpus filtering:** Games with fewer than 10 qualifying users excluded. Users with fewer than 5 or more than 10,000 total hours excluded. Users with fewer than 2 corpus games excluded.
- **Playtime signal:** `log(1 + hours)` — compresses the extreme tail while preserving ordering. Used as weighting in the history pool; never a prediction target.
- **Loss:** Cross-entropy over in-batch negatives (softmax) — each step produces a B×B score matrix; diagonal entries are the correct targets
- **Optimizer:** Adam, lr=0.001, weight_decay=1e-5, CosineAnnealingLR
- **Batch size:** 512 (511 in-batch negatives per example)
- **Steps:** 150,000
- **Training examples:** Rollback construction — for each play event, context = all prior plays sorted by Steam app ID (release-date proxy). Up to 50 examples per user (~1.9M train / 210k val)
""")

        st.header("Offline Evaluation")
        st.markdown(
            "Evaluated on **2,000 held-out users** (never seen during training). "
            "Corpus: 5,442 games. "
            "Each example has one target; Recall@K = Hit Rate@K for single-target eval."
        )
        st.markdown("""
| K | Recall@K | NDCG@K | vs. Random (HR@K) |
|---|---|---|---|
| 1 | 0.0902 | 0.0902 | 491× |
| 5 | 0.2614 | 0.1777 | 284× |
| 10 | 0.3794 | 0.2158 | 206× |
| 20 | 0.5182 | 0.2508 | 141× |
| 50 | 0.7165 | 0.2903 | 78× |

MRR: **0.1845** (random: 0.0017, +109×)
""")

        st.header("Limitations")
        st.markdown("""
- ~6,200-game corpus — games with fewer than 10 qualifying users are excluded
- No timestamps — items are ordered by Steam app ID as a release-date proxy, not actual play order
- Witcher 3, Dark Souls III, Skyrim, Civilization VI and other major titles are absent from this version of the dataset's metadata and cannot be recommended
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
model, fs, be, all_ids, all_embs, all_norm, all_norm_genre, all_norm_tag = load_artifacts()

st.markdown(
    "<small>Two-Tower neural network · Built with "
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
    tab_recommend(model, fs, all_ids, all_embs)

with examples_tab:
    tab_examples(model, fs, all_ids, all_embs)

with similar_tab:
    tab_similar(be, fs, all_ids, all_norm)

with genres_tab:
    tab_explore_genres(model, be, fs, all_ids, all_norm_genre)

with tags_tab:
    tab_explore_tags(model, be, fs, all_ids, all_norm_tag)

with about_tab:
    tab_about()
