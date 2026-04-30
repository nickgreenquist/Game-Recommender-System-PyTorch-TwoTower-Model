"""
Steam Game Recommender — Streamlit app.

Run:      streamlit run streamlit_app.py
Requires: serving/model.pth
          serving/game_embeddings.pt
          serving/feature_store.pt

Generate serving/ with: python main.py export
"""
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from src.evaluate import (
    USER_TYPE_TO_FAVORITE_GAMES,
    USER_TYPE_TO_TAGS,
    SIMULATED_FAV_LOG_HOURS,
    SIMULATED_ANCHOR_LOG_HOURS,
    POPULARITY_ALPHA_INFERENCE_MULTIPLE,
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


def _score_games(user_emb: torch.Tensor, all_ids: list, all_embs: torch.Tensor,
                 fs: dict, exclude_iids: set, top_n: int = 20,
                 mark_iids: set = None):
    """Dot-product rank all corpus games with popularity bias; exclude seeds; return top-n DataFrame.
    mark_iids: item IDs to include but label '  ◀ seed' in the Title column.
    """
    import pandas as pd
    mark_iids  = mark_iids or set()
    raw_scores = (all_embs @ user_emb.T).squeeze(-1)

    # Apply popularity bias — same formula as training and canary inference:
    # training: (u·v)/temp - bias  →  inference ranking: u·v - temp*bias
    cfg   = fs.get('model_config', {})
    alpha = cfg.get('popularity_alpha', 0.0)
    if alpha > 0 and 'game_interaction_counts' in fs:
        counts = fs['game_interaction_counts']
        if isinstance(counts, np.ndarray):
            counts = torch.from_numpy(counts)
        temperature = 0.5 / cfg.get('minibatch_size', 512)

        raw_scores  = raw_scores - temperature * (alpha * POPULARITY_ALPHA_INFERENCE_MULTIPLE) * torch.log1p(counts.float())
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
            user_emb = _build_user_embedding(model, fs, liked_iids, anchor_iids)

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

    profiles = list(USER_TYPE_TO_FAVORITE_GAMES.keys())
    selected = st.selectbox(
        "Profile",
        options=[None] + profiles,
        format_func=lambda x: "Choose a profile..." if x is None else x,
        label_visibility="collapsed",
    )

    if not selected:
        return

    fav_titles = USER_TYPE_TO_FAVORITE_GAMES.get(selected, [])
    tag_names  = USER_TYPE_TO_TAGS.get(selected, [])

    title_to_iid = fs['title_to_item_id']

    missing = [t for t in fav_titles if t not in title_to_iid]
    if missing:
        st.warning("Not found in corpus (check title format): " + ", ".join(missing))

    liked_iids  = [title_to_iid[t] for t in fav_titles if t in title_to_iid]
    anchor_iids = _get_tag_anchors(fs, tag_names, exclude=set(fav_titles))

    with torch.no_grad():
        user_emb = _build_user_embedding(model, fs, liked_iids, anchor_iids, anchor_in_liked=False)

    df = _score_games(user_emb, all_ids, all_embs, fs,
                      exclude_iids=set(liked_iids),
                      mark_iids=set(anchor_iids))

    st.subheader(f"Recommendations for: {selected}")
    if fav_titles:
        st.caption("Because you like: " + ", ".join(fav_titles))
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
            "(~5,437 games, ~4.3M training examples)."
        )
        st.markdown(
            "Trained with full softmax cross-entropy over the entire game corpus, following the YouTube DNN retrieval "
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

        st.header("Item Tower")
        st.markdown(
            "Six sub-embeddings are concatenated (96-dim), then passed through a projection MLP → **128-dim**."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| item_embedding_tower | Game ID (shared lookup with all three user history pools) | Collaborative identity — a learned fingerprint for each game based on who plays it together |
| item_genre_tower | Uniform-weighted genre vector | Broad genre positioning |
| item_tag_tower | TF-IDF Steam tag scores (164 tags) | Community descriptors — granular signals like "Open World", "Rogue-like", "Dark Souls-like" |
| developer_tower | Primary developer index | Developer identity — clusters games by studio |
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
- Liked vs. disliked signals interacting with content tags
- Tag cluster × release era (roguelikes from 2012 vs. 2020 are different products)

Both towers project to the same 128-dim output space — only this final dim needs to match.
The internal concat sizes (160 user, 96 item) are independent of each other.
""")

        st.header("Shared Embeddings")
        st.markdown("""
**item_embedding_lookup** (32-dim) — shared between all four user history pools and the item tower.

The user pools sum raw 32-dim ID embeddings directly (shallow pooling). The item tower additionally
passes the same embedding through a small linear layer before concatenating with other item features.

This means a game you played appears in the same embedding space as the game being scored — the model
learns to align user taste with item identity through training.
""")

        st.header("Training")
        st.markdown("""
- **Dataset:** UCSD Steam — 88k Australian users, ~5,437 corpus games (≥10 users with ≥6 min playtime; ultra-popular Valve titles excluded)
- **Corpus filtering:** Games with fewer than 10 qualifying users excluded. Users with fewer than 5 or more than 10,000 total hours excluded. Users with fewer than 2 corpus games excluded.
- **Playtime signal:** `log(1 + hours)` — used to classify history into Liked/Disliked pools and build genre context. Never a prediction target.
- **Loss:** Full softmax cross-entropy over the entire ~5,437-game corpus every step
- **Optimizer:** Adam, lr=0.001, eps=1e-6, CosineAnnealingLR (eta_min=1e-4)
- **Popularity bias:** alpha=0.4 × log1p(count) at training; 2× multiplier applied at inference
- **Gradient clipping:** max_norm=1.0
- **Batch size:** 512, temperature=0.000977
- **Steps:** 50,000
- **Training examples:** Rollback construction with 3× shuffle augmentation → ~4.3M examples (55k train users)
""")

        st.header("Offline Evaluation")
        st.markdown(
            "Evaluated on **2,000 held-out val users** (never seen during training). "
            "Each example has one target; Recall@K = Hit Rate@K for single-target eval. "
            "Shuffled history — no release-date ordering."
        )

        st.markdown("**V3 PROD** — corpus: 5,437 games (ultra-popular Valve titles removed)")
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

        st.markdown("**V2 PROD** — corpus: 5,442 games (Valve titles included)")
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
            "**Why V3 metrics are lower:** These numbers are not directly comparable. "
            "Ultra-popular Valve games (CS:GO, Garry's Mod, Left 4 Dead 2) appeared in nearly every val user's history "
            "and were trivially easy prediction targets — any model ranks them top-5 for most users, inflating V2 Recall@K. "
            "Removing them from the corpus makes the eval strictly harder: every remaining target requires genuine taste modeling. "
            "V3 canary quality is substantially better — cross-genre Valve recommendations are eliminated and per-genre coherence "
            "improved across all nine user types tested."
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
