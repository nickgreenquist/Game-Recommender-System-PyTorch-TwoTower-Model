"""
Stage 2 — Feature Engineering
Reads base_*.parquet, builds per-game and per-user feature vectors, saves features parquets.
Re-run this (not preprocess) when iterating on feature ideas.

Usage:
    python main.py features
"""
import math
import os
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


FEATURES_VERSION   = 'v1'
VAL_FRACTION       = 0.10   # fraction of users held out for eval
VAL_SPLIT_SEED     = 42
MAX_HISTORY_LEN    = 200    # cap per-user play history (sum-pool handles any length; this caps memory)
N_PRICE_BUCKETS    = 9      # fixed: Free <$5 $5-10 $10-20 $20-30 $30-40 $40-60 >$60 Unknown

# Bucket 5 — sentiment ordinal fallback for games without a known Steam sentiment string.
# Centred on "Mixed" so the Z-scored sentiment_match cross feature reads as "no signal"
# (close to corpus mean once the Z-score buffers are populated from training data).
SENTIMENT_UNKNOWN_FILL = 3.0


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_base(data_dir: str) -> dict:
    files = [
        ('games',        'base_games.parquet'),
        ('vocab',        'base_vocab.parquet'),
        ('interactions', 'base_interactions.parquet'),
        ('game_tags',    'base_game_tags.parquet'),
    ]
    result = {}
    for key, filename in files:
        print(f"  Loading {filename} ...")
        result[key] = pd.read_parquet(os.path.join(data_dir, filename))
    return result


def parse_vocab(vocab_df: pd.DataFrame) -> dict:
    g = vocab_df[vocab_df['type'] == 'genre'].sort_values('index')
    t = vocab_df[vocab_df['type'] == 'tag'].sort_values('index')
    y = vocab_df[vocab_df['type'] == 'year'].sort_values('index')
    d = vocab_df[vocab_df['type'] == 'developer'].sort_values('index')

    return {
        'genres_ordered':     g['value'].tolist(),
        'tags_ordered':       t['value'].tolist(),
        'years_ordered':      y['value'].tolist(),
        'developers_ordered': d['value'].tolist(),
        'genre_to_i':         dict(zip(g['value'], g['index'].astype(int))),
        'tag_to_i':           dict(zip(t['value'], t['index'].astype(int))),
        'year_to_i':          dict(zip(y['value'], y['index'].astype(int))),
        'developer_to_i':     dict(zip(d['value'], d['index'].astype(int))),
    }


# ── Per-game features ─────────────────────────────────────────────────────────

def build_game_features(base: dict, vocab: dict) -> pd.DataFrame:
    """
    One row per game:
      item_id, item_idx, genre_context, tag_context, developer_idx, year_idx, price_bucket, median_hours

    genre_context — float vector length n_genres, uniform weight across listed genres
    tag_context   — float vector length n_tags, TF-IDF scores from base_game_tags.parquet
    developer_idx — int, vocab index; 0 (__unknown__) if developer not in vocab
    year_idx      — int, vocab index; 0 if year not found
    price_bucket  — int, 0-8 (already computed in preprocess)
    median_hours  — float, global median playtime for the game
    """
    games_df   = base['games']
    tags_df    = base['game_tags']

    genre_to_i     = vocab['genre_to_i']
    tag_to_i       = vocab['tag_to_i']
    year_to_i      = vocab['year_to_i']
    developer_to_i = vocab['developer_to_i']
    n_genres = len(genre_to_i)
    n_tags   = len(tag_to_i)

    item_ids    = games_df['item_id'].tolist()
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    # Tag context lookup: item_id → dense float vector
    tag_lookup: dict = {}
    for _, row in tqdm(tags_df.iterrows(), total=len(tags_df), desc="Game tag contexts"):
        vec = [0.0] * n_tags
        for name, score in zip(row['tag_names'], row['scores']):
            if name in tag_to_i:
                vec[tag_to_i[name]] = float(score)
        tag_lookup[row['item_id']] = vec

    rows = []
    for _, grow in tqdm(games_df.iterrows(), total=len(games_df), desc="Game features"):
        iid = grow['item_id']

        # Genre context — uniform weight across the game's genres
        genres = list(grow['genres']) if grow['genres'] is not None else []
        genre_ctx = [0.0] * n_genres
        valid_genres = [g for g in genres if g in genre_to_i]
        if valid_genres:
            w = 1.0 / len(valid_genres)
            for g in valid_genres:
                genre_ctx[genre_to_i[g]] = w

        developer_idx = developer_to_i.get(grow['developer'], 0)
        year_idx      = year_to_i.get(str(grow['year']), 0)
        price_bucket  = int(grow['price_bucket'])
        median_hours  = float(grow['median_hours'])

        rows.append({
            'item_id':       iid,
            'item_idx':      item_to_idx[iid],
            'genre_context': genre_ctx,
            'tag_context':   tag_lookup.get(iid, [0.0] * n_tags),
            'developer_idx': developer_idx,
            'year_idx':      year_idx,
            'price_bucket':  price_bucket,
            'median_hours':  median_hours,
        })

    df = pd.DataFrame(rows)
    print(f"  Game features: {len(df)} games  (genres={n_genres}, tags={n_tags})")
    return df


# ── Per-user features ─────────────────────────────────────────────────────────

def build_user_features(base: dict, vocab: dict, item_to_idx: dict) -> pd.DataFrame:
    """
    One row per user:
      user_id, split, play_history, play_history_weights, avg_log_playtime

    split                — 'train' or 'val' (user-level, 90/10 by user)
    play_history         — list[int] item_idx values, capped to MAX_HISTORY_LEN
    play_history_weights — list[float] log(1+h) weights normalized per user
    avg_log_playtime     — float, mean log(1+h) for the user
    """
    interactions_df = base['interactions']

    # ── User-level train/val split ──
    all_users = interactions_df['user_id'].unique().tolist()
    rng = random.Random(VAL_SPLIT_SEED)
    rng.shuffle(all_users)
    n_val    = int(len(all_users) * VAL_FRACTION)
    val_set  = set(all_users[:n_val])
    train_set = set(all_users[n_val:])
    print(f"  Train users: {len(train_set):,}   Val users: {len(val_set):,}")

    # ── Per-user log-playtime avg ──
    interactions_df = interactions_df.copy()
    interactions_df['log_hours'] = np.log1p(interactions_df['hours'].values)
    user_avg_log = interactions_df.groupby('user_id')['log_hours'].mean().to_dict()

    # ── Per-user play history ──
    history_agg = (interactions_df
                   .groupby('user_id')
                   .agg(item_ids=('item_id', list), log_hours=('log_hours', list))
                   .reset_index())
    history_by_user = {r['user_id']: r for _, r in history_agg.iterrows()}

    rows = []
    for uid in tqdm(all_users, desc="User features"):
        split  = 'val' if uid in val_set else 'train'
        hrow   = history_by_user.get(uid)

        if hrow is not None:
            pairs = [
                (item_to_idx[iid], lh)
                for iid, lh in zip(hrow['item_ids'], hrow['log_hours'])
                if iid in item_to_idx
            ][-MAX_HISTORY_LEN:]
        else:
            pairs = []

        hist_idx  = [p[0] for p in pairs]
        log_hours = [p[1] for p in pairs]

        # Normalize log(1+h) weights per user
        total = sum(log_hours) or 1.0
        hist_weights = [lh / total for lh in log_hours]

        rows.append({
            'user_id':              uid,
            'split':                split,
            'avg_log_playtime':     float(user_avg_log.get(uid, 1.0)),
            'play_history':         hist_idx,
            'play_history_weights': hist_weights,
        })

    df = pd.DataFrame(rows)
    print(f"  User features: {len(df):,} users  "
          f"({df['split'].eq('train').sum():,} train, {df['split'].eq('val').sum():,} val)")
    return df


# ── Parquet writer (handles list columns) ─────────────────────────────────────

def _write_list_parquet(df: pd.DataFrame, path: str) -> None:
    arrays = {}
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        first  = sample[0] if isinstance(sample, list) and len(sample) > 0 else None
        if isinstance(sample, list) and isinstance(first, float):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
        elif isinstance(sample, list) and isinstance(first, int):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.int64()))
        elif isinstance(sample, list):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
        else:
            arrays[col] = pa.array(df[col].tolist())
    pq.write_table(pa.table(arrays), path)


# ── Feature store loader ───────────────────────────────────────────────────────

def load_features(data_dir: str, version: str = FEATURES_VERSION) -> dict:
    """
    Load feature parquets and assemble the FeatureStore dict consumed by
    dataset.py, train.py, evaluate.py, and export.py.
    """
    games_path = os.path.join(data_dir, f'features_games_{version}.parquet')
    users_path = os.path.join(data_dir, f'features_users_{version}.parquet')
    vocab_path = os.path.join(data_dir, 'base_vocab.parquet')

    print(f"  Loading {games_path} ...")
    games_df = pd.read_parquet(games_path)
    print(f"  Loading {users_path} ...")
    users_df = pd.read_parquet(users_path)
    print(f"  Loading base_vocab.parquet ...")
    vocab_df = pd.read_parquet(vocab_path)

    vocab = parse_vocab(vocab_df)

    item_ids    = games_df.sort_values('item_idx')['item_id'].tolist()
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    n_items     = len(item_ids)
    n_genres    = len(vocab['genre_to_i'])
    n_tags      = len(vocab['tag_to_i'])
    n_developers = len(vocab['developer_to_i'])
    n_years     = len(vocab['year_to_i'])

    # Game feature matrices
    games_sorted = games_df.sort_values('item_idx')
    game_genre_matrix  = np.array(games_sorted['genre_context'].tolist(), dtype=np.float32)
    game_tag_matrix    = np.array(games_sorted['tag_context'].tolist(),   dtype=np.float32)
    game_developer_idx = np.array(games_sorted['developer_idx'].tolist(), dtype=np.int64)
    game_year_idx      = np.array(games_sorted['year_idx'].tolist(),      dtype=np.int64)
    game_price_bucket  = np.array(games_sorted['price_bucket'].tolist(),  dtype=np.int64)
    game_median_hours  = np.array(games_sorted['median_hours'].tolist(),  dtype=np.float32)

    # ── Text tower — frozen per-game description embeddings (bge-base-en-v1.5, 768-d) ──
    # Built offline by src.embed_text (mirrors base_game_tags → game_tag_matrix). Aligned
    # to item_idx order here and stored in the FeatureStore so BOTH the CG model and a
    # future ranker text bucket can consume it (rebuilt as a buffer with a pad row in
    # train.build_model / ranker.train._buffers_from_fs).
    text_emb_df     = pd.read_parquet(os.path.join(data_dir, 'base_game_text_emb.parquet'))
    iid_to_text_emb = dict(zip(text_emb_df['item_id'].astype(str), text_emb_df['embedding']))
    missing_text    = [iid for iid in item_ids if str(iid) not in iid_to_text_emb]
    if missing_text:
        raise RuntimeError(
            f"base_game_text_emb.parquet missing {len(missing_text)} corpus games "
            f"(e.g. {missing_text[:3]}). Rebuild with: python -m src.embed_text --force"
        )
    game_text_matrix = np.array([iid_to_text_emb[str(iid)] for iid in item_ids],
                                dtype=np.float32)            # (n_items, text_dim)

    # ── Bucket 5 — per-game numeric scalars for "Numeric Matching" cross features ──
    # All derived from existing base_games + features_games columns, written to FeatureStore
    # as (n_items,) float32 arrays parallel to the other game_* fields above. The ranker
    # registers these as non-persistent buffers (rebuilt from FeatureStore on every load),
    # see ranker/train.py:_buffers_from_fs.
    base_games_for_fs = pd.read_parquet(os.path.join(data_dir, 'base_games.parquet'))
    if 'sentiment_ordinal' not in base_games_for_fs.columns:
        raise RuntimeError(
            "base_games.parquet missing 'sentiment_ordinal' column "
            "(Bucket 5 prerequisite). Rebuild with: python main.py preprocess games"
        )
    # Pull sentiment_ordinal + year by item_id (base_games row order isn't guaranteed
    # to match features_games item_idx order — go through the id→value map).
    iid_to_sentiment = dict(zip(base_games_for_fs['item_id'],
                                base_games_for_fs['sentiment_ordinal']))
    iid_to_year_str  = dict(zip(base_games_for_fs['item_id'], base_games_for_fs['year']))

    sentiment_raw = np.array([iid_to_sentiment[iid] for iid in item_ids], dtype=np.int32)
    game_sentiment = np.where(sentiment_raw >= 0,
                              sentiment_raw.astype(np.float32),
                              SENTIMENT_UNKNOWN_FILL).astype(np.float32)

    year_int = np.array([int(iid_to_year_str[iid]) for iid in item_ids], dtype=np.int32)
    known_year_mask = year_int > 0
    if known_year_mask.any():
        corpus_median_year = float(np.median(year_int[known_year_mask]))
    else:
        corpus_median_year = 2015.0
    game_year_numeric = np.where(known_year_mask,
                                 year_int.astype(np.float32),
                                 corpus_median_year).astype(np.float32)

    game_median_log_hours = np.log1p(game_median_hours).astype(np.float32)

    # ── Bucket 6 — per-game tag/dev rarity buffers for "Niche Feature Crosses" ─
    # All derived from existing FeatureStore inputs (game_tag_matrix + game_developer_idx),
    # written as float32 arrays parallel to the other game_* fields. The ranker registers
    # these as non-persistent buffers (rebuilt from FeatureStore on every load) — see
    # ranker/train.py:_buffers_from_fs.
    #
    # tag_idf is pure IDF per tag (log(N / df)) computed once from corpus tag presence;
    # game_tag_matrix above is RAW TF-IDF (positional × IDF) and shouldn't be substituted
    # for plain IDF here — the niche features want rarity weights independent of how each
    # game listed its tags.
    tag_binary_for_idf = (game_tag_matrix > 0).astype(np.float32)            # (n_items, n_tags)
    tag_df             = tag_binary_for_idf.sum(axis=0)                       # (n_tags,) document frequency
    # Standard IDF with +1 floor on df (every kept tag has MIN_TAG_COUNT presence in preprocess,
    # so df is already ≥ 50 in practice — the floor is paranoid but cheap).
    tag_idf = np.log(n_items / np.maximum(tag_df, 1.0)).astype(np.float32)    # (n_tags,)

    # Shape A buffer for Bucket 6's IDF-overlap features: row-wise IDF-weighted binary tag
    # vector (game_tag_binary * tag_idf[None, :]). Plays the role of game_tag_binary in
    # `weighted_overlap`, with an analogous count denominator = sum of the row's IDF values.
    game_tag_binary_idf = (tag_binary_for_idf * tag_idf[None, :]).astype(np.float32)

    # Shape B per-game scalars (3 niche scalars). Mean/max IDF over an item's tags; the
    # "mean over tags present" uses safe-divide so tag-less items get 0 (harmless — they
    # also have zero IDF mass).
    tag_present_per_game = np.maximum(tag_binary_for_idf.sum(axis=1), 1.0)    # (n_items,) clamp(min=1)
    game_tag_mean_idf    = (game_tag_binary_idf.sum(axis=1) / tag_present_per_game).astype(np.float32)
    game_tag_max_idf     = game_tag_binary_idf.max(axis=1).astype(np.float32)

    # Per-developer corpus catalog size, then per-game lookup → log1p. Pad row appended
    # later (in ranker/train._buffers_from_fs). Unknown-dev games (developer_idx==0) all
    # share the same "unknown" bucket — its catalog count is whatever sum of unknown-dev
    # games we have, so the scalar still has signal (heavy-unknown ≈ lots of small-studio
    # games the developer-string wasn't normalized for).
    dev_catalog_counts = np.bincount(game_developer_idx, minlength=n_developers + 1).astype(np.float32)
    game_dev_log_catalog_size = np.log1p(dev_catalog_counts[game_developer_idx]).astype(np.float32)

    # User dicts
    train_users = users_df[users_df['split'] == 'train']['user_id'].tolist()
    val_users   = users_df[users_df['split'] == 'val']['user_id'].tolist()

    user_to_play_history     = {}
    user_to_play_weights     = {}
    user_to_avg_log_playtime = {}
    user_to_recommend_history = {}

    # Need raw interactions for recommend signal
    interactions_df = pd.read_parquet(os.path.join(data_dir, 'base_interactions.parquet'))
    rec_agg = (interactions_df
               .groupby('user_id')['recommend']
               .apply(list)
               .to_dict())

    for _, row in users_df.iterrows():
        uid = row['user_id']
        user_to_play_history[uid]     = list(row['play_history'])
        user_to_play_weights[uid]     = list(row['play_history_weights'])
        user_to_avg_log_playtime[uid] = float(row['avg_log_playtime'])
        user_to_recommend_history[uid] = rec_agg.get(uid, [None] * len(row['play_history']))

    # item_id → title for canary display
    base_games = pd.read_parquet(os.path.join(data_dir, 'base_games.parquet'))
    item_id_to_title = dict(zip(base_games['item_id'], base_games['title']))

    # Per-game interaction counts across all users (for popularity debiasing)
    game_interaction_counts = np.zeros(n_items, dtype=np.float32)
    for uid, history in user_to_play_history.items():
        for idx in history:
            if 0 <= idx < n_items:
                game_interaction_counts[idx] += 1

    # ── Bucket 5 — per-game popularity scalar (derives from interaction counts) ─
    game_log_count = np.log1p(game_interaction_counts).astype(np.float32)

    # ── Bucket 5 — per-user numeric aggregates ──────────────────────────────────
    # Five floats per user, full-history aggregates (not rollback-prefix-only — matches
    # existing `user_to_avg_log_playtime` convention, the only other per-user scalar
    # consumed by the model). Used by ranker/precompute.py to build the
    # `numeric_match_quintuple` cross feature values for each (label, neg) pair.
    hours_by_user = (interactions_df
                     .groupby('user_id')['hours']
                     .apply(list)
                     .to_dict())
    user_to_mean_price_bucket, user_to_mean_year_numeric    = {}, {}
    user_to_mean_sentiment,    user_to_mean_log_count       = {}, {}
    user_to_median_log_playtime                              = {}
    game_price_bucket_f = game_price_bucket.astype(np.float32)

    for uid, history in user_to_play_history.items():
        valid = [i for i in history if 0 <= i < n_items]
        if valid:
            v = np.asarray(valid, dtype=np.int64)
            user_to_mean_price_bucket[uid] = float(game_price_bucket_f[v].mean())
            user_to_mean_year_numeric[uid] = float(game_year_numeric[v].mean())
            user_to_mean_sentiment[uid]    = float(game_sentiment[v].mean())
            user_to_mean_log_count[uid]    = float(game_log_count[v].mean())
        else:
            # Fallback for users with empty history — `user_to_play_history` should
            # always be non-empty for kept users, but be safe so a missing entry can't
            # crash precompute. Use 0 / SENTIMENT_UNKNOWN_FILL — Z-scoring will
            # re-center these to ~0 after the buffer-populate step.
            user_to_mean_price_bucket[uid] = 0.0
            user_to_mean_year_numeric[uid] = 0.0
            user_to_mean_sentiment[uid]    = SENTIMENT_UNKNOWN_FILL
            user_to_mean_log_count[uid]    = 0.0

        hours = hours_by_user.get(uid, [])
        if hours:
            user_to_median_log_playtime[uid] = float(np.median(np.log1p(np.asarray(hours))))
        else:
            user_to_median_log_playtime[uid] = 0.0

    fs = {
        # Corpus
        'item_ids':           item_ids,
        'item_to_idx':        item_to_idx,
        'item_id_to_title':   item_id_to_title,
        'n_items':            n_items,

        # Vocab sizes (for model construction)
        'n_genres':           n_genres,
        'n_tags':             n_tags,
        'n_developers':       n_developers,
        'n_years':            n_years,
        'n_price_buckets':    N_PRICE_BUCKETS,

        # Vocab maps (for canary / feature lookup)
        'genre_to_i':         vocab['genre_to_i'],
        'tag_to_i':           vocab['tag_to_i'],
        'developer_to_i':     vocab['developer_to_i'],
        'year_to_i':          vocab['year_to_i'],

        # Game feature matrices (numpy, loaded as tensors in model)
        'game_genre_matrix':  game_genre_matrix,
        'game_tag_matrix':    game_tag_matrix,
        'game_text_matrix':   game_text_matrix,
        'game_developer_idx': game_developer_idx,
        'game_year_idx':      game_year_idx,
        'game_price_bucket':  game_price_bucket,
        'game_median_hours':         game_median_hours,
        'game_interaction_counts':   game_interaction_counts,

        # Bucket 5 — per-game numeric scalars (4 new). All (n_items,) float32, parallel
        # to the other game_* arrays. Ranker registers as non-persistent buffers (with a
        # pad row appended) in train._buffers_from_fs. game_price_bucket above is reused
        # for the price-side of price_match (cast to float at use time).
        'game_year_numeric':         game_year_numeric,
        'game_median_log_hours':     game_median_log_hours,
        'game_log_count':            game_log_count,
        'game_sentiment':            game_sentiment,

        # Bucket 6 — per-game rarity buffers for "Niche Feature Crosses" (5 new). All
        # parallel to the other game_* arrays; ranker registers as non-persistent
        # buffers (with pad rows appended) in train._buffers_from_fs. tag_idf is shape
        # (n_tags,) and stored alongside in case future buckets want it directly.
        'tag_idf':                       tag_idf,
        'game_tag_binary_idf':           game_tag_binary_idf,
        'game_tag_mean_idf':             game_tag_mean_idf,
        'game_tag_max_idf':              game_tag_max_idf,
        'game_dev_log_catalog_size':     game_dev_log_catalog_size,

        # User split
        'train_users':        train_users,
        'val_users':          val_users,

        # User features
        'user_to_play_history':      user_to_play_history,
        'user_to_play_weights':      user_to_play_weights,
        'user_to_avg_log_playtime':  user_to_avg_log_playtime,
        'user_to_recommend_history': user_to_recommend_history,

        # Bucket 5 — per-user numeric aggregates (5 new). Consumed by ranker/precompute.py
        # to compute the 5 numeric-match cross features for each (label, neg) candidate.
        'user_to_mean_price_bucket':   user_to_mean_price_bucket,
        'user_to_mean_year_numeric':   user_to_mean_year_numeric,
        'user_to_mean_sentiment':      user_to_mean_sentiment,
        'user_to_mean_log_count':      user_to_mean_log_count,
        'user_to_median_log_playtime': user_to_median_log_playtime,
    }

    print(f"\n  FeatureStore: {n_items:,} games | {n_genres} genres | {n_tags} tags | "
          f"{n_developers:,} developers | {n_years} years")
    print(f"  Users: {len(train_users):,} train | {len(val_users):,} val")
    return fs


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(data_dir: str = 'data', version: str = FEATURES_VERSION) -> None:
    print(f"Loading base parquets from {data_dir}/ ...")
    base  = load_base(data_dir)
    vocab = parse_vocab(base['vocab'])

    print("\n── Building game features ──")
    games_df = build_game_features(base, vocab)

    item_ids    = games_df.sort_values('item_idx')['item_id'].tolist()
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    print("\n── Building user features ──")
    users_df = build_user_features(base, vocab, item_to_idx)

    games_out = os.path.join(data_dir, f'features_games_{version}.parquet')
    users_out = os.path.join(data_dir, f'features_users_{version}.parquet')

    print(f"\nWriting {games_out} ...")
    _write_list_parquet(games_df, games_out)
    print(f"Writing {users_out} ...")
    _write_list_parquet(users_df, users_out)

    print(f"\n✓ features_games_{version}.parquet and features_users_{version}.parquet → {data_dir}/")
