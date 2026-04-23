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
MAX_HISTORY_LEN    = 200    # cap per-user play history (avg-pool handles any length, but caps memory)
N_PRICE_BUCKETS    = 9      # fixed: Free <$5 $5-10 $10-20 $20-30 $30-40 $40-60 >$60 Unknown


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
      item_id, item_idx, genre_context, tag_context, developer_idx, year_idx, price_bucket

    genre_context — float vector length n_genres, uniform weight across listed genres
    tag_context   — float vector length n_tags, TF-IDF scores from base_game_tags.parquet
    developer_idx — int, vocab index; 0 (__unknown__) if developer not in vocab
    year_idx      — int, vocab index; 0 if year not found
    price_bucket  — int, 0-8 (already computed in preprocess)
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

        rows.append({
            'item_id':       iid,
            'item_idx':      item_to_idx[iid],
            'genre_context': genre_ctx,
            'tag_context':   tag_lookup.get(iid, [0.0] * n_tags),
            'developer_idx': developer_idx,
            'year_idx':      year_idx,
            'price_bucket':  price_bucket,
        })

    df = pd.DataFrame(rows)
    print(f"  Game features: {len(df)} games  (genres={n_genres}, tags={n_tags})")
    return df


# ── Per-user features ─────────────────────────────────────────────────────────

def build_user_features(base: dict, vocab: dict, item_to_idx: dict) -> pd.DataFrame:
    """
    One row per user:
      user_id, split, play_history, play_history_weights, genre_context

    split                — 'train' or 'val' (user-level, 90/10 by user)
    play_history         — list[int] item_idx values, capped to MAX_HISTORY_LEN
    play_history_weights — list[float] log(1+h) weights normalized per user
    genre_context        — float vector length 2*n_genres:
                           first half  = debiased avg log-playtime per genre
                           second half = fraction of play history in each genre
    """
    interactions_df = base['interactions']
    games_df        = base['games']

    genre_to_i  = vocab['genre_to_i']
    genres_ord  = vocab['genres_ordered']
    n_genres    = len(genre_to_i)

    item_id_to_genres = {
        r['item_id']: (list(r['genres']) if r['genres'] is not None else [])
        for _, r in games_df.iterrows()
    }

    # ── User-level train/val split ──
    all_users = interactions_df['user_id'].unique().tolist()
    rng = random.Random(VAL_SPLIT_SEED)
    rng.shuffle(all_users)
    n_val    = int(len(all_users) * VAL_FRACTION)
    val_set  = set(all_users[:n_val])
    train_set = set(all_users[n_val:])
    print(f"  Train users: {len(train_set):,}   Val users: {len(val_set):,}")

    # ── Per-user log-playtime avg (for debiasing genre context) ──
    interactions_df = interactions_df.copy()
    interactions_df['log_hours'] = np.log1p(interactions_df['hours'].values)
    user_avg_log = interactions_df.groupby('user_id')['log_hours'].mean().to_dict()

    # ── User genre stats (vectorized) ──
    print("  Computing user genre stats ...")
    _wg = interactions_df[['user_id', 'item_id', 'log_hours']].copy()
    _wg['genre'] = _wg['item_id'].map(item_id_to_genres)
    _wg = _wg.explode('genre').dropna(subset=['genre'])
    _wg = _wg[_wg['genre'].isin(genre_to_i)]

    _agg = (_wg.groupby(['user_id', 'genre'])
               .agg(N=('log_hours', 'count'), S=('log_hours', 'sum'))
               .reset_index())

    total_N  = _agg.groupby('user_id')['N'].sum()
    all_uids = list(total_N.index)
    uid_to_row = {uid: i for i, uid in enumerate(all_uids)}

    _ctx = _agg.copy()
    _ctx['total_N']    = _ctx['user_id'].map(total_N)
    _ctx['avg_log']    = _ctx['user_id'].map(user_avg_log)
    _ctx['avg_g']      = _ctx['S'] / _ctx['N']
    _ctx['val_avg']    = _ctx['avg_g'] - _ctx['avg_log']      # debiased avg log-playtime per genre
    _ctx['val_frac']   = _ctx['N'] / _ctx['total_N']          # play fraction per genre
    _ctx['col_avg']    = _ctx['genre'].map({g: i            for i, g in enumerate(genres_ord)})
    _ctx['col_frac']   = _ctx['genre'].map({g: n_genres + i for i, g in enumerate(genres_ord)})
    _ctx = _ctx.dropna(subset=['col_avg'])
    _ctx[['col_avg', 'col_frac', 'row']] = _ctx[['col_avg', 'col_frac']].astype(int).assign(
        row=_ctx['user_id'].map(uid_to_row).astype(int)
    )

    print("  Building genre context matrix ...")
    genre_ctx_matrix = np.zeros((len(all_uids), 2 * n_genres), dtype=np.float32)
    genre_ctx_matrix[_ctx['row'].values, _ctx['col_avg'].values]  = _ctx['val_avg'].values.astype(np.float32)
    genre_ctx_matrix[_ctx['row'].values, _ctx['col_frac'].values] = _ctx['val_frac'].values.astype(np.float32)

    # ── Per-user play history ──
    history_agg = (interactions_df
                   .groupby('user_id')
                   .agg(item_ids=('item_id', list), log_hours=('log_hours', list))
                   .reset_index())
    history_by_user = {r['user_id']: r for _, r in history_agg.iterrows()}

    rows = []
    for uid in tqdm(all_uids, desc="User features"):
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

        genre_ctx = genre_ctx_matrix[uid_to_row[uid]].tolist() if uid in uid_to_row else [0.0] * (2 * n_genres)

        rows.append({
            'user_id':              uid,
            'split':                split,
            'avg_log_playtime':     float(user_avg_log.get(uid, 1.0)),
            'play_history':         hist_idx,
            'play_history_weights': hist_weights,
            'genre_context':        genre_ctx,
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

    # User dicts
    train_users = users_df[users_df['split'] == 'train']['user_id'].tolist()
    val_users   = users_df[users_df['split'] == 'val']['user_id'].tolist()

    user_to_play_history    = {}
    user_to_play_weights    = {}
    user_to_genre_context   = {}
    user_to_avg_log_playtime = {}

    for _, row in users_df.iterrows():
        uid = row['user_id']
        user_to_play_history[uid]     = list(row['play_history'])
        user_to_play_weights[uid]     = list(row['play_history_weights'])
        user_to_genre_context[uid]    = list(row['genre_context'])
        user_to_avg_log_playtime[uid] = float(row['avg_log_playtime'])

    # item_id → title for canary display
    base_games = pd.read_parquet(os.path.join(data_dir, 'base_games.parquet'))
    item_id_to_title = dict(zip(base_games['item_id'], base_games['title']))

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
        'game_developer_idx': game_developer_idx,
        'game_year_idx':      game_year_idx,
        'game_price_bucket':  game_price_bucket,

        # User split
        'train_users':        train_users,
        'val_users':          val_users,

        # User features
        'user_to_play_history':     user_to_play_history,
        'user_to_play_weights':     user_to_play_weights,
        'user_to_genre_context':    user_to_genre_context,
        'user_to_avg_log_playtime': user_to_avg_log_playtime,
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
