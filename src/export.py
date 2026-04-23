"""
Stage 5 — Export serving artifacts.

serving/model.pth           — state_dict (game_tag_matrix, game_dev_idx excluded)
serving/game_embeddings.pt  — {item_id: {GAME_EMBEDDING_COMBINED, sub-embeddings}}
serving/feature_store.pt    — inference dict (no user data)

game_tag_matrix and game_dev_idx are registered buffers and are excluded
from model.pth. They are restored from feature_store.pt at app startup.

Usage:
    python main.py export
    python main.py export <checkpoint_path>
"""
import glob
import os

import numpy as np
import pandas as pd
import torch

from src.dataset import load_features
from src.evaluate import build_game_embeddings
from src.train import build_model, get_config

SERVING_DIR = 'serving'

PRICE_BUCKET_LABELS = ['Free', '<$5', '$5–10', '$10–20', '$20–30', '$30–40', '$40–60', '>$60', 'Unknown']


def run_export(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if checkpoint_path is None:
        candidates = sorted(
            glob.glob(os.path.join('saved_models', 'best_softmax_*.pth')),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            print("No checkpoint found in saved_models/. Train a model first.")
            return
        checkpoint_path = candidates[0]

    print(f"Checkpoint: {checkpoint_path}")
    config = get_config()

    print("Loading features ...")
    fs = load_features(data_dir, version)

    state_dict = torch.load(checkpoint_path, weights_only=True)
    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()

    print("Building game embeddings ...")
    game_embeddings, _all_ids, _all_combined = build_game_embeddings(model, fs)

    os.makedirs(SERVING_DIR, exist_ok=True)

    # ── model.pth (buffers excluded) ──────────────────────────────────────────
    model_state = {k: v for k, v in model.state_dict().items()
                   if k not in ('game_tag_matrix', 'game_dev_idx')}
    model_path = os.path.join(SERVING_DIR, 'model.pth')
    torch.save(model_state, model_path)
    print(f"Saved {model_path}  ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # ── game_embeddings.pt ────────────────────────────────────────────────────
    emb_path = os.path.join(SERVING_DIR, 'game_embeddings.pt')
    torch.save(game_embeddings, emb_path)
    print(f"Saved {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    # ── Popularity ordering for dropdowns ─────────────────────────────────────
    print("Building metadata ...")
    base_games = pd.read_parquet(os.path.join(data_dir, 'base_games.parquet'))
    item_set   = set(fs['item_ids'])
    games_sorted = base_games.sort_values('n_users', ascending=False)
    popularity_ordered_titles = [
        row['title'] for _, row in games_sorted.iterrows()
        if row['item_id'] in item_set
    ]
    covered = set(popularity_ordered_titles)
    for iid in fs['item_ids']:
        t = fs['item_id_to_title'].get(iid)
        if t and t not in covered:
            popularity_ordered_titles.append(t)

    # ── Per-game display metadata ─────────────────────────────────────────────
    item_id_to_developer = dict(zip(base_games['item_id'], base_games['developer']))

    item_id_to_year = {}
    for _, row in base_games.iterrows():
        y = str(row.get('year', ''))
        item_id_to_year[row['item_id']] = '' if y in ('', '-1', 'None', 'nan') else y

    item_id_to_genres = {}
    for _, row in base_games.iterrows():
        genres = list(row['genres']) if row['genres'] is not None else []
        item_id_to_genres[row['item_id']] = genres

    item_id_to_price_label = {}
    for _, row in base_games.iterrows():
        bucket = int(row.get('price_bucket', 8))
        item_id_to_price_label[row['item_id']] = PRICE_BUCKET_LABELS[min(bucket, 8)]

    # Top tags per game (top 3 by TF-IDF score, vocab-filtered)
    tags_df = pd.read_parquet(os.path.join(data_dir, 'base_game_tags.parquet'))
    tag_to_i = fs['tag_to_i']
    valid_tag_set = set(tag_to_i.keys())
    item_id_to_top_tags = {}
    for _, row in tags_df.iterrows():
        names  = list(row['tag_names']) if row['tag_names'] is not None else []
        scores = list(row['scores'])    if row['scores']    is not None else []
        valid = [(t, s) for t, s in zip(names, scores) if t in valid_tag_set]
        top   = sorted(valid, key=lambda x: x[1], reverse=True)[:3]
        item_id_to_top_tags[row['item_id']] = [t for t, _ in top]

    # ── Vocab lists ───────────────────────────────────────────────────────────
    genre_to_i     = fs['genre_to_i']
    genres_ordered = sorted(genre_to_i.keys(), key=lambda g: genre_to_i[g])
    tags_ordered   = sorted(tag_to_i.keys(),   key=lambda t: tag_to_i[t])

    # ── Registered buffers for model reconstruction ───────────────────────────
    n_items      = fs['n_items']
    n_tags       = fs['n_tags']
    n_developers = fs['n_developers']

    tag_matrix      = fs['game_tag_matrix']          # (n_items, n_tags) numpy float32
    pad_row         = np.zeros((1, n_tags), dtype=np.float32)
    game_tag_matrix = torch.from_numpy(np.vstack([tag_matrix, pad_row]))

    dev_idx_arr  = fs['game_developer_idx']          # (n_items,) int64
    game_dev_idx = torch.from_numpy(np.append(dev_idx_arr, n_developers).astype(np.int64))

    # ── feature_store.pt ──────────────────────────────────────────────────────
    feature_store = {
        # Dropdown ordering
        'popularity_ordered_titles': popularity_ordered_titles,
        # Corpus
        'item_ids':         fs['item_ids'],
        'item_to_idx':      fs['item_to_idx'],
        'item_id_to_title': fs['item_id_to_title'],
        'title_to_item_id': {v: k for k, v in fs['item_id_to_title'].items()},
        'n_items':          n_items,
        # Vocab sizes
        'n_genres':        fs['n_genres'],
        'n_tags':          n_tags,
        'n_developers':    n_developers,
        'n_years':         fs['n_years'],
        'n_price_buckets': fs['n_price_buckets'],
        # Vocab maps and lists
        'genre_to_i':      genre_to_i,
        'tag_to_i':        tag_to_i,
        'genres_ordered':  genres_ordered,
        'tags_ordered':    tags_ordered,
        # Game feature matrices
        'game_genre_matrix': fs['game_genre_matrix'],   # numpy (n_items, n_genres)
        'game_tag_matrix':   game_tag_matrix,           # tensor (n_items+1, n_tags) — buffer
        'game_dev_idx':      game_dev_idx,              # tensor (n_items+1,) — buffer
        # Per-game display metadata
        'item_id_to_developer':  item_id_to_developer,
        'item_id_to_year':       item_id_to_year,
        'item_id_to_genres':     item_id_to_genres,
        'item_id_to_top_tags':   item_id_to_top_tags,
        'item_id_to_price_label': item_id_to_price_label,
        # Model config for reconstruction
        'model_config': config,
    }
    fs_path = os.path.join(SERVING_DIR, 'feature_store.pt')
    torch.save(feature_store, fs_path)
    print(f"Saved {fs_path}  ({os.path.getsize(fs_path) / 1e6:.1f} MB)")

    total_mb = sum(
        os.path.getsize(os.path.join(SERVING_DIR, f)) / 1e6
        for f in ('model.pth', 'game_embeddings.pt', 'feature_store.pt')
    )
    print(f"\nTotal serving/ size: {total_mb:.1f} MB")
    print("Done. Run: streamlit run streamlit_app.py")
