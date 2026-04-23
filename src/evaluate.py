"""
Embedding probes and canary user evaluation.

Usage:
    python main.py probe [checkpoint_path]
    python main.py canary [checkpoint_path]
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GameRecommender
from src.train import build_model, get_config, print_model_summary


# ── Game embedding cache ──────────────────────────────────────────────────────

def build_game_embeddings(model: GameRecommender, fs: dict) -> tuple:
    """
    Pre-compute item tower embeddings for all corpus games.

    Returns:
        game_embeddings : dict  item_id → per-tower + combined tensors
        all_ids         : list[str]   item_ids in index order
        all_combined    : (n_items, dim) tensor
    """
    model.eval()
    item_ids   = fs['item_ids']
    n_items    = len(item_ids)
    batch_size = 512

    all_game_idxs = torch.arange(n_items, dtype=torch.long)
    all_genre_ctx = torch.from_numpy(fs['game_genre_matrix'])
    all_year_idxs = torch.from_numpy(fs['game_year_idx'].astype(np.int64))
    all_dev_idxs  = torch.from_numpy(fs['game_developer_idx'].astype(np.int64))
    all_prices    = torch.from_numpy(fs['game_price_bucket'].astype(np.int64))

    genre_embs = []
    tag_embs   = []
    game_embs  = []
    dev_embs   = []
    year_embs  = []
    price_embs = []

    with torch.no_grad():
        for s in range(0, n_items, batch_size):
            e     = min(s + batch_size, n_items)
            gidxs = all_game_idxs[s:e]
            genre_embs.append(model.item_genre_tower(all_genre_ctx[s:e]))
            tag_embs.append(model.item_tag_tower(model.game_tag_matrix[gidxs]))
            game_embs.append(model.item_embedding_tower(model.item_embedding_lookup(gidxs)))
            dev_embs.append(model.developer_tower(
                model.developer_embedding_lookup(all_dev_idxs[s:e])))
            year_embs.append(model.year_embedding_tower(
                model.year_embedding_lookup(all_year_idxs[s:e])))
            price_embs.append(model.price_embedding_tower(
                model.price_embedding_lookup(all_prices[s:e])))

    genre_all = torch.cat(genre_embs,  dim=0)
    tag_all   = torch.cat(tag_embs,    dim=0)
    game_all  = torch.cat(game_embs,   dim=0)
    dev_all   = torch.cat(dev_embs,    dim=0)
    year_all  = torch.cat(year_embs,   dim=0)
    price_all = torch.cat(price_embs,  dim=0)
    combined  = torch.cat([genre_all, tag_all, game_all, dev_all, year_all, price_all], dim=1)

    game_embeddings = {}
    for i, iid in enumerate(item_ids):
        game_embeddings[iid] = {
            'GAME_GENRE_EMBEDDING':    genre_all[i].unsqueeze(0),
            'GAME_TAG_EMBEDDING':      tag_all[i].unsqueeze(0),
            'GAME_ID_EMBEDDING':       game_all[i].unsqueeze(0),
            'GAME_DEV_EMBEDDING':      dev_all[i].unsqueeze(0),
            'GAME_YEAR_EMBEDDING':     year_all[i].unsqueeze(0),
            'GAME_PRICE_EMBEDDING':    price_all[i].unsqueeze(0),
            'GAME_EMBEDDING_COMBINED': combined[i].unsqueeze(0),
        }

    return game_embeddings, item_ids, combined


# ── Embedding probes ──────────────────────────────────────────────────────────

def probe_genre(model: GameRecommender, genre, game_embeddings: dict,
                fs: dict, top_n: int = 10) -> None:
    """
    Find the most representative games for a genre in item genre embedding space.
    genre may be a single string or a list of strings (multi-hot).
    """
    genres = [genre] if isinstance(genre, str) else genre
    for g in genres:
        if g not in fs['genre_to_i']:
            print(f"Genre '{g}' not in vocabulary.")
            return

    ctx = [0.0] * fs['n_genres']
    for g in genres:
        ctx[fs['genre_to_i'][g]] = 1.0

    with torch.no_grad():
        query_emb = model.item_genre_tower(torch.tensor([ctx])).view(-1)

    sims = {
        iid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            game_embeddings[iid]['GAME_GENRE_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for iid in fs['item_ids']
    }

    label = ' + '.join(genres)
    print(f"\nTop-{top_n} games for genre '{label}':")
    seen = set()
    for iid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen) >= top_n:
            break
        title = fs['item_id_to_title'][iid]
        if title not in seen:
            seen.add(title)
            print(f"  {sim:.4f}  {title}")


def probe_tag(model: GameRecommender, tag_names: list, game_embeddings: dict,
              fs: dict, top_n: int = 10, k_anchors: int = 5) -> None:
    """
    Find games most similar to a tag query in the item tag embedding space.
    Finds top-k_anchors games by raw TF-IDF tag score, averages their GAME_TAG_EMBEDDING
    as the query, then ranks all games by cosine similarity.
    """
    valid_tags = [t for t in tag_names if t in fs['tag_to_i']]
    if not valid_tags:
        print(f"No tags from {tag_names} found in vocabulary.")
        return

    tag_matrix = fs['game_tag_matrix']  # (n_items, n_tags) numpy
    raw_scores = {}
    for i, iid in enumerate(fs['item_ids']):
        raw_scores[iid] = sum(float(tag_matrix[i, fs['tag_to_i'][t]]) for t in valid_tags)

    anchors = sorted(raw_scores, key=raw_scores.get, reverse=True)[:k_anchors]
    query_emb = torch.stack([
        game_embeddings[iid]['GAME_TAG_EMBEDDING'].view(-1) for iid in anchors
    ]).mean(dim=0)

    anchor_titles = [fs['item_id_to_title'][iid] for iid in anchors]
    print(f"\nTag anchors for {valid_tags}: {anchor_titles}")

    sims = {
        iid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            game_embeddings[iid]['GAME_TAG_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for iid in fs['item_ids']
    }

    anchor_set  = set(anchors)
    seen_titles = set()
    print(f"Top-{top_n} games:")
    for iid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen_titles) >= top_n:
            break
        title = fs['item_id_to_title'][iid]
        if title not in seen_titles:
            seen_titles.add(title)
            marker = ' [seed]' if iid in anchor_set else ''
            print(f"  {sim:.4f}  {title}{marker}")


def probe_similar(game_embeddings: dict, fs: dict,
                  all_ids: list, all_norm: torch.Tensor,
                  titles: list, top_n: int = 5,
                  all_norm_id: torch.Tensor = None) -> None:
    """
    For each query title, find the top-N most similar games by cosine similarity.
    Shows results for GAME_EMBEDDING_COMBINED and optionally GAME_ID_EMBEDDING.
    """
    title_to_iid = {v: k for k, v in fs['item_id_to_title'].items()}
    TRUNC = 32

    def trunc(s: str) -> str:
        return s if len(s) <= TRUNC else s[:TRUNC - 1] + '…'

    def get_top_n(norm_matrix: torch.Tensor, emb_key: str, title: str) -> list:
        iid = title_to_iid.get(title)
        if iid is None:
            return []
        query   = F.normalize(game_embeddings[iid][emb_key], dim=1)
        sims    = (norm_matrix @ query.T).squeeze(-1)
        top_idx = sims.argsort(descending=True)
        results = []
        seen    = {title}
        for idx in top_idx:
            candidate = fs['item_id_to_title'][all_ids[idx.item()]]
            if candidate in seen:
                continue
            seen.add(candidate)
            results.append(candidate)
            if len(results) >= top_n:
                break
        return results

    def print_table(label: str, rows: list) -> None:
        valid = [(t, r) for t, r in rows if r]
        if not valid:
            return
        seed_w = max(len(trunc(t)) for t, _ in valid)
        col_w  = TRUNC
        header = f"{'Seed':<{seed_w}}" + "".join(f"  {'#'+str(i+1):<{col_w}}" for i in range(top_n))
        print(f"\n── Most similar games ({label}) ──")
        print(header)
        print('─' * len(header))
        for title, results in rows:
            if not results:
                print(f"{trunc(title):<{seed_w}}  (not in corpus)")
                continue
            row = f"{trunc(title):<{seed_w}}"
            for t in results:
                row += f"  {trunc(t):<{col_w}}"
            print(row)

    combined_rows = [(t, get_top_n(all_norm,    'GAME_EMBEDDING_COMBINED', t)) for t in titles]
    print_table('combined embedding', combined_rows)

    if all_norm_id is not None:
        id_rows = [(t, get_top_n(all_norm_id, 'GAME_ID_EMBEDDING', t)) for t in titles]
        print_table('game ID embedding only', id_rows)


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint_path: str, checkpoint_dir: str):
    if checkpoint_path is not None:
        return checkpoint_path
    candidates = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'best_softmax_*.pth')) +
        glob.glob(os.path.join(checkpoint_dir, 'softmax_*_step_*.pth')),
        key=os.path.getmtime, reverse=True
    )
    if not candidates:
        print("No checkpoint found in saved_models/. Train a model first.")
        return None
    return candidates[0]


def _load_model_and_embeddings(checkpoint_path: str, fs: dict) -> tuple:
    """Build model, load weights, pre-compute game embeddings."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    config     = get_config()
    model      = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()
    print_model_summary(model)

    print("\nBuilding game embeddings ...")
    game_embeddings, all_ids, all_combined = build_game_embeddings(model, fs)

    print("Precomputing normalised embedding matrix ...")
    all_norm    = F.normalize(all_combined, dim=1)
    all_id_embs = torch.cat([game_embeddings[iid]['GAME_ID_EMBEDDING'] for iid in all_ids], dim=0)
    all_norm_id = F.normalize(all_id_embs, dim=1)

    return model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id


# ── Probe titles ──────────────────────────────────────────────────────────────

PROBE_SIMILAR_TITLES = [
    'Counter-Strike: Global Offensive',
    'Portal 2',
    "Garry's Mod",
    'Left 4 Dead 2',
    'Terraria',
    'DARK SOULS™: Prepare To Die™ Edition',
    "Sid Meier's Civilization® V",
    'The Witcher 2: Assassins of Kings Enhanced Edition',
    'Borderlands 2',
    'BioShock™',
    'The Binding of Isaac: Rebirth',
    'Stardew Valley',
    'Rocket League®',
    'XCOM: Enemy Unknown',
    'PAYDAY 2',
    'Warframe',
    'Grand Theft Auto V',
    'Arma 3',
    'The Elder Scrolls IV: Oblivion® Game of the Year Edition',
]


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_probes(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    from src.dataset import load_features
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        return
    print("Loading features ...")
    fs = load_features(data_dir, version)
    model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id = \
        _load_model_and_embeddings(cp, fs)

    print("\n── Genre probes ──")
    probe_genre(model, 'Action',    game_embeddings, fs)
    probe_genre(model, 'RPG',       game_embeddings, fs)
    probe_genre(model, 'Strategy',  game_embeddings, fs)
    probe_genre(model, 'Simulation', game_embeddings, fs)
    probe_genre(model, ['Action', 'RPG'], game_embeddings, fs)

    print("\n── Tag probes ──")
    probe_tag(model, ['FPS', 'Shooter'],                   game_embeddings, fs)
    probe_tag(model, ['RPG', 'Open World'],                game_embeddings, fs)
    probe_tag(model, ['Strategy', 'Turn-Based'],           game_embeddings, fs)
    probe_tag(model, ['Horror', 'Survival Horror'],        game_embeddings, fs)
    probe_tag(model, ['Co-op', 'Multiplayer'],             game_embeddings, fs)
    probe_tag(model, ['Roguelike', 'Roguelite'],           game_embeddings, fs)
    probe_tag(model, ['Puzzle'],                           game_embeddings, fs)

    print("\n── Similarity probes ──")
    probe_similar(game_embeddings, fs, all_ids, all_norm,
                  PROBE_SIMILAR_TITLES, top_n=5, all_norm_id=all_norm_id)


def run_canary(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    print("Canary evaluation not yet implemented.")
