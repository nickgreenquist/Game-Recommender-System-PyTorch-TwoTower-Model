"""
Embedding probes and canary user evaluation.

Usage:
    python main.py probe [checkpoint_path]
    python main.py canary [checkpoint_path]
"""
import glob
import os
from itertools import zip_longest

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GameRecommender
from src.train import build_model, get_config, print_model_summary


# ── Canary user definitions ───────────────────────────────────────────────────
# All game titles must match base_games.parquet exactly (verified against corpus).
# Genres must match base_vocab.parquet type='genre' values exactly.
# Tags must match base_vocab.parquet type='tag' values exactly.

USER_TYPE_TO_FAVORITE_GENRES = {
    'Western RPG Lover': ['RPG'],
    'JRPG Lover':        ['RPG'], 
    'FPS Lover':         [],
    'Civ Lover':         ['Strategy'],
    'Citybuilder Lover': ['Simulation'],
    'Indie Lover':       ['Indie'],
    'Racing Lover':      ['Racing'],
    'Fighting Lover':    []
}

USER_TYPE_TO_FAVORITE_GAMES = {
    'Western RPG Lover': [
        'The Witcher 2: Assassins of Kings Enhanced Edition',
        'The Elder Scrolls IV: Oblivion® Game of the Year Edition',
        'DARK SOULS™: Prepare To Die™ Edition',
        'Divinity: Original Sin (Classic)',
        'Fallout: New Vegas',
        'Mass Effect 2',
    ],
    'JRPG Lover': [
        'FINAL FANTASY VI',
        'Disgaea PC / 魔界戦記ディスガイア PC',
        'FINAL FANTASY VIII'
    ],
    'FPS Lover': [
        'Counter-Strike: Global Offensive',
        'DOOM',
        'Call of Duty®: Black Ops',
        'Battlefield: Bad Company™ 2'
    ],
    'Civ Lover': [
        "Sid Meier's Civilization® V",
        'Civilization IV®: Warlords',
        "Sid Meier's Civilization IV: Colonization",
        'Total War™: ROME II - Emperor Edition'
    ],
    'Citybuilder Lover': [
        'SimCity™ 4 Deluxe Edition',
        'Caesar™ 3',
        'Cities: Skylines',
        'Cities XXL'
    ],
    'Indie Lover': [
        'Terraria',
        'FTL: Faster Than Light',
        'The Binding of Isaac: Rebirth',
        'Rogue Legacy',
        'Spelunky',
    ],
    'Racing Lover': [
        'F1 2012™',
        'Need For Speed: Hot Pursuit',
        'Test Drive Unlimited 2'
    ],
    'Fighting Lover': [
        'Mortal Kombat Komplete Edition',
        'Street Fighter X Tekken',
        'Ultra Street Fighter® IV',
        'Injustice: Gods Among Us Ultimate Edition'
    ]
}

USER_TYPE_TO_TAGS = {
    'Western RPG Lover':  ['Action RPG'],
    'JRPG Lover':         ['JRPG'],
    'FPS Lover':          ['FPS'],
    'Civ Lover':          [],
    'Citybuilder Lover':  ['City Builder'],
    'Indie Lover':        ['Indie', 'Rogue-like', 'Platformer', 'Pixel Graphics'],
    'Racing Lover':       ['Racing'],
    'Fighting Lover':     ['Fighting']
}

SIMULATED_FAV_LOG_HOURS    = 10.0   # weight for favorite games
SIMULATED_ANCHOR_LOG_HOURS =  2.0   # weight for tag-based anchor games
ANCHORS_PER_TAG            =  5


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


# ── Canary user inference ─────────────────────────────────────────────────────

def _get_anchor_titles(fs: dict, tag_names: list, exclude: set) -> list:
    """
    Return up to ANCHORS_PER_TAG top games per tag by raw TF-IDF score,
    skipping titles already in exclude.
    """
    tag_matrix = fs['game_tag_matrix']   # (n_items, n_tags) numpy
    tag_to_i   = fs['tag_to_i']
    item_ids   = fs['item_ids']
    id_to_title = fs['item_id_to_title']

    valid_tags = [t for t in tag_names if t in tag_to_i]
    anchor_titles = []
    seen = set(exclude)

    for tag in valid_tags:
        tag_idx     = tag_to_i[tag]
        sorted_iids = sorted(
            item_ids,
            key=lambda iid: float(tag_matrix[fs['item_to_idx'][iid], tag_idx]),
            reverse=True,
        )
        count = 0
        for iid in sorted_iids:
            if count >= ANCHORS_PER_TAG:
                break
            title = id_to_title[iid]
            if title not in seen:
                anchor_titles.append(title)
                seen.add(title)
                count += 1

    return anchor_titles


def _build_user_embedding(model: GameRecommender, fs: dict, user_type: str) -> torch.Tensor:
    """
    Build a synthetic user embedding for a canary user type.
    Mirrors model.user_embedding() logic exactly — no timestamp tower.

    Simulated play weights:
        favorite games → SIMULATED_FAV_LOG_HOURS
        tag anchor games → SIMULATED_ANCHOR_LOG_HOURS
    Genre context is computed from the synthetic play history using the same
    formula as dataset.py: [debiased_avg_log_playtime | play_frac] per genre.
    """
    fav_genres  = USER_TYPE_TO_FAVORITE_GENRES[user_type]
    fav_titles  = USER_TYPE_TO_FAVORITE_GAMES[user_type]
    tag_names   = USER_TYPE_TO_TAGS.get(user_type, [])

    anchor_titles = _get_anchor_titles(fs, tag_names, exclude=set(fav_titles))

    title_to_iid = {v: k for k, v in fs['item_id_to_title'].items()}
    item_to_idx  = fs['item_to_idx']

    # (title, simulated_log_hours)
    all_games_weighted = (
        [(t, SIMULATED_FAV_LOG_HOURS)    for t in fav_titles]   +
        [(t, SIMULATED_ANCHOR_LOG_HOURS) for t in anchor_titles]
    )

    # Resolve to (item_idx, log_weight), skip titles not in corpus
    history = []
    for title, w in all_games_weighted:
        iid = title_to_iid.get(title)
        if iid is None or iid not in item_to_idx:
            continue
        history.append((item_to_idx[iid], w))

    # ── Genre context (mirrors dataset.py _build_rollback_dataset) ────────────
    n_genres     = fs['n_genres']
    genre_to_i   = fs['genre_to_i']
    genre_matrix = fs['game_genre_matrix']   # (n_items, n_genres) float32

    avg_log       = (sum(w for _, w in history) / max(len(history), 1))
    running_count = np.zeros(n_genres, dtype=np.float32)
    running_sum   = np.zeros(n_genres, dtype=np.float32)

    for item_idx, raw_log in history:
        genre_row = genre_matrix[item_idx]
        for g_idx in np.where(genre_row > 0)[0]:
            running_count[g_idx] += 1
            running_sum[g_idx]   += raw_log

    # Explicit favorite-genre boost: override debiased avg with a strong positive signal
    for g in fav_genres:
        if g in genre_to_i:
            g_idx = genre_to_i[g]
            if running_count[g_idx] == 0:
                running_count[g_idx] = 1.0
                running_sum[g_idx]   = avg_log + SIMULATED_FAV_LOG_HOURS

    genre_ctx = np.zeros(2 * n_genres, dtype=np.float32)
    total_assign = running_count.sum()
    if total_assign > 0:
        mask = running_count > 0
        genre_ctx[:n_genres][mask] = (
            running_sum[mask] / running_count[mask]
        ) - avg_log
        genre_ctx[n_genres:] = running_count / total_assign

    # ── History embedding pool (mirrors model.user_embedding) ─────────────────
    if history:
        hist_idxs = torch.tensor([h[0] for h in history], dtype=torch.long).unsqueeze(0)   # (1, H)
        hist_wts  = torch.tensor([[h[1] for h in history]], dtype=torch.float32)            # (1, H)
        pad_mask  = torch.ones_like(hist_wts).unsqueeze(-1)                                 # (1, H, 1)
        w         = hist_wts.unsqueeze(-1) * pad_mask                                       # (1, H, 1)
        wt_sum    = w.sum(dim=1).clamp(min=1e-6)                                            # (1, 1)
        hist_embs = model.item_embedding_lookup(hist_idxs)                                  # (1, H, D)
        history_emb = (hist_embs * w).sum(dim=1) / wt_sum                                  # (1, D)
    else:
        history_emb = torch.zeros(1, model.item_embedding_lookup.embedding_dim)

    X_genre   = torch.tensor([genre_ctx.tolist()], dtype=torch.float32)
    genre_emb = model.user_genre_tower(X_genre)

    return torch.cat([history_emb, genre_emb], dim=1)   # (1, user_dim)


def run_canary_eval(model: GameRecommender, fs: dict,
                    game_embeddings: dict, all_ids: list, all_combined: torch.Tensor,
                    top_n: int = 10) -> None:
    """Score all games per canary user type and print recommendation tables."""
    model.eval()

    with torch.no_grad():
        for user_type in USER_TYPE_TO_FAVORITE_GENRES:
            user_emb    = _build_user_embedding(model, fs, user_type)
            fav_titles  = USER_TYPE_TO_FAVORITE_GAMES[user_type]
            tag_names   = USER_TYPE_TO_TAGS.get(user_type, [])
            anchor_titles = _get_anchor_titles(fs, tag_names, exclude=set(fav_titles))
            exclude_set   = set(fav_titles) | set(anchor_titles)

            raw_scores  = (all_combined @ user_emb.T).squeeze(-1)   # (n_items,)
            scores      = {all_ids[i]: raw_scores[i].item() for i in range(len(all_ids))}

            recs        = []
            seen_titles = set(exclude_set)
            for iid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if len(recs) >= top_n:
                    break
                title = fs['item_id_to_title'][iid]
                if title not in seen_titles:
                    seen_titles.add(title)
                    recs.append(title)

            fav_genres  = ', '.join(USER_TYPE_TO_FAVORITE_GENRES[user_type]) or '—'
            tags_str    = ', '.join(USER_TYPE_TO_TAGS.get(user_type, [])[:4]) or '—'

            col_w      = min(55, max((len(t) for t in fav_titles), default=20))
            rec_w      = min(55, max((len(r) for r in recs), default=20))
            title_line = f"{user_type}  |  Genre: {fav_genres}  |  Tags: {tags_str}"
            bar_w      = max(col_w + rec_w + 4, len(title_line))

            print(f"\n{'═' * bar_w}")
            print(title_line)
            print(f"{'═' * bar_w}")
            if anchor_titles:
                print(f"Tag anchors (weight={SIMULATED_ANCHOR_LOG_HOURS}):")
                for t in anchor_titles[:10]:
                    print(f"  + {t}")
                print('─' * bar_w)
            header = f"{'Favorite Games':<{col_w}}  Recommendations"
            print(header)
            print('─' * bar_w)
            for a, b in zip_longest(fav_titles, recs, fillvalue=''):
                print(f"{a:<{col_w}}  {b}")


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
    probe_tag(model, ['Rogue-like', 'Rogue-lite'],         game_embeddings, fs)
    probe_tag(model, ['Puzzle'],                           game_embeddings, fs)

    print("\n── Similarity probes ──")
    probe_similar(game_embeddings, fs, all_ids, all_norm,
                  PROBE_SIMILAR_TITLES, top_n=5, all_norm_id=all_norm_id)


def run_canary(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    from src.dataset import load_features
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        return
    print("Loading features ...")
    fs = load_features(data_dir, version)
    model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id = \
        _load_model_and_embeddings(cp, fs)

    print("\n── Canary user evaluation ──")
    run_canary_eval(model, fs, game_embeddings, all_ids, all_combined)
