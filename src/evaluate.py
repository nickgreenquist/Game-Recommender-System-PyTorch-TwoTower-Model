"""
Embedding probes and canary user evaluation V2.
"""
import glob
import os
from itertools import zip_longest
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GameRecommender
from src.train import build_model, get_config, print_model_summary


# ── Canary user definitions ───────────────────────────────────────────────────

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
        'Mugen Souls',
        'OH! RPG!',
        'Ys VI: The Ark of Napishtim',
        'Wild Season',
        'Disgaea PC / 魔界戦記ディスガイア PC',
        'Hyperdevotion Noire: Goddess Black Heart (Neptunia)',
        'FINAL FANTASY® XIII'
    ],
    'FPS Lover': [
        'QUAKE II',
        "Unreal Tournament 2004: Editor's Choice Edition",
        'Quake IV',
        'Final DOOM',
        'Serious Sam HD: The First Encounter',
        'Quake III Arena',
        'Doom 3: BFG Edition'
    ],
    'Civ Lover': [
        "Sid Meier's Civilization® V",
        'Civilization IV®: Warlords',
        "Sid Meier's Civilization IV: Colonization",
        'Total War™: ROME II - Emperor Edition'
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
        'Test Drive Unlimited 2',
        'Ford Street Racing',
        'Test Drive: Ferrari Racing Legends'
    ],
    'Fighting Lover': [
        'Street Fighter X Tekken',
        'Ultra Street Fighter® IV',
        'Injustice: Gods Among Us Ultimate Edition',
        'Divekick',
        'Street Fighter® IV',
        'DEAD OR ALIVE 5 Last Round: Core Fighters',
        'THE KING OF FIGHTERS XIII STEAM EDITION',
    ],
    'Survival Lover': [
        'Unturned',
        'Terraria',
        'Rust',
        'ARK: Survival Evolved'
    ],
    'Management Lover': [
        'Cities: Skylines',
        'Kerbal Space Program',
        'Prison Architect',
        'Game Dev Tycoon',
        'Euro Truck Simulator 2'
    ]
}

USER_TYPE_TO_DISLIKED_GAMES = {
    # 'Western RPG Lover': ['F1 2012™', 'Cities: Skylines'],
    # 'JRPG Lover':        ['Counter-Strike: Global Offensive', 'Arma 3'],
    # 'FPS Lover':         ['Cities: Skylines', 'Game Dev Tycoon'],
    # 'Civ Lover':         ['Need For Speed: Hot Pursuit', 'Street Fighter® IV'],
    # 'Indie Lover':       ['Call of Duty: Black Ops', 'F1 2012™'],
    # 'Racing Lover':      ['The Witcher 2: Assassins of Kings Enhanced Edition', 'Civ V'],
    # 'Fighting Lover':    ['Cities: Skylines', 'Euro Truck Simulator 2'],
    # 'Survival Lover':    ['Street Fighter® IV', 'F1 2012™'],
    # 'Management Lover':  ['DARK SOULS™: Prepare To Die™ Edition', 'DOOM']
}

USER_TYPE_TO_TAGS = {
    'Western RPG Lover':  ['Action RPG'],
    'JRPG Lover':         ['JRPG'],
    'FPS Lover':          ['FPS'],
    'Civ Lover':          ['4X'],
    'Indie Lover':        ['Indie', 'Rogue-like', 'Platformer', 'Pixel Graphics'],
    'Racing Lover':       ['Racing'],
    'Fighting Lover':     ['Fighting'],
    'Survival Lover':     ['Survival'],
    'Management Lover':   ['Management']
}

SIMULATED_FAV_LOG_HOURS    = 10.0   # weight for favorite games
SIMULATED_ANCHOR_LOG_HOURS =  2.0   # weight for tag-based anchor games
SIMULATED_DISLIKE_LOG_HOURS = 0.5   # weight for disliked games
ANCHORS_PER_TAG            =  5


# ── Game embedding cache ──────────────────────────────────────────────────────

def build_game_embeddings(model: GameRecommender, fs: dict) -> tuple:
    model.eval()
    item_ids   = fs['item_ids']
    n_items    = len(item_ids)
    batch_size = 512

    device = next(model.parameters()).device
    all_game_idxs = torch.arange(n_items, device=device)
    all_genre_ctx = torch.from_numpy(fs['game_genre_matrix']).to(device)
    all_year_idxs = torch.from_numpy(fs['game_year_idx'].astype(np.int64)).to(device)
    all_dev_idxs  = torch.from_numpy(fs['game_developer_idx'].astype(np.int64)).to(device)
    all_prices    = torch.from_numpy(fs['game_price_bucket'].astype(np.int64)).to(device)

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
            dev_embs.append(model.developer_tower(model.developer_embedding_lookup(all_dev_idxs[s:e])))
            year_embs.append(model.year_embedding_tower(model.year_embedding_lookup(all_year_idxs[s:e])))
            price_embs.append(model.price_embedding_tower(model.price_embedding_lookup(all_prices[s:e])))

    genre_all = torch.cat(genre_embs,  dim=0)
    tag_all   = torch.cat(tag_embs,    dim=0)
    game_all  = torch.cat(game_embs,   dim=0)
    dev_all   = torch.cat(dev_embs,    dim=0)
    year_all  = torch.cat(year_embs,   dim=0)
    price_all = torch.cat(price_embs,  dim=0)

    # Re-apply item projection MLP
    concat_all = torch.cat([genre_all, tag_all, game_all, dev_all, year_all, price_all], dim=1)
    combined = []
    with torch.no_grad():
        for s in range(0, n_items, batch_size):
            combined.append(model.item_projection(concat_all[s:s + batch_size]))
    combined = torch.cat(combined, dim=0)

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
    tag_matrix = fs['game_tag_matrix']
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
    fav_titles  = USER_TYPE_TO_FAVORITE_GAMES[user_type]
    dis_titles  = USER_TYPE_TO_DISLIKED_GAMES.get(user_type, [])
    tag_names   = USER_TYPE_TO_TAGS.get(user_type, [])

    anchor_titles = _get_anchor_titles(fs, tag_names, exclude=set(fav_titles) | set(dis_titles))

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

    # Triple Pool Construction as per user instruction:
    # favorite games: liked input and also in full history
    # anchor games: full history
    # disliked games: disliked input
    liked_ids    = list(fav_idxs)
    disliked_ids = list(dis_idxs)
    full_ids     = list(fav_idxs) + list(anchor_idxs) + list(dis_idxs)

    # ── Context Calculation (Rolling simulation) ──
    # Mirroring _build_rollback_dataset accumulators on full_ids
    n_genres = fs['n_genres']
    n_tags   = fs['n_tags']
    genre_matrix = fs['game_genre_matrix']
    tag_matrix   = fs['game_tag_matrix']
    
    running_genre_count = np.zeros(n_genres, dtype=np.float32)
    running_genre_sum   = np.zeros(n_genres, dtype=np.float32)
    running_tag_sum     = np.zeros(n_tags, dtype=np.float32)

    # Simulate weights/logs for context calculation
    # favorites = 10.0, anchors = 2.0, disliked = 0.5
    for idx in fav_idxs:
        for g_idx in np.where(genre_matrix[idx] > 0)[0]:
            running_genre_count[g_idx] += 1
            running_genre_sum[g_idx]   += SIMULATED_FAV_LOG_HOURS
        running_tag_sum += tag_matrix[idx]
        
    for idx in anchor_idxs:
        for g_idx in np.where(genre_matrix[idx] > 0)[0]:
            running_genre_count[g_idx] += 1
            running_genre_sum[g_idx]   += SIMULATED_ANCHOR_LOG_HOURS
        running_tag_sum += tag_matrix[idx]

    for idx in dis_idxs:
        for g_idx in np.where(genre_matrix[idx] > 0)[0]:
            running_genre_count[g_idx] += 1
            running_genre_sum[g_idx]   += SIMULATED_DISLIKE_LOG_HOURS
        running_tag_sum += tag_matrix[idx]

    # Compute final genre_ctx and tag_ctx (no explicit override)
    total_items = len(fav_idxs) + len(anchor_idxs) + len(dis_idxs)
    avg_log = (len(fav_idxs)*10.0 + len(anchor_idxs)*2.0 + len(dis_idxs)*0.5) / max(total_items, 1)
    
    total_assign = running_genre_count.sum()
    genre_ctx = np.zeros(2 * n_genres, dtype=np.float32)
    if total_assign > 0:
        mask = running_genre_count > 0
        genre_ctx[:n_genres][mask] = (running_genre_sum[mask] / running_genre_count[mask]) - avg_log
        genre_ctx[n_genres:] = running_genre_count / total_assign

    tag_ctx = running_tag_sum

    device = next(model.parameters()).device
    X_genre_t = torch.from_numpy(np.array([genre_ctx], dtype=np.float32)).to(device)
    X_tag_t   = torch.from_numpy(np.array([tag_ctx],   dtype=np.float32)).to(device)
    
    def to_padded(ids):
        if not ids: ids = [model.game_pad_idx]
        return torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        return model.user_embedding(X_genre_t, X_tag_t, to_padded(liked_ids), to_padded(disliked_ids), to_padded(full_ids))


def run_canary_eval(model: GameRecommender, fs: dict, all_combined: torch.Tensor, all_ids: list, top_n: int = 10) -> None:
    model.eval()
    with torch.no_grad():
        for user_type in USER_TYPE_TO_FAVORITE_GAMES:
            user_emb = _build_user_embedding(model, fs, user_type)
            fav_titles = USER_TYPE_TO_FAVORITE_GAMES[user_type]
            dis_titles = USER_TYPE_TO_DISLIKED_GAMES.get(user_type, [])
            tag_names   = USER_TYPE_TO_TAGS.get(user_type, [])
            anchor_titles = _get_anchor_titles(fs, tag_names, exclude=set(fav_titles) | set(dis_titles))
            exclude_set   = set(fav_titles) | set(anchor_titles)
            
            raw_scores = (all_combined @ user_emb.T).squeeze(-1)
            scores = {all_ids[i]: raw_scores[i].item() for i in range(len(all_ids))}

            recs = []
            seen_titles = set(exclude_set)
            for iid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if len(recs) >= top_n: break
                title = fs['item_id_to_title'][iid]
                if title not in seen_titles:
                    seen_titles.add(title)
                    recs.append(title)

            tags_str = ', '.join(USER_TYPE_TO_TAGS.get(user_type, [])[:4]) or '—'
            title_line = f"{user_type}  |  Tags: {tags_str}"
            bar_w = 100

            print(f"\n{'═' * bar_w}")
            print(title_line)
            print(f"{'═' * bar_w}")
            if dis_titles:
                print(f"Disliked: " + ", ".join(dis_titles))
            if anchor_titles:
                print(f"Anchors: " + ", ".join(anchor_titles[:5]))
            print('─' * bar_w)
            print(f"{'Favorite Games':<50}  Recommendations")
            print("-" * bar_w)
            for a, b in zip_longest(fav_titles, recs, fillvalue=''):
                print(f"{a:<50}  {b}")


# ── Embedding probes ──────────────────────────────────────────────────────────

def probe_genre(model: GameRecommender, genre, game_embeddings: dict, fs: dict, top_n: int = 10) -> None:
    genres = [genre] if isinstance(genre, str) else genre
    ctx = [0.0] * fs['n_genres']
    for g in genres:
        if g in fs['genre_to_i']:
            ctx[fs['genre_to_i'][g]] = 1.0
        else:
            print(f"Genre '{g}' not in vocabulary.")
            return

    device = next(model.parameters()).device
    with torch.no_grad():
        query_emb = model.item_genre_tower(torch.tensor([ctx], dtype=torch.float32, device=device)).view(-1)

    sims = {
        iid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            game_embeddings[iid]['GAME_GENRE_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for iid in fs['item_ids']
    }

    print(f"\nTop-{top_n} games for genre '{' + '.join(genres)}':")
    seen = set()
    for iid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen) >= top_n: break
        title = fs['item_id_to_title'][iid]
        if title not in seen:
            seen.add(title)
            print(f"  {sim:.4f}  {title}")


def probe_tag(model: GameRecommender, tag_names: list, game_embeddings: dict, fs: dict, top_n: int = 10, k_anchors: int = 5) -> None:
    valid_tags = [t for t in tag_names if t in fs['tag_to_i']]
    if not valid_tags:
        print(f"No tags from {tag_names} found.")
        return

    tag_matrix = fs['game_tag_matrix']
    raw_scores = {}
    for i, iid in enumerate(fs['item_ids']):
        raw_scores[iid] = sum(float(tag_matrix[i, fs['tag_to_i'][t]]) for t in valid_tags)

    anchors = sorted(raw_scores, key=raw_scores.get, reverse=True)[:k_anchors]
    query_emb = torch.stack([game_embeddings[iid]['GAME_TAG_EMBEDDING'].view(-1) for iid in anchors]).mean(dim=0)

    print(f"\nTag anchors for {valid_tags}: {[fs['item_id_to_title'][iid] for iid in anchors]}")
    sims = {
        iid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            game_embeddings[iid]['GAME_TAG_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for iid in fs['item_ids']
    }

    seen = set()
    print(f"Top-{top_n} games:")
    anchor_set = set(anchors)
    for iid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        if len(seen) >= top_n: break
        title = fs['item_id_to_title'][iid]
        if title not in seen:
            seen.add(title)
            marker = ' [seed]' if iid in anchor_set else ''
            print(f"  {sim:.4f}  {title}{marker}")


def probe_similar(game_embeddings: dict, fs: dict, all_ids: list, all_norm: torch.Tensor, titles: list, top_n: int = 5, all_norm_id: torch.Tensor = None) -> None:
    title_to_iid = {v: k for k, v in fs['item_id_to_title'].items()}
    TRUNC = 32

    def trunc(s: str) -> str:
        return s if len(s) <= TRUNC else s[:TRUNC - 1] + '…'

    def get_top_n(norm_matrix: torch.Tensor, emb_key: str, title: str) -> list:
        iid = title_to_iid.get(title)
        if iid is None: return []
        query = F.normalize(game_embeddings[iid][emb_key], dim=1)
        sims = (norm_matrix @ query.T).squeeze(-1)
        top_idx = sims.argsort(descending=True)
        results, seen = [], {title}
        for idx in top_idx:
            candidate = fs['item_id_to_title'][all_ids[idx.item()]]
            if candidate not in seen:
                seen.add(candidate)
                results.append(candidate)
                if len(results) >= top_n: break
        return results

    def print_table(label: str, rows: list) -> None:
        valid = [(t, r) for t, r in rows if r]
        if not valid: return
        seed_w = max(len(trunc(t)) for t, _ in valid)
        header = f"{'Seed':<{seed_w}}" + "".join(f"  {'#'+str(i+1):<32}" for i in range(top_n))
        print(f"\n── Most similar games ({label}) ──")
        print(header)
        print('─' * len(header))
        for title, results in rows:
            if not results: continue
            row = f"{trunc(title):<{seed_w}}"
            for t in results: row += f"  {trunc(t):<32}"
            print(row)

    combined_rows = [(t, get_top_n(all_norm, 'GAME_EMBEDDING_COMBINED', t)) for t in titles]
    print_table('combined embedding', combined_rows)
    if all_norm_id is not None:
        id_rows = [(t, get_top_n(all_norm_id, 'GAME_ID_EMBEDDING', t)) for t in titles]
        print_table('game ID embedding only', id_rows)


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint_path: str, checkpoint_dir: str):
    if checkpoint_path is not None:
        return checkpoint_path
    candidates = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'best_triple_full_softmax_*.pth')) +
        glob.glob(os.path.join(checkpoint_dir, 'best_*.pth')),
        key=os.path.getmtime, reverse=True
    )
    return candidates[0] if candidates else None


def _load_model_and_embeddings(checkpoint_path: str, fs: dict) -> tuple:
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    config = get_config()
    model = build_model(config, fs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print_model_summary(model)
    
    print("\nBuilding game embeddings ...")
    game_embeddings, all_ids, all_combined = build_game_embeddings(model, fs)
    
    print("Precomputing normalised embedding matrix ...")
    all_norm = F.normalize(all_combined, dim=1)
    all_id_embs = torch.cat([game_embeddings[iid]['GAME_ID_EMBEDDING'] for iid in all_ids], dim=0)
    all_norm_id = F.normalize(all_id_embs, dim=1)
    
    return model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id


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

def run_canary(data_dir: str = 'data', checkpoint_path: str = None, version: str = 'v1') -> None:
    from src.dataset import load_features
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        print("No checkpoint found.")
        return
    fs = load_features(data_dir, version)
    model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id = _load_model_and_embeddings(cp, fs)
    print("\n── Canary user evaluation ──")
    run_canary_eval(model, fs, all_combined, all_ids)

def run_probes(data_dir: str = 'data', checkpoint_path: str = None, version: str = 'v1') -> None:
    from src.dataset import load_features
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        print("No checkpoint found.")
        return
    fs = load_features(data_dir, version)
    model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id = _load_model_and_embeddings(cp, fs)
    
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
    probe_similar(game_embeddings, fs, all_ids, all_norm, PROBE_SIMILAR_TITLES, top_n=5, all_norm_id=all_norm_id)
