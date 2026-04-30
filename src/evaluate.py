"""
Embedding probes and canary user evaluation V2 (V3: in-model genre/tag context).
"""
import glob
import os
from itertools import zip_longest
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GameRecommender
from src.train import build_model, get_config, load_config_for_checkpoint, print_model_summary


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

SIMULATED_FAV_LOG_HOURS             = 10.0   # weight for favorite games
SIMULATED_ANCHOR_LOG_HOURS          =  2.0   # weight for tag-based anchor games
SIMULATED_DISLIKE_LOG_HOURS         = 0.5    # weight for disliked games
ANCHORS_PER_TAG                     =  5
POPULARITY_ALPHA_INFERENCE_MULTIPLE =  2.0   # apply stronger debias at inference time than we did at training time


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

    # Re-apply item projection MLP + L2 normalize (matches model.item_embedding())
    concat_all = torch.cat([genre_all, tag_all, game_all, dev_all, year_all, price_all], dim=1)
    combined = []
    with torch.no_grad():
        for s in range(0, n_items, batch_size):
            combined.append(F.normalize(model.item_projection(concat_all[s:s + batch_size]), dim=-1))
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

    # Triple Pool Construction:
    liked_ids    = list(fav_idxs)
    disliked_ids = list(dis_idxs)
    full_ids     = list(fav_idxs) + list(anchor_idxs) + list(dis_idxs)

    # Playtime weights for full pool: fav=10.0, anchor=2.0, disliked=0.5, normalized to sum=1
    raw_pw = ([SIMULATED_FAV_LOG_HOURS]    * len(fav_idxs) +
              [SIMULATED_ANCHOR_LOG_HOURS] * len(anchor_idxs) +
              [SIMULATED_DISLIKE_LOG_HOURS] * len(dis_idxs))
    total_pw = sum(raw_pw) or 1.0
    full_playtime_weights = [w / total_pw for w in raw_pw]

    # Compute avg_log for the model to use in debiased affinity calculation
    total_items = len(fav_idxs) + len(anchor_idxs) + len(dis_idxs)
    avg_log = (len(fav_idxs)*SIMULATED_FAV_LOG_HOURS + 
               len(anchor_idxs)*SIMULATED_ANCHOR_LOG_HOURS + 
               len(dis_idxs)*SIMULATED_DISLIKE_LOG_HOURS) / max(total_items, 1)

    device = next(model.parameters()).device
    X_avg_log_t = torch.tensor([[avg_log]], dtype=torch.float32, device=device)
    
    def to_padded(ids):
        if not ids: ids = [model.game_pad_idx]
        return torch.tensor([ids], dtype=torch.long, device=device)

    # Playtime weight tensor: (1, len(full_ids)); pad to match full_ids length
    if full_playtime_weights:
        pw_t = torch.tensor([full_playtime_weights], dtype=torch.float32, device=device)
    else:
        pw_t = torch.zeros(1, 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        return model.user_embedding(X_avg_log_t, to_padded(liked_ids), to_padded(disliked_ids),
                                    to_padded(full_ids), pw_t)


def run_canary_eval(model: GameRecommender, fs: dict, all_combined: torch.Tensor, all_ids: list,
                    popularity_alpha: float = 0.0, temperature: float = None, top_n: int = 10) -> None:
    model.eval()
    device = next(model.parameters()).device

    # Scale bias to dot-product space. Training scores are (u·v)/temp - bias;
    # inference ranking is equivalent to u·v - temp*bias (multiply through by temp).
    if temperature is None:
        temperature = 0.5 / 512
    if popularity_alpha > 0 and 'game_interaction_counts' in fs:
        counts = torch.from_numpy(fs['game_interaction_counts']).to(device)

        pop_bias = temperature * (popularity_alpha * POPULARITY_ALPHA_INFERENCE_MULTIPLE) * torch.log1p(counts)  # (n_items,)
    else:
        pop_bias = None

    with torch.no_grad():
        for user_type in USER_TYPE_TO_FAVORITE_GAMES:
            user_emb = _build_user_embedding(model, fs, user_type)
            fav_titles = USER_TYPE_TO_FAVORITE_GAMES[user_type]
            dis_titles = USER_TYPE_TO_DISLIKED_GAMES.get(user_type, [])
            tag_names   = USER_TYPE_TO_TAGS.get(user_type, [])
            anchor_titles = _get_anchor_titles(fs, tag_names, exclude=set(fav_titles) | set(dis_titles))
            exclude_set   = set(fav_titles) | set(anchor_titles)

            raw_scores = (all_combined @ user_emb.T).squeeze(-1)
            if pop_bias is not None:
                raw_scores = raw_scores - pop_bias
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

def probe_genre(model: GameRecommender, genre, all_norm_genre: torch.Tensor,
                all_ids: list, fs: dict, top_n: int = 10) -> None:
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
        query = model.item_genre_tower(torch.tensor([ctx], dtype=torch.float32, device=device))
        query_norm = F.normalize(query, dim=1)
        sims = (all_norm_genre @ query_norm.T).squeeze(-1)

    print(f"\nTop-{top_n} games for genre '{' + '.join(genres)}':")
    seen = set()
    for idx in sims.argsort(descending=True).tolist():
        if len(seen) >= top_n: break
        title = fs['item_id_to_title'][all_ids[idx]]
        if title not in seen:
            seen.add(title)
            print(f"  {sims[idx].item():.4f}  {title}")


def probe_tag(tag_names: list, all_norm_tag: torch.Tensor, all_ids: list,
              game_embeddings: dict, fs: dict, top_n: int = 10, k_anchors: int = 5) -> None:
    valid_tags = [t for t in tag_names if t in fs['tag_to_i']]
    if not valid_tags:
        print(f"No tags from {tag_names} found.")
        return

    tag_matrix = fs['game_tag_matrix']
    item_to_idx = fs['item_to_idx']
    raw_scores = {iid: sum(float(tag_matrix[item_to_idx[iid], fs['tag_to_i'][t]]) for t in valid_tags)
                  for iid in fs['item_ids']}
    anchors = sorted(raw_scores, key=raw_scores.get, reverse=True)[:k_anchors]

    query = torch.stack([game_embeddings[iid]['GAME_TAG_EMBEDDING'].view(-1) for iid in anchors]).mean(dim=0)
    query_norm = F.normalize(query.unsqueeze(0), dim=1)
    sims = (all_norm_tag @ query_norm.T).squeeze(-1)

    print(f"\nTag anchors for {valid_tags}: {[fs['item_id_to_title'][iid] for iid in anchors]}")
    print(f"Top-{top_n} games:")
    anchor_set = set(anchors)
    seen = set()
    for idx in sims.argsort(descending=True).tolist():
        if len(seen) >= top_n: break
        iid   = all_ids[idx]
        title = fs['item_id_to_title'][iid]
        if title not in seen:
            seen.add(title)
            marker = ' [seed]' if iid in anchor_set else ''
            print(f"  {sims[idx].item():.4f}  {title}{marker}")


def probe_similar(game_embeddings: dict, fs: dict, all_ids: list, all_norm: torch.Tensor, titles: list, top_n: int = 5, all_norm_id: torch.Tensor = None, all_norm_tag: torch.Tensor = None) -> None:
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
    if all_norm_tag is not None:
        tag_rows = [(t, get_top_n(all_norm_tag, 'GAME_TAG_EMBEDDING', t)) for t in titles]
        print_table('tag embedding (32-dim)', tag_rows)


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

    config = load_config_for_checkpoint(checkpoint_path)
    model = build_model(config, fs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print_model_summary(model)
    
    print("\nBuilding game embeddings ...")
    game_embeddings, all_ids, all_combined = build_game_embeddings(model, fs)
    
    print("Precomputing normalised embedding matrices ...")
    all_norm = F.normalize(all_combined, dim=1)
    all_id_embs    = torch.cat([game_embeddings[iid]['GAME_ID_EMBEDDING']    for iid in all_ids], dim=0)
    all_tag_embs   = torch.cat([game_embeddings[iid]['GAME_TAG_EMBEDDING']   for iid in all_ids], dim=0)
    all_genre_embs = torch.cat([game_embeddings[iid]['GAME_GENRE_EMBEDDING'] for iid in all_ids], dim=0)
    all_norm_id    = F.normalize(all_id_embs,    dim=1)
    all_norm_tag   = F.normalize(all_tag_embs,   dim=1)
    all_norm_genre = F.normalize(all_genre_embs, dim=1)

    return model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id, all_norm_tag, all_norm_genre


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
    model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id, all_norm_tag, all_norm_genre = _load_model_and_embeddings(cp, fs)
    cp_config = load_config_for_checkpoint(cp)
    alpha = cp_config.get('popularity_alpha', 0.0)
    temperature = 0.5 / cp_config.get('minibatch_size', 512)
    print("\n── Canary user evaluation ──")
    print(f"  popularity_alpha={alpha} ({'applied' if alpha > 0 else 'disabled'})  temperature={temperature:.6f}")
    run_canary_eval(model, fs, all_combined, all_ids, popularity_alpha=alpha, temperature=temperature)

def run_probes(data_dir: str = 'data', checkpoint_path: str = None, version: str = 'v1') -> None:
    from src.dataset import load_features
    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        print("No checkpoint found.")
        return
    fs = load_features(data_dir, version)
    model, game_embeddings, all_ids, all_combined, all_norm, all_norm_id, all_norm_tag, all_norm_genre = _load_model_and_embeddings(cp, fs)

    print("\n── Genre probes ──")
    probe_genre(model, 'Action',         all_norm_genre, all_ids, fs)
    probe_genre(model, 'RPG',            all_norm_genre, all_ids, fs)
    probe_genre(model, 'Strategy',       all_norm_genre, all_ids, fs)
    probe_genre(model, 'Simulation',     all_norm_genre, all_ids, fs)
    probe_genre(model, ['Action', 'RPG'], all_norm_genre, all_ids, fs)

    print("\n── Tag probes ──")
    probe_tag(['FPS', 'Shooter'],          all_norm_tag, all_ids, game_embeddings, fs)
    probe_tag(['RPG', 'Open World'],       all_norm_tag, all_ids, game_embeddings, fs)
    probe_tag(['Strategy', 'Turn-Based'],  all_norm_tag, all_ids, game_embeddings, fs)
    probe_tag(['Horror', 'Survival Horror'], all_norm_tag, all_ids, game_embeddings, fs)
    probe_tag(['Co-op', 'Multiplayer'],    all_norm_tag, all_ids, game_embeddings, fs)
    probe_tag(['Rogue-like', 'Rogue-lite'], all_norm_tag, all_ids, game_embeddings, fs)
    probe_tag(['Puzzle'],                  all_norm_tag, all_ids, game_embeddings, fs)

    print("\n── Similarity probes ──")
    probe_similar(game_embeddings, fs, all_ids, all_norm, PROBE_SIMILAR_TITLES, top_n=5, all_norm_id=all_norm_id, all_norm_tag=all_norm_tag)
