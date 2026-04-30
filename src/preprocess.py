"""
Stage 1 — Base Preprocessing
Outputs data/base_*.parquet.

Usage:
    python main.py preprocess games         # Step 1: build game corpus → base_games, base_game_tags, base_vocab
    python main.py preprocess interactions  # Step 2: process user items → base_interactions_read/labels
    python main.py preprocess               # Run both steps in order
"""
import ast
import gzip
import math
import os
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_INTERACTIONS_PER_GAME      = 10
MIN_PLAYTIME_PER_USER          = 5        # total hours across corpus games
MAX_PLAYTIME_PER_USER          = 10_000   # removes bots / outliers
MIN_HOURS_PER_GAME             = 0.1      # playtime_forever >= 6 minutes
MIN_TAG_COUNT                  = 50       # tag must appear in this many corpus games


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_gz(path: str):
    """Yield parsed dicts from a gzipped Python-literal JSONL file."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield ast.literal_eval(line)
            except Exception:
                continue


def _parse_year(release_date: str) -> str:
    """Extract year from release_date string like '2018-01-04'."""
    if not release_date:
        return '-1'
    try:
        return str(int(str(release_date).split('-')[0]))
    except (ValueError, IndexError, AttributeError):
        return '-1'


def _parse_price_bucket(price) -> int:
    """
    Bucket price into index:
      0=Free  1=<$5  2=$5-10  3=$10-20  4=$20-30  5=$30-40  6=$40-60  7=>$60  8=Unknown
    """
    if price is None:
        return 8
    if isinstance(price, str):
        s = price.lower().strip()
        if 'free' in s:
            return 0
        s = s.replace('$', '').strip()
        try:
            price = float(s)
        except ValueError:
            return 8
    try:
        p = float(price)
    except (TypeError, ValueError):
        return 8
    if p == 0:   return 0
    if p < 5:    return 1
    if p < 10:   return 2
    if p < 20:   return 3
    if p < 30:   return 4
    if p < 40:   return 5
    if p < 60:   return 6
    return 7


# ── Step 1: Games ─────────────────────────────────────────────────────────────

def run_games(data_dir: str = 'data') -> None:
    """
    Build game corpus + vocabulary.

    Pass 1: count per-game interactions from australian_users_items.json.gz and collect playtimes.
    Pass 2: read steam_games.json.gz, filter to corpus games, parse metadata.
    Writes base_games.parquet, base_game_tags.parquet, base_vocab.parquet.
    """
    os.makedirs(data_dir, exist_ok=True)

    items_path = os.path.join(data_dir, 'australian_users_items.json.gz')
    games_path = os.path.join(data_dir, 'steam_games.json.gz')

    # ── Pass 1: count per-game user interactions and collect playtimes ──
    print("Pass 1: scanning per-game interactions ...")
    game_user_counts: Counter = Counter()
    game_playtimes = defaultdict(list)
    total_users = 0
    for user in tqdm(_read_gz(items_path), desc="  users"):
        total_users += 1
        for item in user.get('items', []):
            hours = item.get('playtime_forever', 0) / 60
            if hours >= MIN_HOURS_PER_GAME:
                gid = str(item['item_id'])
                game_user_counts[gid] += 1
                game_playtimes[gid].append(hours)

    # Top 5 Valve/Gravity wells
    DENYLIST = {'730', '550', '620', '240', '4000'} 

    corpus_ids = {
        gid for gid, cnt in game_user_counts.items() 
        if cnt >= MIN_INTERACTIONS_PER_GAME and gid not in DENYLIST
    }

    # Calculate global medians for corpus games
    print("  Calculating global medians ...")
    game_medians = {
        gid: float(pd.Series(game_playtimes[gid]).median())
        for gid in corpus_ids
    }

    print(f"  Users scanned: {total_users:,}")
    print(f"  Games with any qualifying playtime: {len(game_user_counts):,}")
    print(f"  Corpus games (≥{MIN_INTERACTIONS_PER_GAME} users): {len(corpus_ids):,}")

    # ── Pass 2: read game metadata ──
    print("\nPass 2: reading game metadata ...")
    rows = []
    total_in_metadata = 0
    for game in tqdm(_read_gz(games_path), desc="  games"):
        gid = str(game.get('id', ''))
        total_in_metadata += 1
        if not gid or gid not in corpus_ids:
            continue
        genres = [g for g in (game.get('genres') or []) if g]
        tags   = [t for t in (game.get('tags')   or []) if t]
        rows.append({
            'item_id':      gid,
            'title':        game.get('app_name') or game.get('title', ''),
            'developer':    (game.get('developer') or '').strip(),
            'publisher':    (game.get('publisher') or '').strip(),
            'genres':       genres,
            'tags':         tags,
            'year':         _parse_year(game.get('release_date', '')),
            'price':        game.get('price'),
            'price_bucket': _parse_price_bucket(game.get('price')),
            'n_users':      game_user_counts.get(gid, 0),
            'median_hours': game_medians.get(gid, 0.0),
        })

    found_ids = {r['item_id'] for r in rows}
    missing   = corpus_ids - found_ids
    print(f"\n  Game records in metadata file: {total_in_metadata:,}")
    print(f"  Corpus games found in metadata: {len(rows):,}")
    if missing:
        print(f"  Corpus games missing from metadata (will be excluded): {len(missing):,}")

    games_df = pd.DataFrame(rows)
    games_df['price'] = games_df['price'].astype(str)  # mixed float/str — normalize for parquet

    # ── Tag scores ──
    print("\n── Building tag scores (TF-IDF with inverse-position TF) ──")
    tags_df = _build_game_tag_scores(games_df)

    # ── Vocabulary ──
    print("\n── Building vocabulary ──")
    vocab_df = _build_vocab(games_df, tags_df)

    # ── Write ──
    games_df.to_parquet(os.path.join(data_dir, 'base_games.parquet'),     index=False)
    tags_df.to_parquet( os.path.join(data_dir, 'base_game_tags.parquet'), index=False)
    vocab_df.to_parquet(os.path.join(data_dir, 'base_vocab.parquet'),     index=False)
    print(f"\n✓ Wrote base_games.parquet        ({len(games_df):,} games)")
    print(f"✓ Wrote base_game_tags.parquet    ({len(tags_df):,} games)")
    print(f"✓ Wrote base_vocab.parquet        ({len(vocab_df):,} entries)")


def _build_game_tag_scores(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-game TF-IDF tag scores.
      TF  = 1/(list_position + 1), normalized per game (earlier = more prominent tag)
      IDF = log(N / df),  df = number of corpus games containing the tag
    Only tags in >= MIN_TAG_COUNT corpus games are kept.
    """
    N = len(games_df)

    df_count: Counter = Counter()
    for tags in games_df['tags']:
        for tag in tags:
            df_count[tag] += 1

    valid_tags = {t for t, cnt in df_count.items() if cnt >= MIN_TAG_COUNT}
    print(f"  Total unique tags: {len(df_count):,}")
    print(f"  Tags in ≥{MIN_TAG_COUNT} games (kept): {len(valid_tags):,}")

    rows = []
    for _, game in games_df.iterrows():
        valid = [(i, t) for i, t in enumerate(game['tags']) if t in valid_tags]
        if not valid:
            rows.append({'item_id': game['item_id'], 'tag_names': [], 'scores': []})
            continue
        raw   = {t: 1.0 / (i + 1) for i, t in valid}
        total = sum(raw.values())
        names  = list(raw.keys())
        scores = [(raw[t] / total) * math.log(N / df_count[t]) for t in names]
        rows.append({'item_id': game['item_id'], 'tag_names': names, 'scores': scores})

    return pd.DataFrame(rows)


def _build_vocab(games_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build shared vocabulary table (type, index, value, extra).
    Genres, tags, years: sorted alphabetically/numerically.
    Developers: index 0 = __unknown__, real developers start at 1.
    Price buckets are fixed — not stored in vocab (hardcoded in model).
    """
    # Genres
    all_genres = set()
    for genres in games_df['genres']:
        all_genres.update(genres)
    genres_ordered = sorted(all_genres)

    # Tags (vocab = those that survived MIN_TAG_COUNT filter)
    valid_tags = set()
    for tag_names in tags_df['tag_names']:
        valid_tags.update(tag_names)
    tags_ordered = sorted(valid_tags)

    # Years
    years_ordered = sorted(set(games_df['year'].tolist()))

    # Developers (index 0 = __unknown__)
    devs = sorted({d for d in games_df['developer'].tolist() if d})
    developers_ordered = ['__unknown__'] + devs

    rows = []
    for i, g in enumerate(genres_ordered):
        rows.append({'type': 'genre', 'index': i, 'value': g, 'extra': ''})
    for i, t in enumerate(tags_ordered):
        rows.append({'type': 'tag', 'index': i, 'value': t, 'extra': ''})
    for i, y in enumerate(years_ordered):
        rows.append({'type': 'year', 'index': i, 'value': y, 'extra': ''})
    for i, d in enumerate(developers_ordered):
        rows.append({'type': 'developer', 'index': i, 'value': d, 'extra': ''})

    print(f"  Genres:     {len(genres_ordered):,}")
    print(f"  Tags:       {len(tags_ordered):,}")
    print(f"  Years:      {len(years_ordered):,}")
    print(f"  Developers: {len(developers_ordered):,}  (index 0 = __unknown__)")

    return pd.DataFrame(rows)


# ── Step 2: Interactions ──────────────────────────────────────────────────────

def run_interactions(data_dir: str = 'data') -> None:
    """
    Process user play histories into a flat interaction table.

    Requires base_games.parquet from run_games().
    Joins recommend signal from australian_user_reviews.json.gz where available.
    No timestamps in australian_users_items.json.gz — items kept in file order.
    The 90/10 train/eval split is NOT done here; features.py applies it once when
    building the feature store, so rollback dataset and offline eval use the same split.
    Writes base_interactions.parquet.
    """
    games_path   = os.path.join(data_dir, 'base_games.parquet')
    items_path   = os.path.join(data_dir, 'australian_users_items.json.gz')
    reviews_path = os.path.join(data_dir, 'australian_user_reviews.json.gz')

    if not os.path.exists(games_path):
        raise FileNotFoundError(
            f"{games_path} not found — run 'python main.py preprocess games' first"
        )

    games_df   = pd.read_parquet(games_path)
    corpus_ids = set(games_df['item_id'].tolist())
    print(f"Loaded corpus: {len(corpus_ids):,} games")

    # ── Load review signals ──
    print("\nLoading review signals from australian_user_reviews.json.gz ...")
    review_lookup: dict = {}  # (user_id, item_id) → recommend bool
    total_reviews = 0
    rec_true = rec_false = 0
    for user in tqdm(_read_gz(reviews_path), desc="  review users"):
        uid = str(user.get('user_id', ''))
        for rev in user.get('reviews', []):
            iid = str(rev.get('item_id', ''))
            if iid in corpus_ids:
                rec = rev.get('recommend')
                review_lookup[(uid, iid)] = rec
                total_reviews += 1
                if rec is True:  rec_true  += 1
                if rec is False: rec_false += 1

    print(f"  Corpus-game reviews loaded: {total_reviews:,}")
    if total_reviews:
        print(f"  recommend=True:  {rec_true:,}  ({100*rec_true/total_reviews:.1f}%)")
        print(f"  recommend=False: {rec_false:,}  ({100*rec_false/total_reviews:.1f}%)")

    # ── Stream user items ──
    print("\nProcessing user items ...")
    rows = []
    skipped_too_few  = 0
    skipped_too_many = 0
    skipped_too_small = 0

    for user in tqdm(_read_gz(items_path), desc="  users"):
        uid   = str(user.get('user_id', ''))
        items = user.get('items', [])

        corpus_items = []
        for item in items:
            iid   = str(item.get('item_id', ''))
            hours = item.get('playtime_forever', 0) / 60
            if iid in corpus_ids and hours >= MIN_HOURS_PER_GAME:
                corpus_items.append((iid, round(hours, 4), review_lookup.get((uid, iid))))

        total_hours = sum(h for _, h, _ in corpus_items)
        if total_hours < MIN_PLAYTIME_PER_USER:
            skipped_too_few += 1
            continue
        if total_hours > MAX_PLAYTIME_PER_USER:
            skipped_too_many += 1
            continue
        if len(corpus_items) < 2:
            skipped_too_small += 1
            continue

        for iid, hours, recommend in corpus_items:
            rows.append({'user_id': uid, 'item_id': iid, 'hours': hours, 'recommend': recommend})

    df = pd.DataFrame(rows)
    n_users = df['user_id'].nunique()
    print(f"\n  Surviving users:    {n_users:,}")
    print(f"  Skipped (< {MIN_PLAYTIME_PER_USER}h):   {skipped_too_few:,}")
    print(f"  Skipped (> {MAX_PLAYTIME_PER_USER:,}h): {skipped_too_many:,}")
    print(f"  Skipped (< 2 games): {skipped_too_small:,}")
    print(f"  Total interactions: {len(df):,}  (avg {len(df)/n_users:.1f} games/user)")

    hours_s = df['hours']
    print(f"\n  Playtime per interaction (hours) —"
          f"  median: {hours_s.median():.1f}"
          f"  mean: {hours_s.mean():.1f}"
          f"  p95: {hours_s.quantile(0.95):.1f}"
          f"  max: {hours_s.max():.1f}")

    games_per_user = df.groupby('user_id')['item_id'].count()
    print(f"  Games per user —"
          f"  median: {games_per_user.median():.0f}"
          f"  mean: {games_per_user.mean():.1f}"
          f"  max: {games_per_user.max()}")

    df.to_parquet(os.path.join(data_dir, 'base_interactions.parquet'), index=False)
    print(f"\n✓ Wrote base_interactions.parquet  →  {data_dir}/")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(data_dir: str = 'data', step: str = None) -> None:
    if step == 'games':
        run_games(data_dir)
    elif step == 'interactions':
        run_interactions(data_dir)
    else:
        run_games(data_dir)
        print()
        run_interactions(data_dir)
