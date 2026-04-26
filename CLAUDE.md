# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the Steam dataset. The model predicts game preferences via dot product of user and item embeddings.

This is a sibling project to:
- `/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model` — MovieLens, MSE objective
- `/Users/nickgreenquist/Documents/Book-Recommender-System-PyTorch-TwoTower-Model` — Goodreads, softmax objective (primary reference)

The architecture follows the same two-tower design as the book model. The book model is the primary reference — it uses the softmax objective (3× better than MSE), and adds a domain-specific embedding tower (author) not present in the movie model. Here, the analogous addition is a **developer embedding tower**.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: play history (playtime-weighted avg pooling of game embeddings), genre affinity, and timestamp. Any user can be represented at inference time with just a few games they've played — no retraining required.

**Rating signal: playtime hours.** Steam does not have star ratings. The `playtime_forever` field (total hours played per game, from `australian_user_items.json`) serves as the implicit feedback signal. Playtime is a strong preference proxy — a user who played 500 hours loves that game. Reviews provide a binary `recommend` signal as supplementary explicit feedback. The primary training signal is playtime.

## Running the Code

```bash
python main.py preprocess games          # Step 1: filter games → data/base_games.parquet
python main.py preprocess interactions   # Step 2: process user items → remaining parquets
python main.py preprocess                # Run both steps in order
python main.py explore                   # Explore user/game threshold distributions
python main.py features                  # Stage 2: base parquets → data/features_*.parquet
python main.py dataset                   # Stage 3: features → data/dataset_*_v1.pt
python main.py train                     # Stage 4: train, save checkpoints (softmax)
python main.py canary                    # Canary user recommendations (most recent checkpoint)
python main.py canary <path>             # Canary user recommendations (specific checkpoint)
python main.py probe                     # Embedding probes (most recent checkpoint)
python main.py probe <path>              # Embedding probes (specific checkpoint)
python main.py eval                      # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py eval <path>               # Same, specific checkpoint
python main.py export                    # Export serving artifacts for Streamlit
python main.py export <path>             # Export using specific checkpoint
python main.py                           # Run all stages in order
```

## Dataset

Raw data lives in `data/` (not in git). All files are gzipped JSONL. This project uses the UCSD Steam dataset (McAuley lab). Required files:

- `australian_users_items.json.gz` (**primary**) — 88,310 users: `user_id, items_count, steam_id, user_url, items[{item_id, item_name, playtime_forever, playtime_2weeks}]`
- `australian_user_reviews.json.gz` (**supplementary**) — 25,799 users: `user_id, user_url, reviews[{item_id, recommend, review, posted, helpful, funny}]`
- `steam_games.json.gz` (**item metadata**) — 32,135 games: `id, app_name, title, genres, tags, developer, publisher, release_date, price, discount_price, early_access, sentiment`

**Not used: `steam_reviews.json.gz`** — 7.8M individual V2 review records. Has `hours` and `text` but no `recommend` boolean, no stable `user_id` (uses `username`), and records are not grouped by user. Gives us nothing the other three files don't already provide better.

### Filtering thresholds

```python
MIN_INTERACTIONS_PER_GAME    = 10      # minimum users who have played the game
MIN_PLAYTIME_PER_USER        = 5       # minimum total hours played (across corpus games)
MAX_PLAYTIME_PER_USER        = 10_000  # cap on total hours (removes bots/outliers)
MIN_HOURS_PER_GAME           = 0.1     # minimum hours for a single game to count as an interaction
MIN_TAG_COUNT                = 50      # tag must appear in this many corpus games
MAX_ROLLBACK_EXAMPLES_PER_USER = 50   # rollback cap per user (see note below)
```

Games below `MIN_INTERACTIONS_PER_GAME` are filtered out entirely. Users outside the playtime bounds are dropped.

**Why MIN_INTERACTIONS_PER_GAME=10 (not 100):** Lowering to 10 nearly triples the recommendation corpus (6,258 games vs 2,509) with almost no cost — rollback examples change by less than 1,000. Games with few interactions have undertrained ID embeddings but their content towers (tags, genres, developer) still give them a reasonable representation.

**Why MAX_ROLLBACK_EXAMPLES_PER_USER=50 (not 10):** This dataset is much smaller than Goodreads — 66k users vs 229k, producing only ~575k rollback examples at cap=10 vs Goodreads' 4.7M. Cap=50 brings us to ~1.9M examples (0.4× Goodreads scale) without introducing bias: the user hour filter (5–10,000h) already excludes bots, the median user has 29 corpus games so cap=50 is rarely binding, and we're still far below the unlimited ceiling of 2.9M. The YouTube DNN paper caps to prevent power users from dominating — that concern is already handled by the hour ceiling here.

### Preprocessing pipeline

Two separate steps — run `preprocess games` first to inspect corpus size, then `preprocess interactions`.

**Step 1 (`preprocess games`)** — reads `steam_games.json.gz`, filters by interaction count (requires a first pass over user items to count per-game interactions), collects game metadata (genres, tags, developer, publisher, release year, price). Writes `base_games.parquet` and `base_game_tags.parquet`.

**Step 2 (`preprocess interactions`)** — processes `australian_users_items.json.gz`:
- Only keeps items with `playtime_forever >= MIN_HOURS_PER_GAME * 60` (playtime stored in minutes)
- Filters to corpus games only
- Counts valid playtime per user → builds `valid_users`
- Collects interactions for valid users
- Joins `recommend` signal and `posted` date from `australian_user_reviews.json.gz` where available
- **Sorts each user's items by `item_id` (Steam app ID) before writing** — Steam app IDs are assigned sequentially as games are added to Steam, so item_id order is a weak proxy for game release chronology. This is the only available ordering: `australian_users_items.json.gz` contains no buy date, install date, or first-play timestamp per game. Rollback examples therefore simulate "given older games in library, predict a newer release" rather than true temporal play order.

**Playtime normalization:** Raw `playtime_forever` is in minutes. Convert to hours. Apply log transform for rating weighting: `log(1 + hours)` — compresses the extreme tail (10,000-hour outliers) while preserving order.

### Tag signals

Steam community tags (`tags` field in `steam_games.json.gz`) are granular user-applied labels (e.g. `Roguelike`, `Co-op`, `Open World`, `Dark Souls-like`). Functionally identical to Goodreads shelves. The field is a plain ordered list — no per-tag counts are provided.

**Tag weighting:** Since `steam_games.json.gz` gives only a list (not counts), we treat list position as an implicit relevance signal. Tags listed first are most commonly applied. Weight by inverse position: `weight = 1 / (position + 1)`, then normalize per game. IDF is computed from corpus frequency (how many games carry each tag). TF-IDF = `(positional_weight) * log(N / df)`.

- Only tags appearing `>= MIN_TAG_COUNT` times across corpus games are kept
- Stored in `base_game_tags.parquet`

## Key Differences from Book/Movie Models

| Concept            | MovieLens                              | Goodreads                                    | Steam                                                         |
|--------------------|----------------------------------------|----------------------------------------------|---------------------------------------------------------------|
| Item ID column     | movieId                                | book_id                                      | item_id (app_id)                                              |
| User ID column     | userId                                 | user_id                                      | user_id                                                       |
| Rating signal      | 0.5–5.0 star ratings                   | 1–5 integer ratings                          | `playtime_forever` (minutes) — log-transformed to hours       |
| Explicit feedback  | Star rating                            | Star rating                                  | `recommend` boolean (supplementary)                           |
| Timestamp          | Unix timestamp int                     | `read_at` → `date_updated` → `date_added`    | Not available in user_items — omit or use review `posted`     |
| Year               | Parsed from title (YYYY) regex         | `publication_year` field                     | Parsed from `release_date` string                             |
| Genres             | Pipe-separated string                  | Curated label dict (vote counts)             | `genres` list — broad labels (Action, RPG, Strategy…)         |
| Tags               | Genome scores (dense ML)               | `popular_shelves` (sparse user counts)       | `tags` list (ordered, no counts) — IDF from corpus frequency  |
| Domain tower       | None                                   | Author embedding                             | Developer embedding                                           |
| Price              | Not available                          | Not available                                | `price` field — bucketed into embedding                       |

## Model Architecture

Two-tower design with dot product prediction. Mirrors the book model with **developer embedding** in place of author embedding.

```
User Tower:
  playtime_weighted_avg_pool(item_embeddings[play_history])  → history_emb   (item_id_embedding_size=32)
  user_genre_tower([avg_playtime_per_genre | play_frac])     → genre_emb     (user_genre_embedding_size=32)
  concat → 64-dim
  user_projection(Linear proj_hidden=256 → ReLU → Linear output_dim=128) → 128-dim

Item Tower:
  item_genre_tower(genre_onehot)          → item_genre_emb   (item_genre_embedding_size=8)
  item_tag_tower(tfidf_tag_scores)        → item_tag_emb     (tag_embedding_size=16)
  item_embedding_tower(item_id)           → item_emb         (item_id_embedding_size=32)   [shared with user history pool]
  developer_tower(developer_idx)          → item_dev_emb     (developer_embedding_size=12)
  year_embedding_tower(release_year)      → year_emb         (item_year_embedding_size=8)
  price_embedding_tower(price_bucket)     → price_emb        (price_embedding_size=4)
  concat → 80-dim
  item_projection(Linear proj_hidden=256 → ReLU → Linear output_dim=128) → 128-dim

Prediction: dot_product(user_projection_out, item_projection_out)
```

**Why the projection MLP is required:** A plain concat fed directly into a dot product can only learn additive combinations of the individual sub-embeddings. It cannot model interactions between signals — things like "RPGs from Japanese developers" (genre × developer) or "price sensitivity varies by how many games you've played" (price × history depth) require nonlinearity to learn. Each tower concatenates its sub-embeddings and passes them through a 2-layer MLP before the dot product. Only the final `output_dim` (128) needs to match across towers — the internal concat sizes are decoupled.

**Initialization:** Sub-tower linear layers use `gain=0.1`. Projection layers are re-initialized separately to `gain=1.0` after the rest of the model. Embedding tables use `gain=0.01`. Using `gain=0.01` for the projection layers causes vanishing gradients — the projection adds two more near-zero-gain layers on top of already-small sub-tower outputs, making dot products exactly zero and preventing learning.

### Shared towers

- `item_embedding_tower` — shared between item side and user history avg pool

### Embedding sizes

```python
item_id_embedding_size      = 32   # shared: user history pool + item tower (must match)
user_genre_embedding_size   = 32   # user only
item_genre_embedding_size   = 8    # item only
tag_embedding_size          = 16   # item only
developer_embedding_size    = 12   # item only
item_year_embedding_size    = 8    # item only
price_embedding_size        = 4    # item only

proj_hidden  = 256
output_dim   = 128   # only this must match across towers

# user concat: 32 + 32 = 64  → proj → 128
# item concat: 8 + 16 + 32 + 12 + 8 + 4 = 80  → proj → 128
```

Only `item_id_embedding_size` and `output_dim` are constrained: the former because it is a shared parameter, the latter because the dot product requires equal dimensions.

### Developer tower details

- One developer embedding per game (primary developer only for multi-developer games)
- `nn.Embedding(n_developers + 1, developer_embedding_size)` with padding index = `n_developers`
- Developer index 0 = `__unknown__`
- Developer signal lives on the **item side only** (same decision as author in book model)

### Price tower details

- `original_price` is a float (dollars). Bucket into ~10 price bins: Free, <$5, $5–$10, $10–$20, $20–$30, $30–$40, $40–$60, >$60, Unknown.
- `nn.Embedding(n_price_buckets, price_embedding_size)`
- Price is a real content signal for games — free-to-play vs. premium vs. AAA is a meaningful taste dimension.

## Training Details

**What the model predicts:** "Given this user's play history, which game do they play next?" — a ranking problem, not a regression. Playtime is never a prediction target.

**Playtime role:** Used only as a weighting signal inside the user tower's history avg pool. Each played game's embedding is weighted by `log(1 + hours)` normalized across the user's history — games with 500 hours pull the user embedding harder than games with 2 hours. Playtime does not appear in the loss, the item tower, or anywhere else.

**Note on YouTube DNN:** The YouTube paper predicts watch time, but that is their *ranking* model (stage 2), not the two-tower candidate generation model (stage 1) that we are replicating. Our corpus is ~5,400 games — small enough to score all items exactly at inference time with no ANN needed. A separate ranking stage is unnecessary.

**Primary: In-batch negatives softmax** — same objective as the book model.

- **Loss**: cross-entropy over in-batch negatives. B×B score matrix, diagonal = correct targets.
- **Dataset**: rollback examples — for each play event, context = all prior plays. Up to `MAX_ROLLBACK_EXAMPLES_PER_USER=50` examples per user sampled randomly.
- **Playtime weighting**: `weights = log(1 + hours) / sum(log(1 + hours))` per user → `history_emb = sum(weights[i] * item_emb[i])`.
- **Optimizer**: Adam, `lr=0.001`, `weight_decay=1e-5`
- **Batch size**: 512
- **Temperature**: 0.05
- **Steps**: 150,000
- **`F.normalize` must NOT be used in training.** Raw dot products throughout — matches YouTube DNN paper. See book model CLAUDE.md for details on why cosine similarity causes train/inference mismatch.

**Timestamp:** Steam `australian_user_items.json` does not include timestamps per game interaction. If timestamps are unavailable, omit the `timestamp_embedding_tower` entirely from the user tower (adjust dims accordingly). The review `posted` date is available but only for reviewed games — too sparse to use as a general timestamp.

## Canary Users for Eval

All titles verified against corpus. Genres/tags verified against base_vocab.parquet.

```python
USER_TYPE_TO_FAVORITE_GENRES = {
    'RPG Lover':      ['RPG'],
    'FPS Lover':      ['Action'],
    'Strategy Lover': ['Strategy'],
    'Indie Lover':    ['Indie'],
}
USER_TYPE_TO_FAVORITE_GAMES = {
    'RPG Lover':      ['The Witcher 2: Assassins of Kings Enhanced Edition',
                       'DARK SOULS™: Prepare To Die™ Edition',
                       'Divinity: Original Sin (Classic)',
                       'Fallout: New Vegas', 'Mass Effect 2'],
    'FPS Lover':      ['Counter-Strike: Global Offensive', 'DOOM', 'Left 4 Dead 2',
                       'PAYDAY 2', 'BioShock Infinite'],
    'Strategy Lover': ["Sid Meier's Civilization® V", 'XCOM: Enemy Unknown',
                       'Total War™: ROME II - Emperor Edition',
                       'Crusader Kings II', 'Company of Heroes 2'],
    'Indie Lover':    ['Terraria', 'FTL: Faster Than Light',
                       'The Binding of Isaac: Rebirth', 'Rogue Legacy', 'Spelunky'],
}
USER_TYPE_TO_TAGS = {
    'RPG Lover':      ['RPG', 'Open World', 'Story Rich', 'Character Customization'],
    'FPS Lover':      ['FPS', 'Shooter', 'Action', 'Multiplayer'],
    'Strategy Lover': ['Strategy', 'Turn-Based', '4X', 'Tactical'],
    'Indie Lover':    ['Indie', 'Rogue-like', 'Platformer', 'Pixel Graphics'],
}
```

Canary users are synthetic — no real play timestamps (timestamp tower omitted in this model).

## Offline Evaluation

`python main.py eval [checkpoint_path]` — Recall@K, NDCG@K, Hit Rate@K, MRR at K = 1, 5, 10, 20, 50.

**Protocol:** User-level train/val split. 90% of users are train-only; the remaining 10% are held out entirely and never seen during training.

- **Train users**: all interactions used for rollback training examples — no within-user cut needed, no leakage possible.
- **Val users**: all interactions used for rollback eval examples — no within-user cut needed either, since none of their data was ever in training. Any rollback pair from a val user is valid.

At eval time, val user rollback examples are generated the same way as training examples: for each (context, target) pair, rank all corpus games and measure whether the target appears in top K.

**Why no within-user 90/10 split:** The within-user split exists in the book model to prevent leakage — you can't let the model train on a future read and also use it as an eval label. That concern disappears entirely when val users are held out at the user level: the model has seen zero of their interactions, so there is nothing to leak.

**Why not per-user 90/10 split (book model protocol):** That protocol tests "predict future interactions for users the model has already partially seen." For a no-user-ID model this is a weaker test — the model implicitly learned each user's taste profile from their 90% training context. Held-out users are a stricter and more realistic measure of cold-start generalization, which is the actual inference scenario.

## Relationship to Book Repo

| File            | Status                                                             |
|-----------------|--------------------------------------------------------------------|
| `preprocess.py` | Rewritten — Steam schema, playtime signal, two-step pipeline       |
| `features.py`   | Adapted — same logic, game column names, developer feature added   |
| `dataset.py`    | Adapted — same logic, playtime-weighted rollback examples          |
| `model.py`      | Extended — developer + price towers; timestamp may be omitted      |
| `train.py`      | Identical                                                          |
| `evaluate.py`   | Adapted — canary dicts use game titles and Steam tags              |
| `export.py`     | Adapted — exclude large buffers from model.pth                     |

## Serving / Export Notes

`game_tag_matrix` (registered buffer, n_games × n_tags × float32) and `game_dev_idx` are excluded from `model.pth` and stored in `feature_store.pt`. The Streamlit app reconstructs `GameRecommender` using the buffers from `feature_store.pt` and loads weights with `strict=False`.

Serving artifacts (generated by `python main.py export`):
- `serving/model.pth` — weights only (buffers excluded)
- `serving/game_embeddings.pt` — pre-computed per-game embeddings dict
- `serving/feature_store.pt` — vocab maps, game metadata, buffers, model config

## Streamlit App

`streamlit run streamlit_app.py` — four tabs:
- **Recommend** — pick games you've played → dot-product ranked recommendations
- **Similar** — pick a game → cosine-nearest games in combined embedding space
- **Genres** — pick genres → cosine-nearest games in genre embedding space
- **Tags** — pick Steam tags → anchor-averaged query → cosine-nearest in tag space

Steam cover images fetched live from `cdn.cloudflare.steamstatic.com` using the Steam app ID.

## Planned Work: Item-Pool History (ipool)

Replace the user tower's raw ID-embedding avg pool with a pool over **full projected item embeddings** — i.e., pool the 128-dim output of `item_embedding()` instead of the 32-dim raw ID lookup.

**Motivation:** The current user history avg pool only sees the 32-dim item ID signal. Pooling the full 128-dim projected embedding (which already fuses genre, tags, developer, year, price) gives the user tower a much richer taste representation — "this user's history skews RPG + story-rich + single-player, at ~$20–$40" rather than just opaque learned IDs.

### What changes

| | Old (prod, `gpool`) | New (`ipool_gpool`) |
|---|---|---|
| History pool input | `item_embedding_lookup(hist_idx)` → 32-dim | `item_embedding(...)` → 128-dim |
| User concat | 32 + 32 = 64 → proj → 128 | 128 + 32 = 160 → proj → 128 |
| item_id_embedding_size constraint | Must match user pool dim | Still must match (shared lookup inside item_embedding) |

Only `user_concat_dim` changes. Everything else stays the same.

### Implementation plan

**`src/model.py`**
- Add `use_item_pool_for_history=False` flag to `__init__` (default False = backward compat).
- Register non-persistent buffers needed to call `item_embedding()` from the user side: `genre_buf` (n_games+1, n_genres), `tag_buf` (n_games+1, n_tags), `year_buf` (n_games+1,), `price_buf` (n_games+1,), `dev_buf` (n_games+1,). These are indexed by game embedding index and already available in feature store; pad row = zeros/padding index.
- When `use_item_pool_for_history=True`: in `user_embedding()`, look up `item_embedding(genre_buf[hist], year_buf[hist], hist_idx, dev_buf[hist], price_buf[hist])` for each batch, weighted avg pool over 128-dim outputs.
- Update `user_concat_dim` to `output_dim + user_genre_embedding_size` when flag is True.

**`src/train.py`**
- Pass `use_item_pool_for_history=True` to `build_model()`.
- Populate the new buffers from `fs` before passing to `GameRecommender`.
- Checkpoint naming: `best_ipool_gpool_softmax_*.pth`.

**`src/evaluate.py`**
- `_resolve_checkpoint`: add `ipool_gpool` globs before `proj_softmax` globs.
- `_resolve_config()`: detect new `user_proj_in` size (160 vs 64) to auto-set `use_item_pool_for_history`.
- `_build_user_embedding()`: no change needed — model handles it internally via the flag.

**Files NOT to touch yet:** `streamlit_app.py`, `src/export.py`, serving artifacts. Wait until training is complete and canary results verified.

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

For changes that require retraining to validate (hyperparameters, optimizer, scheduler, loss, dataset logic): write the code, then stop. Do not commit until the user has run training and confirmed the results look better.
