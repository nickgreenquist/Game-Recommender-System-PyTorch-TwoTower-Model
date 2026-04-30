# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the Steam dataset. The model predicts game preferences via dot product of user and item embeddings.

This is a sibling project to:
- `/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model` — MovieLens, MSE objective
- `/Users/nickgreenquist/Documents/Book-Recommender-System-PyTorch-TwoTower-Model` — Goodreads, softmax objective (primary reference)

The architecture follows the same two-tower design as the book model. The book model is the primary reference — it uses the softmax objective (3× better than MSE), and adds a domain-specific embedding tower (author) not present in the movie model. Here, the analogous addition is a **developer embedding tower**.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: four behavior-partitioned play history pools (Liked, Disliked, Full, Playtime-weighted Full), rolling genre affinity, and rolling tag affinity. Any user can be represented at inference time with just a few games they've played — no retraining required.

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

**Valve DENYLIST:** Five ultra-popular titles are hard-filtered from the corpus regardless of interaction count: `{'730', '550', '620', '240', '4000'}` (CS:GO, Left 4 Dead 2, Portal, Counter-Strike, Garry's Mod). These appeared in nearly every user's history and were trivially easy prediction targets that inflated Recall@K metrics and polluted cross-genre recommendations. Removed in `preprocess.py` Pass 1.

**Why MIN_INTERACTIONS_PER_GAME=10 (not 100):** Lowering to 10 nearly triples the recommendation corpus (6,258 games vs 2,509) with almost no cost — rollback examples change by less than 1,000. Games with few interactions have undertrained ID embeddings but their content towers (tags, genres, developer) still give them a reasonable representation.

**Why MAX_ROLLBACK_EXAMPLES_PER_USER=50 (not 10):** This dataset is much smaller than Goodreads — 66k users vs 229k, producing only ~575k rollback examples at cap=10 vs Goodreads' 4.7M. Cap=50 brings us to ~1.4M examples per shuffle pass; with N_SHUFFLES=3 this yields ~4.3M training examples. The YouTube DNN paper caps to prevent power users from dominating — that concern is already handled by the hour ceiling here.

**N_SHUFFLES=3:** Training data is augmented by running `_build_rollback_dataset` 3× per user with independent random shuffles of each user's history. Each shuffle produces genuinely different (context, target) pairs while preserving varied context lengths (early rollback positions still have 1–2 game contexts). Val and offline eval always use `n_shuffles=1` — a single clean pass per user so metrics are not inflated by repeated targets.

**Quality label filter:** Target games must have `hours > 0.5`. Low-quality interactions (very short playtime) stay in the history pools as signal but are never training targets.

### Preprocessing pipeline

Two separate steps — run `preprocess games` first to inspect corpus size, then `preprocess interactions`.

**Step 1 (`preprocess games`)** — reads `steam_games.json.gz`, filters by interaction count (requires a first pass over user items to count per-game interactions), collects game metadata (genres, tags, developer, publisher, release year, price). Also computes **global per-game median playtime** (stored as `median_hours` in `base_games.parquet`) — used by `dataset.py` for Liked/Disliked pool partitioning. Writes `base_games.parquet` and `base_game_tags.parquet`.

**Step 2 (`preprocess interactions`)** — processes `australian_users_items.json.gz`:
- Only keeps items with `playtime_forever >= MIN_HOURS_PER_GAME * 60` (playtime stored in minutes)
- Filters to corpus games only
- Counts valid playtime per user → builds `valid_users`
- Collects interactions for valid users
- Joins `recommend` signal and `posted` date from `australian_user_reviews.json.gz` where available
- **User play history order is not sorted** — `australian_users_items.json.gz` contains no buy date, install date, or first-play timestamp per game. item_id order (Steam app ID ≈ release date) is a spurious proxy that biases targets toward newer games. History is left in ingestion order; `dataset.py` shuffles each user's history with a seeded RNG before building rollback examples, so rollback simulates "given a random subset of games this user plays, predict another" rather than "predict newer releases."

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

Two-tower design with dot product prediction. V2 adds triple history pools, a user tag tower, and shallow sum pooling. A 4th playtime-weighted pool captures intensity of preference beyond the binary liked/disliked split.

```
User Tower:
  liked_pool:    sum(item_id_emb[liked_ids])                              → 32-dim
  disliked_pool: sum(item_id_emb[disliked_ids])                          → 32-dim
  full_pool:     sum(item_id_emb[full_ids])                              → 32-dim
  playtime_pool: sum(item_id_emb[full_ids] * w) where w=log(1+h)/Σlog(1+h) → 32-dim
  user_genre_tower([debiased_avg_log | play_frac]) → genre_emb           (32-dim)
  user_tag_tower(rolling_tag_sum)                  → tag_emb             (32-dim)
  concat → 192-dim
  user_projection(Linear 256 → ReLU → Linear 128) → 128-dim

Item Tower:
  item_genre_tower(genre_onehot)          → item_genre_emb   (item_genre_embedding_size=8)
  item_tag_tower(tfidf_tag_scores)        → item_tag_emb     (tag_embedding_size=32)
  item_embedding_tower(item_id)           → item_emb         (item_id_embedding_size=32)
  developer_tower(developer_idx)          → item_dev_emb     (developer_embedding_size=12)
  year_embedding_tower(release_year)      → year_emb         (item_year_embedding_size=8)
  price_embedding_tower(price_bucket)     → price_emb        (price_embedding_size=4)
  concat → 96-dim
  item_projection(Linear 256 → ReLU → Linear 128) → 128-dim

Prediction: dot_product(user_projection_out, item_projection_out)
```

**Shallow history pooling:** User history pools sum raw 32-dim `item_id` embeddings directly — they do NOT pass through the full item tower. This decouples user history from item tower capacity and is faster. No LayerNorm after pooling — industry standard (YouTube DNN, TikTok); the projection MLP learns the right scale.

**Four pool partitioning** (computed per rollback position in `dataset.py`):
- **Liked:** `recommend==True` OR `hours >= game_median` OR `hours >= user_rolling_median × 2`
- **Disliked:** `recommend==False` OR `(0.1 < hours < 1.0)` OR `hours <= user_rolling_median / 2`
- **Full:** all context items (most recent MAX_HISTORY_LEN=50) — equal-weight sum pool
- **Playtime-weighted Full:** same items as Full; weighted sum where each item's weight is `log(1+hours)` normalized by the context total — captures intensity of engagement beyond the binary liked/disliked split

Items can appear in both Liked and Disliked simultaneously. `game_median` is the global per-game median playtime (computed in `preprocess.py` Pass 1 and stored in `base_games.parquet`); `user_rolling_median` is computed at each rollback step from context hours so far.

**In-model genre/tag context (V3):** Genre and tag context are computed inside `user_embedding()` from `game_genre_matrix` and `game_tag_matrix` registered buffers using the `X_hist_full` indices. `dataset.py` no longer computes rolling genre/tag context — it only provides `X_user_avg_log` (per-user average log playtime scalar, used for genre debiasing inside the model). `features.py` similarly no longer builds the per-user genre context matrix.

**`user_embedding()` signature (V3):**
```python
def user_embedding(self, X_user_avg_log,   # (B, 1)  per-user avg log-playtime
                   X_hist_liked,            # (B, MAX_HISTORY_LEN) pre-padded int64
                   X_hist_disliked,         # (B, MAX_HISTORY_LEN) pre-padded int64
                   X_hist_full,             # (B, MAX_HISTORY_LEN) pre-padded int64
                   X_hist_playtime_weights) # (B, MAX_HISTORY_LEN) pre-padded float32
```

**`item_embedding()` signature (V3):**
```python
def item_embedding(self, target_year_idx, target_game_idx, target_dev_idx, target_price)
# genre looked up internally: self.game_genre_matrix[target_game_idx]
```

**Registered buffers on model:** `game_tag_matrix` (n_games+1, n_tags) and `game_genre_matrix` (n_games+1, n_genres). Both excluded from `model.pth` and stored in `feature_store.pt`.

**Why the projection MLP is required:** A plain concat fed directly into a dot product can only learn additive combinations. Nonlinearity is needed to model interactions (e.g. "RPGs from Japanese developers", "price sensitivity varies by history depth"). Each tower passes its concat through a 2-layer MLP. Only `output_dim=128` must match across towers.

**Initialization:** Sub-tower linear layers use `gain=0.1`. Projection layers re-initialized to `gain=1.0`. Embedding tables use `gain=0.01`. Using `gain=0.01` for projections causes vanishing gradients — the projection layers go on top of already-small sub-tower outputs.

### Shared embedding

- `item_embedding_lookup` (32-dim) — shared between the item tower and all four user history pools. Only the raw lookup is shared; the user pools sum/average it directly, while the item tower passes it through `item_embedding_tower` (Linear 32→32 → ReLU).

### Embedding sizes

```python
item_id_embedding_size      = 32   # shared: all 4 user history pools + item tower
user_genre_embedding_size   = 32   # user only
user_tag_embedding_size     = 32   # user only
item_genre_embedding_size   = 8    # item only
tag_embedding_size          = 32   # item only (was 16 in V1 — increased, tags are rich signal)
developer_embedding_size    = 12   # item only
item_year_embedding_size    = 8    # item only
price_embedding_size        = 4    # item only

proj_hidden  = 256
output_dim   = 128   # only this must match across towers

# user concat: 32+32+32+32 (pools) + 32 (genre) + 32 (tag) = 192 → proj → 128
# item concat: 8+32+32+12+8+4 = 96 → proj → 128
```

Only `item_id_embedding_size` and `output_dim` are constrained across towers.

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

**What the model predicts:** "Given a random subset of this user's play history, which game do they also play?" — a ranking problem, not a regression. Playtime is never a prediction target.

**Playtime role in V3:** Playtime is used to partition history into Liked/Disliked pools and to build the playtime-weighted Full pool. It is also used as the per-user avg_log scalar passed to `user_embedding()` for in-model genre debiasing. Playtime does not appear in the loss or item tower.

**Note on YouTube DNN:** The YouTube paper predicts watch time, but that is their *ranking* model (stage 2), not the two-tower candidate generation model (stage 1) that we are replicating. Our corpus is ~5,400 games — small enough to score all items exactly at inference time with no ANN needed. A separate ranking stage is unnecessary.

**Full softmax over entire corpus:**

- **Loss**: cross-entropy over all ~5,437 items (Valve DENYLIST applied). Every step scores the full corpus.
- **Dataset**: rollback examples with N_SHUFFLES=3 → ~4.3M training examples. 9-tuple of pre-padded tensors (no variable-length lists).
- **Optimizer**: Adam, `lr=0.001`, `weight_decay=0.0`, `eps=1e-6`
- **Scheduler**: CosineAnnealingLR, `T_max=50_000`, `eta_min=1e-4` (floor prevents LR going to zero)
- **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` before each optimizer step
- **Batch size**: 512
- **Temperature**: `0.5 / minibatch_size = 0.000977`
- **Steps**: 50,000
- **Val eval**: fixed 8,192 val examples sampled once at run start — same indices every log step so val loss is comparable across steps (not a fresh random sample each time)
- **Both towers use `F.normalize`** at the end of their projection MLPs — training and inference use cosine similarity consistently. The train/inference mismatch concern (from book model) does not apply here because normalization is applied in both settings.

**Popularity logit adjustment — bias/temperature ordering is critical (Menon et al. 2021).**

The training score formula is:
```python
scores = (U @ V_all.T) / temperature - popularity_bias   # CORRECT
# NOT: (U @ V_all.T - popularity_bias) / temperature    # WRONG — different scale at inference
```

The bias is subtracted **after** dividing by temperature, so it lives in temperature-scaled logit space (magnitudes ~1024 for temperature=0.001). At inference, dot products are in `[-1, 1]` (L2-normalized outputs). Applying the raw `popularity_bias` directly at inference would make it 1000× too large and collapse all users to the same ranking (popularity-sorted only).

**Correct inference formula:**
```python
# Multiply through by temperature to convert training formula to dot-product space:
# training rank: (u·v)/temp - bias  →  inference rank: u·v - temp*bias
pop_bias_inference = temperature * alpha * torch.log1p(counts)
scores = (user_emb @ item_embs.T) - pop_bias_inference
```

**Current implementation uses `log1p(count)` with alpha=0.4.** Valve mega-popular titles (CS:GO, Garry's Mod, L4D2) are hard-filtered from the corpus via DENYLIST in `preprocess.py` (`{'730', '550', '620', '240', '4000'}`), so the popularity bias is only needed to suppress moderately popular games, not mega-popular ones.

**Inference multiplier:** `evaluate.py` and `streamlit_app.py` apply `POPULARITY_ALPHA_INFERENCE_MULTIPLE = 2.0` at serving time, making the effective inference bias `temperature * alpha * 2.0 * log1p(counts)`. `offline_eval.py` uses 1× (matches training) to keep metrics comparable across checkpoints.

Temperature and alpha must be read from the checkpoint's config sidecar (`_config.json`) via `load_config_for_checkpoint()`, not hardcoded.

**Timestamp:** Steam `australian_user_items.json` does not include timestamps per game interaction. Timestamp tower omitted. The review `posted` date is available but only for reviewed games — too sparse.

## Canary Users for Eval

Nine synthetic user types defined in `src/evaluate.py`. All titles verified against corpus. Note that Valve titles (CS:GO, L4D2, etc.) are excluded from the corpus and therefore absent from favorite game lists.

User types: Western RPG Lover, JRPG Lover, FPS Lover, Civ Lover, Indie Lover, Racing Lover, Fighting Lover, Survival Lover, Management Lover.

Each user type has:
- `USER_TYPE_TO_FAVORITE_GAMES` — seed games (high simulated playtime weight = 10.0)
- `USER_TYPE_TO_TAGS` — tags used to find 5 anchor games per tag (anchor weight = 2.0)
- `USER_TYPE_TO_DISLIKED_GAMES` — currently empty dict (all entries commented out)

`POPULARITY_ALPHA_INFERENCE_MULTIPLE = 2.0` is applied during canary scoring. Canary users are synthetic — no real play timestamps.

## Offline Evaluation

`python main.py eval [checkpoint_path]` — Recall@K, NDCG@K, Hit Rate@K, MRR at K = 1, 5, 10, 20, 50.

### Current results

**V4 PROD** (5,437-game corpus — Valve titles removed, no LayerNorm after pools):

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0286 | 0.0286 |
| 5 | 0.0888 | 0.0586 |
| 10 | 0.1422 | 0.0757 |
| 20 | 0.2293 | 0.0976 |
| 50 | 0.3949 | 0.1303 |

MRR: 0.0710 (random: 0.0017)

**V3 PROD** (5,437-game corpus — Valve titles removed, with LayerNorm after pools):

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0278 | 0.0278 |
| 5 | 0.0882 | 0.0581 |
| 10 | 0.1428 | 0.0756 |
| 20 | 0.2287 | 0.0971 |
| 50 | 0.3944 | 0.1299 |

MRR: 0.0706 (random: 0.0017)

**V2 PROD** (5,442-game corpus — Valve titles included):

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0389 | 0.0389 |
| 5 | 0.1138 | 0.0767 |
| 10 | 0.1743 | 0.0962 |
| 20 | 0.2602 | 0.1177 |
| 50 | 0.4256 | 0.1504 |

MRR: 0.0875 (random: 0.0017)

**Why V3 metrics are lower and not directly comparable:** Ultra-popular Valve titles (CS:GO, Garry's Mod, Left 4 Dead 2) appeared in nearly every val user's history and were trivially easy prediction targets — any model ranks them top-5 for most users, inflating V2 Recall@K. Removing them makes the eval strictly harder: every target requires genuine taste modeling. V3 canary quality is substantially better — cross-genre Valve pollution eliminated, per-genre coherence improved across all tested user types.

**Protocol:** User-level train/val split. 90% of users are train-only; the remaining 10% are held out entirely and never seen during training.

- **Train users**: all interactions used for rollback training examples — no within-user cut needed, no leakage possible.
- **Val users**: all interactions used for rollback eval examples — no within-user cut needed either, since none of their data was ever in training. Any rollback pair from a val user is valid.

At eval time, val user rollback examples are generated fresh (not from the saved dataset) using `_build_rollback_dataset` with `n_shuffles=1`. For each (context, target) pair, all corpus games are ranked and whether the target appears in top K is measured. Results are written to `eval_results/<checkpoint_stem>.txt`.

**Why no within-user 90/10 split:** The within-user split exists in the book model to prevent leakage — you can't let the model train on a future read and also use it as an eval label. That concern disappears entirely when val users are held out at the user level: the model has seen zero of their interactions, so there is nothing to leak.

**Why not per-user 90/10 split (book model protocol):** That protocol tests "predict future interactions for users the model has already partially seen." For a no-user-ID model this is a weaker test — the model implicitly learned each user's taste profile from their 90% training context. Held-out users are a stricter and more realistic measure of cold-start generalization, which is the actual inference scenario.

## Relationship to Book Repo

| File            | Status                                                                              |
|-----------------|-------------------------------------------------------------------------------------|
| `preprocess.py` | Rewritten — Steam schema, playtime signal, two-step pipeline, global game medians, Valve DENYLIST |
| `features.py`   | Adapted — game column names, developer feature, avg_log_playtime per user (genre context removed) |
| `dataset.py`    | Rewritten (V3) — pre-allocated numpy arrays, pre-padded tensors, 9-tuple output, genre/tag moved to model |
| `model.py`      | Extended (V3) — 4 pools, in-model genre/tag context, game_genre_matrix buffer, F.normalize on both towers |
| `train.py`      | Extended — full softmax, gradient clipping, cosine schedule, checkpoint config sidecar |
| `evaluate.py`   | Adapted — 9 canary user types, POPULARITY_ALPHA_INFERENCE_MULTIPLE, new signatures |
| `offline_eval.py` | Adapted — V3 9-tuple inputs, pre-padded tensors, 1× popularity bias              |
| `export.py`     | Adapted — exclude game_tag_matrix and game_genre_matrix buffers from model.pth     |

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

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

For changes that require retraining to validate (hyperparameters, optimizer, scheduler, loss, dataset logic, model architecture): write the code, then stop. Do not commit until the user has run training and confirmed the results look better.

**After any model/training change: always wait for the user to run `python main.py train`, then `python main.py canary`, then `python main.py eval`, and confirm the results are acceptable before committing anything.**
