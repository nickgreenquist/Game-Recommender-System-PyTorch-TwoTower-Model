# Game Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [game-recommender-system-two-tower-model.streamlit.app](https://game-recommender-system-two-tower-model.streamlit.app/)

This is a sibling project to the [Book Recommender System](https://github.com/nickgreenquist/Book-Recommender-System-PyTorch-TwoTower-Model) and [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model).

## Introduction

A PyTorch two-tower neural network trained on the [UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html) (~5,437 games, ~4.3M training examples).

Trained with full softmax cross-entropy over the entire game corpus, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant games.

This is a **two-stage** recommender: the two-tower model is the candidate-generation (retrieval) stage, and a **Wide & Deep ranker** reranks its top-100 candidates with richer cross-features. See [Two-Stage Serving](#two-stage-serving-wide--deep-ranker) below.

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: four behavior-partitioned play history pools, genre affinity, and tag affinity computed from play history. Any user can get recommendations from just a few games they've played, with no retraining required.
- **Playtime as the rating signal** — Steam has no star ratings. `log(1 + hours)` compresses the extreme tail while preserving ordering. Used to classify history into Liked/Disliked pools and as the per-user avg scalar for genre debiasing; never a prediction target.
- **Four history pools** — play history is partitioned into Liked (high playtime or explicit recommend), Disliked (bounced off or explicit thumbs-down), Full (all history, equal-weight), and Playtime-weighted Full (same games, weighted by normalized log-playtime). Each pool is a raw sum of 32-dim game ID embeddings — no LayerNorm (industry standard; the projection MLP learns the right scale). This gives the model separate signals for positive taste, negative taste, collaborative fingerprint, and engagement intensity.
- **In-model genre/tag context** — genre affinity and tag context are computed inside `user_embedding()` from `game_genre_matrix` and `game_tag_matrix` registered buffers, using the full history window. This avoids pre-computing context per rollback position in the dataset and enables the model to derive context from any history at inference time.
- **Full softmax over entire corpus** — cross-entropy over all ~5,437 games every training step, rather than in-batch negatives. Denser gradient signal; all items receive updates every step.
- **Valve title filter** — CS:GO, Garry's Mod, Left 4 Dead 2, Portal, and Counter-Strike are hard-removed from the corpus. These appeared in nearly every user's history, were trivially easy prediction targets, and caused cross-genre recommendation pollution.
- **User tag tower** — sum of TF-IDF Steam tag vectors from play history → 32-dim. Captures granular community descriptors like "Open World", "Rogue-like", "Dark Souls-like".
- **Developer embedding tower** — analogous to the author tower in the book model. Clusters games by studio and stylistically similar developers.
- **Price embedding tower** — free-to-play vs. indie vs. AAA is a meaningful taste dimension; bucketed into 9 price tiers.
- **Shuffled history protocol** — Steam provides no per-game timestamps. History is shuffled randomly rather than sorted by release date, which would give the model a temporal shortcut (always predicting newer games). Rollback examples simulate "given a random subset of games this user plays, predict another."
- **Projection MLP in each tower** — each tower concatenates its sub-embeddings and passes them through a 2-layer MLP (→256→ReLU→128), then L2-normalizes the output. Both towers project to the same 128-dim space; dot product of normalized outputs is cosine similarity.

## Model architecture

```
User Tower:
  liked_pool:     sum(32-dim item ID emb[liked games])              → 32-dim
  disliked_pool:  sum(32-dim item ID emb[disliked games])           → 32-dim
  full_pool:      sum(32-dim item ID emb[all history])              → 32-dim
  playtime_pool:  sum(32-dim item ID emb[all history] × log_w)      → 32-dim
  user_genre_tower([debiased_avg_log_playtime | genre_frac])        →            32-dim
  user_tag_tower(sum of TF-IDF tag vectors from history)            →            32-dim
  concat → 192-dim
  user_projection(Linear 256 → ReLU → Linear 128)                  →           128-dim

Item Tower:
  item_embedding_tower(game_id)           → 32-dim  ← shared lookup with user history pools
  item_genre_tower(genre_onehot)          →  8-dim
  item_tag_tower(tfidf_tag_scores)        → 32-dim
  developer_tower(primary_developer_idx)  → 12-dim
  year_embedding_tower(release_year)      →  8-dim
  price_embedding_tower(price_bucket)     →  4-dim
  concat → 96-dim
  item_projection(Linear 256 → ReLU → Linear 128) → 128-dim

Prediction: dot_product(user_projection_out, item_projection_out)
```

**Shared embedding:** `item_embedding_lookup` (32-dim) is shared between all four user history pools and the item tower. The user pools sum it directly (shallow pooling); the item tower additionally passes it through a small linear layer before concatenating with other item features.

**In-model context:** `user_embedding()` takes `X_user_avg_log` (per-user average log-playtime scalar) and pre-padded history tensors — genre and tag context are derived inside the forward pass from `game_genre_matrix` and `game_tag_matrix` registered buffers. `item_embedding()` similarly looks up genre internally rather than taking it as an argument.

## Training

| Hyperparameter | Value |
|---|---|
| Loss | Full softmax cross-entropy (entire ~5,437-game corpus every step) |
| Optimizer | Adam, lr=0.001, eps=1e-6 |
| Scheduler | CosineAnnealingLR, T_max=50,000, eta_min=1e-4 |
| Gradient clipping | max_norm=1.0 |
| Batch size | 512 |
| Temperature | 0.000977 (= 0.5 / 512) |
| Popularity bias | alpha × log1p(count) **added at training** (Menon Path 2); **raw dot products at inference** — no correction. Standalone CG: alpha=0.4; deployed retrieval stage: alpha=0 (see [Two-Stage Serving](#two-stage-serving-wide--deep-ranker)) |
| Training steps | 50,000 |
| Training examples | ~4.3M (N_SHUFFLES=3 rollback augmentation, 55k train users) |
| Val eval | Fixed 8,192-example set, sampled once per run |

**Rollback construction:** For each user, rollback positions are drawn across their shuffled play history — given the first N games, predict game N+1. N_SHUFFLES=3 independent shuffles per user produce genuinely different (context, target) pairs. Val users (10% of all users) are held out entirely and never used in training. The dataset is a 9-tuple of pre-padded tensors (histories padded to MAX_HISTORY_LEN=50); genre/tag context is not stored in the dataset — computed at forward-pass time in the model.

## Offline Evaluation

Evaluated on a sample of 2,000 users drawn from the held-out validation set (10% of all users, never seen during training). Shuffled history protocol — no release-date ordering. Each example has one target; Recall@K = Hit Rate@K for single-target eval.

**Which checkpoint these tables describe.** The `alpha` popularity-bias knob produces two CG variants that matter (same architecture, different training-time bias):

- **Standalone CG (alpha=0.4)** — the tables below. Trades raw recall for cleaner niche-taste lists; this was the pre-ranker single-model serving choice.
- **Deployed retrieval stage (alpha=0)** — recall-maximizing, so it scores *higher* on these offline metrics (NDCG@10 **0.0752** vs 0.0645). This is the retrieval stage in the two-stage system and the baseline the ranker improves on in [Two-Stage Serving](#two-stage-serving-wide--deep-ranker) below.

### V5 PROD — corpus: 5,437 games (Valve titles removed, no LayerNorm, correct Menon Path 2)

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0226 | 0.0226 |
| 5 | 0.0741 | 0.0481 |
| 10 | 0.1253 | 0.0645 |
| 20 | 0.2059 | 0.0848 |
| 50 | 0.3673 | 0.1166 |

MRR: **0.0611** (random: 0.0017, +36×)

### V4 — corpus: 5,437 games (Valve titles removed, no LayerNorm, incorrect Menon sign)

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0294 | 0.0294 |
| 5 | 0.0882 | 0.0589 |
| 10 | 0.1430 | 0.0765 |
| 20 | 0.2280 | 0.0978 |
| 50 | 0.3913 | 0.1300 |

MRR: **0.0715** (random: 0.0017, +42×)

### V3 PROD — corpus: 5,437 games (Valve titles removed, with LayerNorm after pools)

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0278 | 0.0278 |
| 5 | 0.0882 | 0.0581 |
| 10 | 0.1428 | 0.0756 |
| 20 | 0.2287 | 0.0971 |
| 50 | 0.3944 | 0.1299 |

MRR: **0.0706** (random: 0.0017, +41×)

### V2 PROD — corpus: 5,442 games (Valve titles included)

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0389 | 0.0389 |
| 5 | 0.1138 | 0.0767 |
| 10 | 0.1743 | 0.0962 |
| 20 | 0.2602 | 0.1177 |
| 50 | 0.4256 | 0.1504 |

MRR: **0.0875** (random: 0.0017, +51×)

**Why V5 metrics are lower than V4:** V4 was trained with `- popularity_bias` (subtracting), which caused the model to compensate by making popular item embeddings universally closer to all user embeddings — inflating Recall@K for popular targets. V5 uses the correct Menon Path 2 formula (`+ popularity_bias` at training, raw dot products at inference), producing genuinely preference-driven rankings with cleaner per-genre canary quality. **Why V3/V4 metrics are lower than V2:** Ultra-popular Valve games (CS:GO, Garry's Mod, Left 4 Dead 2) were trivially easy prediction targets, inflating V2's Recall@K. Removing them makes every target require genuine taste modeling.

## Two-Stage Serving: Wide & Deep Ranker

The two-tower model above is fast but limited: it can only score a user against an item through a **dot product of two independent embeddings**. It never sees the user and the candidate *together*, so it cannot represent features like "how many of this candidate's genres overlap the user's history" or "how far is this candidate's price from what the user usually plays." A second-stage **ranker** adds exactly those signals.

```
Stage 1 — Retrieval (two-tower CG):   score all ~5,437 games by dot product → take top 100
Stage 2 — Ranking  (Wide & Deep):     rerank those 100 with user × item cross-features
```

Retrieval is the raw **alpha=0** CG — recall-maximizing, no popularity penalty — and the ranker is the final model the user sees. Neither production stage applies a popularity penalty: retrieval maximizes recall, and the ranker reorders purely on its content and cross-features.

### Why Wide & Deep

The ranker is a [Wide & Deep network](https://arxiv.org/abs/1606.07792):

- **Deep side** — mirrors the two-tower model exactly (same per-feature towers, same dimensions), warm-started from the trained CG weights. All sub-embeddings are concatenated (288-dim) and fed through an MLP. This carries over everything the retrieval model already learned.
- **Wide side** — a set of hand-crafted **cross-features** that bypass the MLP and connect directly to the output, each with its own learned weight. A scalar like "genre overlap" would be drowned out among ~290 deep dimensions in the first layer, so it gets a direct path instead.

The cross-features are what the two-tower model structurally cannot compute:

| Group | Examples |
|---|---|
| Categorical overlap | genre / tag / developer overlap between the candidate and the user's history (full, liked, and most-recent-3 slices) |
| Numeric matching | price gap, release-era gap, playtime calibration, popularity match, sentiment match between user averages and the candidate |
| Niche / rarity | IDF-weighted tag overlap and rarest-tag / developer-catalog-scale matching, to suppress popular cross-genre titles leaking into niche-taste lists |

### Training

The ranker is trained with the same sampled-softmax cross-entropy loss as the CG model (1 positive + 999 random negatives per example), warm-started from the retrieval model, then learns the cross-feature weights on top. It reranks within the retrieval model's top-100, so its hit-rate ceiling is the retrieval model's Recall@100.

### Results

Reranking lifts every metric over the retrieval model on the same held-out users (production-realistic, both capped at the retrieval ceiling):

| Metric | Retrieval (raw alpha=0 CG) | + Ranker | Lift |
|---|---:|---:|---:|
| NDCG@10 | 0.0752 | **0.0867** | +15% |
| Hit@10 | 0.1430 | **0.1625** | +14% |
| MRR | 0.0726 | **0.0813** | +12% |

On the pure-reranking subset (cases where retrieval surfaced the target), NDCG@10 rises from 0.1361 → 0.1569 (+15%). Beyond the metrics, the ranker visibly cleans up niche-taste lists — e.g. removing JRPGs that leak into a fighting-game query.

The Streamlit demo's **Recommend** tab shows this live: results appear from retrieval first, and a **⚡ Apply Ranker** button reranks the same 100 candidates side-by-side, with badges showing how far each game moved.

## Usage

```bash
python main.py preprocess   # Build base_games.parquet + interaction parquets
python main.py features     # Build features_*.parquet
python main.py dataset      # Build dataset_*_v1.pt
python main.py train        # Train model, save checkpoint
python main.py canary       # Canary user recommendations (most recent checkpoint)
python main.py eval         # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py export       # Generate serving/ artifacts
streamlit run streamlit_app.py

# Two-stage ranker (reranks the retrieval model's top-100)
python ranker/main.py precompute   # Score candidates with the retrieval model
python ranker/main.py train        # Train the Wide & Deep ranker
python ranker/main.py evaluate     # NDCG@K / Hit@K / MRR vs the retrieval baseline
python ranker/main.py canary       # Side-by-side CG vs ranker for synthetic users
python ranker/main.py export       # Add ranker.pth + ranker_config.json to serving/
```
