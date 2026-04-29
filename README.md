# Game Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [game-recommender-system-two-tower-model.streamlit.app](https://game-recommender-system-two-tower-model.streamlit.app/)

This is a sibling project to the [Book Recommender System](https://github.com/nickgreenquist/Book-Recommender-System-PyTorch-TwoTower-Model) and [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model).

## Introduction

A PyTorch two-tower neural network trained on the [UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html) (~5,400 games, ~4.3M training examples).

Trained with full softmax cross-entropy over the entire game corpus, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant games.

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: three behavior-partitioned play history pools, rolling genre affinity, and rolling tag affinity. Any user can get recommendations from just a few games they've played, with no retraining required.
- **Playtime as the rating signal** — Steam has no star ratings. `log(1 + hours)` compresses the extreme tail while preserving ordering. Used to classify history into Liked/Disliked pools and build genre context; never a prediction target.
- **Triple history pools** — play history is partitioned into Liked (high playtime or explicit recommend), Disliked (bounced off or explicit thumbs-down), and Full (all history). Each pool is a shallow sum of raw 32-dim game ID embeddings + LayerNorm. This gives the model separate signals for positive taste, negative taste, and the broad collaborative fingerprint.
- **Full softmax over entire corpus** — cross-entropy over all ~5,400 games every training step, rather than in-batch negatives. Denser gradient signal; all items receive updates every step.
- **User tag tower** — rolling sum of TF-IDF Steam tag vectors from play history → 32-dim. Captures granular community descriptors like "Open World", "Rogue-like", "Dark Souls-like".
- **Developer embedding tower** — analogous to the author tower in the book model. Clusters games by studio and stylistically similar developers.
- **Price embedding tower** — free-to-play vs. indie vs. AAA is a meaningful taste dimension; bucketed into 9 price tiers.
- **Shuffled history protocol** — Steam provides no per-game timestamps. History is shuffled randomly rather than sorted by release date, which would give the model a temporal shortcut (always predicting newer games). Rollback examples simulate "given a random subset of games this user plays, predict another."
- **Projection MLP in each tower** — each tower concatenates its sub-embeddings and passes them through a 2-layer MLP (→256→ReLU→128). A plain concat fed directly into a dot product can only learn additive combinations; the MLP learns cross-feature interactions (e.g. genre × developer, liked pool × tag cluster). Both towers project to the same 128-dim space.

## Model architecture

```
User Tower:
  liked_pool:    sum(32-dim item ID emb[liked games])   → LayerNorm → 32-dim
  disliked_pool: sum(32-dim item ID emb[disliked games])→ LayerNorm → 32-dim
  full_pool:     sum(32-dim item ID emb[all history])   → LayerNorm → 32-dim
  user_genre_tower([debiased_avg_log_playtime | genre_frac]) →   32-dim
  user_tag_tower(rolling sum of TF-IDF tag vectors)         →   32-dim
  concat → 160-dim
  user_projection(Linear 256 → ReLU → Linear 128)           → 128-dim

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

**Shared embedding:** `item_embedding_lookup` (32-dim) is shared between all three user history pools and the item tower. The user pools sum it directly (shallow pooling); the item tower additionally passes it through a small linear layer before concatenating with other item features.

## Training

| Hyperparameter | Value |
|---|---|
| Loss | Full softmax cross-entropy (entire ~5,400-game corpus every step) |
| Optimizer | Adam, lr=0.001, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR, T_max=50,000, eta_min=1e-4 |
| Gradient clipping | max_norm=1.0 |
| Batch size | 512 |
| Temperature | 0.05 |
| Training steps | 50,000 |
| Training examples | ~4.3M (N_SHUFFLES=3 rollback augmentation, 57k train users) |
| Val eval | Fixed 8,192-example set, sampled once per run |

**Rollback construction:** For each user, rollback positions are drawn across their shuffled play history — given the first N games, predict game N+1. N_SHUFFLES=3 independent shuffles per user produce genuinely different (context, target) pairs. Val users (10% of all users) are held out entirely and never used in training.

## Offline Evaluation

Evaluated on 2,000 held-out val users (never seen during training). Corpus: 5,442 games. Shuffled history protocol — no release-date ordering.

| K | Recall@K | NDCG@K | vs. Random |
|---|---|---|---|
| 1 | 0.0417 | 0.0417 | 231× |
| 5 | 0.1202 | 0.0817 | 134× |
| 10 | 0.1794 | 0.1007 | 99× |
| 20 | 0.2668 | 0.1227 | 72× |
| 50 | 0.4309 | 0.1551 | 47× |

MRR: **0.0918** (random: 0.0017, +54×)

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
```
