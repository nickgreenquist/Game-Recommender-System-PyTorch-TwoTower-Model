# Game Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [game-recommender-system-two-tower-model.streamlit.app](https://game-recommender-system-two-tower-model.streamlit.app/)

This is a sibling project to the [Book Recommender System](https://github.com/nickgreenquist/Book-Recommender-System-PyTorch-TwoTower-Model) and [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model).

## Introduction
A PyTorch Two-Tower neural network trained on the [UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html) (~6,200 games, ~1.9M training examples).

Trained with in-batch negatives softmax loss, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant games.

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: playtime-weighted avg pooling of game embeddings and genre affinity. Any user can get recommendations from just a few games they've played, with no retraining required.
- **Playtime as the rating signal** — Steam has no star ratings. `log(1 + hours)` compresses the extreme tail while preserving ordering. Used to weight the history pool; never a prediction target.
- **In-batch negatives softmax** — cross-entropy over in-batch negatives (batch size 512), following the YouTube DNN approach (Covington et al., 2016).
- **Item-pool history (ipool)** — the user history pool averages the full 128-dim item tower output (after the projection MLP) for each played game, not just the raw 32-dim game ID embedding. This gives the user tower access to every content signal in a game's representation — genre, tags, developer, release year, price — not just its learned identity. User concat: 128 + 32 = 160-dim.
- **Developer embedding tower** — analogous to the author tower in the book model. Clusters games by studio and stylistically similar developers.
- **Price embedding tower** — free-to-play vs. indie vs. AAA is a meaningful taste dimension; bucketed into 9 price tiers.
- **Projection MLP in each tower** — each tower concatenates its sub-embeddings and passes them through a 2-layer MLP (→256→ReLU→128) before the dot product. A plain concat fed directly into a dot product can only learn additive combinations of the individual signals; the MLP learns cross-feature interactions (e.g. genre × developer, price × history depth) that require nonlinearity. Both towers project to the same 128-dim space; only the output dim needs to match, not the internal concat sizes.

## Model architecture

```
User Tower:
  item_pool_avg(item_tower_output[play_history])             → 128-dim  ← full item embedding, not just ID
  user_genre_tower([debiased_avg_log_playtime | play_frac])  →  32-dim
  concat → 160-dim
  user_projection(Linear 256 → ReLU → Linear 128)           → 128-dim

Item Tower:
  item_genre_tower(genre_onehot)          →  8-dim
  item_tag_tower(tfidf_tag_scores)        → 16-dim
  item_embedding_tower(game_id)           → 32-dim  ← shared lookup inside item tower
  developer_tower(primary_developer_idx)  → 12-dim
  year_embedding_tower(release_year)      →  8-dim
  price_embedding_tower(price_bucket)     →  4-dim
  concat → 80-dim
  item_projection(Linear 256 → ReLU → Linear 128)           → 128-dim

Prediction: dot_product(user_projection_out, item_projection_out)
```

The item tower is shared: the same network that encodes a candidate game also encodes every game in the user's play history. Pooling the full 128-dim item tower output (rather than a raw 32-dim ID lookup) means the user representation captures genre, tag, developer, era, and price signals from their history — not just opaque learned IDs.

The projection MLP is critical: a plain concat fed directly into a dot product can only learn additive combinations of the individual sub-embeddings. The MLP learns cross-feature interactions that require nonlinearity. Only the output dim (128) needs to match across towers — the internal concat sizes are independent.

## Offline Evaluation

Evaluated on 2,000 held-out users (never seen during training) across 55,186 rollback examples. Corpus: 5,442 games.

| K | Recall@K | NDCG@K | vs. Random (HR@K) |
|---|---|---|---|
| 1 | 0.0902 | 0.0902 | 491× |
| 5 | 0.2614 | 0.1777 | 284× |
| 10 | 0.3794 | 0.2158 | 206× |
| 20 | 0.5182 | 0.2508 | 141× |
| 50 | 0.7165 | 0.2903 | 78× |

MRR: **0.1845** (random: 0.0017, +109×)

## Usage

```bash
python main.py preprocess   # Build base_games.parquet + interaction parquets
python main.py features     # Build features_*.parquet
python main.py dataset      # Build dataset_*_v1.pt
python main.py train        # Train model, save checkpoint
python main.py export       # Generate serving/ artifacts
python main.py canary       # Evaluate canary users
python main.py probe        # Embedding probes
streamlit run streamlit_app.py
```
