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
- **Developer embedding tower** — analogous to the author tower in the book model. Clusters games by studio and stylistically similar developers.
- **Price embedding tower** — free-to-play vs. indie vs. AAA is a meaningful taste dimension; bucketed into 9 price tiers.
- **Projection MLP in each tower** — each tower concatenates its sub-embeddings and passes them through a 2-layer MLP (→256→ReLU→128) before the dot product. A plain concat fed directly into a dot product can only learn additive combinations of the individual signals; the MLP learns cross-feature interactions (e.g. genre × developer, price × history depth) that require nonlinearity. Both towers project to the same 128-dim space; only the output dim needs to match, not the internal concat sizes.

## Model architecture

```
User Tower:
  playtime_weighted_avg_pool(item_embeddings[play_history])  → 32-dim
  user_genre_tower([debiased_avg_log_playtime | play_frac])  → 32-dim
  concat → 64-dim
  user_projection(Linear 256 → ReLU → Linear 128)           → 128-dim

Item Tower:
  item_genre_tower(genre_onehot)          →  8-dim
  item_tag_tower(tfidf_tag_scores)        → 16-dim
  item_embedding_tower(game_id)           → 32-dim  ← shared with user history pool
  developer_tower(primary_developer_idx)  → 12-dim
  year_embedding_tower(release_year)      →  8-dim
  price_embedding_tower(price_bucket)     →  4-dim
  concat → 80-dim
  item_projection(Linear 256 → ReLU → Linear 128)           → 128-dim

Prediction: dot_product(user_projection_out, item_projection_out)
```

The projection MLP is critical: a plain concat fed directly into a dot product can only learn additive combinations of the individual sub-embeddings. The MLP learns cross-feature interactions that require nonlinearity. Only the output dim (128) needs to match across towers — the internal concat sizes are independent.

## Offline Evaluation

Evaluated on 2,000 held-out users (never seen during training) across 55,186 rollback examples. Corpus: 5,442 games.

| K | Recall@K | NDCG@K | vs. Random (HR@K) |
|---|---|---|---|
| 1 | 0.0634 | 0.0634 | 350× |
| 5 | 0.1904 | 0.1280 | 207× |
| 10 | 0.2751 | 0.1552 | 153× |
| 20 | 0.3832 | 0.1825 | 104× |
| 50 | 0.5587 | 0.2173 | 61× |

MRR: **0.1351** (random: 0.0017, +79×)

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
