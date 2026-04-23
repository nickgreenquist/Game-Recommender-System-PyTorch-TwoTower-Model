# Game Recommender System — PyTorch Two-Tower Model

## Live Demo
**Demo link:** [game-recommender-system-two-tower-model.streamlit.app](https://game-recommender-system-two-tower-model.streamlit.app/)

## Introduction
A PyTorch Two-Tower neural network trained on the [UCSD Steam dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html) (~6,200 games, ~1.9M training examples).

Trained with in-batch negatives softmax loss, following the YouTube DNN retrieval approach (Covington et al., 2016). At inference, a dot product of the user and item embeddings retrieves the most relevant games.

This is a sibling project to the [Book Recommender System](https://github.com/nickgreenquist/Book-Recommender-System-PyTorch-TwoTower-Model) and [Movie Recommender System](https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model).

## Key design choices

- **No user ID embedding** — users are represented entirely by taste signals: playtime-weighted avg pooling of game embeddings and genre affinity. Any user can get recommendations from just a few games they've played, with no retraining required.
- **Playtime as the rating signal** — Steam has no star ratings. `log(1 + hours)` compresses the extreme tail while preserving ordering. Used to weight the history pool; never a prediction target.
- **In-batch negatives softmax** — cross-entropy over in-batch negatives (batch size 512), following the YouTube DNN approach (Covington et al., 2016).
- **Developer embedding tower** — analogous to the author tower in the book model. Clusters games by studio and stylistically similar developers.
- **Price embedding tower** — free-to-play vs. indie vs. AAA is a meaningful taste dimension; bucketed into 9 price tiers.
- **No timestamp tower** — `australian_users_items.json` contains no buy date, install date, or first-play timestamp per game, so this signal is omitted entirely.

## Model architecture

```
User Tower:
  playtime_weighted_avg_pool(item_embeddings[play_history])  → 40-dim
  user_genre_tower([debiased_avg_log_playtime | play_frac])  → 65-dim
  concat → 105-dim user embedding

Item Tower:
  item_genre_tower(genre_onehot)          → 10-dim
  item_tag_tower(tfidf_tag_scores)        → 25-dim
  item_embedding_tower(game_id)           → 40-dim  ← shared with user history pool
  developer_tower(primary_developer_idx)  → 15-dim
  year_embedding_tower(release_year)      → 10-dim
  price_embedding_tower(price_bucket)     →  5-dim
  concat → 105-dim item embedding

Prediction: dot_product(user_embedding, item_embedding)
```

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
