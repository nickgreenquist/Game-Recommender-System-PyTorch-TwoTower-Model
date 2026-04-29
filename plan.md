# Recommender System V2 Plan

This file is used to plan a major next version of the recommender system. It is based on industry proven recommender systems.

## Plan Overview
- **Adam optimizer** (Scale is manageable).
- **Full softmax** instead of in-batch negatives (Corpus is small enough at ~6k items).
- **ReLU** for all activations (Replace TanH).
- **Remove weight decay** (Compression via layer sizes provides sufficient regularization).
- **Sum pooling** for all history aggregation (Replace average pooling).
- **Shallow History Pooling:** Pool directly on item ID embeddings without passing them through the full item tower first.
- **User Tag Context:** Add a tag affinity tower similar to the existing genre affinity tower.
- **Fix User Genre Context:** Move from feature.py to dataset.py to only use rollback history to create it.
- **[BIG CHANGE] Separate Triple-History Inputs:** Partition user history into specific behavior-based pools (Liked, Disliked, and Full history).
- **Quality Labeling:** Filter labels at `hours > 0.5` to increase signal-to-noise ratio.

## 1. Preprocessing (`src/preprocess.py`)
- **Global Median Calculation:** Calculate the median playtime for every game in the corpus and save it as a new column in `base_games.parquet`.
- **Recommend Signal:** Ensure the `recommend` boolean is preserved for use in downstream stages.

## 2. Feature Generation (`src/features.py`)
- **Metadata Propagation:** Ensure the global game median is loaded into the `feature_store.pt` so it is accessible during dataset building.
- **Removed Static Features:** Stop calculating user-level genre context here, as it will be moved to `dataset.py` to avoid data leakage.

## 3. Dataset Building (`src/dataset.py`)
- **Quality Label Filter:** When sampling target games for rollback examples, only select interactions where `hours > 0.5`. This keeps low-quality interactions in the history pools (e.g., as "Disliked" signals) but prevents them from being training targets.
- **Rolling Context Generation:** Inside `_build_rollback_dataset`, implement rolling accumulators for both **Genre Context** and **Tag Context** using the **Full History** of the context. This provides a general taste profile while the partitioned ID pools provide the specific Liked/Disliked signals.
- **Temporal Bucketing (Rollback):** Calculate the **rolling User Median** at each step.
- **Triple Pool Partitioning:** Partition the available context history into three lists per training example:
    - **Liked Pool:** `recommend == True` OR `hours >= Game Median` OR `hours >= (User Median * 2)`.
    - **Disliked Pool:** `recommend == False` OR `(0.1 < hours < 1.0)` OR `hours <= (User Median / 2)`.
    - **Full Pool:** All items in the chronological context.

## 4. Model Architecture (`src/model.py`)
- **ReLU Activations:** Replace all TanH layers with ReLU.
- **Sum Pooling:** Replace average pooling with sum pooling for all history aggregation.
- **Triple History Towers:**
    - Create three separate pool paths: Liked, Disliked, and Full history.
    - **Shared Weights:** All three pools must use the **same** `item_id` embedding lookup table.
    - **Shallow History Pooling:** Sum the raw 32-dim `item_id` embeddings directly.
    - **Concatenation:** Concatenate the three 32-dim pool outputs (96-dim total) before passing them to the user projection MLP.
- **User Tag Tower:** Add a new tower for user tag affinity context (rolling calculation from `dataset.py`).

## 5. Training (`src/train.py`)
- **Full Softmax:** Implement full softmax over the entire corpus instead of in-batch negatives.
- **Optimizer:** Standard Adam (remove weight decay).
- **Loss Function:** Cross-entropy over the full corpus scores.
