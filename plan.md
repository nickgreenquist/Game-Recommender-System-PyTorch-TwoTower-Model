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
- **Remove release-date sort (`src/preprocess.py`):** The original code sorted each user's play history by `item_id` (Steam app ID ≈ release date) before writing to parquet. There is no install date, first-play timestamp, or purchase date in `australian_user_items.json.gz`, so this ordering is a spurious proxy — not real play sequence. The sort has been removed; history is stored in ingestion order.
- **Shuffle User History (`src/dataset.py`):** In `_build_rollback_dataset`, shuffle each user's `(history, weights, recs)` together with the seeded `rng` before building rollback examples. This removes the release-date bias so the model learns "given a random subset of games this user plays, predict another" rather than "predict newer games from older ones." Each of the N shuffle passes uses a fresh independent shuffle.
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
- **Stability Fixes:**
    - **LayerNorm:** Added `LayerNorm` after sum-pooling for each history pool and tag context to stabilize training magnitudes.
    - **Gain Adjustment:** Sub-tower initializations use `gain=0.1`; final projection layers use `gain=1.0` to prevent vanishing/exploding gradients.
- **Triple History Towers:**
...
    - Create three separate pool paths: Liked, Disliked, and Full history.
    - **Shared Weights:** All three pools must use the **same** `item_id` embedding lookup table.
    - **Shallow History Pooling:** Sum the raw 32-dim `item_id` embeddings directly.
    - **Concatenation:** Concatenate the three 32-dim pool outputs (96-dim total) before passing them to the user projection MLP.
- **User Tag Tower:** Add a new tower for user tag affinity context (rolling calculation from `dataset.py`).

## 5. Training (`src/train.py`)
- **Full Softmax:** Implement full softmax over the entire corpus instead of in-batch negatives.
- **Optimizer:** Standard Adam (remove weight decay).
- **Loss Function:** Cross-entropy over the full corpus scores.

## 6. Training Stability Fixes (discovered during implementation)
- **Weight Decay (re-added):** Full softmax sends dense gradients to all ~6k item embeddings every step via Adam's adaptive rates. Without weight decay, embedding norms grow unconstrained and cause loss explosion (~step 10k). Re-enabled at `1e-5`.
- **Gradient Clipping:** Added `clip_grad_norm_(max_norm=1.0)` before `optimizer.step()` as an additional safeguard.
- **Pre-clip grad norm logging:** `clip_grad_norm_` returns the raw norm before clipping; logged as rolling average for diagnostics.

## 7. Apple Silicon GPU (MPS) — already done
- **Device selection:** `train.py` already prefers `torch.device('mps')` over CUDA and CPU, so training runs on the Apple Silicon GPU automatically when available.
- **What benefits:** the full-corpus matmul `U @ V_all.T` (512×128 @ 128×5442) and the item tower forward pass over all 5,442 items per step.
- **What doesn't benefit:** `pad_history_batch` — called 3× per step for the ragged history pools, runs on CPU as a Python loop. This is the main remaining CPU-bound bottleneck.

## 8. Dataset Augmentation: N Shuffles per User
- **N_SHUFFLES = 3:** Run `_build_rollback_dataset` N times per user with independent shuffles. Each shuffle produces a different (context, target) ordering, giving N× more training examples while fully preserving the short-context property (early rollback positions still have 1–2 game contexts).
- **Train only:** `N_SHUFFLES` applies to the train split only. Val and offline eval use `n_shuffles=1` (single clean view per user) so metrics are not inflated by repeated targets across shuffle passes.
