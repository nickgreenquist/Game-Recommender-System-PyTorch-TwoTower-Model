# CLAUDE.md

Guidance for Claude Code working in this repo. **The ranker subsystem under `ranker/` has its own guide at `ranker/CLAUDE.md` — read it before working there.** This file covers the CG model and `src/`.

## Project Overview

A PyTorch Two-Tower recommender trained on the UCSD Steam dataset (McAuley lab). Predicts game preferences via dot product of user and item embeddings.

**Two-stage serving.** The two-tower model is the candidate-generation (CG) / retrieval stage documented here; a **Wide & Deep ranker** (`ranker/`) reranks its top-100. Streamlit and the canary run the full pipeline: raw α=0 CG retrieval → ranker rerank.

**No user ID embedding.** Users are represented entirely by taste signals: four behavior-partitioned play-history pools (Liked, Disliked, Full, Playtime-weighted Full) + rolling genre and tag affinity. Any user can be represented at inference from just a few games — no retraining.

**Rating signal: playtime hours.** Steam has no star ratings. `playtime_forever` (minutes → hours, `log(1+hours)` for weighting) is the implicit feedback signal; the review `recommend` boolean is supplementary explicit feedback. Playtime is the primary training signal but is never a prediction target.

Sibling projects (same two-tower design): `../Movie-Recommender-System-PyTorch-TwoTower-Model` (MovieLens, MSE) and `../Book-Recommender-System-PyTorch-TwoTower-Model` (Goodreads, softmax — **primary reference**). The book model's author tower maps to a **developer tower** here.

## Running the Code

```bash
python main.py preprocess games          # filter games → data/base_games.parquet
python main.py preprocess interactions   # process user items → remaining parquets
python main.py preprocess                # both steps in order
python main.py features                  # base parquets → data/features_*.parquet
python main.py dataset                   # features → data/dataset_*_v1.pt
python main.py train                     # train, save checkpoints (softmax)
python main.py canary [path]             # canary user recs (latest or specific checkpoint)
python main.py probe  [path]             # embedding probes
python main.py eval   [path]             # offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py export [path]             # export serving artifacts for Streamlit
python main.py                           # all stages in order
```

`streamlit run streamlit_app.py` — tabs: **Recommend** (pick games → α=0 CG top-100; ⚡ Apply Ranker reranks side-by-side with rank-delta badges, degrades to CG-only if `serving/ranker.pth` absent), **Similar** / **Genres** / **Tags** (cosine-nearest in the respective embedding space). Steam covers fetched live from `cdn.cloudflare.steamstatic.com`.

## Dataset

Raw gzipped-JSONL files live in `data/` (not in git). UCSD Steam dataset:
- `australian_users_items.json.gz` (**primary**, 88,310 users) — `items[{item_id, playtime_forever, playtime_2weeks}]`
- `australian_user_reviews.json.gz` (**supplementary**, 25,799 users) — `reviews[{item_id, recommend, posted, ...}]`
- `steam_games.json.gz` (**item metadata**, 32,135 games) — `id, app_name, genres, tags, developer, publisher, release_date, price, sentiment`
- **Not used:** `steam_reviews.json.gz` — no `recommend` boolean, no stable `user_id`, not grouped by user.

### Filtering thresholds

```python
MIN_INTERACTIONS_PER_GAME       = 10      # users who played the game (100→10 ~triples corpus, near-free)
MIN_PLAYTIME_PER_USER           = 5       # min total hours
MAX_PLAYTIME_PER_USER           = 10_000  # cap (removes bots/outliers)
MIN_HOURS_PER_GAME              = 0.1     # min hours for a single game to count
MIN_TAG_COUNT                   = 50      # tag must appear in this many corpus games
MAX_ROLLBACK_EXAMPLES_PER_USER  = 50      # cap per user (small dataset; hour ceiling handles power users)
```

**Valve DENYLIST:** `{'730','550','620','240','4000'}` (CS:GO, L4D2, Portal, Counter-Strike, Garry's Mod) are hard-filtered in `preprocess.py` Pass 1 regardless of count — they appeared in nearly every history, were trivial prediction targets that inflated Recall@K and polluted cross-genre recs.

**N_SHUFFLES=3:** `_build_rollback_dataset` runs 3× per user with independent seeded shuffles → ~4.3M training examples, each yielding different (context, target) pairs at varied context lengths. **Val and offline eval always use n_shuffles=1** so metrics aren't inflated by repeated targets.

**Quality label filter:** target games need `hours > 0.5`. Low-playtime interactions stay in history pools as signal but are never targets.

### Preprocessing pipeline (two steps)

- **Step 1 `preprocess games`** — reads `steam_games.json.gz`, filters by interaction count (first pass over user items to count), collects metadata, computes **global per-game median playtime** (`median_hours` in `base_games.parquet`, used for Liked/Disliked partitioning). Writes `base_games.parquet`, `base_game_tags.parquet`.
- **Step 2 `preprocess interactions`** — processes user items (keep `playtime ≥ MIN_HOURS_PER_GAME*60`, corpus games only), joins `recommend`/`posted` from reviews. **History is NOT sorted** — Steam has no per-game timestamp; item_id order ≈ release date is a spurious newer-game bias. `dataset.py` shuffles each user's history with a seeded RNG, so rollback = "given a random subset, predict another."

### Tag signals

Steam community `tags` are an ordered list, no counts. Treat list position as relevance: `weight = 1/(position+1)`, normalized per game; TF-IDF = `positional_weight * log(N/df)`. Only tags with `≥ MIN_TAG_COUNT` corpus occurrences are kept, stored in `base_game_tags.parquet`.

## Model Architecture

Two-tower, dot-product prediction. Both towers end in `F.normalize` (cosine sim consistently at train + inference). Only `item_id_embedding_size=32` (shared) and `output_dim=128` must match across towers.

```
User Tower (concat 192 → proj 256→ReLU→128):
  liked / disliked / full / playtime-weighted pools — each sum(item_id_emb[ids])   4×32
  user_genre_tower([debiased_avg_log | play_frac])                                  32
  user_tag_tower(rolling_tag_sum)                                                   32

Item Tower (V6a, concat 128 → proj 256→ReLU→128):
  item_genre_tower(genre_onehot)      8     developer_tower(dev_idx)       12
  item_tag_tower(tfidf)              32     year_embedding_tower            8
  item_embedding_tower(item_id)      32     price_embedding_tower           4
  item_text_tower(text_emb_768)      32 (V6a)

Prediction: dot(user_proj, item_proj)
```

**Shallow history pooling:** user pools sum the raw shared 32-dim `item_embedding_lookup` directly — they do NOT pass through the item tower. No LayerNorm after pooling (the projection MLP learns the scale; industry standard).

**Four pools** (per rollback position in `dataset.py`; an item can be in both Liked and Disliked):
- **Liked:** `recommend==True` OR `hours ≥ game_median` OR `hours ≥ user_rolling_median×2`
- **Disliked:** `recommend==False` OR `0.1 < hours < 1.0` OR `hours ≤ user_rolling_median/2`
- **Full:** all context items (most recent MAX_HISTORY_LEN=50), equal-weight sum
- **Playtime-weighted Full:** same items, weight `log(1+hours)` normalized by context total

`game_median` is the global per-game median (`base_games.parquet`); `user_rolling_median` is computed at each rollback step from context-so-far.

**In-model genre/tag context (V3):** computed inside `user_embedding()` from `game_genre_matrix` / `game_tag_matrix` registered buffers using `X_hist_full` indices. `dataset.py` only supplies `X_user_avg_log` (per-user avg log-playtime scalar, for in-model genre debiasing). These two buffers + `game_text_matrix` are excluded from `model.pth` and stored in `feature_store.pt`.

**Item text tower (V6a, item-side only):** each game's store description (`short_description` + `about_the_game`, fetched offline from the Steam Storefront API) is encoded ONCE by frozen `BAAI/bge-base-en-v1.5` (768-d) into `data/base_game_text_emb.parquet`. At train/serve it's a buffer lookup (`game_text_matrix`) → L2-norm → trainable adapter `item_text_tower` (768→128→32). The encoder never runs at train/inference. Gated by `use_item_text`. **Offline-flat for CG** (item-side-only text has no text-vs-text path in a dot product) but kept — `game_text_matrix` is shared into `feature_store.pt` and consumed by the ranker's `item_text_tower` + `text_cosine` cross feature. A user-history text pool (V6b) was tested and **dropped** (regressed offline).

**Why the projection MLP:** a plain concat into a dot product only learns additive combinations; the 2-layer MLP models interactions (e.g. "RPGs from Japanese devs"). **Init:** sub-tower linears `gain=0.1`, projections `gain=1.0`, embeddings `gain=0.01` (gain=0.01 on projections vanishes gradients).

**Developer tower:** primary developer only; `nn.Embedding(n_developers+1, 12)`, padding idx `n_developers`, idx 0 = `__unknown__`. Item-side only. **Price tower:** `original_price` bucketed into ~10 bins (Free, <$5, …, >$60, Unknown) → `nn.Embedding(n_buckets, 4)`.

## Training Details

Predicts "given a random subset of play history, which game do they also play?" — a ranking problem. **Full softmax over the entire ~5,437-game corpus** (Valve DENYLIST applied; corpus small enough to score all items exactly — no ANN, no separate ranking stage needed for CG).

- **Loss:** cross-entropy over all items. **Optimizer:** Adam `lr=0.001, weight_decay=0.0, eps=1e-6`. **Scheduler:** CosineAnnealingLR `T_max=50_000, eta_min=1e-4`. **Grad clip:** `max_norm=1.0`. **Batch:** 512. **Temperature:** `0.5/batch = 0.000977`. **Steps:** 50,000.
- **Val eval:** fixed 8,192 examples sampled once at start (same indices every log step, comparable across steps).
- **Checkpoint selection (2026-05-25):** best on **val NDCG@10**, not val CE (CE keeps falling via confidence calibration after ranks plateau). `val_loss` still logged as a diagnostic.

**Popularity logit adjustment — Menon et al. 2021 Path 2 (add at training, raw dot product at inference):**
```python
scores = (U @ V_all.T)/temperature + alpha*log1p(count)   # training (Menon Path 2)
scores = user_emb @ item_embs.T                            # inference — no correction anywhere
```
Adding the bias during training forces the model to learn preference *beyond* popularity, so popular items are naturally suppressed at inference. Temperature and alpha are read from the checkpoint config sidecar (`_config.json`) via `load_config_for_checkpoint()` — never hardcoded.

**Alpha trade-off — α=0 wins offline, α=0.4 wins canary.** α=0 gives materially better offline metrics (+25% Recall@1) but worse canary on niche tastes (popular cross-genre titles leak in). **In the two-stage world (2026-05-23) both stages ship raw α=0** — the α=0 CG IS the deployed retrieval stage, the ranker reranks its top-100. A ranker-side α=0.2 penalty was tested and rejected (hurt offline ~27%, no better canary). No popularity penalty anywhere in the deployed pipeline. **α is the last knob: A/B new features at α=0, calibrate popularity last.**

## Evaluation

**Canary** (`src/evaluate.py`): nine synthetic user types (Western RPG, JRPG, FPS, Civ, Indie, Racing, Fighting, Survival, Management Lover). Each has `USER_TYPE_TO_FAVORITE_GAMES` (seed weight 10.0) and `USER_TYPE_TO_TAGS` (5 anchor games/tag, weight 2.0); disliked dict currently empty. `POPULARITY_ALPHA_INFERENCE_MULTIPLE = 0.0` (raw dot products). All titles verified in corpus.

**Offline** (`python main.py eval [path]`): Recall@K, NDCG@K, Hit Rate@K, MRR at K=1,5,10,20,50,100, raw dot products. **Protocol:** user-level split — 90% train-only, 10% held out entirely (stricter cold-start test than a per-user split for a no-user-ID model; no leakage so no within-user cut needed). Val rollback examples generated fresh with `n_shuffles=1`; results → `eval_results/<stem>.txt`.

**V6a DEPLOYED** (item text, α=0, NDCG@10-selected — served retrieval stage as of 2026-05-25):

| K | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| Recall | 0.0279 | 0.0875 | 0.1444 | 0.2299 | 0.3968 | 0.5575 |
| NDCG   | 0.0279 | 0.0576 | 0.0758 | 0.0972 | 0.1302 | 0.1562 |

MRR 0.0704 (random ≈ 0.0017). Flat vs the no-text α=0 baseline (every Δ ≤ .0014) — text adds no CG lift but is kept for ranker reuse. **Use this α=0 row (not the α=0.4 prod CG) as the offline yardstick for ranker comparisons.**

## Serving / Export

Registered buffers (`game_tag_matrix`, `game_genre_matrix`, `game_text_matrix`) and `game_dev_idx` are excluded from `model.pth` and stored in `feature_store.pt`; the app reconstructs `GameRecommender` from those buffers and loads weights `strict=False`.

**Deployed (2026-05-25):** `serving/` exported from α=0 item-text CG `best_triple_full_softmax_text_popularity_alpha_0_20260525_100022.pth`; PROD ranker `ranker_wd_alpha_0_20260525_113932.pth` retrained against it (24 wide features, deep_in 320, `text_cosine` cross feature; NDCG@10 +0.0120 over the text CG). See `ranker/CLAUDE.md`.

- `python main.py export` → `serving/model.pth` (CG weights), `serving/game_embeddings.pt` (per-game CG embeddings), `serving/feature_store.pt` (vocab maps, metadata, CG buffers, config, **+ 9 ranker source arrays** rebuilt by `ranker.train._buffers_from_fs`).
- `python ranker/main.py export [ranker.pth]` re-exports the CG then adds `serving/ranker.pth` + `serving/ranker_config.json`. The app rebuilds the ranker purely from serving artifacts (no `saved_models/`, no `get_config()` glob).

## Working Style and Guidelines

### Git workflow
- Never commit and push in the same command. Commit first, then ask before pushing. Solo repo — no PRs.
- **Any change that affects training behavior** (hyperparameters, optimizer, scheduler, loss, dataset logic, architecture): write the code, then **stop**. Don't commit until the user has run `train` → `canary` → `eval` and confirmed results. Don't update the results table here until the user reports numbers — smoke tests verify shapes/imports, not metrics.

### Behavioral (project-specific; supplements the system prompt)
- **Match the house style.** Long docstring headers, NamedTuple bundles, multi-paragraph comment banners on training-loop functions, named slice offsets over magic numbers, vertically-aligned parquet column comments. New ranker buckets mirror the previous bucket exactly.
- **Surgical changes.** Touch only what the task requires; every changed line traces to the request. Mention adjacent dead code / refactor opportunities — don't act on them.
- **Surface tradeoffs early** in 1-2 sentences; pick the option you think is right with a stated assumption rather than opening a multi-option question for routine calls.
- **Multi-step work in one session → TaskCreate** (not a plan file). **Multi-session features → a visible `*_PLAN.md` in the repo + TaskCreate** (never stash the plan in agent memory — the user can't see it).
