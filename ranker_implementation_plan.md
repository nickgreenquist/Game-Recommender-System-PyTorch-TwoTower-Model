# Steam Ranker: Implementation Plan

> **Sibling project context.** A ranker has been built on the Movie-Recommender repo (`../Movie-Recommender-System-PyTorch-TwoTower-Model/ranker_implementation_plan.md`). That work established the methodology — CG-parity baseline, warm-start init from prod CG, Wide & Deep architecture, BCE + Mixed Negative Sampling — and surfaced one important finding: **the Menon popularity correction (α) does not transfer from softmax CG training to BCE ranker training.** The games ranker should reuse that methodology, but **must not assume** the α finding transfers — different dataset, different popularity dynamics, different corpus size.
>
> Games is faster to train than movies or books — this is intentionally the fastest sandbox for ranker iteration. Once techniques are validated here, port them to movies/books.

---

## 1. Pipeline

Two-stage retrieve-and-rank:
1. **CG** (v5 softmax two-tower, 4-pool user tower, F.normalize on both towers, 128-dim) — retrieves top-K candidates per rollback example.
2. **Ranker** (Wide & Deep MLP) — reranks the K candidates using richer features.

### CG v5 baseline (current prod, target to beat)

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 0.0226 | 0.0226 |
| 5 | 0.0741 | 0.0481 |
| 10 | 0.1253 | 0.0645 |
| 20 | 0.2059 | 0.0848 |
| 50 | 0.3673 | 0.1166 |

MRR: 0.0611 (random: 0.0017). Trained with `popularity_alpha = 0.4` (Menon Path 2 — add at training, raw dot at inference).

**Recall@K-ceiling target** is K=250 (matching movies) but games corpus is only 5,437 items; consider K=100 or even K=50 to keep the ranker problem well-conditioned. Final K to be chosen at precompute time.

---

## 2. Core Principles

### CG-parity baseline first

The ranker must contain every CG input/feature, projected through the same per-feature towers CG uses, with the same dimensions and the same initialization. Cross features come *on top* — they are the differentiator, not a replacement. A ranker that sees less than CG cannot be expected to beat CG regardless of clever cross features. (Industry practice: Google/YouTube/Netflix rankers always receive every signal CG had, plus more.)

### Wide & Deep architecture

Unlike CG (separate user/item towers + late F.normalize'd dot product), the ranker concatenates all per-feature embeddings (user-side + item-side) into ONE vector that feeds a deep MLP. Cross features bypass the MLP and go straight to the head — giving each one a direct learned weight.

### No CG coupling at runtime

The ranker owns its own copies of every parameter (its own `item_id_lookup`, `developer_lookup`, `developer_tower`, `item_tag_tower`, etc. — see §3 warm-start map for the full list). It just *replicates the architecture* of each CG sub-tower. **Warm-starting weights from CG at init is encouraged** — it's a one-time copy at construction; the tensors then live entirely in the ranker `state_dict` and train freely. What is NOT allowed: shared `nn.Module` objects, frozen CG tensors referenced at runtime, cross-graph references after construction. The ranker's only runtime connection to CG is (1) the candidate indices from precompute, (2) precomputed features in the parquet, (3) static feature data both models read from disk.

### Fair-α comparison (READ THIS)

**Critical methodological rule.** CG v5 is trained with `popularity_alpha = 0.4`. The α adjustment intentionally trades offline metrics (Recall, NDCG, MRR) for canary quality — a model trained at α=0 has *better offline numbers* but worse genre discrimination on niche tastes.

So: **do not compare a ranker trained at α=0 against CG trained at α=0.4 and declare victory.** That is not a fair comparison — you're measuring against a CG deliberately handicapped on offline metrics. The +Δ NDCG is the gap between α=0 and α=0.4 retrieval, not real ranker value.

**Two scoreboards, two CG checkpoints.** Offline metrics and canary quality answer different questions and need different comparators:

1. **Offline metrics (Recall/NDCG/MRR)** — compare ranker α=0 against a CG α=0 checkpoint. This is the *only* meaningful fair comparison for offline numbers, because the α adjustment intentionally suppresses popular items at training time (Menon Path 2), which is exactly what offline metrics reward retrieving. Any ranker vs. CG α=0.4 offline delta is mostly retrieval headroom, not ranker value.
2. **Canary quality** — compare ranker output against prod CG α=0.4 lists side-by-side. Different question: "does the ranker's content modeling produce qualitatively better recommendations than what we ship today?"

Do not conflate the two. The throwaway α=0 CG is *not* a deployment candidate — it has worse canary quality on niche tastes (which is why prod is α=0.4 in the first place). It exists solely to give the ranker an honest offline yardstick.

**Throwaway CG α=0 conventions:**
- Save as `checkpoints/cg_alpha0_eval_baseline_<date>.pth` (or similar — make the role obvious from the filename).
- Add `"do_not_export": true` and `"role": "ranker_offline_baseline"` to the `_config.json` sidecar.
- Never run `python main.py export` against it. Never promote to streamlit. Future-you in 6 months will not remember which checkpoint is which without these guardrails.

**On BCE+Menon (ranker α>0):** the movie repo found BCE+Menon incompatible (NDCG collapse, train/val loss gap blew open). Re-testing on games is *optional* — if you only care about offline lift, skip it and go straight to the ranker α=0 vs CG α=0 comparison. Only train ranker α=0.4 if you want a canary-parity ranker for the second scoreboard. Games trains fast enough that re-deriving the finding is cheap if you're curious, but it's not on the critical path.

**So the planned CG side-quest is: train one throwaway α=0 CG checkpoint.** That's the offline comparator for every ranker experiment that follows.

---

## 3. Architecture

All dims and tower shapes below match the current `src/model.py` (V5) and `src/train.get_config()`. Confirmed values: `tag_embedding_size=32`, `user_genre_embedding_size=32`, `user_tag_embedding_size=32`, `item_genre_embedding_size=8`, `developer_embedding_size=12`, `item_year_embedding_size=8`, `price_embedding_size=4`, `item_id_embedding_size=32`. Tower hidden dims (hardcoded inside `src/model.py`): `tag_hidden=128` for `item_tag_tower`, `tag_ctx_hidden=256` for `user_tag_tower`, `genre_hidden=128` for `user_genre_tower`.

### Per-game static buffers (registered on the ranker)

The ranker owns its own buffers built from `FeatureStore` at construction (same pattern as movies' `year_buffer`). All non-persistent — rebuilt on load from `feature_store.pt`. Indexing by `cand_idx` is how `item_embedding()` gets per-game metadata at score time, so the call signature is simply `item_embedding(cand_idx)` (no separate `target_year_idx` / `target_dev_idx` / `target_price` args — matches movies' ranker idiom):

```python
self.register_buffer('game_year_idx',     year_idx,     persistent=False)  # (n_games+1,) int64
self.register_buffer('game_dev_idx',      dev_idx,      persistent=False)  # (n_games+1,) int64
self.register_buffer('game_price_idx',    price_idx,    persistent=False)  # (n_games+1,) int64
self.register_buffer('game_tag_matrix',   tag_matrix,   persistent=False)  # (n_games+1, n_tags)   float32 — TF-IDF
self.register_buffer('game_genre_matrix', genre_matrix, persistent=False)  # (n_games+1, n_genres) float32 — one-hot
```

CG already registers `game_tag_matrix` and `game_genre_matrix` on the model — the ranker mirrors these exactly (same names, same dtypes) so the towers consuming them stay semantically valid after warm-start. The padding row (index `n_games`) is appended to each buffer.

### Deep concat layout (mirrors v5 CG towers)

```
USER SIDE — mirrors v5 CG user tower (no LayerNorm; v5 removed it):
  pool_liked        : 32   sum(item_id_lookup[liked_ids])                         ← shared lookup
  pool_disliked     : 32   sum(item_id_lookup[disliked_ids])
  pool_full         : 32   sum(item_id_lookup[full_ids])
  pool_playtime     : 32   playtime-weighted sum (weights pre-normalized in dataset)
  user_genre_emb    : 32   2-layer: Linear(2*n_genres → 128) → ReLU → Linear(128 → 32) → ReLU
                            input = in-model genre debiasing — uses game_genre_matrix[X_hist_full],
                            X_hist_playtime_weights, X_user_avg_log (same code path as CG)
  user_tag_emb      : 32   2-layer: Linear(n_tags → 256) → ReLU → Linear(256 → 32) → ReLU
                            input = sum of game_tag_matrix[X_hist_full] (in-model)
  X_user_avg_log    :  1   raw per-user avg-log-playtime scalar — see invariant 8 (parity+1)

ITEM SIDE — mirrors v5 CG item tower:
  item_id_emb       : 32   item_id_lookup(cand_idx) → Linear(32 → 32) → ReLU       ← shared lookup
  item_genre_emb    :  8   Linear(n_genres → 8) → ReLU
                            input = game_genre_matrix[cand_idx]
  item_tag_emb      : 32   2-layer: Linear(n_tags → 128) → ReLU → Linear(128 → 32) → ReLU
                            input = game_tag_matrix[cand_idx]
  item_dev_emb      : 12   dev_lookup(game_dev_idx[cand_idx]) → Linear(12 → 12) → ReLU
  year_emb          :  8   year_lookup(game_year_idx[cand_idx]) → Linear(8 → 8) → ReLU
  price_emb         :  4   price_lookup(game_price_idx[cand_idx]) → Linear(4 → 4) → ReLU

TOTAL deep concat:
  user side  = 4×32 + 32 + 32 + 1 = 193
  item side  = 32 + 8 + 32 + 12 + 8 + 4 = 96
  total      = 289

Deep MLP:  Linear(289 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 64) → ReLU
           → deep_out (64)
Wide:      cat(cross_features)  — bypasses MLP, direct to head
Head:      Linear(64 + |wide| → 1) → raw logit
```

### Tower architectures (match CG exactly)

CG has a mix of single-layer and 2-layer towers. Using single-Linear ranker towers where CG has 2 would silently drop the second-layer warm-start to random init — losing the bulk of CG's content-tower learning. Mirror these exact shapes:

**2-layer towers** (hidden dims hardcoded in `src/model.py`):

| Tower | Architecture |
|---|---|
| `item_tag_tower` | `Linear(n_tags → 128) → ReLU → Linear(128 → 32) → ReLU` |
| `user_tag_tower` | `Linear(n_tags → 256) → ReLU → Linear(256 → 32) → ReLU` |
| `user_genre_tower` | `Linear(2*n_genres → 128) → ReLU → Linear(128 → 32) → ReLU` |

**1-layer towers** (`Linear → ReLU`):

| Tower | Architecture |
|---|---|
| `item_id_tower` | `Linear(32 → 32) → ReLU` |
| `item_genre_tower` | `Linear(n_genres → 8) → ReLU` |
| `developer_tower` | `Linear(12 → 12) → ReLU` |
| `year_tower` | `Linear(8 → 8) → ReLU` |
| `price_tower` | `Linear(4 → 4) → ReLU` |

Every lookup-fronted feature (id, dev, year, price) is `Embedding → Linear+ReLU` as **two separate `nn.Module`s**, not a single fused Embedding. That's how CG names them and what the warm-start map below copies.

### Implementation invariants

1. **Ranker owns its own `item_id_lookup`** (`nn.Embedding(n_games+1, 32, padding_idx=n_games)`). Shared across all 4 user pools and the item-side `item_id_tower`. Same shared-lookup pattern as CG.
2. **No LayerNorm on pools.** V5 CG removed LayerNorm after pools — the ranker must do the same. Partial parity invalidates ablations.
3. **No `F.normalize` anywhere in the ranker.** CG applies `F.normalize` at the end of each projection MLP because CG's score is cosine. The ranker scores `BCE + raw logits` and has no projection MLP at all — user_concat and item_concat go straight into the deep MLP. `F.normalize` would distort the BCE objective.
4. **Sub-tower init:** Xavier uniform `gain=0.1` on per-feature linears; `gain=1.0` on deep MLP + head; Embedding tables `gain=0.01`. (Same recipe as CG, minus the final projection — which the ranker doesn't have.)
5. **Year / dev / price are bucketed and embedded** — the ranker never sees raw values, only bucket indices from `game_year_idx` / `game_dev_idx` / `game_price_idx` buffers. Bucket boundaries are read from FeatureStore and must match CG's exactly (otherwise warm-started Embedding rows index the wrong buckets).
6. **No timestamp tower.** Steam interactions have no timestamps. Do not add one in the ranker.
7. **In-model genre context is computed inside the user-side forward pass** — same debiasing logic as CG's `user_embedding` (uses `game_genre_matrix[X_hist_full]`, `X_hist_playtime_weights`, and `X_user_avg_log`), feeding `user_genre_tower`. `X_user_avg_log` is therefore already a required user-side input — see invariant 8.
8. **`X_user_avg_log` is parity+1.** CG uses this scalar internally for genre debiasing and never exposes it. The ranker also concatenates it as a 1-dim raw scalar into the user concat. Zero compute cost, gives the deep MLP / head a direct read of "heavy player vs light player" without having to infer it through the genre tower. This is a deliberate +1 expansion beyond strict CG parity — flag in run notes, not a bug.
9. **Buffers built from FeatureStore, not transferred from CG.** `game_tag_matrix`, `game_genre_matrix`, and the new `game_year_idx` / `game_dev_idx` / `game_price_idx` buffers are constructed at ranker init from `feature_store.pt` (non-persistent). They must use the same vocab orderings CG used — otherwise the warm-started towers consume rows in the wrong slots.

### Warm-start mapping (init from v5 CG state_dict)

Every CG parameter with a shape-equivalent home in the ranker is copied at construction. **Both** `*_lookup.weight` **and** `*_tower.0.{weight,bias}` (plus `.2.{weight,bias}` for 2-layer towers) must transfer — missing a row silently drops that component to random init.

**Lookups** (Embedding tables):

| CG key | Ranker key |
|---|---|
| `item_embedding_lookup.weight` | `item_id_lookup.weight` |
| `developer_embedding_lookup.weight` | `developer_lookup.weight` |
| `year_embedding_lookup.weight` | `year_lookup.weight` |
| `price_embedding_lookup.weight` | `price_lookup.weight` |

**1-layer towers** (one Linear+ReLU):

| CG keys | Ranker keys |
|---|---|
| `item_embedding_tower.0.{weight,bias}` | `item_id_tower.0.{weight,bias}` |
| `item_genre_tower.0.{weight,bias}` | `item_genre_tower.0.{weight,bias}` |
| `developer_tower.0.{weight,bias}` | `developer_tower.0.{weight,bias}` |
| `year_embedding_tower.0.{weight,bias}` | `year_tower.0.{weight,bias}` |
| `price_embedding_tower.0.{weight,bias}` | `price_tower.0.{weight,bias}` |

**2-layer towers** (two Linears — copy both):

| CG keys | Ranker keys |
|---|---|
| `item_tag_tower.0.{weight,bias}` | `item_tag_tower.0.{weight,bias}` |
| `item_tag_tower.2.{weight,bias}` | `item_tag_tower.2.{weight,bias}` |
| `user_tag_tower.0.{weight,bias}` | `user_tag_tower.0.{weight,bias}` |
| `user_tag_tower.2.{weight,bias}` | `user_tag_tower.2.{weight,bias}` |
| `user_genre_tower.0.{weight,bias}` | `user_genre_tower.0.{weight,bias}` |
| `user_genre_tower.2.{weight,bias}` | `user_genre_tower.2.{weight,bias}` |

**Not transferred** (random init in ranker):
- Deep MLP, head, cross-feature weights — no CG counterpart.
- `user_projection.*` and `item_projection.*` from CG — the ranker has no projection MLP (the deep MLP takes its place, and is structurally different: different input dim, no normalization, no cosine score).
- All buffers (`game_tag_matrix`, `game_genre_matrix`, `game_year_idx`, `game_dev_idx`, `game_price_idx`) — built fresh from FeatureStore.

`_warm_start_from_cg` should log transferred-vs-fallback per key; the transferred count should equal the row count of the three tables above (4 + 5 + 6 = 15 unique CG keys, expanding to 15 tensor-pair transfers). Anything short of that means a shape drift to fix before training.

### Why Wide & Deep

Cross-feature scalars (tag cosine, genre Jaccard, developer match, etc.) get a single learned weight in the head — a direct gradient path. Without the wide bypass, those signals would have to compete against ~290 dims for attention in the first hidden layer and get washed out during backprop. Each cross feature is essentially a hand-crafted interaction the deep MLP would otherwise have to learn from scratch.

---

## 4. E2E Evaluation Rule (Golden Rule of Two-Stage Systems)

**Ranker_Hit@K ≤ CG_Recall@K_cand in production.** If CG didn't retrieve the label, the ranker never sees it.

In offline eval: if `cg_label_rank >= n_cand`, rank is set to `n_cand + 1` (score = 0 for all metrics). Both CG baseline and ranker numbers use this same ceiling so comparison is apples-to-apples.

`cg_label_rank = n_cand` is ambiguous (could be rank K or > K); treated conservatively as "not found."

Eval outputs `Recall@n_cand` as the production ceiling.

---

## 5. Repo Structure (proposed)

```
ranker/
├── precompute.py     ← CG scoring: builds ranker_candidates_{train,val}.parquet
├── dataset.py        ← RankerDataset, sample_batch (Mixed Negative Sampling)
├── model.py          ← WideDeepRanker
├── train.py          ← BCE training loop
├── evaluate.py       ← NDCG@K, MRR, Hit@K, CG baseline, E2E ceiling
├── canary.py         ← side-by-side CG vs Ranker top-N for synthetic users
├── main.py           ← entry point
├── eval_results/
└── canary_results/

data/
├── ranker_candidates_train.parquet
├── ranker_candidates_val.parquet
└── ranker_game_stats.parquet         ← global avg_log_playtime, log1p(interaction_count) per game
```

Do not modify anything in `src/` — the CG code is read-only.

---

## 6. Stage 0: Precompute (NOT STARTED)

**Configuration:** Pick `TOP_K_CANDIDATES`. The games corpus is only 5,437 items, so K=250 covers ~4.6% — much denser coverage than movies (250 of 9,375 = 2.7%). Recommend K=100 to keep the reranking problem meaningful; verify Recall@100 first via offline_eval before locking in.

**Train/val split:** Reuse the existing 90/10 user-level split from CG with the same seed. Do not introduce a new split.

**Parquet schema (target):**

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | |
| `rollback_n` | int | position in user's history (after seeded shuffle, as CG eval does) |
| `label_item_idx` | int | positive item corpus index |
| `neg_item_idxs` | list[int] | K-1 hard negatives in CG score order |
| `cg_label_rank` | int | label's rank in full corpus (1-indexed, capped at K) |
| `cg_label_score` | float | CG dot for the label, pre-masking |
| `cg_neg_scores` | list[float] | CG dot per negative |
| `tag_cosine_label` | float | cosine(user_tag_pool, label_tag_vec) |
| `tag_cosine_negs` | list[float] | tag cosine per negative |
| `user_avg_log_playtime` | float | per-user mean log(1+hours) |
| `user_interaction_count` | int | |
| `X_hist_liked` | list[int] | padded indices, MAX_HISTORY_LEN |
| `X_hist_disliked` | list[int] | padded indices, MAX_HISTORY_LEN |
| `X_hist_full` | list[int] | padded indices, MAX_HISTORY_LEN |
| `X_hist_playtime_weights` | list[float] | normalized weights for playtime pool, MAX_HISTORY_LEN |

**Tag cosine choice:** Compute over the **raw TF-IDF tag vectors** stored in `game_tag_matrix` (the registered buffer CG already uses) — *not* off the tag tower outputs.

This is the direct analog of movies' `genome_cosine`. Movies' genome scores are already dense, per-tag scores in [0, 1] — cosine over the raw vector is meaningful. Games' `game_tag_matrix` rows are TF-IDF-weighted tag vectors (`positional_weight * log(N / df)`, per `preprocess.py`) — same shape of signal: dense per-game weighted scores. Using raw TF-IDF gives a model-independent precompute (the parquet column doesn't break if the tag tower's hidden dims change) and matches the precompute idiom in `ranker/precompute.py` from movies.

```
user_tag_pool_tfidf  = playtime-weighted sum of game_tag_matrix[X_hist_full] rows
                       (or rating-weighted to mirror movies' genome pool more directly)
tag_cosine_label     = cosine(user_tag_pool_tfidf, game_tag_matrix[label_idx])
tag_cosine_negs[i]   = cosine(user_tag_pool_tfidf, game_tag_matrix[neg_idxs[i]])
```

The tag tower output is still used in the deep concat (`user_tag_emb`, `item_tag_emb`) — but the wide-bypass cross feature reads from the raw matrix.

**Stability concern:** seed the CG model load + rollback shuffle. Steam's no-timestamp dataset means rollback order is shuffle-determined — different shuffles produce different parquets. Use the seed CG eval uses (or fix one in `precompute.py` and document it).

---

## 7. Training Stack (planned)

### Mixed Negative Sampling (MNS)

`sample_batch` uses 50% hard / 50% easy negatives:
- **Hard** (50%): CG-retrieved candidates from parquet.
- **Easy** (50%): uniform random corpus items (anchor global decision boundary).

Easy negatives get `tag_cosine = 0.0` (CG never scored them). Adjust `easy_neg_frac` if the model struggles — movie repo notes easy negatives can become uninformative.

### Loss

`BCEWithLogitsLoss` only. Never `BCELoss`. Never sigmoid in `WideDeepRanker.forward()`.

### Training config (proposed defaults — tune after first run)

```
lr:               1e-3  (Adam + cosine annealing, T_max = training_steps)
weight_decay:     0.0
batch_size:       4096
training_steps:   100_000      ← start lower than movies; games dataset is ~3x smaller
grad_clip:        1.0
hidden_dims:      [256, 128, 64]
dropout:          0.0
popularity_alpha: see §2 fair-α rule — start with 0.4 to match CG, verify BCE+Menon works
easy_neg_frac:    0.5
```

Save arch params (hidden_dims, embedding sizes, cross-feature count, **popularity_alpha**) in a `_config.json` sidecar alongside each checkpoint. The α value matters at eval time (for choosing the right CG comparator).

### Eval

- NDCG@K, MRR, Hit@K for K ∈ {1, 5, 10, 20, 50, 100}.
- CG baseline computed with the same E2E ceiling.
- `Recall@K_cand` printed as the production ceiling.
- Auto-find the most recent checkpoint; read arch params from the JSON sidecar.
- **Always report which CG baseline is being compared against** (α=0 vs α=0.4). Print the comparator's α in the eval output header.

---

## 8. Popularity Alpha — Plan of Record

**Default: ranker α=0, compared against throwaway CG α=0** (the offline-metrics scoreboard). This is the critical-path comparison and the only one needed to validate ranker lift.

**Optional side-quest: ranker α=0.4 (BCE+Menon viability test).** Movie repo finding: BCE + Menon logit adjustment caused NDCG collapse (−7.1% vs CG), train/val loss gap blew open. Hypothesis was that in BCE at 1:K-1 ratio, the model is trained to undervalue popular items (which appear constantly as true positives), the opposite of what Menon does in softmax over all items.

Reasons games might differ:

- Games corpus is smaller (5.4k vs 9.4k items). Different positive:negative density per batch.
- Games dataset has the Valve DENYLIST — the most extreme popularity heads are already removed. Menon may have less work to do, less harm to inflict.
- Games uses cosine similarity (F.normalize on both towers) — different score-scale dynamics than movies.

Only worth running if you specifically want a canary-parity ranker to compare against prod CG α=0.4. Skip otherwise — the α=0 vs α=0 offline comparison answers the "does the ranker help?" question on its own.

If you do run it: train ranker α=0.4, measure offline + canary. If train/val loss diverges or NDCG collapses, the finding transfers and there's nothing more to do. If it works cleanly, you have an α=0.4 ranker for canary-parity comparison against prod CG.

---

## 9. CG Score — Gated, Re-enable Last

Like in movies, do not feed the CG score (`cg_label_score` / `cg_neg_scores`) into the ranker until the ranker beats a fair CG on content features alone. The CG score is a circular feature — the ranker can trivially learn "follow CG" and "beat" CG only because it's wrapping CG's output. Re-enable as the final improvement, after content-based wins are demonstrated.

---

## 10. Cross-Feature Roadmap

Each cross feature is added one at a time and measured against the previous run.

| Priority | Feature | Path | Formula |
|----------|---------|------|---------|
| 1 | **Tag Cosine** | Wide | precomputed: `cosine(user_tag_pool, item_tag_vec)` |
| 2 | **Genre Jaccard** | Wide | `Jaccard(user_genre_set, item_genre_set)` — sharp genre signal |
| 3 | **Developer Affinity** | Wide | `1 if item_dev in user_history_devs else 0` (or count-weighted) — strong games signal: developer loyalty is a real preference axis |
| 4 | **Price Match** | Wide | `abs(user_mean_price_bucket - item_price_bucket)` — F2P vs AAA buyer distinction |
| 5 | **Era Gap** | Wide | `abs(user_mean_year_norm - item_year_norm)` — new release vs retro preference |
| 6 | **Playtime Calibration** | Wide | `user_avg_log_playtime - item_global_avg_log_playtime` — heavy-hours user vs short-session user |
| 7 | **Dislike Similarity** | Wide | `cosine(user_disliked_pool, item_id_emb)` — veto signal |
| 8 | **Tag Peak Match** | Wide | `max(user_tag_profile * item_tag_vec)` — single dominant tag overlap |
| 9 | **Recent Game Similarity** | Wide | `dot(last_played_id_emb, item_id_emb)` — sequential next-game signal (no timestamps, so "last" = end of shuffled context) |
| 10 | **Popularity Match** | Wide | `abs(user_avg_log_count - item_log_count)` — user preference for popular vs obscure |
| 11 | **CG Score** | Wide | raw CG dot-product (re-enable last; see §9) |
| 12 | **Genre Diversity** / **Tag Entropy** / **History Confidence** | Deep | user-state scalars added to the deep concat |
| 13 | **DCN V2 cross network** | architecture | replace deep MLP with explicit cross layers — only if features stop helping |

### Wide-feature normalization

Wide features beyond cosines/Jaccard must be **normalized before concatenation** using fixed statistics registered as **persistent** model buffers (not BatchNorm — train/eval batch composition differs). Compute per-feature mean/std from a single pass over training data, register with `register_buffer(name, tensor)` (persistent — the default). **Do not pass `persistent=False`** — that excludes the buffer from `state_dict`, so the stats reset to constructor defaults on load and silently degrade the model.

```
Expected ranges (pre-normalization):
  Tag Cosine           [−1, 1]       → no normalization
  Genre Jaccard        [0, 1]        → Z-score
  Developer Affinity   [0, 1]        → Z-score (heavy zero mass — may need separate treatment)
  Price Match          [0, ~8]       → Z-score
  Era Gap              [0, 1]        → Z-score (after normalizing year to [0,1])
  Playtime Calibration [~−5, +5]     → Z-score
  Dislike Similarity   [−1, 1]       → no normalization
  Tag Peak Match       [0, 1]        → Z-score
  Recent Game Sim      [unbounded]   → Z-score
  Popularity Match     [0, ~8]       → Z-score
```

Initialize new wide-feature weights at 0.1 — small non-zero signal without swamping the deep path.

### Domain-specific notes

- **Developer Affinity is a strong signal in games** that has no analog in movies. Studios have recognizable styles (FromSoftware, Larian, Paradox) and users follow them. Worth prioritizing early.
- **Price Match is a games-specific signal.** Users who play only free-to-play vs users who buy $60 AAA titles are very different segments. Movies don't have this.
- **No "genome scores."** Tags are the dense content signal here, and they're already in the deep concat. Tag-related cross features (cosine, peak match) are the equivalent of movie's genome features.
- **No timestamps means no time-decay features.** "Recent" can only mean "end of shuffled context" — useful but noisier than real recency.

---

## 11. Ablation Sequence (planned)

| Phase | Experiment | Hypothesis |
|-------|-----------|------------|
| 0 | **Throwaway CG α=0 reference run** | Establish the offline-metrics scoreboard (§2). Save as `cg_alpha0_eval_baseline_<date>.pth` with `"do_not_export": true` in the sidecar. Never deploy. |
| 1 | **CG-parity baseline + Tag Cosine wide bypass** | Recover at least CG-equivalent offline metrics. |
| 2 | **+ Genre Jaccard** | Sharpen genre precision. |
| 3 | **+ Developer Affinity** | Capture studio-loyalty signal absent from CG. |
| 4 | **+ Price Match** | Distinguish F2P / indie / AAA buyer segments. |
| 5 | **+ Playtime Calibration** | Match heavy-hours users to long-tail engagement games. |
| 6 | **+ Dislike Similarity** | Veto channel — leverages the disliked pool more aggressively. |
| 7 | **+ Tag Peak Match** | "One tag spark" signal beyond global cosine. |
| 8 | **CG Score re-enable** | Final retrieval-signal boost once content features have proven independent value. |
| Later | **DCN V2** | If content features saturate. |

**Change one thing per experiment.** Measure NDCG@10 delta vs the previous run before proceeding.

---

## 12. Experiment Discipline Rules

1. **One change at a time.** Every run isolates exactly one variable.
2. **Fair-α rule.** Always report which CG baseline (α=?) you are comparing against. Beating a CG that's tuned for canary quality on offline metrics is not a real win.
3. **Beat CG on content features before re-enabling CG score.** Earn improvements from independent signal.
4. **No `src/` modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never sigmoid in `forward()`.
7. **Batch sampling is across all tuples** — not within rollback groups (avoids K-1:1 imbalance dominating gradients).
8. **E2E ceiling always enforced** in both ranker eval and CG baseline.
9. **§3 reflects v5 CG as of this writing.** If `src/model.py` or `src/train.get_config()` changes (tower hidden dims, embedding sizes, new sub-towers), re-derive §3 *first* before changing the ranker — partial drift will silently break warm-start.
