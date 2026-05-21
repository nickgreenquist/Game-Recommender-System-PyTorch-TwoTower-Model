# Steam Ranker: Implementation Plan

## Status (2026-05-21)

Phase A ✓, Phase B Bucket 1 ✓, Phase B Bucket 2 ✓, Phase B Bucket 3 ✗ (dropped), Phase B Bucket 4 ✗ (dropped), Phase B Bucket 5 ✓, Phase B Bucket 6 ✓.

**A-N2** (Phase A exit, tag_cosine only): NDCG@10 0.0741 vs CG α=0 0.0752 (Δ −1.5%). Proved a Wide & Deep MLP can effectively match the CG two-tower dot product on identical features.

**Bucket 1** (A-N2 + genre_overlap + tag_overlap + dev_affinity, 4 wide cross features): **NDCG@10 0.0822** vs CG α=0 0.0752 (Δ **+9.3%**) and vs A-N2 0.0741 (Δ **+10.9%**). MRR 0.0779 (+7.3% vs CG). Pure-reranking subset NDCG@10 0.1487 vs 0.1361 (+9.3%) — the lift is genuine reranking signal, not an E2E ceiling artifact. The three new categorical-overlap features together earned their seat and ranker now decisively beats the α=0 CG yardstick.

**Bucket 2** (Bucket 1 + 6 features: genre/tag/dev × {Liked, Recent-3 Liked}, 10 wide cross features total): **NDCG@10 0.0828** vs Bucket 1 0.0822 (Δ **+0.7%**). MRR 0.0784 (+0.6%). Pure-reranking NDCG@10 0.1499 vs 0.1487 (+0.8%). Direction is positive and consistent across every metric (no regressions), but magnitudes are tiny compared to Bucket 1's +10.9%. Bucket 1 already extracted most of the categorical-overlap signal from this data — Liked and Recent-3 slices add a thin layer on top. Bundle earns its seat, barely.

**Bucket 3 — DROPPED** (Bucket 2 + 3 features: genre/tag/dev × Disliked, 13 wide cross features total): **NDCG@10 0.0824** vs Bucket 2 0.0828 (Δ **−0.5%**), MRR 0.0781 vs 0.0784 (−0.4%), pure-reranking NDCG@10 0.1492 vs 0.1499 (−0.5%). Regressed on every headline metric vs Bucket 2 and essentially tied with Bucket 1 (+0.2% / +0.3%). The 3 disliked columns took capacity away from the existing 10 and contributed no usable signal. Root cause: Steam's "disliked" partition is too noisy to earn its columns — the `recommend==False` signal is sparse (most users don't review), the `0.1 < hours < 1.0` heuristic captures "tried it, didn't stick" which is ambiguous, and `hours ≤ user_rolling_median / 2` flags below-average games which for heavy players can include genuine favorites. Reverted in code; disliked variants are now dropped from later buckets (e.g. Bucket 4's deliberate exclusion stays permanent).

**Bucket 4 — DROPPED** (Bucket 2 + 6 features: genre/tag × {full, liked, recent-3} on per-developer averaged genre/tag vectors, 16 wide cross features total): **NDCG@10 0.0827** vs Bucket 2 0.0828 (Δ **−0.1%**), MRR 0.0785 vs 0.0784 (+0.1%), pure-rerank NDCG@10 0.1496 vs 0.1499 (−0.2%). Mixed-sign micro-deltas, regressions on the two headline NDCG metrics. Two independent training runs (same seed) reproduced the flat pattern — not seed variance. Root cause: the deep tower's `developer_lookup` already encodes studio identity end-to-end via warm-start; averaging genre/tag vectors per developer and inner-producting against the candidate adds no signal the dev embedding wasn't already representing implicitly. 6 columns of capacity displacement, ~0 net lift. Reverted in code; the dev-catalog signal class is permanently dropped (see §10 — don't retry in a different shape).

**Bucket 5** (Bucket 2 + 5 numeric-match scalars: price / era / playtime-cal-median / popularity / sentiment, 15 wide cross features total): **NDCG@10 0.0866** vs Bucket 2 0.0828 (Δ **+4.6%**). MRR 0.0814 (+3.8%). Pure-reranking NDCG@10 0.1567 vs 0.1499 (+4.5%). **Uniform +4–5% lift across every headline metric** in both E2E and pure-rerank views — first bucket since Bucket 1 to deliver meaningful lift on top of the prior baseline. Hit@50 E2E jumped +0.0138 where Bucket 2 had flattened (+0.0030 vs CG), showing the numeric-match features are surfacing real targets that pure overlap couldn't. Canary quality holds or improves on every user type (JRPG, Racing, Survival, Management materially cleaner than CG; no catastrophic regressions on niche tastes). First bucket to use the Z-score normalization infrastructure (persistent `wide_norm_mean` / `wide_norm_std` buffers populated once at training start from train-parquet stats). Reset the next-bucket baseline.

**Bucket 6** (Bucket 5 + 8 niche feature crosses: 4 concepts × {full, liked}, 23 wide cross features total): **NDCG@10 0.0867** vs Bucket 5 0.0866 (Δ **+0.1%**). Offline metrics are flat at the noise floor — every E2E and pure-rerank delta sits in ±0.001 across K∈{1, 5, 10, 20, 50}, both signs scattered. **Canary tells a different story:** Fighting Lover ranker top-10 drops all 3 Final Fantasy / Tales JRPGs that B5 was surfacing (FF Type-0, FF VIII, Tales of Symphonia) and replaces them with METAL SLUG, METAL SLUG X, UNDER NIGHT IN-BIRTH, Naruto Storm 4 — proper fighters/arcade. Civ Lover promotes Galactic Civ III (anchor sibling) #3 → #1 and brings Master of Orion + Victoria II up. Western RPG, FPS show smaller niche-coherence improvements. **Kept on canary alone** — offline-flat-but-canary-better is the inverse of Bucket 4 ✗ (flat-everywhere) and falls under the V5 PROD vs V5 α=0 precedent: canary quality on niche tastes is what users notice. Inference cost: negligible (~120k extra FLOPs/query, sub-millisecond on MPS — see §8 Bucket 6 entry). Bucket 7 (Item-Intrinsic Priors) was dropped in planning; 2 of its 3 features absorbed here as `niche_dev_match` and `max_tag_idf_match`.

**Bucket 7 (Item-Intrinsic Priors) dropped in planning (2026-05-20)** — 2 of its 3 features (`dev_catalog_size`, `candidate_max_tag_idf`) absorbed into Bucket 6 as proper user × item scalar crosses (`niche_dev_match`, `max_tag_idf_match`), which is the correct shape for the linear wide head. The remaining feature (`dev_specialization`) is the only one without an obvious user-side analog and isn't worth a bucket on its own. Could return later in a different shape (e.g. `|user_genre_specialization − item_dev_specialization|`) if downstream buckets disappoint.

**Ranker is not yet wired into serving.** Streamlit currently runs CG-only retrieval; integrating the ranker requires non-trivial app work (load both checkpoints, shared feature engineering for the user-side cross feature inputs, two-stage scoring pipeline). Deliberately deferred — Phase B keeps focus on offline lift until the bucket roadmap saturates.

> **Terminology note:** earlier drafts called these "pools." That's wrong — "pool" in this codebase means an embedding aggregation (the deep tower's `pool_liked` / `pool_disliked` / `pool_full` / `pool_playtime`, which sum item embeddings). The cross features compute weighted overlap and categorical affinity *directly over the history arrays* — no embedding aggregation happens. The code uses `weighted_overlap` / `dev_affinity` with `history_indices` / `history_weights`.

---

## 1. Pipeline

Two-stage retrieve-and-rank:

1. **CG** (v5 softmax two-tower, 4-pool user tower, F.normalize on both towers, 128-dim) — retrieves top-100 candidates per rollback example.
2. **Ranker** (Wide & Deep MLP) — reranks the 100 candidates using richer features.

### CG checkpoints

- **Prod CG α=0.4** — what serves real users. Trained with Menon Path 2 popularity correction; trades offline metrics for canary quality on niche tastes. Not the right comparator for the ranker on offline metrics (α=0.4 is deliberately handicapped on Recall/NDCG/MRR).
- **Throwaway CG α=0** — `best_triple_full_softmax_popularity_alpha_00_<date>.pth`. Never exported, never promoted. Exists solely as the honest offline-metrics yardstick. The ranker runs at α=0 and is measured against this checkpoint.

CG v5 prod baseline (α=0.4, for reference):

| K | Recall@K | NDCG@K |
|---|---:|---:|
| 1 | 0.0226 | 0.0226 |
| 5 | 0.0741 | 0.0481 |
| 10 | 0.1253 | 0.0645 |
| 20 | 0.2059 | 0.0848 |
| 50 | 0.3673 | 0.1166 |

MRR: 0.0611 (random: 0.0017). The α=0 throwaway sits ~10–25% above these on Recall/NDCG (see `CLAUDE.md` for the full α=0 table).

---

## 2. Core Principles

### CG-parity baseline first

The ranker contains every CG input/feature, projected through the same per-feature towers CG uses, with the same dimensions and the same warm-started weights. Cross features come *on top* — they are the differentiator, not a replacement.

### Wide & Deep architecture

The ranker concatenates all per-feature embeddings (user-side + item-side) into ONE vector that feeds a deep MLP. Cross features bypass the MLP and go straight to the head — giving each one a direct learned weight. Without the wide bypass, cross-feature scalars would have to compete against ~290 dims for attention in the first hidden layer and get washed out during backprop.

### No CG coupling at runtime

The ranker owns its own copies of every parameter (its own `item_id_lookup`, `developer_lookup`, `developer_tower`, `item_tag_tower`, etc.). Warm-starting weights from CG at init is one-time copy at construction; the tensors then live entirely in the ranker `state_dict` and train freely. The ranker's only runtime connection to CG is (1) the candidate indices from precompute, (2) precomputed features in the parquet, (3) static feature data both models read from disk.

### Fair-α rule

Always report which CG checkpoint (α=0 vs α=0.4) the ranker is being compared against. Ranker α=0 vs CG α=0 is the only meaningful offline comparison. Ranker α=0 vs CG α=0.4 is not a real win — the delta is mostly retrieval headroom from α=0.4's deliberate handicap.

---

## 3. Architecture

All dims and tower shapes match `src/model.py` (V5) and `src/train.get_config()`: `tag_embedding_size=32`, `user_genre_embedding_size=32`, `user_tag_embedding_size=32`, `item_genre_embedding_size=8`, `developer_embedding_size=12`, `item_year_embedding_size=8`, `price_embedding_size=4`, `item_id_embedding_size=32`. Tower hidden dims (hardcoded inside `src/model.py`): `tag_hidden=128`, `tag_ctx_hidden=256`, `genre_hidden=128`.

### Per-game static buffers (registered on the ranker)

Built from `FeatureStore` at construction. All non-persistent — rebuilt on load from `feature_store.pt`. Indexing by `cand_idx` is how `item_embedding()` gets per-game metadata at score time.

```python
self.register_buffer('game_year_idx',     year_idx,     persistent=False)  # (n_games+1,) int64
self.register_buffer('game_dev_idx',      dev_idx,      persistent=False)  # (n_games+1,) int64
self.register_buffer('game_price_idx',    price_idx,    persistent=False)  # (n_games+1,) int64
self.register_buffer('game_tag_matrix',   tag_matrix,   persistent=False)  # (n_games+1, n_tags)   float32 — TF-IDF
self.register_buffer('game_genre_matrix', genre_matrix, persistent=False)  # (n_games+1, n_genres) float32 — one-hot
```

CG already registers `game_tag_matrix` and `game_genre_matrix` — the ranker mirrors them exactly (same names, dtypes, vocab orderings) so warm-started towers consume the right rows.

### Deep concat layout (strict v5 CG parity)

```
USER SIDE — mirrors v5 CG user tower (no LayerNorm):
  pool_liked        : 32   sum(item_id_lookup[liked_ids])                      ← shared lookup
  pool_disliked     : 32   sum(item_id_lookup[disliked_ids])
  pool_full         : 32   sum(item_id_lookup[full_ids])
  pool_playtime     : 32   playtime-weighted sum (weights pre-normalized in dataset)
  user_genre_emb    : 32   2-layer: Linear(2*n_genres → 128) → ReLU → Linear(128 → 32) → ReLU
                            input = in-model genre debiasing — uses game_genre_matrix[X_hist_full],
                            X_hist_playtime_weights, X_user_avg_log (same code path as CG).
                            X_user_avg_log is consumed internally for the debiasing math
                            but NOT concatenated into user_concat — strict CG parity.
  user_tag_emb      : 32   2-layer: Linear(n_tags → 256) → ReLU → Linear(256 → 32) → ReLU
                            input = sum of game_tag_matrix[X_hist_full] (in-model)

ITEM SIDE — mirrors v5 CG item tower:
  item_id_emb       : 32   item_id_lookup(cand_idx) → Linear(32 → 32) → ReLU   ← shared lookup
  item_genre_emb    :  8   Linear(n_genres → 8) → ReLU                          input = game_genre_matrix[cand_idx]
  item_tag_emb      : 32   2-layer: Linear(n_tags → 128) → ReLU → Linear(128 → 32) → ReLU
                            input = game_tag_matrix[cand_idx]
  item_dev_emb      : 12   dev_lookup(game_dev_idx[cand_idx]) → Linear(12 → 12) → ReLU
  year_emb          :  8   year_lookup(game_year_idx[cand_idx]) → Linear(8 → 8) → ReLU
  price_emb         :  4   price_lookup(game_price_idx[cand_idx]) → Linear(4 → 4) → ReLU

TOTAL deep concat:
  user side  = 4×32 + 32 + 32 = 192    ← matches CG exactly
  item side  = 32 + 8 + 32 + 12 + 8 + 4 = 96
  total      = 288

Deep MLP:  Linear(288 → 256) → ReLU → Linear(256 → 128)   ← NO final ReLU/activation
           → deep_out (128)
Wide:      cat(cross_features)  — bypasses MLP, direct to head
Head:      Linear(128 + |wide| → 1) → raw logit
Score:     logit / temperature(=0.1)  — applied before softmax CE, matches CG
```

### Tower architectures

CG mixes 1-layer and 2-layer towers; use the matching shape so warm-start copies the full state.

**2-layer towers:**

| Tower | Architecture |
|---|---|
| `item_tag_tower` | `Linear(n_tags → 128) → ReLU → Linear(128 → 32) → ReLU` |
| `user_tag_tower` | `Linear(n_tags → 256) → ReLU → Linear(256 → 32) → ReLU` |
| `user_genre_tower` | `Linear(2*n_genres → 128) → ReLU → Linear(128 → 32) → ReLU` |

**1-layer towers (`Linear → ReLU`):**

| Tower | Architecture |
|---|---|
| `item_id_tower` | `Linear(32 → 32) → ReLU` |
| `item_genre_tower` | `Linear(n_genres → 8) → ReLU` |
| `developer_tower` | `Linear(12 → 12) → ReLU` |
| `year_tower` | `Linear(8 → 8) → ReLU` |
| `price_tower` | `Linear(4 → 4) → ReLU` |

Every lookup-fronted feature (id, dev, year, price) is `Embedding → Linear+ReLU` as two separate `nn.Module`s, not a fused Embedding. That's how CG names them and what the warm-start map below copies.

### Implementation invariants

1. **Ranker owns its own `item_id_lookup`** (`nn.Embedding(n_games+1, 32, padding_idx=n_games)`). Shared across all 4 user pools and the item-side `item_id_tower`.
2. **No LayerNorm on pools** (v5 CG removed it).
3. **No `F.normalize` anywhere.** CG uses cosine; the ranker uses softmax CE on `logits / temperature` (T=0.1).
4. **No final activation on the deep MLP.** A trailing ReLU would clamp `deep_out ≥ 0`, breaking parity with CG's dot product which freely spans negative values. This was an actual bug fixed during Phase A bring-up.
5. **Sub-tower init:** Xavier uniform `gain=0.1` on per-feature linears; `gain=1.0` on deep MLP + head; Embedding tables `gain=0.01`. Same recipe as CG.
6. **Year / dev / price are bucketed and embedded** — bucket boundaries from FeatureStore must match CG's exactly.
7. **No timestamp tower.** Steam has no timestamps.
8. **In-model genre context** is computed inside the user-side forward pass — same debiasing logic as CG.
9. **Buffers built from FeatureStore, not transferred from CG.** Vocab orderings must match.

### Warm-start mapping (init from v5 CG state_dict)

> **Default ON** (`get_config()['warm_start'] = True`). A from-scratch ablation produced materially worse NDCG — the deep MLP needs CG's content-tower head start.

26 tensor transfers expected (4 lookups × 1 + 5 one-layer towers × 2 + 3 two-layer towers × 4). Anything short means a shape drift or key mismatch.

**Lookups:**

| CG key | Ranker key |
|---|---|
| `item_embedding_lookup.weight` | `item_id_lookup.weight` |
| `developer_embedding_lookup.weight` | `developer_lookup.weight` |
| `year_embedding_lookup.weight` | `year_lookup.weight` |
| `price_embedding_lookup.weight` | `price_lookup.weight` |

**1-layer towers:**

| CG keys | Ranker keys |
|---|---|
| `item_embedding_tower.0.{w,b}` | `item_id_tower.0.{w,b}` |
| `item_genre_tower.0.{w,b}` | `item_genre_tower.0.{w,b}` |
| `developer_tower.0.{w,b}` | `developer_tower.0.{w,b}` |
| `year_embedding_tower.0.{w,b}` | `year_tower.0.{w,b}` |
| `price_embedding_tower.0.{w,b}` | `price_tower.0.{w,b}` |

**2-layer towers (copy both Linears):**

| CG keys | Ranker keys |
|---|---|
| `item_tag_tower.{0,2}.{w,b}` | `item_tag_tower.{0,2}.{w,b}` |
| `user_tag_tower.{0,2}.{w,b}` | `user_tag_tower.{0,2}.{w,b}` |
| `user_genre_tower.{0,2}.{w,b}` | `user_genre_tower.{0,2}.{w,b}` |

**Not transferred:** Deep MLP, head, cross-feature weights (random init — no CG counterpart); `user_projection.*` / `item_projection.*` (the ranker has no projection MLP, the deep MLP takes its place); all buffers (built fresh from FeatureStore).

---

## 4. E2E Evaluation Rule

**Ranker_Hit@K ≤ CG_Recall@K_cand in production.** If CG didn't retrieve the label, the ranker never sees it.

In offline eval: if `cg_label_rank >= n_cand`, rank is set to `n_cand + 1` (score = 0 for all metrics). Both CG baseline and ranker numbers use this ceiling so comparison is apples-to-apples.

`cg_label_rank = n_cand` is ambiguous; treated conservatively as "not found." Eval outputs `Recall@n_cand` as the production ceiling.

---

## 5. Repo Structure

```
ranker/
├── precompute.py     ← CG scoring: builds ranker_candidates_{train,val}.parquet
├── dataset.py        ← RankerDataset, sample_batch (sampled softmax)
├── model.py          ← WideDeepRanker
├── train.py          ← Sampled softmax CE training loop
├── evaluate.py       ← NDCG@K, MRR, Hit@K, CG baseline, E2E ceiling
├── canary.py         ← side-by-side CG vs Ranker top-N for synthetic users
├── main.py           ← entry point
├── eval_results/
└── canary_results/

data/
├── ranker_candidates_train.parquet
└── ranker_candidates_val.parquet
```

`src/` is read-only — the ranker doesn't modify CG code.

---

## 6. Precompute

`TOP_K_CANDIDATES=100` (1 label + 99 hard negs per row). Train/val split reuses CG's 90/10 user-level split with the same seed.

**Parquet schema:**

| Column | Type | Description |
|---|---|---|
| `user_id` | int | |
| `rollback_n` | int | position in user's history (after seeded shuffle) |
| `label_item_idx` | int | positive item corpus index |
| `neg_item_idxs` | list[int] | K-1 hard negatives in CG score order |
| `cg_label_rank` | int | label's rank in full corpus (1-indexed, capped at K) |
| `cg_label_score` | float | CG dot for the label |
| `cg_neg_scores` | list[float] | CG dot per negative |
| `tag_cosine_label` | float | `cosine(user_tag_pool_tfidf, game_tag_matrix[label])` |
| `tag_cosine_negs` | list[float] | tag cosine per negative |
| `user_avg_log_playtime` | float | per-user mean log(1+hours) |
| `user_interaction_count` | int | |
| `X_hist_liked` | list[int] | padded indices, MAX_HISTORY_LEN |
| `X_hist_disliked` | list[int] | padded indices, MAX_HISTORY_LEN |
| `X_hist_full` | list[int] | padded indices, MAX_HISTORY_LEN |
| `X_hist_playtime_weights` | list[float] | normalized weights for playtime pool |

**Tag cosine:** raw TF-IDF cosine over `game_tag_matrix` rows (the buffer CG already uses) — not over tag-tower outputs. Model-independent, so the precompute parquet doesn't break if the tag tower's hidden dims change. The tag tower output is still used in the deep concat; the wide-bypass cross feature reads from the raw matrix.

**Settings:**
- `N_SHUFFLES=3` for train, `1` for val — matches CG.
- `MAX_ROLLBACK_EXAMPLES_PER_USER=50` — matches CG.
- Label filter: `raw_hours > 0.5` AND `history[i] not in history[:i]` (dedupe guard avoids the ~0.2% Steam history duplicates from leaking the label into its own context).
- Steam has no timestamps — rollback order is shuffle-determined. Seed the CG load + rollback shuffle.

---

## 7. Training Stack (Active Config)

### Negative sampling

**Sampled softmax**, 1 label + 999 random corpus items = 1000-cand pool. No hard negs (`n_hard_negs=0`). Set in `get_config()`.

Hard-neg infrastructure (parquet `neg_item_idxs` column, `n_hard_negs` knob in `sample_batch`) is kept — Phase B may revisit the hard-neg mix once new cross features change the gradient landscape. Cheap to re-enable.

### Loss

```python
F.cross_entropy(scores / temperature, target)   # target=0 (label at col 0), temperature=0.1
```

No BCE, no `pos_weight`, no sigmoid in `forward()`.

### Config block

```
lr:               1e-3
weight_decay:     0.0
adam_eps:         1e-6
batch_size:       512
training_steps:   50_000
log_every:        500
grad_clip:        1.0
temperature:      0.1
scheduler:        CosineAnnealingLR, T_max=training_steps, eta_min=1e-4
hidden_dims:      [256, 128]                ← matches CG's projection shape (no final ReLU)
dropout:          0.0
popularity_alpha: 0.0                        ← matches the α=0 throwaway CG comparator
n_random_negs:    999
n_hard_negs:      0
warm_start:       True
```

`_config.json` sidecar alongside each checkpoint records arch params + `popularity_alpha` + `temperature` + `n_random_negs` + `n_hard_negs`. `n_hard_negs` distinguishes A-N1 (hard kept) from A-N2 (hard dropped) checkpoints.

### Eval

NDCG@K, MRR, Hit@K for K ∈ {1, 5, 10, 20, 50, 100}. CG baseline uses the same E2E ceiling. `Recall@K_cand` printed as the production ceiling. Eval output header reports which CG checkpoint (α=0 vs α=0.4) the comparison uses.

---

## 8. Phase Log

### Phase 0 ✓ — CG α=0 throwaway baseline

Trained `best_triple_full_softmax_popularity_alpha_00_20260515_084320.pth` as the offline-metrics comparator. Sidecar marked `"do_not_export": true`, `"role": "ranker_offline_baseline"`. Never deployed.

### Phase A ✓ — Strict CG parity

Goal: prove a Wide & Deep MLP can effectively match a two-tower dot product on identical data and features. Sampled softmax CE + temperature=0.1, warm-start from α=0 CG, one cross feature (`tag_cosine`), ranker α=0, all 8 per-feature towers in CG parity.

#### Chronology (full-val NDCG@10 unless noted as sampled)

| Run | Composition | Total cands | Result | Notes |
|---|---|---:|---:|---|
| Try 1 | 1 label + 99 hard | 100 | ~0.066 (sampled) | Pure listwise over CG-confusables only. Rising knee of the N-curve, not the plateau. |
| Try 2 | 1 label + 5,436 corpus | 5,437 | n/a | Effective full softmax. ~9 it/s, infeasible. Abandoned. |
| Try 3 | 1 label + 99 hard + 400 random | 500 | 0.070 (sampled) | Hard negs cover CG-confusables; 400 random cover broad-landscape tail. ~6.8% below CG α=0. |
| A-N1 | 1 label + 99 hard + 999 random | 1099 | (not finalized) | Trained 2026-05-16; superseded by A-N2 before a full-val eval was logged. Sampled-val trajectory tracked A-N2 within noise. |
| **A-N2** ✓ | 1 label + 999 random (no hard) | 1000 | **0.0741** | RUN 2026-05-17. CG α=0 is 0.0752 → Δ −1.5%. Pure-reranking subset (n=82,364): NDCG@10 0.1342 vs 0.1361. **Phase A exit baseline.** |

Pointwise BCE was also tried during early bring-up; converged to predicting the class prior (NDCG random) and was abandoned. Discipline rule 6 keeps softmax CE as the only loss.

#### Phase A outcome

A-N2 hit near-parity with CG α=0 (within ~2% on every offline metric). The remaining 1.5% gap is the inherent cost of joint-MLP-on-concat vs dot-product on identical features — the dot product carries a useful inductive bias for similarity that the MLP has to re-learn from data, and there is no cross-feature signal yet for the MLP to exploit in return. **The Phase A purpose is achieved**: the architecture works, warm-start works, the training stack is sound. The path to beating CG runs through Phase B.

#### Empirical findings from Phase A

1. **Drop hard negs at large N.** A-N1 → A-N2 (1099 → 1000 cands, hard negs removed) was a net improvement. Once N is large enough to cover the broad-landscape tail, hard negs over-concentrate gradient at the CG-confusable boundary; uniform random sampling provides the easy-vs-label gradient the model actually needs. The "1 label + N hard + M random" recipe is small-N era advice — at N ≥ ~1000, the hard-neg term is redundant and slightly hurts.
2. **Warm-start is load-bearing.** A from-scratch ablation produced materially worse NDCG. The deep MLP needs CG's content-tower head start.
3. **N is the lever for sampled-softmax lift, not architecture or hard-neg mining.** Going from N=100 → N=500 → N=1000 closed the gap from ~12% → ~6.8% → ~1.5%. Loss-family changes (BCE / CCL / WARP) are not the bottleneck.

### Phase B — Cross features

Cross features added in **buckets** of related signals, measured against the previous Phase B baseline (A-N2 for the first). Bundle-level NDCG is the verdict; per-feature attribution within a bucket is only done as a drop-one diagnostic *if the bucket disappoints* (see §10 rule 1). See §9 for the full bucketed roadmap.

#### Bucket 1 ✓ — Content / Categorical Overlap (2026-05-18)

Added three set-membership / categorical features CG mathematically cannot represent: genre_overlap (B-1), tag_overlap (B-1b), dev_affinity (B-2). All three sit on the wide bypass (each is one column in `head.weight`); none are concatenated into the deep MLP.

Implementation lives in `ranker/cross_features.py` — two utility functions (`weighted_overlap`, `dev_affinity`) parameterized on `(history_indices, history_weights, cand_idx)` so Buckets 2–4 can reuse them with different history slices (liked / last-3-liked / disliked). Precompute and train both call the same utils → bit-exact identity between parquet and on-the-fly compute (verified to 2.98e-8 FP roundoff).

| Metric | CG α=0 | A-N2 | Bucket 1 | Δ vs CG | Δ vs A-N2 |
|---|---:|---:|---:|---:|---:|
| NDCG@1 | 0.0278 | — | 0.0305 | +9.7% | — |
| NDCG@10 | 0.0752 | 0.0741 | **0.0822** | **+9.3%** | **+10.9%** |
| NDCG@20 | 0.0968 | — | 0.1029 | +6.3% | — |
| MRR | 0.0726 | — | **0.0779** | **+7.3%** | — |
| Hit@10 | 0.1430 | — | 0.1540 | +7.7% | — |
| Hit@100 | 0.5525 | — | 0.5525 | +0.0% (ceiling) | — |

Pure-reranking subset (n=82,364 where CG retrieved the label):
- NDCG@10: 0.1361 → 0.1487 (+9.3%)
- MRR: 0.1235 → 0.1330 (+7.7%)

Hit@100 unchanged at the CG ceiling — as expected, the ranker can't recover labels CG didn't surface. Lift is concentrated at low K (NDCG@1 +9.7%, NDCG@10 +9.3%, narrowing to +6.3% at K=20) — exactly where categorical overlap signal matters most.

**Bucket 1 outcome:** content/categorical cross features deliver real lift independent of any CG-score signal (rule 3 satisfied). The wide-bypass architecture works as intended — three scalars in a 132-dim head are not drowned out by the 128-dim deep output.

### Phase B — Bucket 2 ✓ (2026-05-19)

**Liked + Recent-3 Liked**: Bucket 1's three categorical-overlap features computed over two new history slices in a single training run — six new wide-head columns (cols 4-9):

- **Liked slice:** `history_indices=X_hist_liked`, `history_weights=X_hist_liked_playtime_weights` (new precompute column).
- **Recent-3 slice:** last 3 non-pad positions of `X_hist_liked` with playtime weights re-normalized over those 3 (extracted by new `last_n_history` util in `ranker/cross_features.py`).

Both slices reuse the same `weighted_overlap` / `dev_affinity` utils — different `history_indices` / `history_weights` only. Bit-exact identity between precompute (parquet) and train (on-the-fly) verified.

| Metric | CG α=0 | Bucket 1 | Bucket 2 | Δ vs B1 |
|---|---:|---:|---:|---:|
| NDCG@1 | 0.0278 | 0.0305 | 0.0308 | +1.0% |
| NDCG@10 | 0.0752 | 0.0822 | **0.0828** | **+0.7%** |
| NDCG@20 | 0.0968 | 0.1029 | 0.1037 | +0.8% |
| MRR | 0.0726 | 0.0779 | 0.0784 | +0.6% |
| Hit@10 | 0.1430 | 0.1540 | 0.1549 | +0.6% |
| Pure-rerank NDCG@10 | 0.1361 | 0.1487 | 0.1499 | +0.8% |
| Pure-rerank MRR | 0.1235 | 0.1330 | 0.1339 | +0.7% |

**Bucket 2 outcome:** lift is real but small. Direction is consistent positive across every metric (no regressions), so the 6 features earn their seat. But the magnitude (+0.7% NDCG@10) is an order of magnitude smaller than Bucket 1's +10.9% over A-N2. Honest read: Bucket 1 already extracted most of the categorical-overlap signal — the Liked and Recent-3 slices add a thin layer on top, not the second-step lift the experiment hoped for. Recalibrates expectations for downstream buckets toward the +0.5-1% range.

### Phase B — Bucket 3 ✗ DROPPED (2026-05-19)

**Disliked History — tried and dropped.** Three categorical-overlap features on the `X_hist_disliked` slice (genre/tag/dev affinity, columns 10-12 in the trial), with a new `X_hist_disliked_playtime_weights` precompute column. Hypothesis: disliked-as-negative-signal would let the wide head learn negative weights ("high overlap with dislikes → lower score").

**Result:** regressed on every headline metric vs Bucket 2.

| Metric | Bucket 1 | Bucket 2 | Bucket 3 | Bucket 3 Δ vs 2 |
|---|---|---|---|---|
| NDCG@10 | 0.0822 | **0.0828** | 0.0824 | −0.5% |
| MRR | 0.0779 | **0.0784** | 0.0781 | −0.4% |
| Pure-rerank NDCG@10 | 0.1487 | **0.1499** | 0.1492 | −0.5% |
| Pure-rerank MRR | 0.1330 | **0.1339** | 0.1333 | −0.4% |

**Root cause: Steam's "disliked" partition is too noisy to earn its columns.** The partition rule is `recommend==False  OR  0.1 < hours < 1.0  OR  hours <= user_rolling_median / 2`. Each clause has problems on Steam:
- `recommend==False` is sparse (most users don't review).
- The hours-band heuristic captures "tried it, didn't stick" which is genuinely ambiguous (a user might bounce off a great game in their first hour).
- The relative-to-median rule flags below-average games — for heavy players, this can include games they actually liked, just not as much as their absolute favorites.

Net result: the disliked slice is a mix of true dislikes and noise. Three noisy wide features can't help, and they take capacity away from the 10 working features.

**Code reverted on disk** (2026-05-19): n_cross_features=13 → 10, parquet column `X_hist_disliked_playtime_weights` dropped, dataset/train/evaluate/canary back to 10-feature roster. `categorical_overlap_triple` and `OverlapBuffers` utils kept — they were the right shared-compute design and Bucket 4 will reuse them.

**Permanent rule from this:** disliked-history variants are dropped from later buckets entirely. Bucket 4's deliberate exclusion of disliked dev-catalog signals stays. Don't retry the disliked slice unless the partition rule itself improves (e.g. richer dislike signal from somewhere outside the current dataset).

### Phase B — Bucket 4 ✗ DROPPED (2026-05-19)

**Developer Catalog Signals — tried and dropped.** Six categorical-overlap features on `X_hist_full / X_hist_liked / last_n_history(X_hist_liked, 3)` × {genre, tag}, but with the per-item buffers replaced by **developer-catalog-averaged** versions (`game_dev_genre_avg`, `game_dev_tag_avg` — per-developer mean of `game_genre_binary` / `game_tag_binary`). Columns 10-15 in the trial. Hypothesis: existing `dev_affinity` is identity-match ("user has playtime on this exact studio's games"); dev-catalog overlap adds similarity-match ("user likes studios that *make games like* this one's studio does"). Should help especially when candidate's own tags are sparse but the studio's catalog disambiguates.

**Result:** flat-to-negative vs Bucket 2 on every headline metric. Two independent training runs reproduced the pattern.

| Metric | Bucket 2 | Bucket 4 (run 1) | Bucket 4 (run 2) | Bucket 4 Δ vs 2 (run 2) |
|---|---|---|---|---|
| NDCG@10 | **0.0828** | 0.0824 | 0.0827 | −0.1% |
| MRR | **0.0784** | 0.0781 | 0.0785 | +0.1% |
| Hit@10 | **0.1549** | 0.1545 | 0.1543 | −0.4% |
| NDCG@20 | **0.1037** | 0.1033 | 0.1036 | −0.1% |
| Pure-rerank NDCG@10 | **0.1499** | 0.1492 | 0.1496 | −0.2% |
| Pure-rerank MRR | **0.1339** | 0.1333 | 0.1340 | +0.1% |

Two seeds, same flat pattern: micro-delta on MRR (+0.1%), regression on the two headline NDCG metrics (−0.1% and −0.2%). Not seed variance.

**Root cause: the deep tower's `developer_lookup` already encodes studio identity end-to-end.** The dev embedding is 12-dim per developer, trained jointly with the rest of the deep MLP and warm-started from CG's `developer_embedding_lookup`. It implicitly learns whatever catalog-level signal exists (because two games by the same studio share an embedding, the head can already condition on "studio X tends to make Z-like games"). Hand-crafting a per-dev average of `game_genre_binary` / `game_tag_binary` and dotting it against the user's history doesn't add an independent axis — it duplicates a signal the deep path already carries. 6 wide columns of capacity displacement, ~0 net lift.

**Code reverted on disk** (2026-05-19): n_cross_features=16 → 10, the 12 new parquet columns (label + negs × 6 features) dropped, the 4 dev-catalog buffers (`game_dev_genre_avg / count`, `game_dev_tag_avg / count`) removed from `WideDeepRanker`, dataset/train/evaluate/canary/precompute back to 10-feature roster. `DevCatalogBuffers` NamedTuple, `build_dev_catalog_buffers` constructor, and `dev_catalog_overlap_pair` util in `ranker/cross_features.py` all removed — not kept as shared infra because the dev-catalog signal class is permanently dropped (see below).

**Permanent rule from this:** the dev-catalog signal class is dropped from future buckets entirely. Don't retry in a different shape (e.g. publisher-catalog, tag-tfidf-weighted dev catalog, dev embedding cosine cross feature) — the structural reason is that `developer_lookup` already encodes studio identity in the deep tower, and any hand-crafted per-dev aggregate of content features will be approximately redundant with what the deep path learns end-to-end. If a future bucket wants to extract more dev-side signal, the lever is on the deep tower (e.g. richer dev features inside the deep concat), not a wide-bypass cross feature.

### Phase B — Bucket 5 ✓ (2026-05-20)

**Numeric Matching**: five scalar-arithmetic differences between per-user and per-item numeric stats — first bucket whose features aren't weighted reductions over categorical buffers, and first to use the Z-score normalization infrastructure (persistent `wide_norm_mean` / `wide_norm_std` model buffers, populated once at training start from train-parquet column stats). Wide-head columns 10–14:

- **col 10 — price_match**: `|user_mean_price_bucket − item_price_bucket|`
- **col 11 — era_gap**: `|user_mean_year_numeric − item_year_numeric|`
- **col 12 — playtime_calibration_median (signed)**: `user_median_log_playtime − item_median_log_hours`
- **col 13 — popularity_match**: `|user_mean_log_count − item_log_count|`
- **col 14 — sentiment_match**: `|user_mean_sentiment − item_sentiment_ordinal|` (sentiment ordinal 0–7 from Steam community sentiment string, unknown → 3.0)

Four buffers added to the model (non-persistent, rebuilt from FeatureStore on load): `game_year_numeric`, `game_median_log_hours`, `game_log_count`, `game_sentiment`. Five new per-user scalar dicts in FeatureStore: `user_mean_price_bucket`, `user_mean_year_numeric`, `user_median_log_playtime`, `user_mean_log_count`, `user_mean_sentiment`. Single new cross-feature util `numeric_match_quintuple` in `ranker/cross_features.py`, used by precompute, train, canary — bit-exact identity across all three call sites by construction. Mean-playtime calibration was considered alongside median but rejected (Pearson ≥ 0.85 with median on Steam's whale-heavy distribution — colinear wide-bypass columns can't extract independent signal); median wins, mean dropped permanently.

| Metric | Bucket 2 | Bucket 5 | Δ vs B2 |
|---|---:|---:|---:|
| NDCG@1 | 0.0308 | **0.0324** | +5.2% |
| NDCG@10 | 0.0828 | **0.0866** | **+4.6%** |
| NDCG@20 | 0.1037 | **0.1083** | +4.4% |
| NDCG@50 | 0.1356 | **0.1410** | +4.0% |
| Hit@10 | 0.1549 | **0.1620** | +4.6% |
| Hit@20 | 0.2380 | **0.2484** | +4.4% |
| Hit@50 | 0.3999 | **0.4137** | +3.5% |
| MRR | 0.0784 | **0.0814** | +3.8% |
| Pure-rerank NDCG@10 | 0.1499 | **0.1567** | +4.5% |
| Pure-rerank MRR | 0.1339 | **0.1393** | +4.0% |
| Pure-rerank Hit@10 | 0.2804 | **0.2932** | +4.6% |
| Pure-rerank Hit@50 | 0.7238 | **0.7489** | +3.5% |

**Bucket 5 outcome:** clear ship signal. Uniform +4–5% lift across every headline metric in both E2E and pure-rerank views — first bucket since Bucket 1 (+10.9% over A-N2) to clear the +0.5–1% magnitude that Bucket 2 set as the new normal. The two preceding flat-or-negative buckets (3 ✗, 4 ✗) both added more user × item categorical-overlap features; Bucket 5's signal class is fundamentally different — scalar-arithmetic cross of two independent numeric quantities — and the lift confirms that the head extracts independent value from this axis. Hit@50 E2E in particular jumped +0.0138 where Bucket 2 was already flattening at +0.0030 vs CG, suggesting the numeric-match features pull in correct deep-tail labels that pure overlap couldn't surface.

**Canary read:** all 9 user types hold or improve niche coherence vs CG. JRPG / Racing / Survival / Management materially cleaner (drop several cross-genre titles the CG was surfacing); Western RPG / Civ / Indie / FPS / Fighting comparable. No catastrophic regressions. Specifically, the popularity_match feature does not cause obvious popularity-leak on niche-taste canaries — the ranker α=0 keeps showing niche, non-mega-popular titles where CG α=0 already does.

**Implementation note: Z-score normalization architecture.** Two persistent buffers `wide_norm_mean (n_wide_normalized,)` and `wide_norm_std (n_wide_normalized,)` are registered on the model and populated once before the optimizer loop by `populate_wide_norm_buffers(model, train_parquet_path)` — single pass reading the 5 label-only columns (`price_match_label`, `era_gap_label`, `playtime_cal_median_label`, `popularity_match_label`, `sentiment_match_label`), compute float64 mean/std, copy in with std clamped to 1.0 if variance is near zero. `_normalize_wide(cross_features)` then Z-scores only the trailing `n_wide_normalized=5` columns (10–14) — cols 0–9 are bounded ([−1,1] / [0,1]) and pass through raw. Pattern reusable for Buckets 6–8 (just extend `n_wide_normalized` and the label-column list).

---

### Phase B — Bucket 6 ✓ (2026-05-21)

**Niche Feature Crosses**: 8 user × item cross features structured as **4 niche/rarity concepts × 2 history slices** (`X_hist_full` and `X_hist_liked` with their respective playtime-weight columns) — `tag_overlap_idf` (Shape A, IDF-reweighted Bucket 1 B-1b), `niche_tag_match` (Shape B, weighted-mean tag IDF diff), `max_tag_idf_match` (Shape B, weighted-max tag IDF diff), `niche_dev_match` (Shape B, log dev-catalog-size diff). Wide-head columns 15–22. Five new per-game buffers landed (`game_tag_binary_idf`, `game_tag_count_idf`, `game_tag_mean_idf`, `game_tag_max_idf`, `game_dev_log_catalog_size`) — all non-persistent, rebuilt from FeatureStore. Single new compute util `niche_scalar_triple` in `ranker/cross_features.py` plus reuse of `weighted_overlap` for the IDF overlap pair — bit-exact identity across precompute, train, canary by construction.

| Metric | Bucket 5 | Bucket 6 | Δ vs B5 |
|---|---:|---:|---:|
| NDCG@1 | 0.0324 | 0.0319 | −1.5% |
| NDCG@10 | 0.0866 | 0.0867 | +0.1% |
| NDCG@20 | 0.1083 | 0.1083 | 0.0% |
| NDCG@50 | 0.1410 | 0.1410 | 0.0% |
| Hit@10 | 0.1620 | 0.1625 | +0.3% |
| Hit@20 | 0.2484 | 0.2486 | +0.1% |
| Hit@50 | 0.4137 | 0.4140 | +0.1% |
| MRR | 0.0814 | 0.0813 | −0.1% |
| Pure-rerank NDCG@10 | 0.1567 | 0.1569 | +0.1% |
| Pure-rerank MRR | 0.1393 | 0.1391 | −0.1% |
| Pure-rerank Hit@10 | 0.2932 | 0.2941 | +0.3% |
| Pure-rerank Hit@50 | 0.7489 | 0.7494 | +0.1% |

**Bucket 6 outcome:** offline-flat (every delta in ±0.001 noise band, both signs scattered) **but canary-better on niche tastes.** Per-slice canary read (ranker-vs-ranker, identical CG control):

- **Fighting Lover** — clean win. B5 ranker top-10 had 3 JRPGs (FF Type-0 #5, FF VIII #6, Tales of Symphonia #7) bleeding into a fighting-tag query. B6 ranker removes all three and surfaces METAL SLUG, METAL SLUG X, UNDER NIGHT IN-BIRTH, Naruto Storm 4 — proper fighters and arcade. Cleanest single-slice improvement in any bucket since Bucket 1.
- **Civ Lover (4X)** — win. Galactic Civilizations III (sibling of the StarDrive/GalCiv II anchor tags) promoted #3 → #1; Master of Orion #11 → #5, Victoria II #12 → #7. B5's Might & Magic Heroes VI (a non-4X outlier) drops out of the top 10.
- **Western RPG** — slight win. Pillars of Eternity #10 → #4, Kingdoms of Amalur #14 → #8. Both buckets still surface Terraria (Terraria pollution is a separate axis).
- **FPS** — slight win. Half-Life 2 promoted #9 → #5. Both still have Oddworld pollution.
- **JRPG / Indie / Racing / Survival / Management** — comparable to B5; minor reorders inside already-good lists.

The 4 niche concepts target exactly the failure mode the Fighting slice exhibited — an anime-adjacent CG cluster pulls JRPGs into fighting lists, and the niche scalars (`max_tag_idf_match`, `niche_dev_match` on `liked`) penalize that bleed by checking whether the candidate's rarest tag matches the user's rarest-tag fingerprint and whether the candidate's developer-catalog scale matches the user's. **Kept on canary alone.** This is the inverse of Bucket 4 ✗ (flat-everywhere) and consistent with the V5 PROD vs V5 α=0 precedent — canary quality on niche tastes is what users notice. Bucket 7's `dev_specialization` (item-only, no user-side analog) was dropped in planning; the other two B7 features absorbed here as proper user × item crosses.

**Inference cost:** negligible. Per query (n_cand=100, B=1) Bucket 6 adds 2× `weighted_overlap` matmuls + 2× `niche_scalar_triple` reductions ≈ ~120k FLOPs total — sub-millisecond on MPS, dwarfed by CG retrieval (~700k FLOPs) and the deep tower forward. **Training cost:** Bucket 6 columns flow through the wide bypass alongside the existing 15; adds 8 × hidden_dim weights to the head, no architectural changes. **Precompute cost:** materially higher than Bucket 5 (8 new columns × 4.3M rows × 1000 candidates), motivated the chunked-streaming-write + sync-reduction work in `ranker/precompute.py` (CHUNK_SIZE=250_000, single per-batch stack-and-sync). Peak precompute memory dropped from ~103 GB → ~8–10 GB.

**Lesson:** when offline-flat coincides with canary-better, ship. Flat offline metrics don't penalize within-top-20 reorders that don't flip the held-out target across the K boundary, but those reorders are exactly what list-shape evaluation (canary) sees. Bucket 4 ✗ was flat-everywhere (flat offline AND flat canary); Bucket 6 is flat offline but canary-better — different signal entirely.

---

## 9. Cross-Feature Roadmap (Phase B)

Features are grouped into **buckets** that share a semantic theme. Each bucket is one training experiment, measured against the previous Phase B baseline (or A-N2 for the first). Bundle-level NDCG is the verdict; per-feature attribution is only done as a drop-one diagnostic if a bucket disappoints (see §10 rule 1).

All Phase B cross features sit on the **wide bypass** — each feature is one column in `head.weight` with a direct gradient path, *not* concatenated into the deep MLP (which would drown a single scalar in ~290 dims).

### B-0 (in A-N2) — Tag Cosine

| Feature | Formula |
|---|---|
| **Tag Cosine** | precomputed: `cosine(user_tag_pool_tfidf, item_tag_vec)` — direction match on the user's full tag profile vs candidate. Magnitude-blind. |

### Buckets 1–2 (landed): Categorical-overlap history variants

The same three categorical-overlap features (genre_overlap, tag_overlap, dev_affinity) computed over **three different slices of the user's interaction history**, grouped into two buckets. All slices reuse `ranker/cross_features.py`'s `weighted_overlap` and `dev_affinity` utils — only `history_indices` and `history_weights` change. There is no pooling here (no embedding aggregation): the utils index per-item categorical buffers directly and reduce by playtime weights to one scalar per (history slice, candidate) pair.

Bucket 2 rolls **Liked** and **Recent-3 Liked** into a single training run (6 features). Each bucket independently answers "does this history-variant family add signal on top of the previous baseline?"

**Bucket 3 (Disliked) and Bucket 4 (Developer Catalog Signals) were both tried and dropped** — see dedicated ✗ sections below for eval tables and root causes. Bucket 4's drop is the second consecutive "more user × item categorical overlap" bucket to flatten, recalibrating downstream expectations away from this signal class.

**Bucket 5 ✓ (Numeric Matching) broke the flat streak** — 5 scalar-arithmetic cross features (price / era / playtime-calibration-median / popularity / sentiment match) landed with a uniform +4–5% lift across every headline metric vs Bucket 2 (see §8 Bucket 5 entry for the table). Lesson: when an overlap-style bucket flattens, the next bucket should be a different signal class entirely, not a re-shaping of the same signal.

**Bucket 6 ✓ (Niche Feature Crosses) landed offline-flat but canary-better** — 8 user × item cross features (4 niche concepts × {full, liked}) landed at NDCG@10 0.0867 vs Bucket 5's 0.0866 (deltas in ±0.001 noise band on every metric), but the canary side-by-side shows clear per-slice improvements (Fighting Lover JRPG bleed eliminated, Civ Lover Galactic Civ III promoted to #1, smaller niche-coherence wins on Western RPG and FPS). Kept on canary alone — see §8 Bucket 6 entry for the full table and per-slice canary analysis. **Bucket 8 (Engagement-Level Cross)** is the remaining user × item cross feature in the roadmap and runs next. Bucket 8 is the least-confident of the original Phase B roadmap (the deep MLP may already capture engagement signal via dev/genre embeddings + user_avg_log).

**Bucket 7 (Item-Intrinsic Priors) dropped in planning (2026-05-20).** Two of its three features (`dev_catalog_size`, `candidate_max_tag_idf`) absorbed into Bucket 6 as user × item scalar crosses (`niche_dev_match`, `max_tag_idf_match`) — the correct shape for the linear wide head, which can't extract modulation from item-only inputs. The remaining feature (`dev_specialization`) is the only item-only scalar without an obvious user-side analog and isn't worth a bucket on its own.

Naming convention for the cross-feature column slots (must stay stable across checkpoints — see `dataset.compute_cross_features` ordering):

```
col 0   : tag_cosine                          (B-0,  Phase A)
col 1   : genre_overlap_full                  (B-1,  Bucket 1 ✓)
col 2   : tag_overlap_full                    (B-1b, Bucket 1 ✓)
col 3   : dev_affinity_full                   (B-2,  Bucket 1 ✓)
col 4   : genre_overlap_liked                 (Bucket 2 ✓)
col 5   : tag_overlap_liked                   (Bucket 2 ✓)
col 6   : dev_affinity_liked                  (Bucket 2 ✓)
col 7   : genre_overlap_recent3               (Bucket 2 ✓)
col 8   : tag_overlap_recent3                 (Bucket 2 ✓)
col 9   : dev_affinity_recent3                (Bucket 2 ✓)
col 10  : price_match                         (Bucket 5)
col 11  : era_gap                             (Bucket 5)
col 12  : playtime_calibration_median         (Bucket 5)
col 13  : popularity_match                    (Bucket 5)
col 14  : sentiment_match                     (Bucket 5)
col 15  : tag_overlap_idf_full                (Bucket 6)
col 16  : tag_overlap_idf_liked               (Bucket 6)
col 17  : niche_tag_match_full                (Bucket 6)
col 18  : niche_tag_match_liked               (Bucket 6)
col 19  : max_tag_idf_match_full              (Bucket 6)
col 20  : max_tag_idf_match_liked             (Bucket 6)
col 21  : niche_dev_match_full                (Bucket 6)
col 22  : niche_dev_match_liked               (Bucket 6)
col 23  : user_dev_engagement_cross           (Bucket 8)
col 24  : user_genre_engagement_cross         (Bucket 8)
col 25  : cg_score                            (Bucket 9, kept solo — was Bucket 6 in earlier drafts)
```
(Bucket 3's columns 10-12 were the disliked-slice triple; that bucket was dropped — see Bucket 3 ✗ section. Bucket 4 then took cols 10-15 for the dev-catalog triple-times-two and also dropped — see Bucket 4 ✗ section. Bucket 5 onward reclaims the slots in roadmap order. Bucket 7 was dropped in planning 2026-05-20: its 3 features `dev_specialization` / `dev_catalog_size` / `candidate_max_tag_idf` were item-only (not real crosses); 2 of those 3 were absorbed into Bucket 6 as proper user × item scalar crosses (`niche_dev_match`, `max_tag_idf_match`) on both full and liked slices, and the third (`dev_specialization`) was dropped as not worth a slot on its own. Bucket 6 grew from 2 features to 8 (4 niche concepts × 2 slices) as a result, shifting Buckets 8 and 9 to cols 23-25. Bucket 9 CG Score is held to the end of the roadmap per Discipline Rule 3.)

**Bucket 5 design note:** Bucket 5 is the first bucket whose features are *scalar arithmetic* on numeric user/item stats rather than weighted reductions over categorical buffers. Five Z-scored differences land in cols 10-14 (see Bucket 5 detail section). Mean-playtime calibration was considered alongside median-playtime calibration but rejected — the two scalars are heavily correlated (Pearson ≥ 0.85 on Steam's whale-heavy distribution), and the head can't extract independent signal from colinear wide-bypass columns. Median is the right stat for Steam's distribution (whale-distorted means mislead on "typical engagement"), so median wins; mean is dropped permanently.

#### Bucket 1 ✓ — Full History (2026-05-18)

History: `X_hist_full`, weighted by `X_hist_playtime_weights`. **Outcome:** NDCG@10 0.0741 → 0.0822 (+10.9% vs A-N2). See §8 Phase Log for the full metric table.

| # | Feature | Formula |
|---|---|---|
| B-1 | **Genre Overlap (Full)** | `(user_genre_w · item_genre_binary) / item_genre_count` — mean over the candidate's genres of "user's playtime fraction in that genre." [0, 1]. Buffers: `game_genre_binary`, `game_genre_count`. |
| B-1b | **Tag Overlap (Full)** | `(user_tag_w · item_tag_binary) / item_tag_count` — same structure as B-1 but on tags. **Magnitude-aware and complementary to B-0 Tag Cosine** (cosine is direction-only; overlap is "average user weight on candidate's tags"). Buffers: `game_tag_binary`, `game_tag_count`. |
| B-2 | **Developer Affinity (Full)** | `playtime-weighted fraction of user's history under this developer` (i.e. `Σ_i h_pw[i] · 1[hist_dev[i] == item_dev]`). Captures studio loyalty (FromSoftware, Larian, Paradox) — categorical membership CG can't do. |

#### Bucket 2 ✓ — Liked + Recent-3 Liked (2026-05-19)

Six new features in one training run: Bucket 1's three categorical-overlap features computed over two slices of the user's liked-only history. **Outcome:** NDCG@10 0.0822 → 0.0828 (+0.7% vs Bucket 1). Direction-consistent across every metric but small magnitude; see §8 Phase Log Bucket 2 entry for the full metric table and honest read.

**Slice A — Liked (full):** `history_indices=X_hist_liked`, `history_weights=X_hist_liked_playtime_weights` (**new precompute column**, parallel to `X_hist_liked`, MAX_HISTORY_LEN, looking up each liked entry's playtime weight from the full-history computation). Filters out games the user disliked or barely touched, so the user-side weight vector reflects only genuine positive preferences.

**Slice B — Recent-3 Liked:** last 3 non-pad positions of `X_hist_liked` with playtime weights re-normalized to sum to 1 over those 3. Captures recency drift on positive-signal context — "what has the user been *enjoying* lately" — sharper than the full-liked slice. Steam has no timestamps, so "last" = the last 3 entries in `X_hist_liked` (filled in shuffled-prefix order, seeded so consistent across runs). If a user has fewer than 3 liked games in the context, use however many exist; if 0, the recent-3 features fall through to 0 and the model relies on the liked-full slice.

Both slices reuse the `X_hist_liked_playtime_weights` precompute column — landed once, two consumers.

**Hypothesis:** sharper signal-to-noise on the liked-only side should improve discrimination, especially for users with bimodal libraries (e.g. plays both AAA and indie — full-history dev_affinity gets smeared across both, liked-history concentrates on the studios they *kept* playing). Recent-3 layered on top should pick up taste drift the full-liked slice averages out. Measured against Bucket 1.

| # | Feature | Formula |
|---|---|---|
| B-3a | **Genre Overlap (Liked)** | Same as B-1, history restricted to liked games. |
| B-3b | **Tag Overlap (Liked)** | Same as B-1b, history restricted to liked games. |
| B-3c | **Developer Affinity (Liked)** | Same as B-2, history restricted to liked games. |
| B-4a | **Genre Overlap (Recent-3)** | Same as B-1, history = last 3 non-pad of `X_hist_liked` with weights re-normalized. |
| B-4b | **Tag Overlap (Recent-3)** | Same as B-1b, last 3 liked. |
| B-4c | **Developer Affinity (Recent-3)** | Same as B-2, last 3 liked. "Are you currently on a studio kick?" |

#### Bucket 3 ✗ — Disliked History (DROPPED 2026-05-19)

**Tried with playtime-weight option (a) and dropped.** Three categorical-overlap features on `X_hist_disliked` (genre/tag/dev affinity, B-5a/b/c), with a new `X_hist_disliked_playtime_weights` precompute column. Hypothesis was that the wide head would learn negative weights ("high overlap with dislikes → lower score") and add real signal if user dislikes are coherent.

**Outcome:** regressed vs Bucket 2 on every headline metric (NDCG@10 −0.5%, MRR −0.4%, pure-rerank NDCG@10 −0.5%); essentially tied with Bucket 1. See §8 Bucket 3 ✗ DROPPED entry for the full metric table and root-cause analysis.

**Why it failed:** Steam's "disliked" partition is too noisy to earn its columns. The partition rule `recommend==False  OR  0.1 < hours < 1.0  OR  hours <= user_rolling_median / 2` mixes true dislikes with noise — `recommend` is sparse on Steam, the hours-band heuristic captures ambiguous "tried it didn't stick" behavior, and the relative-to-median rule flags below-average games which for heavy players can include genuine favorites. Three noisy features can't help and slightly displace the signal carried by the 10 Bucket 2 features.

**Permanent decision:** disliked-history variants stay out of future buckets. Bucket 4's deliberate exclusion of disliked dev-catalog signals becomes the rule, not the exception. Open weighting options (a/b/c) are no longer worth pursuing — the noise lives in the partition rule itself, not in the weighting.

#### Bucket 4 ✗ — Developer Catalog Signals (DROPPED 2026-05-19)

**Tried and dropped.** Six categorical-overlap features mirroring Bucket 2's slice structure but with per-item buffers swapped from `game_genre_binary` / `game_tag_binary` to **developer-catalog-averaged** versions: `game_dev_genre_avg[item, g]` = fraction of the dev's catalog carrying genre g (and similarly for tags). Two new per-item buffers (genre + tag) × three history slices (full / liked / recent-3 liked) = 6 wide-head columns (cols 10-15 in the trial). Hypothesis was that existing `dev_affinity` is *identity-match* ("user has playtime on this exact studio") and dev-catalog overlap would add *similarity-match* ("user likes studios that make games *like* this one's studio"), helping especially for sparse-tag candidates and unseen-but-similar studios.

**Outcome:** flat-to-negative vs Bucket 2 on every headline metric, two independent training runs reproducing the same pattern (NDCG@10 −0.1%, MRR +0.1%, pure-rerank NDCG@10 −0.2%). See §8 Bucket 4 ✗ DROPPED entry for the full metric table and root-cause analysis.

**Why it failed:** the deep tower's `developer_lookup` (12-dim embedding per developer, warm-started from CG, trained jointly with the deep MLP) already encodes whatever catalog-level signal exists — because two games by the same studio share an embedding, the head can already condition on "studio X tends to make Z-like games." Hand-crafting a per-dev average of `game_genre_binary` / `game_tag_binary` and dotting it against the user's history doesn't add an independent axis; it duplicates a signal the deep path already carries. 6 wide columns of capacity displacement, ~0 net lift.

**Permanent decision:** the dev-catalog signal class is dropped from future buckets entirely. Don't retry in a different shape (publisher-catalog, tag-tfidf-weighted dev catalog, dev embedding cosine cross feature, etc.) — the structural reason is that `developer_lookup` in the deep tower already encodes studio identity, and any hand-crafted per-dev aggregate of content features will be approximately redundant with what the deep path learns end-to-end. If a future bucket wants more dev-side signal, the lever is on the deep tower (richer dev features inside the deep concat), not a wide-bypass cross feature.

### Bucket 5 ✓ — Numeric Matching (2026-05-20)

**Outcome:** NDCG@10 0.0828 → **0.0866** (+4.6% vs Bucket 2). Uniform +4–5% across every headline metric in both E2E and pure-rerank views, plus clean canary quality on all 9 user types. See §8 Bucket 5 entry for the full metric table. Resets the Phase B baseline; Bucket 6 measures against this.

Five scalar-arithmetic differences on user-vs-item numeric stats. All require Z-score normalization with fixed train-set mean/std stored as persistent buffers on the model (see "Wide-feature normalization" below) — the raw scalars have wildly different scales (price 0-8, year ~2000s, log-count up to ~8, sentiment 0-7, log-playtime 0-7) so one feature would dominate the head gradient without normalization. First bucket to require this infrastructure; subsequent Buckets 6-8 reuse the same Z-score buffer pattern.

| # | Feature | Formula | Sign |
|---|---|---|---|
| B-7a | **Price Match** | `abs(user_mean_price_bucket - item_price_bucket)` — F2P / indie / AAA buyer segments. | abs |
| B-7b | **Era Gap** | `abs(user_mean_year_numeric - item_year_numeric)` — new release vs retro preference. | abs |
| B-7c | **Playtime Calibration (Median)** | `user_median_log_playtime - item_median_log_hours` — heavy-session vs short-session compatibility. Median (not mean) because Steam playtime is whale-heavy and means are systematically distorted by long-tail outliers. | **signed** |
| B-7d | **Popularity Match** | `abs(user_mean_log_count - item_log_count)` — user preference for popular vs obscure. | abs |
| B-7e | **Sentiment Match** | `abs(user_mean_sentiment - item_sentiment_ordinal)` — sentiment ordinal 0-7 (Overwhelmingly Negative → Overwhelmingly Positive). Captures "quality bar" preference (user who plays only Very Positive+ vs user who plays Mixed cult curios). | abs |

**Signed vs abs:** Playtime Calibration is signed so the head can learn asymmetric weights for "user heavier than this candidate's typical" vs "user lighter than this candidate's typical." The other four are symmetric — mismatch in either direction is roughly equivalent at population level.

**Per-game item-side buffers (NEW for Bucket 5)** — registered non-persistent on the ranker, rebuilt from FeatureStore on load. Each shaped `(n_items+1,)` float32 with pad row appended (pad value = 0; never indexed since `cand_idx` stays in `[0, n_items)`):
- `game_year_numeric` — release year as float (e.g. `2018.0`); unknown → corpus median year.
- `game_median_log_hours` — `log1p(median_hours)`; existing `game_median_hours` already in FeatureStore, just log-transformed.
- `game_log_count` — `log1p(interaction_count)`; same `game_interaction_counts` already in FeatureStore.
- `game_sentiment` — sentiment ordinal as float (0-7); unknown → 3.0 ("Mixed", roughly corpus median).

(`game_price_idx` already on the model as int64 for the Embedding lookup — cast to float at use time in the cross-feature compute, no new buffer needed.)

**Per-user scalar aggregates (NEW for Bucket 5)** — five floats per user added to FeatureStore as dicts keyed by `user_id`. Each is a single-pass aggregation over the user's full history (not the rollback prefix — consistent with existing `user_avg_log_playtime`, the only other per-user scalar passed into the model):
- `user_mean_price_bucket`, `user_mean_year_numeric`, `user_mean_sentiment`, `user_mean_log_count` — straightforward means over `game_X[i]` for `i in user_history`.
- `user_median_log_playtime` — median of `log1p(hours[i])` for `i in user_history`.

**Cross-feature compute** — single new util `numeric_match_quintuple` in `ranker/cross_features.py`. Takes `(Bucket5GameBuffers, user_b5_scalars (B, 5), cand_idx (B, n_cand))`, returns the 5 features each shaped `(B, n_cand)`. Used by precompute (CPU, on-device tensors built locally), train (model's registered buffers), and canary (model's buffers + synthetic-user-side scalars built per `_build_synthetic_user_inputs`). Bit-exact identity across all 3 call sites by construction.

**Implementation cost:** 4 new per-game buffers (~22 KB total — 5,438 × 4 floats × 4 buffers); 5 new per-user dicts (~3.5 MB — 88,310 users × 5 floats × 8 bytes for the dict overhead); 10 new parquet columns (5 features × {label, negs} per row, ~10 GB on the train parquet); 2 new persistent Z-score model buffers (40 bytes — `wide_norm_mean(5,)` + `wide_norm_std(5,)`); 5 lines of subtract-and-abs in cross-feature compute; 1 new helper for populating the Z-score buffers on training start (single pass over train parquet, ~5-10 sec one-time).

### Bucket 6 ✓ — Niche Feature Crosses (2026-05-21)

**Outcome:** offline-flat (every E2E and pure-rerank delta in ±0.001 noise band vs Bucket 5), but canary-better on niche tastes — Fighting Lover JRPG bleed eliminated, Civ Lover Galactic Civ III sibling promoted to #1, smaller niche-coherence wins on Western RPG / FPS. Kept on canary alone. See §8 Bucket 6 entry for the full metric table and per-slice canary analysis. Implementation details below are the design as shipped.

Eight user × item cross features, structured as **4 niche/rarity concepts × 2 history slices** (`X_hist_full` and `X_hist_liked` with their respective playtime-weight columns). Each concept measures a different facet of "how well does this user's niche/rarity taste match this candidate's niche/rarity profile?" — and computing on both slices mirrors Bucket 6's existing IDF-overlap pattern (Bucket 2 already established the full/liked structure for categorical-overlap features).

**The 4 concepts:**

| Concept | Shape | What it measures |
|---|---|---|
| **IDF Tag Overlap** | A (overlap) | Bucket 1's `tag_overlap` reweighted by tag IDF — "Roguelike" weighted higher than "Action." Magnitude-aware AND rarity-weighted; the missing axis between B-0 (direction-only cosine) and B-1b (magnitude-aware rarity-blind). |
| **Niche Tag Match** | B (scalar) | `|user_mean_tag_idf − item_mean_tag_idf|` — aggregate tag-rarity match. User who plays mostly-niche-tag games matches a candidate with mostly-niche tags. |
| **Max Tag IDF Match** | B (scalar) | `|user_max_tag_idf − item_max_tag_idf|` — sharpest-tag match. Picks out users with at least one rare-tag preference vs candidates that carry at least one rare tag. |
| **Niche Dev Match** | B (scalar) | `|user_mean_log_dev_catalog_size − log(item_dev_catalog_size)|` — heavy small-dev/indie user matches small-dev candidate. Absorbs the would-be Bucket 7 `dev_catalog_size` as a proper user × item cross. |

**All 8 features (4 concepts × {full, liked}):**

| # | Feature | History slice | User-side aggregate |
|---|---|---|---|
| B-9a | **Tag Overlap (IDF, Full)**     | `X_hist_full`,  `X_hist_playtime_weights`        | `Σ user_tag_w[t] · game_tag_binary_idf[cand, t]` |
| B-9b | **Tag Overlap (IDF, Liked)**    | `X_hist_liked`, `X_hist_liked_playtime_weights` | same, liked slice |
| B-9c | **Niche Tag Match (Full)**      | `X_hist_full`                                    | `Σ_i h_pw[i] · game_tag_mean_idf[hist[i]]` |
| B-9d | **Niche Tag Match (Liked)**     | `X_hist_liked`                                   | same, liked slice |
| B-9e | **Max Tag IDF Match (Full)**    | `X_hist_full`                                    | `max_i (h_pw[i] · game_tag_max_idf[hist[i]])`  *(weighted max)* |
| B-9f | **Max Tag IDF Match (Liked)**   | `X_hist_liked`                                   | same, liked slice |
| B-9g | **Niche Dev Match (Full)**      | `X_hist_full`                                    | `Σ_i h_pw[i] · game_dev_log_catalog_size[hist[i]]` |
| B-9h | **Niche Dev Match (Liked)**     | `X_hist_liked`                                   | same, liked slice |

**Per-item buffers (NEW):**
- `game_tag_binary_idf` shape `(n_items+1, n_tags)` float32 — for Shape A overlap (B-9a/b). `game_tag_binary * tag_idf[None, :]`.
- `game_tag_mean_idf` shape `(n_items+1,)` float32 — mean IDF of the item's tags (used as item-side scalar for B-9c/d, AND looked up per history item for the user-side weighted mean).
- `game_tag_max_idf` shape `(n_items+1,)` float32 — max IDF of the item's tags (B-9e/f).
- `game_dev_log_catalog_size` shape `(n_items+1,)` float32 — `log1p(# corpus games by this game's dev)` (B-9g/h).

**Tag IDF source:** `tag_idf` already computed in `features.py` for TF-IDF. No new IDF computation needed — just gather the per-tag IDF values into the four new per-game buffers at preprocess time.

**Per-user values:** None new. Each user-side aggregate is computed on the fly per rollback example by indexing the per-item buffers with the slice's history indices (`X_hist_full` or `X_hist_liked`) and reducing by the slice's playtime weights — same pattern as `weighted_overlap` and `dev_affinity`. No FeatureStore changes for Bucket 6's user side; everything flows from the rollback history arrays already passed into the cross-feature compute.

**Cross-feature compute:** Two new utils in `ranker/cross_features.py` reusing the Bucket 1 pattern:
- `weighted_overlap_idf(buffers, hist_indices, hist_weights, cand_idx)` — same shape as `weighted_overlap`, just uses the IDF-weighted tag buffer (B-9a/b).
- `niche_scalar_triple(buffers, hist_indices, hist_weights, cand_idx)` — returns 3 scalars per (slice, candidate): niche_tag_match, max_tag_idf_match, niche_dev_match. Called once per slice (full + liked) → 6 features.

Both utils take a `NicheBuffers` NamedTuple (`game_tag_binary_idf`, `game_tag_mean_idf`, `game_tag_max_idf`, `game_dev_log_catalog_size`) and are reused by precompute, train, and canary — bit-exact identity by construction (same pattern as Bucket 5's `numeric_match_quintuple`).

**Hypothesis:** complementary to B-0 (direction-only) and B-1b / B-3b (magnitude-aware, rarity-blind). 4 concepts × 2 slices gives the head 8 levers on the niche-match axis. Bucket 1's tag features were the biggest single Phase B win — adding IDF reweighting on top of them plus 3 aggregate-rarity scalars is the highest-confidence next bucket. Pattern A (per-context full/liked) chosen over Pattern B (static per-user) because niche-ness is plausibly context-dependent (user might normally play Action but be currently on a Roguelike kick — full-history niche differs from liked-history niche).

**Implementation cost:** 4 new per-game buffers; 16 new parquet columns (8 features × {label, negs} per row, ~16 GB on the train parquet); 8 new Z-score normalization slots (`n_wide_normalized` grows 5 → 13); 2 new utils in `ranker/cross_features.py` (~30 lines); the `populate_wide_norm_buffers` helper extends to read the 8 new label columns. Cross-feature compute is `n_slices × n_concepts = 2 × 4 = 8` gather-and-reduce ops per batch — comparable to Bucket 2's compute footprint.

### Bucket 8 — Engagement-Level Cross

Two scalars crossing the user's overall engagement level (`X_user_avg_log` — already passed into `user_forward`) with the candidate's intrinsic engagement level. Captures *"high-hours user evaluating a high-hours studio"* compatibility.

| # | Feature | Formula |
|---|---|---|
| B-11a | **User × Dev Engagement** | `X_user_avg_log[b] · dev_mean_log_playtime[cand]` |
| B-11b | **User × Genre Engagement** | `X_user_avg_log[b] · mean(genre_mean_log_playtime[g] for g in cand's genres)` |

`dev_mean_log_playtime[d]` = global mean `log(1+hours)` across all observed interactions with games by dev `d` (extension of `ranker_game_stats.parquet`'s `avg_log_playtime` aggregated to dev level). `genre_mean_log_playtime[g]` is the analogous per-genre quantity. Both built at preprocess.

**Hypothesis:** the deep tower already sees `X_user_avg_log` (for in-model genre debiasing) and the dev/genre embeddings, but does *not* see explicit dev/genre engagement-level scalars. The cross gives the wide head a direct "engagement compatibility" lever the deep path can't easily reconstruct.

**Risk acknowledged:** of the remaining user × item cross features in the roadmap (Buckets 6, 8), this is the lower-confidence one — the deep MLP may already capture enough of this signal via dev embeddings + user pool. Fair to deprioritize if Bucket 6 underwhelms; the engagement-cross hypothesis is interesting but its independence from the deep path is unproven.

**Implementation cost:** 2 new scalar buffers (`dev_mean_log_playtime` shape `(n_devs+1,)`, `genre_mean_log_playtime` shape `(n_genres,)`); 4 new parquet columns; compute is one elementwise multiply + per-cand gather/mean.

### Bucket 9 — CG Score (kept solo)

*Renumbered from Bucket 6 in earlier drafts; Buckets 6-8 above slotted in between Numeric Matching and CG Score per the "earn independent signal first" discipline.*

| # | Feature | Formula |
|---|---|---|
| B-8 | **CG Score** | raw CG dot, per candidate. **Re-enable LAST.** CG score is circular ("follow CG and beat CG"); only earn it after content features have proven independent value. Kept solo to make its lift cleanly attributable. |

### Deep-concat additions (future, not in any bucket above)

| Feature | Path | Note |
|---|---|---|
| Genre Diversity / Tag Entropy / History Confidence | Deep | User-state scalars added to the deep concat. Only consider after the wide-path roadmap saturates. |

### Removed from the roadmap

- ~~Dislike Similarity~~ (was B-6, `cosine(user_disliked_pool, item_id_emb)`) — dropped as a one-off scalar. Bucket 3 ✗ later tried the structured-columns version of the disliked-history signal (3 categorical-overlap features on `X_hist_disliked`) and that also failed; root cause was the noisy disliked partition itself, not the feature shape. (The cosine name `user_disliked_pool` here refers to the deep-tower's `pool_disliked` — that one IS a legit embedding pool.)
- ~~Tag Peak Match~~ (was B-7, `max(user_tag_profile * item_tag_vec)`) — dropped. Tag Cosine (B-0) and Tag Overlap variants (Buckets 1–4 each carry one) cover every useful tag-affinity view; the "single peak" angle is redundant.
- ~~Recent Game Similarity~~ (was B-8, `dot(last_played_id_emb, item_id_emb)`) — **replaced** by the Recent-3 slice inside Bucket 2 (Recent Genre Overlap / Tag Overlap / Developer Affinity on the last 3 LIKED games). Item-ID dot is a weak proxy; mirroring Bucket 1's three features on a recent-liked window is the meaningful recency signal.

### Wide-feature normalization

Wide features beyond cosines/Jaccard must be normalized before concatenation using fixed statistics registered as **persistent** model buffers (not BatchNorm — train/eval batch composition differs). Compute mean/std from a single pass over training data, register with `register_buffer(name, tensor)` (persistent — the default; do not pass `persistent=False`, that excludes from `state_dict` and silently degrades the model on load).

```
Expected ranges (pre-normalization):
  Tag Cosine                              [−1, 1]   → no normalization
  Genre/Tag Overlap (Full / Liked /       [0, 1]    → no normalization
    Recent-3 / Disliked)
  Developer Affinity (Full / Liked /      [0, 1]    → Z-score (heavy zero mass — may need
    Recent-3 / Disliked)                              separate treatment; all four history variants
                                                      share this concern)
  Price Match                             [0, ~8]   → Z-score (Bucket 5)
  Era Gap                                 [0, ~30]  → Z-score (raw year diff; Z-score handles the scale)
  Playtime Calibration (Median)           [~−5, +5] → Z-score (signed)
  Popularity Match                        [0, ~8]   → Z-score
  Sentiment Match                         [0, ~7]   → Z-score (Bucket 5)
  Tag Overlap (IDF, Full / Liked)         [0, ~3]   → Z-score (Bucket 6; range depends on IDF max)
  Niche Tag Match (Full / Liked)          [0, ~3]   → Z-score (Bucket 6; weighted-mean IDF diff)
  Max Tag IDF Match (Full / Liked)        [0, ~4]   → Z-score (Bucket 6; weighted-max IDF diff)
  Niche Dev Match (Full / Liked)          [0, ~5]   → Z-score (Bucket 6; log dev catalog size diff)
  User × Dev Engagement                   [~−5, +5] → Z-score (Bucket 8; signed cross product)
  User × Genre Engagement                 [~−5, +5] → Z-score
  CG Score                                [~−1, 1]  → no normalization (already cosine-bounded)
```

Initialize new wide-feature weights at 0.1 — small non-zero signal without swamping the deep path.

### Other future phases

- **Phase C — Label quality.** After Phase B saturates, re-precompute with stricter `is_liked` filter (`raw_hours ≥ game_median` OR `≥ 2× user_median` OR `recommend=True`). Slashes label set by 40-60% but every label is a genuinely-engaged game. Expected to lift NDCG; cleanly attributable because Phases A+B are already proven.
- **DCN V2.** Replace deep MLP with explicit cross layers — only if cross features stop helping.

### Open infrastructure TODOs

- **Fix `evaluate_only` memory like we just fixed train (2026-05-21).** `evaluate_only` in `ranker/train.py` calls `load_splits('data')` which still loads `train_ds` in full mode — pulling all 23 × {label, negs} cross features for 4.3M training rows into RAM (~40 GB) even though full-val eval only ever touches `val_ds`. Same fix pattern as the train-mode work that landed today: either skip `train_ds` entirely in the eval-only path, or load it in `train_only` mode (or a future `metadata_only` mode that drops even the history arrays). Expected peak memory after fix: ~1.5 GB (just val_ds full) instead of 60+ GB.
- **Share parquet-loading utilities.** `_scalar_to_numpy`, `_fixed_list_to_numpy`, and the `_USER_AND_LABEL_COLS` / `_EVAL_ONLY_COLS` / `_CROSS_FEATURE_COLS` manifests in `ranker/dataset.py` are the right home for parquet → numpy conversion. They're already reused via `RankerDataset` (used by both train and eval). One open duplication: the cross-feature column manifest currently lives in 3 places — `ranker/dataset._CROSS_FEATURE_COLS` (manifest of attr names), `ranker/precompute._CROSS_COLS` (manifest of column names + dtypes for parquet write), and `ranker/train._WIDE_NORM_PARQUET_COLS` (manifest of just the Z-scored label column names). All three need to stay in lockstep when a bucket lands. Consolidate into a single canonical manifest in `ranker/dataset.py` that the other two import — or push down into a new `ranker/_columns.py` if both `dataset.py` and `precompute.py` want to consume it without a circular import. Lower priority than the `evaluate_only` memory fix; cosmetic until someone forgets to update one of the three lists on a new bucket.

---

## 10. Discipline Rules

1. **One bucket at a time.** Phase B groups cross features into buckets that share a semantic theme (see §9). Each bucket is one training experiment, measured against the previous baseline. Bundle-level NDCG is the verdict — *do not* attempt per-feature attribution by default. Drop-one ablation (train bucket-minus-feature-X for each X) is only triggered when a bucket disappoints, as a diagnostic. Rationale: 30h to test 10 features individually vs ~8h to test 4 buckets, with strictly correct per-bucket verdicts. (Note: inference-time ablation — zeroing a feature's head weight after training — is NOT a valid substitute. A model trained without X shifts its deep representation and may recover most of the lost signal; inference-zero only measures "how much is this feature alive in *this* trained head," not "is X worth including.")
2. **Fair-α rule.** Always report which CG checkpoint (α=?) you're comparing against. Phase B experiments compare against A-N2 (NDCG@10 0.0741), which compares against CG α=0.
3. **Beat CG on content features before re-enabling CG score.** Earn improvements from independent signal.
4. **No `src/` modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **Softmax CE only.** Never sigmoid in `forward()`. Pointwise BCE was tried during early bring-up and abandoned.
7. **E2E ceiling always enforced** in both ranker eval and CG baseline.
8. **If `src/model.py` or `src/train.get_config()` changes** (tower hidden dims, embedding sizes, new sub-towers): re-derive §3 *first* before changing the ranker — partial drift silently breaks warm-start.
9. **Permanently dropped wide-feature classes:** (a) disliked-history variants (Bucket 3 ✗, see §8 — partition rule too noisy on Steam); (b) dev-catalog signals in any shape (Bucket 4 ✗, see §8 — redundant with `developer_lookup` in the deep tower). Don't re-propose either class as a new bucket. If a future need points at dev-side signal, the lever is on the deep tower, not the wide bypass.

---

## 11. Appendix — Loss family rationale

Why sampled softmax CE and not something else.

**The N-curve** (Klenitskiy & Vasilev, "Turning Dross Into Gold Loss," RecSys'23 — [arxiv 2309.07602](https://arxiv.org/abs/2309.07602)). On ML-1M with vanilla SASRec, swapping BCE+1-neg for sampled softmax CE with N=3000 lifts NDCG@10 by +38% (0.1341 → 0.1857). The curve rises monotonically up to N≈1000 then plateaus; N=3000 sampled slightly *beats* full softmax (sampling acts as a mild regularizer). Steam's Phase A reproduced the same curve shape: Try 1 (N=100) ≈ 0.066, Try 3 (N=500) = 0.070, A-N2 (N=1000) = 0.0741. The lever was always N.

**Why not BCE / DeepFM / DCN-V2.** Pointwise CTR rankers assume rating binarization (Steam has no star ratings) or impression logs (Steam has no shown-but-not-played signal — every row in `australian_users_items.json` is a positive). Sampled negs + BCE produces overconfidence pathologies (Petrov & Macdonald, gSASRec RecSys'23) and Klenitskiy table 2 shows BCE strictly loses to softmax CE on NDCG. We tested it directly during bring-up — converged to class prior, NDCG random.

**Why not SimpleX / CCL.** Replaces softmax with margin-based hinge, breaks parity with CG, and the "large N" insight is the same lesson softmax CE already exploits.

**Why not WARP / LightFM.** Adaptive iterative sampling is fundamentally serial — does not vectorize on GPU. Superseded by sampled softmax with large N.

**Parking lot.** *In-batch negatives + LogQ correction* ([arxiv 2507.09331](https://arxiv.org/abs/2507.09331)) would give 511 free negs per row at batch=512. Not on the critical path — Phase A is already at the dot-product ceiling on identical features; loss-family changes can't close that. Worth revisiting only if Phase B saturates and we want to test "is the loss family itself the ceiling?"
