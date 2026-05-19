# Steam Ranker: Implementation Plan

## Status (2026-05-18)

Phase A ✓ and Phase B Bucket 1 ✓.

**A-N2** (Phase A exit, tag_cosine only): NDCG@10 0.0741 vs CG α=0 0.0752 (Δ −1.5%). Proved a Wide & Deep MLP can effectively match the CG two-tower dot product on identical features.

**Bucket 1** (A-N2 + genre_overlap + tag_overlap + dev_affinity, 4 wide cross features): **NDCG@10 0.0822** vs CG α=0 0.0752 (Δ **+9.3%**) and vs A-N2 0.0741 (Δ **+10.9%**). MRR 0.0779 (+7.3% vs CG). Pure-reranking subset NDCG@10 0.1487 vs 0.1361 (+9.3%) — the lift is genuine reranking signal, not an E2E ceiling artifact. The three new categorical-overlap features together earned their seat and ranker now decisively beats the α=0 CG yardstick.

**Next: Bucket 2 (Numeric Matching)** — Price Match, Era Gap, Playtime Calibration, Popularity Match. Measured against Bucket 1 (NDCG@10 0.0822).

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

Implementation lives in `ranker/cross_features.py` — two utility functions (`overlap_pool`, `dev_affinity_pool`) parameterized on `(pool_indices, pool_weights, cand_idx)` so Bucket 3 can reuse them with last-3-liked pool args. Precompute and train both call the same utils → bit-exact identity between parquet and on-the-fly compute (verified to 2.98e-8 FP roundoff).

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

### Phase B — Bucket 2 (NEXT)

Numeric matching: Price Match + Era Gap + Playtime Calibration + Popularity Match. Measured against Bucket 1 (NDCG@10 0.0822). All four require Z-score normalization with fixed train-set mean/std stored as persistent model buffers — see §9 "Wide-feature normalization."

---

## 9. Cross-Feature Roadmap (Phase B)

Features are grouped into **buckets** that share a semantic theme. Each bucket is one training experiment, measured against the previous Phase B baseline (or A-N2 for the first). Bundle-level NDCG is the verdict; per-feature attribution is only done as a drop-one diagnostic if a bucket disappoints (see §10 rule 1).

All Phase B cross features sit on the **wide bypass** — each feature is one column in `head.weight` with a direct gradient path, *not* concatenated into the deep MLP (which would drown a single scalar in ~290 dims).

### B-0 (in A-N2) — Tag Cosine

| Feature | Formula |
|---|---|
| **Tag Cosine** | precomputed: `cosine(user_tag_pool_tfidf, item_tag_vec)` — direction match on the user's full tag profile vs candidate. Magnitude-blind. |

### Bucket 1 ✓ — Content / Categorical Overlap (2026-05-18)

Set-membership and overlap signals CG mathematically cannot represent (dot products can't do per-item normalization or categorical membership tests). **Outcome:** NDCG@10 0.0741 → 0.0822 (+10.9% vs A-N2). See §8 Phase Log for full metric table.

| # | Feature | Formula |
|---|---|---|
| B-1 | **Genre Overlap** | `(user_genre_w · item_genre_binary) / item_genre_count` — mean over the candidate's genres of "user's playtime fraction in that genre." [0, 1]. Buffers: `game_genre_binary`, `game_genre_count`. |
| B-1b | **Tag Overlap** | `(user_tag_w · item_tag_binary) / item_tag_count` — same structure as B-1 but on tags. **Magnitude-aware and complementary to B-0 Tag Cosine** (cosine is direction-only; overlap is "average user weight on candidate's tags"). Buffers: `game_tag_binary`, `game_tag_count`. |
| B-2 | **Developer Affinity** | `playtime-weighted fraction of user's history under this developer` (i.e. `Σ_i h_pw[i] · 1[hist_dev[i] == item_dev]`). Captures studio loyalty (FromSoftware, Larian, Paradox) — categorical membership CG can't do. |

### Bucket 2 — Numeric Matching (NEXT)

Absolute-difference scalars on user-vs-item numeric stats. All require Z-score normalization with fixed train-set mean/std stored as persistent buffers (see "Wide-feature normalization" below).

| # | Feature | Formula |
|---|---|---|
| B-3 | **Price Match** | `abs(user_mean_price_bucket - item_price_bucket)` — F2P / indie / AAA buyer segments. |
| B-4 | **Era Gap** | `abs(user_mean_year_norm - item_year_norm)` — new release vs retro preference. |
| B-5 | **Playtime Calibration** | `user_avg_log_playtime - item_global_avg_log_playtime` — heavy-hours user vs short-session user. |
| B-9 | **Popularity Match** | `abs(user_avg_log_count - item_log_count)` — user preference for popular vs obscure. |

### Bucket 3 — Recency (mirror of Bucket 1, last 3 LIKED games)

Same three features as Bucket 1, but the user-side pool is restricted to the **last 3 non-pad positions of `X_hist_liked`** (not the full history). Captures recency drift on positive-signal context — "what has the user been *enjoying* lately" — separate from the full-history signal in Bucket 1.

Steam has no timestamps, so "last" = the last 3 entries in `X_hist_liked`, which is filled in shuffled-prefix order. Shuffle-determined but seeded, so consistent across runs. Playtime weights re-normalized to sum to 1 over those 3 positions.

If a user has fewer than 3 liked games in the context, use however many exist; if 0, all three features fall through to 0 (no recency signal, model relies on Bucket 1).

**New precompute column required:** `X_hist_liked_playtime_weights` (parallel to `X_hist_liked`, MAX_HISTORY_LEN), looking up each liked entry's playtime weight from the full pool. Otherwise the recency pool can't be playtime-weighted consistently with Bucket 1.

| # | Feature | Formula |
|---|---|---|
| B-8a | **Recent Genre Overlap** | Same as B-1 Genre Overlap (`(user_genre_w · item_genre_binary) / item_genre_count`), but user_genre_w is computed from the last 3 non-pad positions of `X_hist_liked` with their playtime weights re-normalized over those 3. |
| B-8b | **Recent Tag Overlap** | Same as B-1b Tag Overlap, but on the last 3 liked (re-normalized playtime weights). |
| B-8c | **Recent Developer Affinity** | Same as B-2 Developer Affinity (`Σ_i h_pw_recent[i] · 1[hist_dev[i] == item_dev]`), but on the last 3 liked. Captures "are you currently on a studio kick" — sharper than the full-history version. |

### Bucket 4 — CG Score (kept solo)

| # | Feature | Formula |
|---|---|---|
| B-10 | **CG Score** | raw CG dot, per candidate. **Re-enable LAST.** CG score is circular ("follow CG and beat CG"); only earn it after content features have proven independent value. Kept solo to make its lift cleanly attributable. |

### Deep-concat additions (future, not in any bucket above)

| Feature | Path | Note |
|---|---|---|
| Genre Diversity / Tag Entropy / History Confidence | Deep | User-state scalars added to the deep concat. Only consider after the wide-path roadmap saturates. |

### Removed from the roadmap

- ~~Dislike Similarity~~ (was B-6, `cosine(user_disliked_pool, item_id_emb)`) — dropped.
- ~~Tag Peak Match~~ (was B-7, `max(user_tag_profile * item_tag_vec)`) — dropped. Tag Cosine (B-0) and Tag Overlap (B-1b) cover the two useful tag-affinity views; the "single peak" angle is redundant.
- ~~Recent Game Similarity~~ (was B-8, `dot(last_played_id_emb, item_id_emb)`) — **replaced** by Bucket 3 above (Recent Genre Overlap / Tag Overlap / Developer Affinity on the last 3 LIKED games). Item-ID dot is a weak proxy; mirroring Bucket 1's three features on a recent-liked window is the meaningful recency signal.

### Wide-feature normalization

Wide features beyond cosines/Jaccard must be normalized before concatenation using fixed statistics registered as **persistent** model buffers (not BatchNorm — train/eval batch composition differs). Compute mean/std from a single pass over training data, register with `register_buffer(name, tensor)` (persistent — the default; do not pass `persistent=False`, that excludes from `state_dict` and silently degrades the model on load).

```
Expected ranges (pre-normalization):
  Tag Cosine                  [−1, 1]   → no normalization
  Genre Overlap               [0, 1]    → no normalization
  Tag Overlap                 [0, 1]    → no normalization
  Developer Affinity          [0, 1]    → Z-score (heavy zero mass — may need separate treatment)
  Price Match                 [0, ~8]   → Z-score
  Era Gap                     [0, 1]    → Z-score (after normalizing year to [0,1])
  Playtime Calibration        [~−5, +5] → Z-score
  Popularity Match            [0, ~8]   → Z-score
  Recent Genre Overlap        [0, 1]    → no normalization
  Recent Tag Overlap          [0, 1]    → no normalization
  Recent Developer Affinity   [0, 1]    → Z-score (same heavy-zero treatment as full-history B-2)
  CG Score                    [~−1, 1]  → no normalization (already cosine-bounded)
```

Initialize new wide-feature weights at 0.1 — small non-zero signal without swamping the deep path.

### Other future phases

- **Phase C — Label quality.** After Phase B saturates, re-precompute with stricter `is_liked` filter (`raw_hours ≥ game_median` OR `≥ 2× user_median` OR `recommend=True`). Slashes label set by 40-60% but every label is a genuinely-engaged game. Expected to lift NDCG; cleanly attributable because Phases A+B are already proven.
- **DCN V2.** Replace deep MLP with explicit cross layers — only if cross features stop helping.

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

---

## 11. Appendix — Loss family rationale

Why sampled softmax CE and not something else.

**The N-curve** (Klenitskiy & Vasilev, "Turning Dross Into Gold Loss," RecSys'23 — [arxiv 2309.07602](https://arxiv.org/abs/2309.07602)). On ML-1M with vanilla SASRec, swapping BCE+1-neg for sampled softmax CE with N=3000 lifts NDCG@10 by +38% (0.1341 → 0.1857). The curve rises monotonically up to N≈1000 then plateaus; N=3000 sampled slightly *beats* full softmax (sampling acts as a mild regularizer). Steam's Phase A reproduced the same curve shape: Try 1 (N=100) ≈ 0.066, Try 3 (N=500) = 0.070, A-N2 (N=1000) = 0.0741. The lever was always N.

**Why not BCE / DeepFM / DCN-V2.** Pointwise CTR rankers assume rating binarization (Steam has no star ratings) or impression logs (Steam has no shown-but-not-played signal — every row in `australian_users_items.json` is a positive). Sampled negs + BCE produces overconfidence pathologies (Petrov & Macdonald, gSASRec RecSys'23) and Klenitskiy table 2 shows BCE strictly loses to softmax CE on NDCG. We tested it directly during bring-up — converged to class prior, NDCG random.

**Why not SimpleX / CCL.** Replaces softmax with margin-based hinge, breaks parity with CG, and the "large N" insight is the same lesson softmax CE already exploits.

**Why not WARP / LightFM.** Adaptive iterative sampling is fundamentally serial — does not vectorize on GPU. Superseded by sampled softmax with large N.

**Parking lot.** *In-batch negatives + LogQ correction* ([arxiv 2507.09331](https://arxiv.org/abs/2507.09331)) would give 511 free negs per row at batch=512. Not on the critical path — Phase A is already at the dot-product ceiling on identical features; loss-family changes can't close that. Worth revisiting only if Phase B saturates and we want to test "is the loss family itself the ceiling?"
