# Steam Ranker: Implementation Plan

## Status (2026-05-23)

Phase A ✓. Phase B: Buckets 1 ✓, 2 ✓, 3 ✗, 4 ✗, 5 ✓, 6 ✓, 8 ✗, 9 ✗. **Roadmap complete — Bucket 9 was the last planned bucket. Bucket 6 is the final/PROD ranker.**

**Best-so-far ranker = FINAL ranker** (Bucket 6, last kept): `saved_models/ranker/ranker_wd_alpha_0_20260520_204654.pth` — NDCG@10 0.0867, MRR 0.0813, pure-rerank NDCG@10 0.1569 / MRR 0.1391. CG comparator (α=0): `saved_models/best_triple_full_softmax_popularity_alpha_00_20260515_084320.pth`. No further feature buckets planned; next work is serving integration.

Headline metric is full-val NDCG@10, each bucket measured against the previous *kept* baseline. Full tables + per-slice canary analysis live in §8; this is the scoreboard.

| Bucket | Adds | NDCG@10 | Δ vs prev kept | Verdict |
|---|---|---:|---:|---|
| **A-N2** (Phase A exit) | tag_cosine only | 0.0741 | — (CG α=0: 0.0752) | baseline — MLP ≈ dot product |
| **1** ✓ | genre/tag/dev overlap (full) | **0.0822** | **+10.9%** | ship — biggest single win |
| **2** ✓ | overlap × {liked, recent-3} | **0.0828** | +0.7% | ship, barely — B1 took most of it |
| **3** ✗ | overlap × disliked | 0.0824 | −0.5% | **drop** — disliked partition too noisy |
| **4** ✗ | dev-catalog genre/tag overlap | 0.0827 | −0.1% | **drop** — redundant with `developer_lookup` |
| **5** ✓ | 5 numeric-match scalars | **0.0866** | **+4.6%** | ship — uniform +4–5%, new signal class |
| **6** ✓ | 8 niche/IDF crosses | 0.0867 | +0.1% (flat) | **ship on canary** — offline-flat, canary-better |
| **8** ✗ | 2 engagement crosses | 0.0870 | +0.03% (flat) | **drop** — flat offline AND canary-regressed (RPG/Fighting) |
| **9** ✗ | CG corpus log-rank (solo, last) | 0.0864 | −0.3% (flat) | **drop** — no lift (circular) + frozen-CG serving cost |

**Permanently dropped signal classes** (don't re-propose in any shape — see §10 rule 9): disliked-history variants (Bucket 3); dev-catalog aggregates (Bucket 4); engagement-level crosses (Bucket 8 — popularity-leak channel, deep MLP already captures it). **Bucket 7** (Item-Intrinsic Priors) was dropped in planning; 2 of its 3 features were absorbed into Bucket 6 as proper user × item crosses (`niche_dev_match`, `max_tag_idf_match`), the third (`dev_specialization`, item-only) dropped.

**Ranker is not yet wired into serving.** Streamlit runs CG-only retrieval; integration (load both checkpoints, shared user-side feature engineering, two-stage scoring) is the real remaining work now that the bucket roadmap is complete. Note: the FINAL ranker (Bucket 6) has **no frozen-CG runtime dependency** — Bucket 9 would have introduced one and was dropped, so serving only needs the ranker's own content buffers + the CG retrieval already in place.

> **Terminology:** "pool" means an embedding aggregation (the deep tower's `pool_liked/disliked/full/playtime`). Cross features compute weighted overlap / categorical affinity *directly over the history arrays* — no embedding aggregation. Code uses `weighted_overlap` / `dev_affinity` with `history_indices` / `history_weights`.

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

Tower shapes are listed inline in the deep-concat layout above (2-layer: item_tag / user_tag / user_genre; 1-layer: item_id / item_genre / developer / year / price). Every lookup-fronted feature (id, dev, year, price) is `Embedding → Linear+ReLU` as two separate `nn.Module`s, not a fused Embedding — that's how CG names them, which the warm-start map relies on.

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

**Default ON** (`get_config()['warm_start'] = True`). A from-scratch ablation produced materially worse NDCG — the deep MLP needs CG's content-tower head start. The full CG→ranker key map lives in `train._CG_TO_RANKER_KEY_MAP`: 26 tensor transfers (4 lookups + 5 one-layer towers × 2 + 3 two-layer towers × 4); anything short means shape drift or key mismatch. **Not transferred:** deep MLP, head, cross-feature weights (random init — no CG counterpart); `user_projection.*` / `item_projection.*` (ranker has no projection MLP); all buffers (rebuilt from FeatureStore).

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

### Phase B — Bucket 8 ✗ (2026-05-22)

**Engagement-Level Cross** (dropped): 2 user × item scalars crossing `X_user_avg_log` with each candidate's intrinsic engagement level — `user_dev_engagement_cross` (× `dev_mean_log_playtime`) and `user_genre_engagement_cross` (× `genre_mean_log_playtime`). Wide-head cols 23–24. Two new per-game buffers (`game_dev_mean_log_playtime`, `game_genre_mean_log_playtime`) built in `features.load_features`; compute via `engagement_cross_pair` in `ranker/cross_features.py`.

| Metric | Bucket 6 | Bucket 8 | Δ vs B6 |
|---|---:|---:|---:|
| NDCG@1 | 0.0319 | 0.0324 | +1.6% |
| NDCG@10 | 0.0867 | 0.0870 | +0.3% |
| NDCG@20 | 0.1083 | 0.1088 | +0.5% |
| NDCG@50 | 0.1410 | 0.1413 | +0.2% |
| Hit@10 | 0.1625 | 0.1626 | +0.1% |
| Hit@20 | 0.2486 | 0.2494 | +0.3% |
| MRR | 0.0813 | 0.0817 | +0.5% |
| Pure-rerank NDCG@10 | 0.1569 | 0.1574 | +0.3% |
| Pure-rerank MRR | 0.1391 | 0.1398 | +0.5% |

**Bucket 8 outcome:** offline-flat (every delta in the ±0.001 noise band — sign is uniformly positive but magnitudes are 3rd–4th decimal, indistinguishable from a lucky-seed B6) **and canary-regressed on the two niche slices that matter most.** Per-slice canary read (ranker-vs-ranker, identical CG control):

- **Western RPG** — ✗ regression. Terraria leaps to **#1**; Civ V #8, PlanetSide 2 #9, PAYDAY 2 #13, Portal #16, Warframe #17, Rust #20 all leak into the flagship RPG list. Pillars of Eternity, Icewind Dale: EE, Dragon Age: Origins, Might & Magic X — all present in B6 — drop out.
- **Fighting** — ✗ regression, and it *undoes Bucket 6's headline win*. FF Type-0 #6, Borderlands 2 #7, Warframe #8, Counter-Strike #12, FF VIII #14, FF XIII #20 — the exact JRPG + shooter bleed B6 was kept for cleaning up.
- **Civ (4X)** — slight win. Beyond Earth #7, EU IV #14, EU III #19, TW: Warhammer #20 (genuine 4X/grand-strategy) vs B6's EVE Online / GoT Genesis / X3 outliers.
- **JRPG / FPS / Indie / Racing / Survival / Management** — comparable to B6; minor reorders inside already-good lists, no clear direction.

**Why it regressed:** the cross is effectively a popularity-leak channel. `dev_mean_log_playtime` / `genre_mean_log_playtime` are highest for high-engagement mass-market titles (Terraria, Borderlands 2, Warframe, Counter-Strike, PlanetSide), so the wide head's positive weight on `X_user_avg_log × cand_engagement` boosts those titles *across every genre* regardless of taste fit — the precise cross-genre pollution the α=0.4 popularity correction and Bucket 6's niche scalars exist to suppress. **Confirms the pre-train hypothesis** (§9): the deep MLP already sees `X_user_avg_log` plus the dev/genre embeddings, so the engagement signal was redundant; re-expressing it as a multiplicative wide cross only added a popularity lever that does net harm. **Dropped — code reverted via git, not promoted.**

**Lesson:** "offline-flat" alone isn't the ship signal — Bucket 6 was flat offline *and canary-better*; Bucket 8 was flat offline *and canary-worse*. The canary is the tiebreaker in both directions, and a uniformly-positive-but-tiny offline delta is not evidence of a real win when canary list-shape regresses. A flat bucket whose mechanism is "boost high-engagement items" is a popularity leak in disguise — it should be expected to pollute niche slices, not help them.

### Phase B — Bucket 9 ✗ (2026-05-23) — FINAL bucket

**CG corpus log-rank** (dropped): one CG-derived wide feature — `log1p(corpus rank)` of each candidate under a frozen CG for that user (col 23). Implemented as `cross_features.cg_corpus_log_rank` + a frozen CG threaded into train/eval (canary already had CG). Decided on **corpus** rank, not within-pool rank, because training uses random negs while eval uses the CG-top-K pool — only a pool-independent rank is consistent across both (see "Why log-rank" + the implement-time scope correction below).

| Metric | Bucket 6 | Bucket 9 | Δ vs B6 |
|---|---:|---:|---:|
| NDCG@1 | 0.0319 | 0.0314 | −1.6% |
| NDCG@5 | 0.0675 | 0.0669 | −0.9% |
| NDCG@10 | 0.0867 | 0.0864 | −0.3% |
| NDCG@50 | 0.1410 | 0.1407 | −0.2% |
| Hit@10 | 0.1625 | 0.1628 | +0.2% |
| Hit@20 | 0.2486 | 0.2495 | +0.4% |
| MRR | 0.0813 | 0.0809 | −0.5% |
| Pure-rerank NDCG@10 | 0.1569 | 0.1564 | −0.3% |
| Pure-rerank MRR | 0.1391 | 0.1383 | −0.6% |

**Bucket 9 outcome:** flat — every NDCG and MRR number fractionally **down** (Hit@K fractionally up), all in the ±0.001 noise band. **No lift.** This is precisely the "circular — only re-imports CG's ranking" result the bucket was gated against: the ranker warm-started from CG and its content features (Buckets 1/5/6) already subsume everything CG's *ordering* encodes, so handing the linear head CG's rank as an explicit column adds no independent signal (and the tiny NDCG/MRR dips suggest it slightly distracts the head). **Dropped — code reverted via git, never committed.** Canary not run: for a *circular* feature the verdict criterion is "keep only on lift," and offline showed none.

**Serving-cost asymmetry (why the bar was higher than Bucket 6).** Unlike every content bucket, Bucket 9 imposes a hard runtime dependency: the ranker cannot score a candidate without a **frozen CG forward** to compute corpus ranks. A flat-but-harmless content feature can ride along cheaply (Bucket 6 precedent); a flat *and serving-expensive* feature must clear a real win to justify itself. Flat offline → clear drop. The FINAL ranker (Bucket 6) therefore has **no frozen-CG dependency at inference**.

**Implement-time scope correction (worth keeping).** The plan originally called Bucket 9 "wire-only" — wrong. Training uses **random** negatives (`n_hard_negs=0`), which have no precomputed CG score, so a CG feature (rank *or* raw score) can only be sourced by running a frozen CG live at forward time. The parquet's `cg_*` columns cover only the eval pool. This is why Bucket 9 needed a frozen CG in train + eval, not a parquet read. The capped `cg_label_rank` was still usable for the feature's Z-stats (mean→head bias, std→weight scale are absorbed by a single linear weight, so the top-K cap is harmless there).

**Lesson:** a feature that re-imports an upstream model's own score/rank is only worth its keep if it beats that model on signal the reranker doesn't already hold — and a warm-started reranker with strong content features usually already holds it. Pair "no lift" with "adds a serving dependency" and the drop is unambiguous. This closes the bucket roadmap.

---

## 9. Cross-Feature Roadmap (Phase B)

Cross features are grouped into **buckets** that share a semantic theme. Each bucket is one training experiment, measured against the previous *kept* baseline; bundle-level NDCG is the verdict, with drop-one attribution only as a diagnostic when a bucket disappoints (§10 rule 1). All cross features sit on the **wide bypass** — one column in `head.weight` with a direct gradient path, *not* concatenated into the deep MLP (where a single scalar would drown in ~290 dims).

**Landed/dropped bucket designs (1–6) are realized in `ranker/cross_features.py` + the precompute/train/canary call sites — code is the source of truth.** Outcomes, eval tables, and drop/pass reasoning live in §8. The scoreboard is in the Status section. This section keeps only the canonical column ordering + the final bucket design (9) + cross-cutting reference (normalization, removed ideas, future phases).

### Column slots (stable across checkpoints — see `dataset.compute_cross_features`)

```
col 0   : tag_cosine                          (Phase A — cosine(user_tag_pool_tfidf, item_tag_vec))
col 1-3 : genre/tag/dev overlap, FULL slice   (Bucket 1 ✓)
col 4-6 : genre/tag/dev overlap, LIKED slice  (Bucket 2 ✓)
col 7-9 : genre/tag/dev overlap, RECENT-3     (Bucket 2 ✓)
col 10  : price_match                         (Bucket 5 ✓)
col 11  : era_gap                             (Bucket 5 ✓)
col 12  : playtime_calibration_median         (Bucket 5 ✓, signed)
col 13  : popularity_match                    (Bucket 5 ✓)
col 14  : sentiment_match                     (Bucket 5 ✓)
col 15-16: tag_overlap_idf, full/liked        (Bucket 6 ✓)
col 17-18: niche_tag_match, full/liked        (Bucket 6 ✓)
col 19-20: max_tag_idf_match, full/liked      (Bucket 6 ✓)
col 21-22: niche_dev_match, full/liked        (Bucket 6 ✓)
(col 23 : cg_log_rank — Bucket 9 ✗, dropped + reverted; PROD ranker stops at col 22)
```
**FINAL PROD ranker = 23 features (cols 0–22), n_wide_normalized=13.** Cols are append-only — never reorder, or older checkpoints mis-align at load. Cols 10–22 are Z-scored at forward time (`wide_norm_mean/std` persistent buffers); cols 0–9 are bounded ([−1,1]/[0,1]) and pass through raw. Dropped buckets 3/4/8/9 transiently occupied cols 10+ (Bucket 8 held cols 23–24, Bucket 9 held col 23) and were reverted, so those slots are free again. The overlap features reuse `weighted_overlap` / `dev_affinity` across full/liked/recent-3 slices (only `history_indices`/`history_weights` change); Bucket 5 introduced the Z-score infra; Bucket 6 added IDF/rarity scalars (`niche_scalar_triple`).

### Bucket 8 — Engagement-Level Cross ✗ (dropped 2026-05-22)

**Dropped — flat offline AND canary-regressed. Code reverted via git, never committed. See §8 for the full result + per-slice canary read.** Tried 2 scalars crossing the user's engagement level (`X_user_avg_log`) with the candidate's intrinsic engagement (`dev_mean_log_playtime` / `genre_mean_log_playtime`). The mechanism turned out to be a popularity-leak channel — high-engagement mass-market titles got boosted across all genres — so it polluted the Western RPG and Fighting niche slices rather than helping. Confirmed the pre-train hypothesis: the deep MLP already captures engagement via `X_user_avg_log` + dev/genre embeddings, leaving no independent signal for the wide cross. **Do not re-propose engagement-level crosses in any shape** (§10 rule 9).

### Bucket 9 — CG Corpus Log-Rank ✗ (dropped 2026-05-23) — FINAL bucket

**Dropped — flat offline (no lift) + a frozen-CG serving dependency. Code reverted via git, never committed. See §8 for the result table + full reasoning.** Tried one CG-derived feature: `log1p(corpus rank)` of each candidate under a frozen CG for that user (col 23). The implement-time correction landed on **corpus** rank (pool-independent → consistent between training's random negs and eval's CG-top-K pool) computed from a frozen CG live at forward time — *not* the "within-pool, wire-only" framing this section originally carried, which only held for the eval path. Offline came back flat-to-slightly-negative: the warm-started ranker's content features already subsume CG's ordering, so re-importing CG's rank adds no independent signal. Combined with the serving cost (the ranker would need a frozen CG forward to score anything), the drop was unambiguous. **This closes the bucket roadmap — Bucket 6 is the final ranker.**

> **Dropped from scope:** the previously-considered deep-concat additions (Genre Diversity / Tag Entropy / History Confidence — user-state scalars on the deep path) are *not* being pursued. They're user-only signals that don't need the ranker's cross structure, and the buildout is wrapping up. If user-state signal is ever wanted, the higher-leverage place is CG's user tower (raises the retrieval ceiling), not the ranker reranker.

### Removed from the roadmap

- ~~Dislike Similarity~~ / ~~disliked-history columns~~ — dropped; the disliked partition itself is too noisy (Bucket 3 ✗, §8).
- ~~Tag Peak Match~~ (`max(user_tag_profile * item_tag_vec)`) — redundant with Tag Cosine + Tag Overlap variants.
- ~~Recent Game Similarity~~ (`dot(last_id_emb, item_emb)`) — **replaced** by the Recent-3 slice in Bucket 2.

### Wide-feature normalization

Wide features beyond cosines/overlaps are Z-scored before the head using fixed train-set mean/std in **persistent** model buffers (`wide_norm_mean/std` — not BatchNorm; train/eval batch composition differs). Populated by `populate_wide_norm_buffers` in one pass over the train parquet at training start, std clamped to 1.0 near zero variance. `_normalize_wide` applies it to the trailing `n_wide_normalized` columns only (cols 10–22 in the FINAL ranker, n_wide_normalized=13); cols 0–9 (cosine/overlap, bounded [−1,1]/[0,1]) pass through raw. New wide-head weights init at 0.1.

### Other future phases

- **DCN V2.** Replace deep MLP with explicit cross layers — only if cross features stop helping.

### Open infrastructure TODOs

- **✓ Fixed `evaluate_only` memory (2026-05-21).** `evaluate_only` in `ranker/train.py` used to call `load_splits('data')`, loading `train_ds` in full mode — pulling all cross features for 4.3M training rows into RAM (~40 GB) even though full-val eval only ever touches `val_ds`. Fix: added `train_mode='skip'` to `load_splits` (returns `None` for train_ds, never opens the train parquet) and switched the eval-only call to `load_splits('data', train_mode='skip')`. Expected peak memory after fix: ~1.5 GB (just val_ds full) instead of 60+ GB. Train path (line ~520) is unaffected — still uses `'full'`/`'train_only'`.
- **Possible improvement — share parquet-loading utilities / consolidate cross-feature manifests.** `_scalar_to_numpy`, `_fixed_list_to_numpy`, and the `_USER_AND_LABEL_COLS` / `_EVAL_ONLY_COLS` / `_CROSS_FEATURE_COLS` manifests in `ranker/dataset.py` are the right home for parquet → numpy conversion. They're already reused via `RankerDataset` (used by both train and eval). One open duplication: the cross-feature column manifest currently lives in 3 places — `ranker/dataset._CROSS_FEATURE_COLS` (manifest of attr names), `ranker/precompute._CROSS_COLS` (manifest of column names + dtypes for parquet write), and `ranker/train._WIDE_NORM_PARQUET_COLS` (manifest of just the Z-scored label column names). All three need to stay in lockstep when a bucket lands. Consolidate into a single canonical manifest in `ranker/dataset.py` that the other two import — or push down into a new `ranker/_columns.py` if both `dataset.py` and `precompute.py` want to consume it without a circular import. Cosmetic / drift-prevention only — no runtime cost, deferred until someone forgets to update one of the three lists on a new bucket.

---

## 10. Discipline Rules

1. **One bucket at a time.** Phase B groups cross features into buckets that share a semantic theme (see §9). Each bucket is one training experiment, measured against the previous baseline. Bundle-level NDCG is the verdict — *do not* attempt per-feature attribution by default. Drop-one ablation (train bucket-minus-feature-X for each X) is only triggered when a bucket disappoints, as a diagnostic. Rationale: 30h to test 10 features individually vs ~8h to test 4 buckets, with strictly correct per-bucket verdicts. (Note: inference-time ablation — zeroing a feature's head weight after training — is NOT a valid substitute. A model trained without X shifts its deep representation and may recover most of the lost signal; inference-zero only measures "how much is this feature alive in *this* trained head," not "is X worth including.")
2. **Fair-α rule.** Always report which CG checkpoint (α=?) you're comparing against. Phase B experiments compare against A-N2 (NDCG@10 0.0741), which compares against CG α=0.
3. **Beat CG on content features before re-enabling the CG signal (now CG log-rank, Bucket 9).** Earn improvements from independent signal first.
4. **No `src/` modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **Softmax CE only.** Never sigmoid in `forward()`. Pointwise BCE was tried during early bring-up and abandoned.
7. **E2E ceiling always enforced** in both ranker eval and CG baseline.
8. **If `src/model.py` or `src/train.get_config()` changes** (tower hidden dims, embedding sizes, new sub-towers): re-derive §3 *first* before changing the ranker — partial drift silently breaks warm-start.
9. **Permanently dropped wide-feature classes:** (a) disliked-history variants (Bucket 3 ✗, see §8 — partition rule too noisy on Steam); (b) dev-catalog signals in any shape (Bucket 4 ✗, see §8 — redundant with `developer_lookup` in the deep tower); (c) engagement-level crosses (Bucket 8 ✗, see §8 — `X_user_avg_log × cand-engagement` is a popularity-leak channel that pollutes niche slices; the deep MLP already captures engagement). Don't re-propose any of these as a new bucket. If a future need points at dev-side or engagement signal, the lever is on the deep tower, not the wide bypass.

---

## 11. Appendix — Loss family rationale

Why sampled softmax CE and not something else.

**The N-curve** (Klenitskiy & Vasilev, "Turning Dross Into Gold Loss," RecSys'23 — [arxiv 2309.07602](https://arxiv.org/abs/2309.07602)). On ML-1M with vanilla SASRec, swapping BCE+1-neg for sampled softmax CE with N=3000 lifts NDCG@10 by +38% (0.1341 → 0.1857). The curve rises monotonically up to N≈1000 then plateaus; N=3000 sampled slightly *beats* full softmax (sampling acts as a mild regularizer). Steam's Phase A reproduced the same curve shape: Try 1 (N=100) ≈ 0.066, Try 3 (N=500) = 0.070, A-N2 (N=1000) = 0.0741. The lever was always N.

**Why not BCE / DeepFM / DCN-V2.** Pointwise CTR rankers assume rating binarization (Steam has no star ratings) or impression logs (Steam has no shown-but-not-played signal — every row in `australian_users_items.json` is a positive). Sampled negs + BCE produces overconfidence pathologies (Petrov & Macdonald, gSASRec RecSys'23) and Klenitskiy table 2 shows BCE strictly loses to softmax CE on NDCG. We tested it directly during bring-up — converged to class prior, NDCG random.

**Why not SimpleX / CCL.** Replaces softmax with margin-based hinge, breaks parity with CG, and the "large N" insight is the same lesson softmax CE already exploits.

**Why not WARP / LightFM.** Adaptive iterative sampling is fundamentally serial — does not vectorize on GPU. Superseded by sampled softmax with large N.

**Parking lot.** *In-batch negatives + LogQ correction* ([arxiv 2507.09331](https://arxiv.org/abs/2507.09331)) would give 511 free negs per row at batch=512. Not on the critical path — Phase A is already at the dot-product ceiling on identical features; loss-family changes can't close that. Worth revisiting only if Phase B saturates and we want to test "is the loss family itself the ceiling?"
