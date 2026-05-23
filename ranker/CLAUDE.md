# Ranker Subsystem (`ranker/`)

Guidance for Claude Code when working in the `ranker/` directory. This is the **Wide & Deep ranker** — stage 2 of the two-stage recommender. The root `CLAUDE.md` documents the candidate-generation (CG) two-tower model; this file documents the reranker that sits on top of it. The CG model in `src/` is read-only from here.

## Status (2026-05-23)

**Buildout complete. Bucket 6 is the FINAL/PROD ranker and it is now served on Streamlit.**

Phase A ✓. Phase B: Buckets 1 ✓, 2 ✓, 3 ✗, 4 ✗, 5 ✓, 6 ✓, 8 ✗, 9 ✗. Roadmap complete — Bucket 9 was the last planned bucket.

**FINAL ranker** (Bucket 6, last kept): `saved_models/ranker/ranker_wd_alpha_0_20260520_204654.pth` — NDCG@10 0.0867, MRR 0.0813, pure-rerank NDCG@10 0.1569 / MRR 0.1391. CG comparator (α=0): `saved_models/best_triple_full_softmax_popularity_alpha_00_20260515_084320.pth`.

**Popularity penalty stays OFF the ranker (α=0).** A ranker trained with `popularity_alpha=0.2` was tested 2026-05-23 and rejected: it hurt offline ~27% (the ranker's lift over CG dropped from +0.0115 to +0.0084 NDCG@10) and did not produce meaningfully better canary lists (cleaner on Fighting/Western-RPG, but worse on Survival — GTA V/PAYDAY climbed, genre-defining DayZ fell). The CG stays raw α=0 as the fixed retrieval stage and the ranker stays raw α=0; neither stage applies a popularity penalty in the shipped config. The Menon Path 2 plumbing (`popularity_alpha`, `warm_start_alpha`) remains in `train.py` set to 0.

Headline metric is full-val NDCG@10, each bucket measured against the previous *kept* baseline.

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

**Permanently dropped signal classes** (don't re-propose in any shape — see §10 rule 9): disliked-history variants (Bucket 3); dev-catalog aggregates (Bucket 4); engagement-level crosses (Bucket 8). **Bucket 7** (Item-Intrinsic Priors) was dropped in planning; 2 of its 3 features were absorbed into Bucket 6 as proper user × item crosses (`niche_dev_match`, `max_tag_idf_match`), the third (`dev_specialization`, item-only) dropped.

The FINAL ranker (Bucket 6) has **no frozen-CG runtime dependency** — Bucket 9 would have introduced one and was dropped, so serving only needs the ranker's own content buffers + the CG retrieval already in place.

> **Terminology:** "pool" means an embedding aggregation (the deep tower's `pool_liked/disliked/full/playtime`). Cross features compute weighted overlap / categorical affinity *directly over the history arrays* — no embedding aggregation. Code uses `weighted_overlap` / `dev_affinity` with `history_indices` / `history_weights`.

---

## 1. Pipeline

Two-stage retrieve-and-rank:

1. **CG** (v5 softmax two-tower, 4-pool user tower, F.normalize on both towers, 128-dim) — retrieves top-100 candidates per rollback example.
2. **Ranker** (Wide & Deep MLP) — reranks the 100 candidates using richer features.

### CG checkpoints

- **Raw CG α=0** — `best_triple_full_softmax_popularity_alpha_00_<date>.pth`. This is now the **deployed retrieval stage**: it retrieves recall-maximizing top-100 and the ranker reranks them. It is also the honest offline-metrics yardstick — the ranker runs at α=0 and is measured against this checkpoint.
- **CG α=0.4** — the pre-ranker standalone-serving compromise (popularity correction baked into the CG to keep niche-taste lists clean). In the two-stage world the CG no longer carries the popularity work, so α=0.4 is no longer deployed; it remains the reference for the CG α-tradeoff (see root `CLAUDE.md`).

CG v5 prod baseline (α=0.4, for reference):

| K | Recall@K | NDCG@K |
|---|---:|---:|
| 1 | 0.0226 | 0.0226 |
| 5 | 0.0741 | 0.0481 |
| 10 | 0.1253 | 0.0645 |
| 20 | 0.2059 | 0.0848 |
| 50 | 0.3673 | 0.1166 |

MRR: 0.0611 (random: 0.0017). The α=0 CG sits ~10–25% above these on Recall/NDCG (see root `CLAUDE.md` for the full α=0 table).

---

## 2. Core Principles

### CG-parity baseline first

The ranker contains every CG input/feature, projected through the same per-feature towers CG uses, with the same dimensions and the same warm-started weights. Cross features come *on top* — they are the differentiator, not a replacement.

### Wide & Deep architecture

The ranker concatenates all per-feature embeddings (user-side + item-side) into ONE vector that feeds a deep MLP. Cross features bypass the MLP and go straight to the head — giving each one a direct learned weight. Without the wide bypass, cross-feature scalars would have to compete against ~290 dims for attention in the first hidden layer and get washed out during backprop.

### No CG coupling at runtime

The ranker owns its own copies of every parameter (its own `item_id_lookup`, `developer_lookup`, `developer_tower`, `item_tag_tower`, etc.). Warm-starting weights from CG at init is one-time copy at construction; the tensors then live entirely in the ranker `state_dict` and train freely. The ranker's only runtime connection to CG is (1) the candidate indices from retrieval, (2) precomputed features in the parquet, (3) static feature data both models read from disk.

### Fair-α rule

Always report which CG checkpoint (α=0 vs α=0.4) the ranker is being compared against. Ranker α=0 vs CG α=0 is the only meaningful offline comparison. Ranker α=0 vs CG α=0.4 is not a real win — the delta is mostly retrieval headroom from α=0.4's deliberate handicap.

---

## 3. Architecture

All dims and tower shapes match `src/model.py` (V5) and `src/train.get_config()`: `tag_embedding_size=32`, `user_genre_embedding_size=32`, `user_tag_embedding_size=32`, `item_genre_embedding_size=8`, `developer_embedding_size=12`, `item_year_embedding_size=8`, `price_embedding_size=4`, `item_id_embedding_size=32`. Tower hidden dims (hardcoded inside `src/model.py`): `tag_hidden=128`, `tag_ctx_hidden=256`, `genre_hidden=128`.

### Per-game static buffers (registered on the ranker)

Built from `FeatureStore` at construction. All non-persistent — rebuilt on load from `feature_store.pt` (serving rebuilds them via `ranker.train._buffers_from_fs`). Indexing by `cand_idx` is how `item_embedding()` gets per-game metadata at score time.

```python
self.register_buffer('game_year_idx',     year_idx,     persistent=False)  # (n_games+1,) int64
self.register_buffer('game_dev_idx',      dev_idx,      persistent=False)  # (n_games+1,) int64
self.register_buffer('game_price_idx',    price_idx,    persistent=False)  # (n_games+1,) int64
self.register_buffer('game_tag_matrix',   tag_matrix,   persistent=False)  # (n_games+1, n_tags)   float32 — TF-IDF
self.register_buffer('game_genre_matrix', genre_matrix, persistent=False)  # (n_games+1, n_genres) float32 — one-hot
```

CG already registers `game_tag_matrix` and `game_genre_matrix` — the ranker mirrors them exactly (same names, dtypes, vocab orderings) so warm-started towers consume the right rows. The persistent `wide_norm_mean/std` buffers ARE saved in the checkpoint.

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

`warm_start_alpha` decouples the warm-start CG source from the penalty: the α=0 ranker warm-starts from the raw α=0 CG. (Used by the rejected α=0.2 experiment to keep its warm-start source raw; now both are α=0.)

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
├── cross_features.py ← wide-bypass cross-feature compute utils (shared by precompute/train/canary/serving)
├── train.py          ← Sampled softmax CE training loop; build_ranker, _buffers_from_fs, warm-start
├── evaluate.py       ← NDCG@K, MRR, Hit@K, CG baseline, E2E ceiling
├── serving.py        ← shared rerank pipeline used by canary AND streamlit (build_user_inputs_from_indices, rerank_candidates)
├── canary.py         ← side-by-side CG vs Ranker top-N for synthetic users (delegates to serving.py)
├── export.py         ← serving artifact export (re-exports α=0 CG via src.export, then adds ranker.pth + ranker_config.json)
├── main.py           ← entry point (precompute / train / evaluate / canary / export)
├── eval_results/
└── canary_results/

data/
├── ranker_candidates_train.parquet
└── ranker_candidates_val.parquet
```

`src/` is read-only — the ranker doesn't modify CG code.

### Commands

```bash
python ranker/main.py precompute            # Stage 0: candidates (default raw α=0 CG)
python ranker/main.py precompute <cg.pth>   # override CG checkpoint
python ranker/main.py train                 # train Wide & Deep ranker
python ranker/main.py evaluate [ranker.pth] # eval-only (auto-finds most recent if omitted)
python ranker/main.py canary [ranker.pth]   # CG vs ranker top-N for all canaries
python ranker/main.py export [ranker.pth]   # serving artifacts (α=0 CG + given/latest ranker)
```

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

**Tag cosine:** raw TF-IDF cosine over `game_tag_matrix` rows (the buffer CG already uses) — not over tag-tower outputs. Model-independent, so the precompute parquet doesn't break if the tag tower's hidden dims change.

**Settings:**
- `N_SHUFFLES=3` for train, `1` for val — matches CG.
- `MAX_ROLLBACK_EXAMPLES_PER_USER=50` — matches CG.
- Label filter: `raw_hours > 0.5` AND `history[i] not in history[:i]` (dedupe guard avoids the ~0.2% Steam history duplicates from leaking the label into its own context).
- Steam has no timestamps — rollback order is shuffle-determined. Seed the CG load + rollback shuffle.
- Precompute peak memory was reduced ~103 GB → ~8–10 GB via chunked streaming writes (`CHUNK_SIZE=250_000`, single per-batch stack-and-sync) — landed with Bucket 6's 8 extra columns.

---

## 7. Training Stack (Active Config)

### Negative sampling

**Sampled softmax**, 1 label + 999 random corpus items = 1000-cand pool. No hard negs (`n_hard_negs=0`). Set in `get_config()`. Hard-neg infrastructure (parquet `neg_item_idxs`, `n_hard_negs` knob) is kept but disabled — at N ≥ ~1000 the hard-neg term is redundant and slightly hurts (see §8 Phase A finding 1).

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
popularity_alpha: 0.0                        ← raw; the α=0.2 experiment was rejected
warm_start_alpha: 0.0                        ← warm-start source CG α (decoupled from penalty)
n_random_negs:    999
n_hard_negs:      0
warm_start:       True
```

`_config.json` sidecar alongside each checkpoint records arch params + `popularity_alpha` + `temperature` + `n_random_negs` + `n_hard_negs`.

### Eval

NDCG@K, MRR, Hit@K for K ∈ {1, 5, 10, 20, 50, 100}. CG baseline uses the same E2E ceiling. `Recall@K_cand` printed as the production ceiling. Eval output header reports which CG checkpoint (α=0 vs α=0.4) the comparison uses.

---

## 8. Phase Log

### Phase 0 ✓ — CG α=0 baseline

Trained `best_triple_full_softmax_popularity_alpha_00_20260515_084320.pth` as the offline-metrics comparator. Sidecar originally marked `"do_not_export": true` / `"role": "ranker_offline_baseline"`. **Now promoted: it is the deployed retrieval stage in the two-stage architecture.**

### Phase A ✓ — Strict CG parity

Goal: prove a Wide & Deep MLP can effectively match a two-tower dot product on identical data and features. Sampled softmax CE + temperature=0.1, warm-start from α=0 CG, one cross feature (`tag_cosine`), ranker α=0, all 8 per-feature towers in CG parity.

#### Chronology (full-val NDCG@10 unless noted as sampled)

| Run | Composition | Total cands | Result | Notes |
|---|---|---:|---:|---|
| Try 1 | 1 label + 99 hard | 100 | ~0.066 (sampled) | Pure listwise over CG-confusables only. Rising knee of the N-curve, not the plateau. |
| Try 2 | 1 label + 5,436 corpus | 5,437 | n/a | Effective full softmax. ~9 it/s, infeasible. Abandoned. |
| Try 3 | 1 label + 99 hard + 400 random | 500 | 0.070 (sampled) | Hard negs cover CG-confusables; 400 random cover broad-landscape tail. ~6.8% below CG α=0. |
| A-N1 | 1 label + 99 hard + 999 random | 1099 | (not finalized) | Trained 2026-05-16; superseded by A-N2 before a full-val eval was logged. |
| **A-N2** ✓ | 1 label + 999 random (no hard) | 1000 | **0.0741** | RUN 2026-05-17. CG α=0 is 0.0752 → Δ −1.5%. Pure-reranking subset (n=82,364): NDCG@10 0.1342 vs 0.1361. **Phase A exit baseline.** |

Pointwise BCE was also tried during early bring-up; converged to predicting the class prior (NDCG random) and was abandoned. Discipline rule 6 keeps softmax CE as the only loss.

#### Phase A outcome

A-N2 hit near-parity with CG α=0 (within ~2% on every offline metric). The remaining 1.5% gap is the inherent cost of joint-MLP-on-concat vs dot-product on identical features. **The Phase A purpose is achieved**: the architecture works, warm-start works, the training stack is sound. The path to beating CG runs through Phase B.

#### Empirical findings from Phase A

1. **Drop hard negs at large N.** A-N1 → A-N2 (1099 → 1000 cands, hard negs removed) was a net improvement. Once N is large enough to cover the broad-landscape tail, hard negs over-concentrate gradient at the CG-confusable boundary; uniform random sampling provides the easy-vs-label gradient the model actually needs. "1 label + N hard + M random" is small-N era advice — at N ≥ ~1000 the hard-neg term is redundant and slightly hurts.
2. **Warm-start is load-bearing.** A from-scratch ablation produced materially worse NDCG. The deep MLP needs CG's content-tower head start.
3. **N is the lever for sampled-softmax lift, not architecture or hard-neg mining.** N=100 → 500 → 1000 closed the gap from ~12% → ~6.8% → ~1.5%. Loss-family changes (BCE / CCL / WARP) are not the bottleneck.

### Phase B — Cross features

Cross features added in **buckets** of related signals, measured against the previous Phase B baseline (A-N2 for the first). Bundle-level NDCG is the verdict; per-feature attribution within a bucket is only done as a drop-one diagnostic *if the bucket disappoints* (§10 rule 1).

#### Bucket 1 ✓ — Content / Categorical Overlap (2026-05-18)

Three set-membership / categorical features CG mathematically cannot represent: genre_overlap, tag_overlap, dev_affinity. All three sit on the wide bypass; none concatenated into the deep MLP. Implementation: `weighted_overlap` + `dev_affinity` in `ranker/cross_features.py`, parameterized on `(history_indices, history_weights, cand_idx)` so later buckets reuse them. Precompute and train call the same utils → bit-exact identity (verified to 2.98e-8 FP roundoff).

| Metric | CG α=0 | A-N2 | Bucket 1 | Δ vs CG | Δ vs A-N2 |
|---|---:|---:|---:|---:|---:|
| NDCG@1 | 0.0278 | — | 0.0305 | +9.7% | — |
| NDCG@10 | 0.0752 | 0.0741 | **0.0822** | **+9.3%** | **+10.9%** |
| NDCG@20 | 0.0968 | — | 0.1029 | +6.3% | — |
| MRR | 0.0726 | — | **0.0779** | **+7.3%** | — |
| Hit@10 | 0.1430 | — | 0.1540 | +7.7% | — |

Pure-reranking subset (n=82,364): NDCG@10 0.1361 → 0.1487 (+9.3%), MRR 0.1235 → 0.1330 (+7.7%). Lift concentrated at low K — exactly where categorical overlap matters. **Outcome:** content cross features deliver real lift independent of any CG-score signal; the wide-bypass architecture works (three scalars in a 132-dim head are not drowned by the 128-dim deep output).

#### Bucket 2 ✓ (2026-05-19) — Liked + Recent-3 Liked

Bucket 1's three features over two new history slices (cols 4–9): Liked slice (`X_hist_liked` + its playtime weights) and Recent-3 slice (last 3 non-pad positions of `X_hist_liked`, weights re-normalized over those 3, via `last_n_history`).

| Metric | Bucket 1 | Bucket 2 | Δ vs B1 |
|---|---:|---:|---:|
| NDCG@10 | 0.0822 | **0.0828** | +0.7% |
| MRR | 0.0779 | 0.0784 | +0.6% |
| Pure-rerank NDCG@10 | 0.1487 | 0.1499 | +0.8% |

**Outcome:** lift is real but small — consistent positive across every metric (no regressions), so the 6 features earn their seat, but Bucket 1 already extracted most of the categorical-overlap signal. Recalibrates expectations toward the +0.5–1% range.

#### Bucket 3 ✗ DROPPED (2026-05-19) — Disliked History

Three categorical-overlap features on `X_hist_disliked` (cols 10–12). Hypothesis: disliked-as-negative-signal lets the head learn negative weights. **Regressed on every headline metric** (NDCG@10 −0.5%, MRR −0.4%). **Root cause:** Steam's "disliked" partition (`recommend==False OR 0.1<hours<1.0 OR hours<=user_rolling_median/2`) is too noisy — sparse reviews, ambiguous "tried it didn't stick", relative-to-median flags below-average-but-liked games. **Permanent rule:** don't retry the disliked slice unless the partition rule itself improves.

#### Bucket 4 ✗ DROPPED (2026-05-19) — Developer Catalog Signals

Six features on full/liked/recent-3 × {genre, tag} with per-item buffers replaced by **developer-catalog-averaged** versions. Hypothesis: similarity-match ("user likes studios that make games *like* this one's"). **Flat-to-negative across two seeds** (NDCG@10 −0.1%). **Root cause:** the deep tower's `developer_lookup` (12-dim, warm-started, trained jointly) already encodes studio identity end-to-end; hand-crafted per-dev content averages duplicate it. **Permanent rule:** dev-catalog signal class dropped in any shape — the lever for more dev signal is the deep tower, not a wide cross.

#### Bucket 5 ✓ (2026-05-20) — Numeric Matching

Five scalar-arithmetic differences between per-user and per-item numeric stats (cols 10–14) — first non-categorical bucket, first to use Z-score normalization:

- col 10 — `price_match`: `|user_mean_price_bucket − item_price_bucket|`
- col 11 — `era_gap`: `|user_mean_year_numeric − item_year_numeric|`
- col 12 — `playtime_calibration_median` (signed): `user_median_log_playtime − item_median_log_hours`
- col 13 — `popularity_match`: `|user_mean_log_count − item_log_count|`
- col 14 — `sentiment_match`: `|user_mean_sentiment − item_sentiment_ordinal|`

Four non-persistent buffers (`game_year_numeric`, `game_median_log_hours`, `game_log_count`, `game_sentiment`); five per-user scalar dicts in FeatureStore; util `numeric_match_quintuple`. Mean-playtime calibration rejected (Pearson ≥ 0.85 with median → colinear).

| Metric | Bucket 2 | Bucket 5 | Δ vs B2 |
|---|---:|---:|---:|
| NDCG@10 | 0.0828 | **0.0866** | **+4.6%** |
| MRR | 0.0784 | **0.0814** | +3.8% |
| Hit@50 | 0.3999 | **0.4137** | +3.5% |
| Pure-rerank NDCG@10 | 0.1499 | **0.1567** | +4.5% |

**Outcome:** clear ship — uniform +4–5% across every metric, a fundamentally different signal class (scalar-arithmetic cross). Canary: all 9 types hold or improve; JRPG/Racing/Survival/Management materially cleaner; `popularity_match` did NOT cause popularity-leak. **Z-score infra:** persistent `wide_norm_mean/std` buffers populated once by `populate_wide_norm_buffers` over the train parquet; `_normalize_wide` Z-scores the trailing `n_wide_normalized` columns only (cols 0–9 are bounded and pass through raw).

#### Bucket 6 ✓ (2026-05-21) — Niche Feature Crosses — FINAL

8 user × item features = **4 niche/rarity concepts × 2 history slices** (`X_hist_full`, `X_hist_liked`): `tag_overlap_idf` (IDF-reweighted Bucket-1 tag overlap), `niche_tag_match` (weighted-mean tag IDF diff), `max_tag_idf_match` (weighted-max tag IDF diff), `niche_dev_match` (log dev-catalog-size diff). Cols 15–22. Five new non-persistent buffers (`game_tag_binary_idf`, `game_tag_count_idf`, `game_tag_mean_idf`, `game_tag_max_idf`, `game_dev_log_catalog_size`); util `niche_scalar_triple` + reuse of `weighted_overlap`.

| Metric | Bucket 5 | Bucket 6 | Δ vs B5 |
|---|---:|---:|---:|
| NDCG@10 | 0.0866 | 0.0867 | +0.1% |
| MRR | 0.0814 | 0.0813 | −0.1% |
| Pure-rerank NDCG@10 | 0.1567 | 0.1569 | +0.1% |

**Outcome:** offline-flat (every delta in ±0.001 noise band) **but canary-better on niche tastes — kept on canary alone:**
- **Fighting** — clean win. B5 had 3 JRPGs (FF Type-0, FF VIII, Tales of Symphonia) bleeding into a fighting query; B6 removes all three, surfaces METAL SLUG / UNDER NIGHT IN-BIRTH / Naruto Storm 4. Cleanest single-slice improvement since Bucket 1.
- **Civ (4X)** — win. Galactic Civilizations III #3→#1, Master of Orion #11→#5, Victoria II #12→#7; non-4X Heroes VI drops out of top 10.
- **Western RPG / FPS** — slight wins (Pillars #10→#4; Half-Life 2 #9→#5).
- **JRPG/Indie/Racing/Survival/Management** — comparable, minor reorders.

The 4 niche concepts target exactly the failure mode the Fighting slice exhibited (anime-adjacent CG cluster pulling JRPGs into fighting lists). Inference cost negligible (~120k FLOPs/query). **Lesson: when offline-flat coincides with canary-better, ship** — flat offline metrics don't penalize within-top-20 reorders that don't flip the target across the K boundary, but those reorders are exactly what list-shape (canary) sees.

#### Bucket 8 ✗ DROPPED (2026-05-22) — Engagement-Level Cross

2 scalars crossing `X_user_avg_log` with each candidate's intrinsic engagement (`dev_mean_log_playtime` / `genre_mean_log_playtime`), cols 23–24. **Offline-flat AND canary-regressed on the two niche slices that matter most:** Western RPG (Terraria → #1; Civ V, PlanetSide 2, PAYDAY 2, Portal, Warframe, Rust all leak in), Fighting (FF Type-0, Borderlands 2, Warframe, Counter-Strike bleed back — *undoes Bucket 6's win*). **Why:** the cross is a popularity-leak channel — `*_mean_log_playtime` is highest for mass-market titles, so a positive head weight boosts them across all genres. **Permanent rule:** don't re-propose engagement-level crosses; the deep MLP already captures engagement via `X_user_avg_log` + dev/genre embeddings. **Lesson:** "offline-flat" alone isn't the ship signal — the canary is the tiebreaker in both directions, and a flat bucket whose mechanism is "boost high-engagement items" is a popularity leak in disguise.

#### Bucket 9 ✗ DROPPED (2026-05-23) — CG corpus log-rank — closes roadmap

One CG-derived feature: `log1p(corpus rank)` of each candidate under a frozen CG (col 23). **Flat — every NDCG/MRR fractionally down, all in ±0.001 noise.** This is the "circular — only re-imports CG's ranking" result it was gated against: the warm-started ranker + content features (1/5/6) already subsume CG's ordering. **Serving-cost asymmetry:** unlike content buckets, this imposes a hard runtime dependency (a frozen CG forward to score anything) — so a flat *and* serving-expensive feature must clear a real win; flat → unambiguous drop. **Implement-time correction (kept):** training uses random negs (no precomputed CG score), so a CG feature must run a frozen CG live at forward time — *not* a parquet read (the parquet's `cg_*` columns cover only the eval pool). **Lesson:** a feature that re-imports an upstream model's score/rank is only worth its keep if it beats that model on signal the reranker doesn't already hold — a warm-started reranker with strong content features usually already holds it.

---

## 9. Cross-Feature Reference

### Column slots (stable across checkpoints — see `dataset.compute_cross_features`)

```
col 0   : tag_cosine                          (Phase A)
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

**FINAL PROD ranker = 23 features (cols 0–22), n_wide_normalized=13.** Cols are append-only — never reorder, or older checkpoints mis-align at load. Cols 10–22 are Z-scored at forward time (`wide_norm_mean/std` persistent buffers); cols 0–9 are bounded ([−1,1]/[0,1]) and pass through raw. Dropped buckets 3/4/8/9 transiently occupied cols 10+ and were reverted, so those slots are free again. New wide-head weights init at 0.1.

### Wide-feature normalization

Wide features beyond cosines/overlaps are Z-scored before the head using fixed train-set mean/std in **persistent** model buffers (`wide_norm_mean/std` — not BatchNorm; train/eval batch composition differs). Populated by `populate_wide_norm_buffers` in one pass over the train parquet at training start, std clamped to 1.0 near zero variance.

### Removed / not-pursued ideas

- ~~Dislike Similarity~~ / disliked-history columns — dropped, partition too noisy (Bucket 3 ✗).
- ~~Tag Peak Match~~ (`max(user_tag_profile * item_tag_vec)`) — redundant with Tag Cosine + Tag Overlap.
- ~~Recent Game Similarity~~ — replaced by the Recent-3 slice (Bucket 2).
- ~~Genre Diversity / Tag Entropy / History Confidence~~ (deep-concat user-state scalars) — not pursued; if user-state signal is ever wanted, the higher-leverage place is CG's user tower (raises the retrieval ceiling), not the reranker.
- **DCN V2** — replace deep MLP with explicit cross layers; only if cross features stop helping.

### Open infrastructure TODOs

- **✓ Fixed `evaluate_only` memory (2026-05-21).** `load_splits('data', train_mode='skip')` returns `None` for `train_ds` so eval-only never opens the 4.3M-row train parquet (~40 GB → ~1.5 GB peak).
- **Cross-feature column manifest lives in 3 places** — `dataset._CROSS_FEATURE_COLS`, `precompute._CROSS_COLS`, `train._WIDE_NORM_PARQUET_COLS` — all must stay in lockstep when a bucket lands. Consolidate into one canonical manifest if a future bucket reopens this. Cosmetic / drift-prevention only; deferred.

---

## 10. Discipline Rules

1. **One bucket at a time.** Each bucket is one training experiment vs the previous *kept* baseline; bundle-level NDCG is the verdict. Drop-one ablation only as a diagnostic when a bucket disappoints. (Inference-time zeroing of a head weight is NOT a valid substitute — a model trained without X shifts its deep representation.)
2. **Fair-α rule.** Always report which CG checkpoint (α=?) you're comparing against.
3. **Beat CG on content features before re-enabling any CG signal.** Earn improvements from independent signal first.
4. **No `src/` modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes until a model is verified better by eval + canary.**
6. **Softmax CE only.** Never sigmoid in `forward()`. Pointwise BCE was tried and abandoned (converges to class prior).
7. **E2E ceiling always enforced** in both ranker eval and CG baseline.
8. **If `src/model.py` or `src/train.get_config()` changes** (tower hidden dims, embedding sizes, new sub-towers): re-derive §3 *first* — partial drift silently breaks warm-start.
9. **Permanently dropped wide-feature classes:** (a) disliked-history variants (Bucket 3 ✗ — partition too noisy on Steam); (b) dev-catalog signals in any shape (Bucket 4 ✗ — redundant with `developer_lookup`); (c) engagement-level crosses (Bucket 8 ✗ — popularity-leak channel). Don't re-propose. If a future need points at dev-side or engagement signal, the lever is the deep tower, not the wide bypass.
10. **Popularity penalty stays off the ranker.** The α=0.2 experiment (2026-05-23) hurt offline ~27% with no meaningfully better canary. Both CG and ranker ship raw α=0. Don't re-propose a ranker-side popularity penalty.

---

## 11. Serving

The ranker is wired into both canary and Streamlit through one shared path in `ranker/serving.py` (so the demo and the eval use identical feature engineering).

**Two-stage flow:** build user context → raw α=0 CG retrieves top-100 → ranker reranks those 100 → return reranked order. `serving.rerank_candidates` builds all 23 cross features (Buckets 0/1/2/5/6) for the candidate set and calls `ranker.score_pairs`.

**Artifacts** (generated by `python ranker/main.py export [ranker.pth]` — re-exports the CG from the α=0 checkpoint via `src.export`, then adds the ranker):
- `serving/model.pth`, `serving/game_embeddings.pt` — raw α=0 CG (retrieval stage)
- `serving/feature_store.pt` — vocab maps, game metadata, CG buffers, model config, **+ 9 ranker source arrays** (`game_developer_idx`, `game_year_numeric`, `game_median_log_hours`, `game_log_count`, `game_sentiment`, `game_tag_binary_idf`, `game_tag_mean_idf`, `game_tag_max_idf`, `game_dev_log_catalog_size`) consumed by `train._buffers_from_fs` to rebuild the ranker's non-persistent buffers at load
- `serving/ranker.pth` — `WideDeepRanker` state_dict (params + persistent `wide_norm` buffers; non-persistent `game_*` buffers rebuilt on load)
- `serving/ranker_config.json` — reconstruction config (emb dims / `n_cross_features` / `n_wide_normalized` / α + provenance)

The app rebuilds the ranker purely from serving artifacts — no `saved_models/` and no `get_config()` glob (prod has neither). It degrades gracefully to CG-only if `serving/ranker.pth` is absent.

---

## 12. Appendix — Loss family rationale

Why sampled softmax CE and not something else.

**The N-curve** (Klenitskiy & Vasilev, "Turning Dross Into Gold Loss," RecSys'23 — [arxiv 2309.07602](https://arxiv.org/abs/2309.07602)). On ML-1M with vanilla SASRec, swapping BCE+1-neg for sampled softmax CE with N=3000 lifts NDCG@10 by +38%. The curve rises monotonically up to N≈1000 then plateaus. Steam's Phase A reproduced the shape: N=100 ≈ 0.066, N=500 = 0.070, N=1000 = 0.0741. The lever was always N.

**Why not BCE / DeepFM / DCN-V2.** Pointwise CTR rankers assume rating binarization (Steam has no stars) or impression logs (every row in `australian_users_items.json` is a positive). Sampled negs + BCE produces overconfidence pathologies (Petrov & Macdonald, gSASRec); tested directly during bring-up → converged to class prior, NDCG random.

**Why not SimpleX / CCL.** Margin-based hinge breaks parity with CG, and the "large N" insight is the same lesson softmax CE already exploits.

**Why not WARP / LightFM.** Adaptive iterative sampling is serial — does not vectorize on GPU.

**Parking lot.** *In-batch negatives + LogQ correction* ([arxiv 2507.09331](https://arxiv.org/abs/2507.09331)) would give 511 free negs/row at batch=512. Not on the critical path — Phase A is already at the dot-product ceiling on identical features. Worth revisiting only if Phase B saturates.
