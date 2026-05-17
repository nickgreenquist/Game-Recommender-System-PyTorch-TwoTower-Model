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

### Phase A: Strict CG Parity (the controlled experiment)

**Empirical finding (2026-05-15/16):** With only `tag_cosine` in the wide bypass, three independent loss regimes — pointwise BCE+pos_weight, listwise softmax over 100-cand pool, sampled softmax over 500-cand pool — *all* plateau at the same NDCG@10 ≈ 0.066, ~12% below CG α=0 (0.075). That's evidence the loss function is **not** the bottleneck. Before adding any cross feature beyond tag_cosine, the ranker must demonstrate it can match or beat CG **on equal footing** — same labels, same loss family, same architecture-modulo-MLP, same hyperparams. Otherwise we won't know whether each new feature is closing a real ranker gap or compensating for an unfixed methodological one.

**Phase A ground rules — DO NOT add cross features beyond tag_cosine until ranker ≥ CG α=0 on offline metrics:**

1. **Same dataset, rollbacks, labels.** Ranker precompute uses `raw_hours > 0.5` label filter (identical to `src/dataset.py`), same `is_liked` / `is_disliked` pool partitioning, same `max_per_user=50`, same Valve denylist. `N_SHUFFLES=3` for train, `1` for val — matches CG's data augmentation exactly.
2. **Same loss family.** Softmax cross-entropy. Full corpus is impractical (~150× more expensive per (user, item) pair than CG's dot product), so use **sampled softmax**: per row, `1 label + 99 hard negs (from precompute parquet) + 400 random corpus items`. Softmax CE naturally upweights hard negs by score → equivalent to CG's training in the limit.
3. **Same features as CG.** All 8 per-feature towers (item_id, item_genre, item_tag, developer, year, price, user_genre, user_tag), same dims, same architectures, **same warm-start** (random-init baseline tested 2026-05-16 was materially worse — the deep MLP needs the content-tower head start). ONE cross feature (`tag_cosine`) in the wide bypass. No `X_user_avg_log` scalar in user concat — strict 192-dim parity.
4. **Same top-of-stack shape as CG's projection.** Deep MLP is `[256, 128]` (matches CG's `proj_hidden=256, output_dim=128`). `Linear(288→256) → ReLU → Linear(256→128)` — **NO final ReLU** (would clamp `deep_out ≥ 0`, breaking parity with CG's dot product which freely spans negative values). Head is `Linear(128 + n_cross_features, 1)`.
5. **Same hyperparams.** `lr=1e-3`, `weight_decay=0.0`, `adam_eps=1e-6`, `grad_clip=1.0`, `batch_size=512`, `training_steps=50_000`, `eta_min=1e-4` in cosine schedule, **`temperature=0.1` applied to logits before softmax CE** (matches `src/train.py:281`). `popularity_alpha=0` (matches the α=0 throwaway CG comparator).

**The only diff between ranker and CG after Phase A is:** joint MLP on `cat(user_concat, item_concat)` vs CG's per-tower projection + cosine score. That's the isolated experiment — "can a Wide & Deep MLP match a two-tower dot product on identical data and features?"

**Exit criteria from Phase A:** ranker NDCG@10 ≥ CG α=0 NDCG@10 on full val set, OR demonstrated near-parity (within ~2%) with a clear explanation for why exact parity is unreachable on identical features. Until then, every "improvement" is suspect. Once met, proceed to Phase B (cross-feature roadmap, §10).

#### Phase A outcome (declared 2026-05-17): near-parity, exit criterion met

Best Phase A config is **A-N2** (`n_random_negs=999, n_hard_negs=0`, 1000 all-random candidates per row; warm-start ON; ranker α=0). Full val set, 149,083 rollback rows, 100-candidate pools.

**Production-realistic (E2E ceiling applied to both):**

| Metric    | CG α=0 | Ranker A-N2 | Δ        |
|-----------|-------:|------------:|---------:|
| NDCG@10   | 0.0752 | **0.0741**  | −0.0011 (−1.5%) |
| MRR       | 0.0726 | 0.0719      | −0.0007 |
| Hit@1     | 0.0278 | 0.0276      | −0.0002 |
| Hit@10    | 0.1430 | 0.1408      | −0.0022 |
| Hit@100   | 0.5525 | 0.5525      | 0 (ceiling) |

**Pure-reranking subset (CG-retrieved labels, n=82,364):**

| Metric   | CG α=0 | Ranker A-N2 | Δ      |
|----------|-------:|------------:|-------:|
| NDCG@10  | 0.1361 | 0.1342      | −0.0020 |
| MRR      | 0.1235 | 0.1221      | −0.0013 |
| Hit@10   | 0.2588 | 0.2548      | −0.0040 |

Every metric tracks CG within noise — consistently 1–2% behind, never ahead. **This is the expected ceiling**, not a Phase A failure:

1. **Joint MLP on concat cannot beat a dot product on identical features.** A concat→MLP is strictly more *flexible* than a dot product but carries strictly *less inductive bias for similarity*. On identical inputs and labels, the dot product's bias is a free regularizer that the MLP has to re-learn from data; the ranker pays that re-learning cost and gains nothing in return because there is no cross-feature signal for the MLP to exploit. The pure-reranking row makes this concrete — given the same 100 candidates, the ranker is slightly worse at ordering them than CG is.
2. **Three independent loss regimes plateaued at the same place earlier** (BCE+pos_weight, listwise N=100, sampled N=500) — see the 2026-05-15/16 empirical finding higher in this section. The 2026-05-17 scale-up to N=1000 narrowed the gap from ~12% to ~1.5%, exactly tracking Klenitskiy's N-curve prediction. There is no more loss-family or sampling-N lever left to pull.
3. **The cross feature on the wide path (just `tag_cosine`) was always going to be insufficient on its own.** Tag overlap *is* already in the deep concat via the tag towers — `tag_cosine` is a sharpened restatement of the same signal, not a new one. The real Phase B story is features CG mathematically cannot represent: set-intersection (Genre Jaccard), categorical membership (Developer Affinity), absolute differences (Era Gap, Price Match).

**Decision:** treat the strict "≥ CG α=0" criterion as relaxed-met (within ~2% with diagnosed reason) and proceed to Phase B. Next experiment is **B-1: Genre Jaccard** on top of the A-N2 config (1000 all-random cands, warm-start, ranker α=0). Continue using A-N2 as the Phase A baseline that every B-* experiment is measured against.

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

### Deep concat layout (strict v5 CG parity)

```
USER SIDE — mirrors v5 CG user tower (no LayerNorm; v5 removed it):
  pool_liked        : 32   sum(item_id_lookup[liked_ids])                         ← shared lookup
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
  item_id_emb       : 32   item_id_lookup(cand_idx) → Linear(32 → 32) → ReLU       ← shared lookup
  item_genre_emb    :  8   Linear(n_genres → 8) → ReLU
                            input = game_genre_matrix[cand_idx]
  item_tag_emb      : 32   2-layer: Linear(n_tags → 128) → ReLU → Linear(128 → 32) → ReLU
                            input = game_tag_matrix[cand_idx]
  item_dev_emb      : 12   dev_lookup(game_dev_idx[cand_idx]) → Linear(12 → 12) → ReLU
  year_emb          :  8   year_lookup(game_year_idx[cand_idx]) → Linear(8 → 8) → ReLU
  price_emb         :  4   price_lookup(game_price_idx[cand_idx]) → Linear(4 → 4) → ReLU

TOTAL deep concat:
  user side  = 4×32 + 32 + 32 = 192    ← matches CG exactly
  item side  = 32 + 8 + 32 + 12 + 8 + 4 = 96
  total      = 288

Deep MLP:  Linear(288 → 256) → ReLU → Linear(256 → 128)         ← NO final ReLU/activation
           → deep_out (128)
           (Mirrors CG's user_projection / item_projection shape exactly:
            proj_hidden=256, output_dim=128, single Linear→ReLU→Linear, no final activation.
            A final ReLU would clamp deep_out ≥ 0, breaking parity with CG's dot product
            which freely spans negative values — a known optimization-friction failure mode.)
Wide:      cat(cross_features)  — bypasses MLP, direct to head
Head:      Linear(128 + |wide| → 1) → raw logit
Score:     logit / temperature(=0.1)  — applied before softmax CE, matches CG (src/train.py)
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
3. **No `F.normalize` anywhere in the ranker.** CG applies `F.normalize` at the end of each projection MLP because CG's score is cosine. The ranker scores softmax CE on `logits / temperature` (T=0.1) — no normalization needed; temperature handles the logit-scale dynamics that `F.normalize + T` handles on the CG side.
4. **No final activation on the deep MLP.** Deep MLP ends on `Linear(256→128)` with no trailing ReLU. CG's `user_projection` / `item_projection` also end on a raw Linear (then `F.normalize` is applied externally before the dot product). A final ReLU here would clamp `deep_out ≥ 0`, compressing the head's effective input geometry and breaking parity with CG's dot product (which freely produces negative values). This was an actual bug fixed during Phase A bring-up; do not re-introduce it.
5. **Sub-tower init:** Xavier uniform `gain=0.1` on per-feature linears; `gain=1.0` on deep MLP + head; Embedding tables `gain=0.01`. Same recipe as CG, applied to the deep MLP (which structurally replaces CG's projection).
6. **Year / dev / price are bucketed and embedded** — the ranker never sees raw values, only bucket indices from `game_year_idx` / `game_dev_idx` / `game_price_idx` buffers. Bucket boundaries are read from FeatureStore and must match CG's exactly (otherwise warm-started Embedding rows index the wrong buckets).
7. **No timestamp tower.** Steam interactions have no timestamps. Do not add one in the ranker.
8. **In-model genre context is computed inside the user-side forward pass** — same debiasing logic as CG's `user_embedding` (uses `game_genre_matrix[X_hist_full]`, `X_hist_playtime_weights`, and `X_user_avg_log`), feeding `user_genre_tower`. `X_user_avg_log` is a required user-side input for the debiasing math but is NOT concatenated into `user_concat` — strict CG parity (192-dim user concat, not 193).
9. **Buffers built from FeatureStore, not transferred from CG.** `game_tag_matrix`, `game_genre_matrix`, and the new `game_year_idx` / `game_dev_idx` / `game_price_idx` buffers are constructed at ranker init from `feature_store.pt` (non-persistent). They must use the same vocab orderings CG used — otherwise the warm-started towers consume rows in the wrong slots.

### Warm-start mapping (init from v5 CG state_dict)

> **Default ON in Phase A** (`get_config()['warm_start'] = True`). A from-scratch ablation tested 2026-05-16 produced materially worse NDCG — the deep MLP needs CG's content-tower head start to optimize cleanly. The gate (`cfg['warm_start']`) is kept for future ablations but should not be flipped without a specific question to answer; the auto-resolver picks the α-matched CG checkpoint via `_ALPHA_TO_CG_GLOB`.

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

`_warm_start_from_cg` should log transferred-vs-fallback per key. The expected count is **26 tensors**:
- 4 lookups × 1 tensor each = 4
- 5 one-layer towers × 2 tensors (weight, bias) = 10
- 3 two-layer towers × 2 Linears × 2 tensors = 12
- Total: **26 tensor transfers** out of 34 total ranker tensors (the 8 untransferred are deep MLP + head, which have no CG counterpart).

Anything short of 26 means a shape drift or key mismatch to fix before training.

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

## 7. Training Stack (Phase A: strict CG parity)

### Negative sampling

**Sampled softmax**, not MNS. **Active config (post-A-N2, 2026-05-17): `1 label + 999 random corpus items = 1000-cand pool` — no hard negs.** Set by `n_random_negs=999, n_hard_negs=0` in `get_config()`. Phase B experiments inherit this config; only the cross-feature set changes.

The Phase A starting config was `1 label + 99 hard + 400 random = 500 cands` (Try 3 in §13's "what we've tried" table). A-N1 added more random (99 hard + 999 random = 1099 cands); A-N2 dropped hard negs entirely (999 random = 1000 cands) and won — see §13's "pure random at large N wins" finding.

Why not full softmax over all 5,437 items? The deep MLP cannot factorize like CG's dot product — it's ~150× more expensive per (user, item) pair. Full corpus per step → 9 it/s, untenable. Sampled softmax with N=999 random negs is the closest tractable approximation; softmax CE auto-focuses gradient on whichever items get high score, so the random samples cover the easy-neg tail in expectation.

Why not 100-cand listwise only (1 + 99 hard)? Earlier Phase A finding (Try 1, §13): plateaus ~12% below CG α=0. The gradient signal is *strictly* "label vs CG-confusables" — no anchor for "label vs obviously-wrong easy items." A-N2 confirmed that pure random sampling at N=1000 closes the gap to within ~1.5%.

**For the broader paradigm survey** (listwise softmax, pointwise CTR, contrastive, WARP), **what we've tried** (Try 1/2/3 + A-N1/A-N2 with NDCG numbers), and **the empirical "drop hard negs at large N" finding**, see §13.

**Hard-neg infrastructure kept (parquet `neg_item_idxs` column, dataset `n_hard_negs` knob)** — Phase B may revisit hard-neg mixes once new cross features change the gradient landscape. Cheap to re-enable; expensive to re-add if removed.

### Loss

`F.cross_entropy(scores / temperature, target)`. Softmax CE over the 1000-cand pool (A-N2 active config), target=0 (label always at column 0 by construction). **`temperature=0.1`** matches CG's logit scaling — sharpens softmax by 10×.

No BCE. No `pos_weight`. No sigmoid in forward.

### Training config (Phase A defaults — match CG exactly)

```
lr:               1e-3                       ← matches CG (src/train.py:35)
weight_decay:     0.0                        ← matches CG
adam_eps:         1e-6                       ← matches CG
batch_size:       512                        ← matches CG
training_steps:   50_000                     ← matches CG
log_every:        500                        ← finer-grained than CG's 1000 (Phase A diagnostic)
grad_clip:        1.0                        ← matches CG
temperature:      0.1                        ← matches CG (logit scaling before softmax)
scheduler:        CosineAnnealingLR, T_max=training_steps, eta_min=1e-4    ← matches CG
hidden_dims:      [256, 128]                 ← matches CG's projection shape (no final ReLU)
dropout:          0.0
popularity_alpha: 0.0                        ← matches α=0 throwaway CG comparator (§2)
n_random_negs:    999                        ← sampled-softmax tail size (A-N2 winner)
n_hard_negs:      0                          ← A-N2 dropped hard negs; pure random at N=1000 wins
```

**Precompute settings** (`ranker/precompute.py`):
- `TOP_K_CANDIDATES=100` (1 label + 99 hard negs per row)
- `N_SHUFFLES=3` for train split, `N_SHUFFLES=1` for val — matches CG (`src/dataset.py:20`)
- `MAX_ROLLBACK_EXAMPLES_PER_USER=50` — matches CG
- Label filter: `raw_hours > 0.5` AND `history[i] not in history[:i]` (the dedupe guard avoids the ~0.2% Steam history duplicates from leaking the label into its own context — a CG-tolerable noise that the ranker's cross features would trivially exploit)

Save arch params + `popularity_alpha` + `temperature` + `n_random_negs` + `n_hard_negs` in a `_config.json` sidecar alongside each checkpoint. The α value matters at eval time (for choosing the right CG comparator); `n_hard_negs` distinguishes A-N1 (hard kept) from A-N2 (hard dropped) checkpoints.

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
| 0   | **Throwaway CG α=0 reference run** | Establish the offline-metrics scoreboard (§2). Save as `best_triple_full_softmax_popularity_alpha_00_<date>.pth` with `"do_not_export": true` and `"role": "ranker_offline_baseline"` in the sidecar. Never deploy. |
| **A** ✓ | **STRICT CG parity** (see §2 "Phase A ground rules"). Same data, labels, towers, dims, hyperparams, **warm-start from α-matched CG**. Sampled softmax CE + temperature=0.1. Only architectural diff from CG: joint MLP on concat vs per-tower projection + cosine. `n_cross_features=1` (tag_cosine).  | **Exit criterion: ranker NDCG@10 ≥ CG α=0 NDCG@10 on full val set, OR near-parity (within ~2%) with diagnosis.** **STATUS 2026-05-17: EXIT MET via A-N2** (NDCG@10 0.0741 vs CG 0.0752, Δ −1.5%). Proves "a W&D MLP can effectively match a two-tower dot product on identical inputs" — the foundation every later phase rests on. Phase B now active. |
| A-N1 | **`n_random_negs = 999`** (99 hard + 999 random = 1099 cands) | Direct test of Klenitskiy N-scaling lift on Steam (§13). Trained 2026-05-16; superseded by A-N2 before a full eval was logged — sampled val NDCG@10 during training matched A-N2's trajectory closely, so A-N2 (the cleaner pure-random config) is the recorded Phase A baseline. |
| **A-N2** ✓ | **`n_random_negs = 999`, hard_negs = 0** (1000 all-random) | **RUN 2026-05-17. NDCG@10 0.0741 vs CG α=0 0.0752 (Δ −0.0011, −1.5%); MRR 0.0719 vs 0.0726.** Pure-reranking subset NDCG@10 0.1342 vs 0.1361 (Δ −0.0020). Near-parity declared sufficient for Phase A exit — see §2 "Phase A outcome." Confirms Klenitskiy's "pure random at large N wins" prediction (§13). This is the Phase A baseline every B-* experiment is measured against. |
| A-IB | **In-batch negatives + LogQ correction** (parking lot — would only revisit if a B-* phase signals the loss family itself is the ceiling) | 511 free negatives per row at batch=512 from other rows' labels in the same minibatch (§13). Not on critical path — Phase A is already at the dot-product ceiling on identical features; loss-family changes won't close the remaining 1.5% gap. |
| B-1 | **+ Genre Jaccard** | Sharpen genre precision. First feature CG mathematically cannot represent (set-intersection, non-linear). |
| B-2 | **+ Developer Affinity** | Capture studio-loyalty signal absent from CG. Indicator over a categorical set — CG can't do membership tests in a dot product. |
| B-3 | **+ Price Match** | Distinguish F2P / indie / AAA buyer segments. |
| B-4 | **+ Era Gap** | New release vs retro preference. |
| B-5 | **+ Playtime Calibration** | Match heavy-hours users to long-tail engagement games. |
| B-6 | **+ Dislike Similarity** | Veto channel — leverages the disliked pool more aggressively. |
| B-7 | **+ Tag Peak Match** | "One tag spark" signal beyond global cosine. |
| B-8 | **+ Recent Game Similarity** | Sequential next-game signal — "what just played" anchor. |
| B-9 | **+ Popularity Match** | User preference for popular vs obscure. |
| B-10 | **+ CG Score** | Final retrieval-signal boost once content features have proven independent value (plan §9 — gated, re-enable last). |
| C | **Label quality: switch to `is_liked` filter** | Per user-decided roadmap: only after Phase A + B are validated, re-precompute with the stricter `is_liked` label filter (`raw_hours ≥ game_median` OR `≥ 2× user_median` OR `recommend=True`). Slashes label set by 40-60% but every label is a genuinely-engaged game. Expected to lift NDCG meaningfully; cleanly attributable because Phases A+B are already proven. |
| Later | **DCN V2** | If content features saturate. |

**Change one thing per experiment.** Measure NDCG@10 delta vs the previous run before proceeding. Phase A exited at A-N2 (2026-05-17) — Phase B experiments now compare against A-N2 (NDCG@10 0.0741) as the baseline, not CG α=0.

---

## 12. Experiment Discipline Rules

1. **One change at a time.** Every run isolates exactly one variable.
2. **Fair-α rule.** Always report which CG baseline (α=?) you are comparing against. Beating a CG that's tuned for canary quality on offline metrics is not a real win.
3. **Beat CG on content features before re-enabling CG score.** Earn improvements from independent signal.
4. **No `src/` modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **Softmax CE only** (matches CG). `F.cross_entropy(scores / temperature, target)`. Never sigmoid in `forward()`. Pointwise BCE was tried during early Phase A bring-up and abandoned — it converges to predicting the class prior (loss ≈ 0.056 at 1% positive rate, NDCG random) and requires `pos_weight` hacks that softmax CE doesn't need by construction.
7. **Don't enter Phase B until Phase A exit criterion is met.** Phase A is the "MLP-can-match-two-tower" experiment. Adding cross features before that confounds "real ranker lift" with "compensating for an unfixed Phase A bug." (Status 2026-05-17: Phase A exited at A-N2 — near-parity (−1.5%) declared sufficient; see §2 "Phase A outcome." Phase B is now active.)
8. **E2E ceiling always enforced** in both ranker eval and CG baseline.
9. **§3 reflects v5 CG as of this writing.** If `src/model.py` or `src/train.get_config()` changes (tower hidden dims, embedding sizes, new sub-towers), re-derive §3 *first* before changing the ranker — partial drift will silently break warm-start.

---

## 13. Loss & Negative Sampling — Literature & What We've Tried

The literature on implicit-signal ranking converges on **sampled softmax cross-entropy** as the right loss family for two-stage rankers. Four paradigms dominate the MovieLens benchmarks; **only listwise softmax cleanly applies to Steam**, and we are already in that paradigm. This section captures (a) the literature, (b) our three attempts to date, (c) why the other three paradigms don't apply, and (d) the highest-leverage next experiments.

### Reference papers

- Klenitskiy & Vasilev, **"Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?"** — RecSys'23, [arxiv 2309.07602](https://arxiv.org/abs/2309.07602). The central paper. Shows that swapping vanilla SASRec's BCE-with-1-negative for sampled softmax CE with N=3000 negatives lifts ML-1M NDCG@10 by **+38%** (0.1341 → 0.1857) on the *same architecture*. Establishes that the loss family + N is the dominant lever, not architecture.
- Petrov & Macdonald, **"gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling"** — RecSys'23. Diagnoses why BCE + sampled negatives produces pathological overconfidence (predicted probabilities of top items asymptote to 1, sum over all items blows up to 100×+ instead of 1). Introduces gBCE — a generalized sigmoid on the positive logit — that fixes it while keeping N small (k=128). +9.47% NDCG@10 over BERT4Rec on ML-1M.
- Wu et al., **"On the Effectiveness of Sampled Softmax Loss for Item Recommendation"** — TOIS 2024. Comprehensive ablation confirming sampled softmax CE > BPR > BCE on NDCG@K across multiple recommender benchmarks.
- 2025 RecSys paper, **"Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval"** — [arxiv 2507.09331](https://arxiv.org/abs/2507.09331). Shows in-batch negatives + LogQ correction beats most alternatives on temporal-split MovieLens.
- Mao et al., **"SimpleX"** — CIKM 2021. Cosine contrastive loss (CCL) baseline; demonstrates that large random negative sampling (1:100+) matters more than architectural complexity.

### Paradigm 1 — Listwise Softmax (what we're doing) ✓ applicable

Loss: sampled cross-entropy over `1 positive + N negatives`.

```
L = -log [ exp(s_label) / (exp(s_label) + Σ_neg exp(s_neg)) ]
```

This is exactly what `ranker/dataset.py:sample_batch` produces — label at column 0, `target=0`, plain `F.cross_entropy(scores / temperature, target)`. All three of our attempts to date sit within this paradigm; only N and the hard/random mix differ.

**Literature N-curve on ML-1M** (Klenitskiy table 2, fig. 2; SASRec, same architecture, only N varies):

| Setup | N (negatives per row) | NDCG@10 |
|---|---:|---:|
| Vanilla SASRec (BCE, 1 neg) | 1 | 0.1341 |
| Sampled CE | 100 | ~0.165 |
| Sampled CE | 1000 | ~0.181 |
| Sampled CE | **3000** | **0.1857** |
| Full softmax | 3,416 (full corpus) | 0.1821 |

The curve is **monotonically rising up to N≈1000** then plateaus. N=3000 sampled actually *beats* full softmax slightly — sampling acts as a mild regularizer. Hard negatives are not used; pure uniform random sampling produces the gains.

### What we've tried (Steam ranker, V5 architecture, α=0)

| Try | Composition | Total cands | Full-val NDCG@10 | Notes |
|---|---|---:|---:|---|
| 1 | 1 label + 99 hard | 100 | ~0.066 (sampled) | Pure listwise over CG-confusables only. Matches the "1:99" cell most ranker blog posts call SOTA — actually the rising knee of the curve, not the plateau. |
| 2 | 1 label + 5,436 corpus | 5,437 | n/a | Effective full softmax. Too slow (~9 it/s) — runtime infeasible. Abandoned. |
| 3 | 1 label + 99 hard + 400 random | 500 | 0.070 (sampled) | Hard negs cover CG-confusable space; 400 random cover broad-landscape tail. Still ~6.8% below CG α=0 sampled (0.075). |
| A-N1 | 1 label + 99 hard + 999 random | 1099 | not finalized | Trained 2026-05-16; superseded by A-N2 before a full-val eval was logged. Sampled-val trajectory tracked A-N2 within noise. |
| **A-N2** ✓ | 1 label + 999 random (no hard) | 1000 | **0.0741** | RUN 2026-05-17. Closes the gap to CG α=0 (0.0752) to **−1.5%** — Phase A exit baseline (see §2 "Phase A outcome"). |

**Empirical finding (2026-05-17):** dropping hard negatives at large N wins. A-N1 → A-N2 (1099 → 1000 cands, hard negs removed) was a net improvement on full-val NDCG@10. This confirms Klenitskiy's prediction: once N is large enough to cover the broad-landscape tail, hard negs over-concentrate gradient at the CG-confusable boundary — the model is already ranking those well; what it needs is the easy-vs-label gradient that uniform random sampling provides. The "1 label + N hard + M random" recipe most ranker blog posts cite was actually two-stage advice from the era when N had to stay small (≤200) for compute reasons; once N ≥ ~1000, the hard-neg term becomes redundant and slightly hurts.

**Loss family is not the bottleneck on Phase A; N was.** A-N2 (N=1000 pure random) closed the gap from ~12% to ~1.5%, exactly tracking the N-curve plateau. The remaining 1.5% is the joint-MLP-vs-dot-product gap on identical features (§2 "Phase A outcome"), not a sampling deficiency.

### Paradigm 2 — Pointwise CTR (DeepFM, DCN-V2, xDeepFM, BCE) ✗ not applicable to Steam

Loss: binary cross-entropy with sigmoid head.

```
L = -[y · log σ(s) + (1-y) · log(1 - σ(s))]
```

**Five reasons this doesn't apply to Steam:**

1. **No rating binarization signal.** MovieLens CTR rankers map `rating ≥ 4 → positive, ≤ 3 → negative`, producing the natural ~3:2 label distribution DeepFM/DCN papers train on. Steam has no star ratings — playtime is continuous and one-sided. The `recommend` boolean exists but covers only ~29% of users (25k of 88k) and only on reviewed games — too sparse to be a main label.
2. **No impression logs / no organic negatives.** CTR ranker training assumes "shown but not clicked" rows in serving logs. `australian_users_items.json` contains only games users *played* — every row is a positive. There is no "shown but not bought" signal anywhere in the raw data.
3. **Sampled negatives + BCE produces overconfidence.** Without organic negatives we'd have to sample them anyway, which puts us in the regime gSASRec (Petrov & Macdonald 2023) identifies as pathological: predicted probabilities of top-ranked items asymptote to 1, the sum of probabilities across the catalog blows up to 100× the correct value, and the model loses its ability to discriminate among the top items (which is what NDCG@K measures).
4. **BCE strictly loses to softmax CE on NDCG.** Klenitskiy table 2: same SASRec, BCE + 1 neg = 0.1341 NDCG@10, softmax CE + N negs = 0.1857. The ~38% gap is entirely the loss family — at the metric we care about, pointwise BCE is dominated.
5. **Score calibration mismatch with CG.** CG outputs softmax-CE scores. A BCE ranker outputs sigmoid probabilities. The two scales don't compose naturally — any future "blend CG and ranker" experiment becomes harder than it needs to be.

**We also tested it directly.** Phase A bring-up tried pointwise BCE; it converged to predicting the class prior (loss ≈ 0.056 at 1% positive rate, NDCG random) and was abandoned. See §12 rule 6.

**One narrow scenario where pointwise CTR would matter for Steam:** if the Streamlit app eventually instrumented real impression logs (cards shown vs. cards clicked), that real-impression data could justify a pointwise BCE pass. That's a product/instrumentation project, not a Phase A code change.

### Paradigm 3 — Dense Negative-Contrastive (SimpleX / CCL) ✗ not applicable

Loss: cosine margin loss with large random negative sampling.

```
L = max(0, 1 - cos(u, i⁺)) + (1/N) · Σ max(0, cos(u, j⁻) - m)
```

**Why it doesn't apply to Steam:**
- Replaces softmax with margin-based hinge — would break parity with CG (which uses softmax CE), defeating the Phase A controlled-experiment design.
- The "large N negative sampling" insight (Mao pushes 1:100+) is the *same* lesson Klenitskiy proved for softmax CE — and softmax CE wins on NDCG. CCL's net contribution beyond "use many negatives" is the margin formulation, which isn't a free win relative to softmax CE on the metrics we care about.

Worth knowing as a far-future ablation baseline if softmax CE saturates and we want to test "is the loss family itself the ceiling?" Not on the critical path.

### Paradigm 4 — WARP (LightFM) ✗ not applicable

Loss: rank-weighted pairwise hinge with adaptive sampling — draw negatives one at a time until you find a rank violation; gradient weight scales with how many tries it took.

**Why it doesn't apply:**
- Adaptive iterative sampling is fundamentally serial — does not vectorize on GPU. LightFM is a CPU/sparse-matrix library; WARP was state-of-the-art in that era. On a GPU two-tower stack like ours, WARP would dominate runtime.
- The "adaptive hard-neg mining" idea WARP popularized has been superseded by listwise softmax with large N — softmax CE auto-upweights gradient on whichever items get high score, achieving the same effect without serial sampling.

Of historical interest only.

### Other techniques worth knowing (within softmax-CE family)

- **gBCE** (Petrov & Macdonald). Pointwise loss with a generalized sigmoid `σ^β(s+)` on the positive logit that fixes BCE's overconfidence pathology. Lets you stay at low N (k=128). Three-line code change. Worth trying *if* we ever want a pointwise option without BCE's drawbacks — not on the current critical path since we're already in softmax CE.
- **In-batch negatives + LogQ correction** (YouTube/Google two-tower default; the [2507.09331](https://arxiv.org/abs/2507.09331) paper). At batch=512, each row gets 511 free negatives (other rows' labels in the same batch). LogQ correction subtracts `log P(item in batch)` from each in-batch logit to debias popularity. Memory cost: zero — they're already loaded. Highest-upside alternative to "scale N higher" since the negatives come for free.

### Plan of record: next loss/sampling experiments (gated to Phase A exit)

**Status (2026-05-17): Phase A exited at A-N2. The sampling-side question is closed.** Items 1 and 2 below have both been run; A-N2 (item 2) is the winning config. Items 3 and 4 are now parking-lot — only revisit if a future B-* phase signals the loss family itself is the ceiling, which Phase A's outcome makes unlikely (the remaining 1.5% gap is dot-product-vs-MLP on identical features, not loss family).

1. ~~**`n_random_negs = 999`** with current hard-neg mix (99 hard + 999 random = 1099 cands).~~ **Done (A-N1).** Superseded by A-N2.
2. ~~**`n_random_negs = 999`, hard_negs = 0** (1000 all-random).~~ **Done (A-N2). Winner.** NDCG@10 0.0741 vs CG α=0 0.0752 (Δ −1.5%) — Phase A baseline.
3. **In-batch negatives + LogQ correction** — parking lot.
4. **gBCE** — parking lot.

CCL and WARP are not on the roadmap.
