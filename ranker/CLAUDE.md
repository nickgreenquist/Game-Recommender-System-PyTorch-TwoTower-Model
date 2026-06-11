# Ranker Subsystem (`ranker/`)

Guidance for Claude Code in `ranker/` — the **Wide & Deep ranker**, stage 2 of the two-stage recommender. The root `CLAUDE.md` documents the candidate-generation (CG) two-tower model; this documents the reranker on top of it. `src/` is **read-only** from here.

## Status (2026-05-25)

**SHIPPED / PROD / FINAL — item-text ranker:** `ranker_wd_alpha_0_20260525_113932.pth`, warm-started + precomputed from the served V6a item-text CG (`…text_popularity_alpha_0_…100022.pth`), with deep-tower item-text parity (`item_text_tower` 768→128→32, item concat 96→128, deep_in 288→320, warm-start map 26→30) + a `text_cosine` cross feature (col 23; n_cross 23→24, n_wide_normalized 13→14). Replaces the prior Bucket-6 no-text ranker.

- **Results (vs the text CG, fair-α):** prod-realistic NDCG@10 **0.0873** / MRR **0.0820** / Hit@10 0.1629; pure-rerank NDCG@10 **0.1577** / MRR 0.1401. Lift over the text CG **+0.0120** NDCG@10 (beats Bucket 6's +0.0115 over the no-text CG).
- **Canary:** comparable-or-better on all 9 archetypes — Civ clearly better, FPS/Indie/Racing slightly better, rest even. Point & Click was the cleanest list (where distinctive descriptions help most).
- **Prior FINAL (Bucket 6, no-text):** `ranker_wd_alpha_0_20260520_204654.pth` (NDCG@10 0.0867) — superseded.

**Popularity penalty stays OFF the ranker (α=0).** A `popularity_alpha=0.2` ranker was tested 2026-05-23 and rejected: hurt offline ~27% (lift over CG dropped +0.0115→+0.0084) with no meaningfully better canary. Both CG and ranker ship raw α=0 — no popularity penalty anywhere. The Menon Path 2 plumbing (`popularity_alpha`, `warm_start_alpha`) stays in `train.py` set to 0.

**Bucket roadmap complete.** Headline metric is full-val NDCG@10, each bucket vs the previous *kept* baseline. This table is the canonical per-bucket record:

| Bucket | Adds | NDCG@10 | Δ vs prev kept | Verdict |
|---|---|---:|---:|---|
| **A-N2** (Phase A exit) | tag_cosine only | 0.0741 | — (CG α=0: 0.0752) | baseline — MLP ≈ dot product |
| **1** ✓ | genre/tag/dev overlap (full) | **0.0822** | **+10.9%** | ship — biggest single win |
| **2** ✓ | overlap × {liked, recent-3} | **0.0828** | +0.7% | ship, barely — B1 took most of it |
| **3** ✗ | overlap × disliked | 0.0824 | −0.5% | **drop** — disliked partition too noisy |
| **4** ✗ | dev-catalog genre/tag overlap | 0.0827 | −0.1% | **drop** — redundant with `developer_lookup` |
| **5** ✓ | 5 numeric-match scalars | **0.0866** | **+4.6%** | ship — uniform +4–5%, new signal class |
| **6** ✓ | 8 niche/IDF crosses | 0.0867 | +0.1% (flat) | **ship on canary** — offline-flat, canary-better |
| **8** ✗ | 2 engagement crosses | 0.0870 | +0.03% (flat) | **drop** — flat offline AND canary-regressed (popularity leak) |
| **9** ✗ | CG corpus log-rank | 0.0864 | −0.3% (flat) | **drop** — circular + frozen-CG serving cost |

**Permanently dropped signal classes (don't re-propose in any shape — rule 9):** disliked-history variants (B3 — partition too noisy on Steam); dev-catalog aggregates (B4 — deep tower's `developer_lookup` already encodes it); engagement-level crosses (B8 — popularity-leak channel). Bucket 7 (Item-Intrinsic Priors) dropped in planning; 2 of its 3 features became Bucket 6 crosses (`niche_dev_match`, `max_tag_idf_match`). The FINAL ranker has **no frozen-CG runtime dependency** (B9 would have added one).

> **Terminology:** "pool" = an embedding aggregation (deep tower's `pool_liked/disliked/full/playtime`). "Cross feature" = weighted overlap / categorical affinity computed *directly over history arrays* (`weighted_overlap`/`dev_affinity` on `history_indices`/`history_weights`) — no embedding aggregation.

---

## 1. Pipeline

Two-stage retrieve-and-rank: **CG** (V6a softmax two-tower, 4-pool user tower, F.normalize both towers, 128-dim) retrieves top-100 per rollback example → **Ranker** (Wide & Deep MLP) reranks the 100 with richer features.

**CG checkpoints:**
- **Raw CG α=0** — as of 2026-05-25 the V6a item-text CG (`…text_popularity_alpha_0_…100022.pth`); `_ALPHA_TO_CG_GLOB[0.0]` points here so warm-start / precompute / canary / export all use it. The **deployed retrieval stage** AND the honest offline yardstick (ranker runs α=0, measured against this).
- **CG α=0.4** — the pre-ranker standalone-serving compromise; no longer deployed in the two-stage world. Reference only (root `CLAUDE.md` has the α-tradeoff + baseline tables).

---

## 2. Core Principles

- **CG-parity baseline first.** The ranker contains every CG input/feature, through the same per-feature towers, same dims, same warm-started weights. Cross features come *on top* — the differentiator, not a replacement.
- **Wide & Deep.** All per-feature embeddings concat into ONE vector → deep MLP. Cross features bypass the MLP straight to the head, each getting a direct learned weight — otherwise scalar crosses get washed out competing against ~320 dims in the first hidden layer.
- **No CG coupling at runtime.** The ranker owns its own copies of every param (`item_id_lookup`, `developer_lookup`, etc.); warm-start is a one-time copy at construction, then tensors live in the ranker `state_dict` and train freely. Runtime ties to CG: (1) candidate indices from retrieval, (2) precomputed parquet features, (3) static feature data both read from disk.
- **Fair-α rule.** Always report which CG checkpoint (α=0 vs α=0.4) the ranker is compared against. Ranker α=0 vs CG α=0 is the only meaningful offline comparison; vs CG α=0.4 is not a real win (delta is mostly α=0.4's deliberate retrieval handicap).

---

## 3. Architecture

All dims match `src/model.py` (V6a) and `src/train.get_config()`: `item_id=32`, `tag=32`, `user_genre=32`, `user_tag=32`, `item_genre=8`, `developer=12`, `item_year=8`, `price=4`, `text=32`. Tower hidden dims (hardcoded in `src/model.py`): `tag_hidden=128`, `tag_ctx_hidden=256`, `genre_hidden=128`.

### Per-game static buffers (registered on the ranker)

Built from `FeatureStore` at construction, all **non-persistent** — rebuilt on load from `feature_store.pt` via `ranker.train._buffers_from_fs`. Indexed by `cand_idx` to get per-game metadata at score time.

```python
game_year_idx     (n_games+1,)        int64
game_dev_idx      (n_games+1,)        int64
game_price_idx    (n_games+1,)        int64
game_tag_matrix   (n_games+1, n_tags) float32  TF-IDF
game_genre_matrix (n_games+1, n_genres) float32  one-hot
game_text_matrix    (n_games+1, 768)  float32  RAW desc emb (V6a)
game_text_matrix_l2 (n_games+1, 768)  float32  L2-normalized rows
```

`game_text_matrix` (RAW) feeds the user-side `text_cosine` pool (sum-then-normalize); `game_text_matrix_l2` feeds BOTH the deep `item_text_tower` (== CG's `F.normalize(raw)`) and the `text_cosine` candidate side — same two-buffer split as the tag pair. Names/dtypes/vocab orderings must mirror CG exactly so warm-started towers consume the right rows. Persistent `wide_norm_mean/std` ARE saved in the checkpoint.

### Deep concat layout (strict CG parity)

```
USER SIDE (mirrors CG user tower, no LayerNorm):
  pool_liked / pool_disliked / pool_full   : 32 each  sum(item_id_lookup[ids])  ← shared lookup
  pool_playtime                            : 32       playtime-weighted sum (weights pre-normalized in dataset)
  user_genre_emb : 32  Linear(2*n_genres→128)→ReLU→Linear(128→32)→ReLU
                       in-model genre debiasing from game_genre_matrix[X_hist_full],
                       X_hist_playtime_weights, X_user_avg_log (X_user_avg_log used internally
                       for debiasing but NOT concatenated — strict CG parity)
  user_tag_emb   : 32  Linear(n_tags→256)→ReLU→Linear(256→32)→ReLU; in = sum game_tag_matrix[X_hist_full]

ITEM SIDE (mirrors CG item tower):
  item_id_emb    : 32  item_id_lookup(cand_idx)→Linear(32→32)→ReLU            ← shared lookup
  item_genre_emb :  8  Linear(n_genres→8)→ReLU                                  in = game_genre_matrix[cand_idx]
  item_tag_emb   : 32  Linear(n_tags→128)→ReLU→Linear(128→32)→ReLU             in = game_tag_matrix[cand_idx]
  item_dev_emb   : 12  dev_lookup(game_dev_idx[cand_idx])→Linear(12→12)→ReLU
  year_emb       :  8  year_lookup(...)→Linear(8→8)→ReLU
  price_emb      :  4  price_lookup(...)→Linear(4→4)→ReLU
  item_text_emb  : 32  item_text_tower(game_text_matrix_l2[cand_idx]): Linear(768→128)→ReLU→Linear(128→32)→ReLU  (V6a)

  user side = 4×32 + 32 + 32 = 192     item side = 32+8+32+12+8+4+32 = 128 (V6a +text)     total = 320

Deep MLP: Linear(320→256)→ReLU→Linear(256→128)  ← NO final activation → deep_out (128)
Wide:     cat(cross_features) bypasses MLP, direct to head
Head:     Linear(128 + |wide| → 1) → raw logit;  Score = logit / temperature(0.1)
```

Every lookup-fronted feature (id/dev/year/price) is `Embedding → Linear+ReLU` as two separate `nn.Module`s (not a fused Embedding) — that's how CG names them, which the warm-start map relies on.

### Implementation invariants

1. Ranker owns its `item_id_lookup` (`nn.Embedding(n_games+1, 32, padding_idx=n_games)`), shared across all 4 user pools + item-side tower.
2. **No LayerNorm on pools** (CG removed it).
3. **No `F.normalize` anywhere** — CG uses cosine; ranker uses softmax CE on `logits/temperature` (T=0.1).
4. **No final activation on the deep MLP.** A trailing ReLU clamps `deep_out ≥ 0`, breaking parity with CG's dot product (spans negatives). Was an actual Phase A bug.
5. **Init:** Xavier uniform `gain=0.1` on per-feature linears, `gain=1.0` on deep MLP + head, embeddings `gain=0.01`. Same recipe as CG.
6. Year/dev/price bucket boundaries from FeatureStore must match CG's exactly.
7. No timestamp tower (Steam has none).
8. In-model genre context computed in the user-side forward — same debiasing as CG.
9. Buffers built from FeatureStore, not transferred from CG — vocab orderings must match.

### Warm-start mapping (init from CG state_dict)

**Default ON** (`get_config()['warm_start']=True`); a from-scratch ablation was materially worse. `train._CG_TO_RANKER_KEY_MAP` does **30 tensor transfers** (4 lookups + 5 one-layer towers + 4 two-layer towers — the 4th two-layer is V6a `item_text_tower`); fewer means shape drift / key mismatch. **Not transferred:** deep MLP, head, cross-feature weights (random init); `*_projection.*` (ranker has none); all buffers (rebuilt from FeatureStore). `warm_start_alpha` decouples the warm-start CG source from the penalty (now both α=0).

---

## 4. E2E Evaluation Rule

**Ranker_Hit@K ≤ CG_Recall@K_cand in production** — if CG didn't retrieve the label, the ranker never sees it. In offline eval, if `cg_label_rank >= n_cand`, rank → `n_cand + 1` (score 0). Both CG baseline and ranker use this ceiling (apples-to-apples). `cg_label_rank == n_cand` is treated as "not found"; eval prints `Recall@n_cand` as the production ceiling.

---

## 5. Repo Structure

```
ranker/
├── precompute.py     ← CG scoring → ranker_candidates_{train,val}.parquet
├── dataset.py        ← RankerDataset, sample_batch (sampled softmax)
├── model.py          ← WideDeepRanker
├── cross_features.py ← wide-bypass cross-feature utils (shared: precompute/train/canary/serving)
├── train.py          ← sampled-softmax CE loop; build_ranker, _buffers_from_fs, warm-start
├── evaluate.py       ← NDCG@K, MRR, Hit@K, CG baseline, E2E ceiling
├── serving.py        ← shared rerank pipeline (canary AND streamlit)
├── canary.py         ← side-by-side CG vs ranker top-N (delegates to serving.py)
├── export.py         ← serving export (re-exports α=0 CG via src.export, then adds ranker)
└── main.py           ← precompute / train / evaluate / canary / export
data/ranker_candidates_{train,val}.parquet
```

```bash
python ranker/main.py precompute [cg.pth]   # Stage 0: candidates (default raw α=0 CG)
python ranker/main.py train                 # train Wide & Deep ranker
python ranker/main.py evaluate [ranker.pth] # eval-only (auto-finds latest)
python ranker/main.py canary   [ranker.pth] # CG vs ranker top-N for all canaries
python ranker/main.py export   [ranker.pth] # serving artifacts (α=0 CG + given/latest ranker)
```

---

## 6. Precompute

`TOP_K_CANDIDATES=100` (1 label + 99 hard negs/row). Train/val reuses CG's 90/10 user-level split + seed. `N_SHUFFLES=3` train / `1` val and `MAX_ROLLBACK_EXAMPLES_PER_USER=50` — both match CG. Label filter: `raw_hours > 0.5` AND `history[i] not in history[:i]` (dedupe guard for the ~0.2% Steam history dups). Peak memory cut ~103 GB → ~8–10 GB via chunked streaming writes (`CHUNK_SIZE=250_000`).

**Parquet schema** (`precompute._CROSS_COLS`): `user_id`, `rollback_n`, `label_item_idx`, `neg_item_idxs` (K-1 hard negs in CG order), `cg_label_rank` (1-indexed, capped at K), `cg_label_score`, `cg_neg_scores`, `tag_cosine_{label,negs}`, `text_cosine_{label,negs}` (V6a), Bucket 1/2/5/6 cross cols, `user_avg_log_playtime`, `user_interaction_count`, and the four padded history arrays `X_hist_{liked,disliked,full,playtime_weights}` (MAX_HISTORY_LEN).

**Tag/text cosine** are raw cosines over the `game_tag_matrix` / `game_text_matrix` buffers (not tower outputs) — model-independent, so the parquet doesn't break if tower hidden dims change.

---

## 7. Training Stack

**Negative sampling:** sampled softmax, 1 label + 999 random corpus items = 1000-cand pool, **no hard negs** (`n_hard_negs=0`). At N ≥ ~1000 the hard-neg term is redundant and slightly hurts (§8). **Loss:** `F.cross_entropy(scores/temperature, target=0)`, temperature=0.1. No BCE/sigmoid in `forward()`.

```
lr 1e-3 | weight_decay 0.0 | adam_eps 1e-6 | batch 512 | steps 50_000 | log_every 500 | grad_clip 1.0
temperature 0.1 | scheduler CosineAnnealingLR T_max=steps eta_min=1e-4 | hidden_dims [256,128] (no final ReLU)
n_cross_features 24 (Bucket 6's 23 + text_cosine col 23) | n_wide_normalized 14 (cols 10-23 Z-scored) | dropout 0.0
popularity_alpha 0.0 | warm_start_alpha 0.0 | n_random_negs 999 | n_hard_negs 0 | warm_start True
```

`_config.json` sidecar records arch params + `popularity_alpha` + `temperature` + `n_random_negs` + `n_hard_negs`. **Eval:** NDCG@K/MRR/Hit@K for K∈{1,5,10,20,50,100}, same E2E ceiling; header reports the comparison CG's α.

---

## 8. Phase Findings

Per-bucket results are the §Status table. The durable lessons:

**Phase A (strict CG parity):** A Wide & Deep MLP reached near-parity with the dot product on identical features (A-N2 NDCG@10 0.0741 vs CG α=0 0.0752, −1.5%). Three findings:
1. **Drop hard negs at large N.** Once N covers the broad-landscape tail, hard negs over-concentrate gradient at the CG-confusable boundary; uniform random gives the easy-vs-label gradient the model needs. "1 label + N hard + M random" is small-N advice.
2. **Warm-start is load-bearing** — from-scratch was materially worse; the deep MLP needs CG's content-tower head start.
3. **N is the lever for sampled-softmax lift**, not architecture or hard-neg mining (N=100→500→1000 closed the gap ~12%→~6.8%→~1.5%). Loss-family changes aren't the bottleneck. Pointwise BCE was tried and converged to the class prior (NDCG random) — abandoned.

**Phase B (cross features), durable rules drawn from drops:**
- **When offline-flat coincides with canary-better, ship (Bucket 6).** Flat offline doesn't penalize within-top-20 reorders that don't flip the target across K — but those reorders are exactly what list-shape (canary) sees.
- **"Offline-flat" alone isn't the ship signal — the canary is the tiebreaker both ways (Bucket 8).** A flat bucket whose mechanism is "boost high-engagement items" is a popularity leak in disguise (`*_mean_log_playtime` is highest for mass-market titles).
- **A feature re-importing an upstream model's score/rank earns its keep only if it beats that model on signal the reranker doesn't already hold (Bucket 9).** A warm-started reranker with strong content features usually already holds CG's ordering — and a CG feature imposes a frozen-CG runtime forward (random-neg training has no precomputed CG score to read).

---

## 9. Cross-Feature Reference

```
col 0    : tag_cosine                       (Phase A)
col 1-3  : genre/tag/dev overlap, FULL      (Bucket 1)
col 4-6  : genre/tag/dev overlap, LIKED     (Bucket 2)
col 7-9  : genre/tag/dev overlap, RECENT-3  (Bucket 2)
col 10   : price_match                      (Bucket 5)
col 11   : era_gap                          (Bucket 5)
col 12   : playtime_calibration_median      (Bucket 5, signed)
col 13   : popularity_match                 (Bucket 5)
col 14   : sentiment_match                  (Bucket 5)
col 15-16: tag_overlap_idf, full/liked      (Bucket 6)
col 17-18: niche_tag_match, full/liked      (Bucket 6)
col 19-20: max_tag_idf_match, full/liked    (Bucket 6)
col 21-22: niche_dev_match, full/liked      (Bucket 6)
col 23   : text_cosine                      (item-text CG, SHIPPED 2026-05-25)
```

**24 features (cols 0–23), n_wide_normalized=14.** Cols are **append-only — never reorder**, or older checkpoints mis-align at load. Cols 10–23 are Z-scored at forward (`wide_norm_mean/std` persistent buffers, populated once over the train parquet by `populate_wide_norm_buffers`, std clamped to 1.0 near zero variance); cols 0–9 are bounded ([−1,1]/[0,1]) and pass raw. text_cosine (col 23) is a bounded cosine but, appended last, lands in the trailing Z-scored block — harmless and keeps normalized cols contiguous (the "trailing N" `_normalize_wide` design can't host a raw col after a normalized one). Dropped buckets 3/4/8/9 transiently used cols 10+ and were reverted. New wide-head weights init at 0.1.

**Column manifest lives in 3 places** — `dataset._CROSS_FEATURE_COLS`, `precompute._CROSS_COLS`, `train._WIDE_NORM_PARQUET_COLS` — keep in lockstep when a bucket lands.

**Not-pursued ideas:** Tag Peak Match (redundant with tag cosine + overlap); Recent Game Similarity (became the Recent-3 slice); Genre Diversity / Tag Entropy / History Confidence (deep-concat user-state scalars — if ever wanted, the higher-leverage place is CG's user tower, raising the retrieval ceiling); DCN V2 (explicit cross layers — only if cross features stop helping).

---

## 10. Discipline Rules

1. **One bucket at a time** vs the previous *kept* baseline; bundle-level NDCG is the verdict. Drop-one ablation only as a diagnostic when a bucket disappoints (inference-time weight-zeroing is NOT a substitute — a model trained without X shifts its deep representation).
2. **Fair-α rule** — always report the comparison CG's α.
3. **Beat CG on content features before re-enabling any CG signal.**
4. **No `src/` modifications** — ranker is self-contained, CG code read-only.
5. **No streamlit/export changes until a model is verified better by eval + canary.**
6. **Softmax CE only** — never sigmoid in `forward()`. Pointwise BCE converges to the class prior.
7. **E2E ceiling always enforced** in both ranker eval and CG baseline.
8. **If `src/model.py` or `src/train.get_config()` changes** (tower dims, embedding sizes, new sub-towers): re-derive §3 *first* — partial drift silently breaks warm-start.
9. **Permanently dropped wide-feature classes** (don't re-propose): disliked-history (B3), dev-catalog (B4), engagement-level crosses (B8). If a future need points at dev-side or engagement signal, the lever is the deep tower, not the wide bypass.
10. **Popularity penalty stays off the ranker** — the α=0.2 experiment hurt offline ~27% with no better canary. Both stages ship raw α=0.

---

## 11. Serving

The ranker is wired into canary AND Streamlit through one shared path in `ranker/serving.py` (demo and eval use identical feature engineering). **Flow:** build user context → raw α=0 CG retrieves top-100 → ranker reranks → return reranked order. `serving.rerank_candidates` builds all 24 cross features (Buckets 0/1/2/5/6 + `text_cosine`) and calls `ranker.score_pairs`.

**Artifacts** (`python ranker/main.py export [ranker.pth]` — re-exports the CG via `src.export`, then adds the ranker):
- `serving/model.pth`, `serving/game_embeddings.pt` — raw α=0 CG (retrieval stage)
- `serving/feature_store.pt` — vocab maps, metadata, CG buffers, config, **+ 9 ranker source arrays** (`game_developer_idx`, `game_year_numeric`, `game_median_log_hours`, `game_log_count`, `game_sentiment`, `game_tag_binary_idf`, `game_tag_mean_idf`, `game_tag_max_idf`, `game_dev_log_catalog_size`) consumed by `train._buffers_from_fs` to rebuild the ranker's non-persistent buffers
- `serving/ranker.pth` — `WideDeepRanker` state_dict (params + persistent `wide_norm` buffers; `game_*` buffers rebuilt on load)
- `serving/ranker_config.json` — reconstruction config (emb dims / `n_cross_features` / `n_wide_normalized` / α + provenance)

The app rebuilds the ranker purely from serving artifacts — no `saved_models/`, no `get_config()` glob. Degrades gracefully to CG-only if `serving/ranker.pth` is absent.

---

## 12. Loss-Family Rationale

Why sampled softmax CE (the N-curve: Klenitskiy & Vasilev, "Turning Dross Into Gold Loss," RecSys'23, [arxiv 2309.07602](https://arxiv.org/abs/2309.07602) — NDCG rises with N up to ~1000 then plateaus; Phase A reproduced it). **Not BCE/DeepFM/DCN-V2** — pointwise CTR rankers assume rating binarization (Steam has no stars) or impression logs; sampled negs + BCE overconfidence → class prior. **Not SimpleX/CCL** — margin hinge breaks CG parity, same large-N lesson. **Not WARP/LightFM** — serial adaptive sampling doesn't vectorize on GPU. **Parking lot:** in-batch negatives + LogQ correction ([arxiv 2507.09331](https://arxiv.org/abs/2507.09331)) → 511 free negs/row at batch 512; revisit only if Phase B saturates.
