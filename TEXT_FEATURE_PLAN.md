# Text Feature Plan — game-description embeddings in the CG two-tower model

Living plan for adding Steam game-description text (`short_description` + `about_the_game`)
as a new content signal to the candidate-generation two-tower model. Multi-session work.
Approach decided 2026-05-24, staged A/B decided 2026-05-25, with user sign-off.

> This is the canonical, human-facing plan and the single source of truth — it is NOT
> mirrored in agent memory (memory isn't human-readable). Per the repo Git workflow, model
> changes are code-only until the user runs train + canary + eval and reports numbers.

## Strategy — frozen precomputed embeddings + trainable projection adapter

- Encode each game's text ONCE offline with a frozen sentence encoder (`BAAI/bge-base-en-v1.5`, 768-d). The encoder never runs at train or serve time.
- Only a small projection MLP (the "adapter") trains inside the model. Project down to a modest dim (~16–32) so text does **not** drown the small categorical sub-towers (genre 8, dev 12) in the item concat — a localized information bottleneck.
- L2-normalize the frozen vector before the adapter; init the sub-tower at `gain=0.1` (existing house rule) so large-magnitude text embeddings don't wash out the ID/genre signals.
- **Rejected:** semantic IDs / RQ-VAE / TIGER (over-engineering for a 5,437-item corpus scored exhaustively). **Rejected:** PCA/SVD pre-reduction (unsupervised, blind to the rec loss — the trainable adapter learns a better task-conditional reduction).

## Key design decisions

- **User-tower parity (the easy-to-forget piece).** An item-only text tower lets the model match item text only against *non-text* user signals. True text-vs-text matching needs a **user history text pool**: masked-mean the frozen text embeddings of the user's history items → a user-side text projection → user concat. Mirrors the existing 4 item-id history pools. Whether it earns its parameters is what Stage B measures.
- **No categoricals folded into the text string.** Genres/tags keep their own towers; the text tower stays the *residual* (prose mood/setting/mechanics the categorical towers miss).
- **Text string ordering for 512-token truncation:** `"Title: {t}. Short Description: {s}. About the Game: {a}"` — high-density tokens first, so truncation only ever clips long-form About flavor text at the tail.
- **Encoder prefix is model-family-specific.** BGE-en-v1.5 → documents get NO prefix (only short queries get an instruction). E5 → both docs and queries get `passage:`/`query:`. We treat both game text and user-history text as documents, so encode symmetrically (BGE: empty prefix both sides).
- **Reusable for the ranker (NOT CG-only) — hard requirement.** The `game_text_matrix` / text-embedding artifact must land in `serving/feature_store.pt` as a SHARED source array (same pattern as the 9 existing ranker source arrays consumed by `ranker.train._buffers_from_fs`), so a future ranker text bucket (text-similarity cross feature, or pooled history-text into the deep tower) can consume it without re-embedding. Build the buffer-assembly recipe so BOTH `main.py export` and `ranker/main.py export` pick it up.

## Validation — staged, sequential A/B

Run sequentially so each stage's eval isolates exactly one change:

- **Stage A — V6a = V5 + item text tower ONLY.** Train, canary, eval vs V5. Answers "does *any* text help?"
- **Stage B — V6b = V6a + user history text pool.** Train, canary, eval vs **V6a** (not V5). Isolates whether the direct text-vs-text path earns its parameters. If V6b ≈ V6a, ship item-only (simpler); if V6b > V6a, the user pool is justified.

Identical offline protocol + the archetype canary users **plus the Nick real-user canary** each stage; α=0 as the offline yardstick, α=0.4 for canary judgment. Only the winning variant gets wired to the ranker and documented in CLAUDE.md.

*Why staged, not bundled:* in a dot-product two-tower, item-side-only text is a weak/indirect signal (it must route through the id-pools via the projection); the user pool turns it into a content-based taste-centroid match, strongest on the long tail — but it overlaps with the item-id history pools, so its lift is an empirical question, not a given.

## Step sequence

**Shared setup**
1. Embed full corpus → `data/base_game_text_emb.parquet` (`python -m src.embed_text`). **USER-RUN**, after the appdetails fetch completes.
2. Build the shared `game_text_matrix` buffer in `features.py` (ranker-reusable source array).

**Stage A — item text only (V6a)**
3. Add the item text tower to `model.py` (register buffer, L2-norm → adapter → item concat).
4. Smoke-test V6a (shapes, forward pass, 1 train step).
5. **USER-RUN:** train V6a + canary + eval vs V5 → record results.

**Stage B — add user pool (V6b), gated on Stage A being measured**
6. Add the user history text pool to `model.py` (masked-mean of `game_text_matrix[X_hist_full]` → user adapter → user concat; in-model, like genre/tag context).
7. Verify `dataset.py` needs no change (`X_hist_full` already provided in V3).
8. Smoke-test V6b.
9. **USER-RUN:** train V6b + canary + eval vs V6a → record results.

**Finalize**
10. Wire CG + ranker export to carry the winning variant's text artifact.
11. Verdict + update CLAUDE.md (winning Offline Eval row + Architecture section) + commit `embed_text.py` + requirements + model changes. The shared artifact then unblocks a future ranker text bucket.

## Data pipeline (offline prereqs, standalone — not in `main.py`)

1. `python -m src.fetch_appdetails` — caches the full Steam Storefront `appdetails` payload, one JSON per app_id, to `data/appdetails/`. Resumable, rate-limited, 429/connection-reset backoff.
2. `python -m src.embed_text` — reads `data/appdetails/`, builds the text string, encodes with frozen `bge-base-en-v1.5`, writes `data/base_game_text_emb.parquet` (keyed by `item_id`) + a `_meta.json` provenance sidecar. Delisted/missing games fall back to title-only text (never a zero vector).

## Progress

- **2026-05-25:** On branch `text-feature`. Fetch ~2,300/5,437 (resumable bg job). Generated a *dev* `base_game_text_emb.parquet` (dim 768; 2,168 real store text / 3,269 title-only) from the partial cache — for wiring + smoke only, to be regenerated `--force` at full quality once the fetch lands.
  - **V6a (item text tower) fully wired and smoke-tested, code-only:** `model.py` (frozen `game_text_matrix` buffer + `item_text_tower` 768→128→32 + item concat → 128), `features.py` (`game_text_matrix` built into the FeatureStore, ranker-reusable), `train.py` (config `text_input_dim`/`text_embedding_size` + `build_model` buffer load), `evaluate.py` (`build_game_embeddings` now includes the text component → covers offline_eval / canary / export, the shared reconstruction path).
  - Smoke: forward + backward grads flow through `item_text_tower`; eval-path produces correct (n_items, 128) L2-normalized embeddings. `dataset.py` needs **no change** (text looked up in-model from `target_game_idx`).
  - `src/embed_text.py` committed (`9ff42ec`); **V6a wiring + Nick canary + use_item_text flag are UNCOMMITTED** (held per workflow — model changes commit after the user's Stage A train + eval).
- **2026-05-25 (later):** Fetch COMPLETE — 5,437/5,437 (0 failed, 114 delisted). Regenerated `base_game_text_emb.parquet` at full quality (`--force`): 5,303 with store text, 134 title-only.
  - Added `use_item_text` config flag (default True): `False` builds the exact V5 item tower (96-d concat, no text tower) so the A/B has a **same-branch control**. Wired through model.py / train.get_config / build_model / evaluate.build_game_embeddings. TODO in model.py: delete the flag + branches once text is validated.
  - Added **Nick** real-user canary (`NICK_PLAYTIME`, 52 corpus-matched games with hand-overridden console playtimes; playtime-driven liked/full pools, no disliked) — surfaces in `run_canary_eval` once a V6 checkpoint exists.
  - Smoke-tested both arms: V6a (128-d, text tower) and V5-control (96-d, no tower) both build + forward + eval clean.
  - Extended CG `offline_eval.py` `ks` to include **100** (matches ranker range).
  - Added `_load_model_and_embeddings`/`build_model` **backward-compat**: legacy checkpoints (no `use_item_text`/text dims in sidecar) build as the exact V5 tower and load under strict `load_state_dict`; text buffer registered only when `use_item_text=True`.
- **2026-05-25 — BASELINES LOCKED (no-text V5, with @100):** legacy checkpoints loaded as `use_item_text=False` controls, reproducing documented metrics exactly (so no V5-control retrain needed). Saved to `eval_results/` + `canary_results/`.
  - **α=0** (`best_triple_full_softmax_popularity_alpha_00_20260515_084320`): R@1 .0283, @10 .1435, @50 .3952, **@100 .5552**, MRR .0704 — the offline yardstick.
  - **α=0.4 PROD** (`PROD_best_triple_full_softmax_popularity_alpha_04_20260502_203217`): R@1 .0226, @10 .1253, @50 .3673, **@100 .5278**, MRR .0611 — the canary baseline; Nick's α=0.4 recs are the qualitative bar (franchise-adjacent: Civ IV base, DA:O base, Torchlight, KOTOR/Mass Effect cluster).
  - **Next actionable — USER runs Stage A (#11) on `text-feature`:** train V6a `use_item_text=True` at α=0 (→ `eval` vs α=0 baseline) and at α=0.4 (→ `canary` vs α=0.4 baseline + Nick). If text helps → Stage B (#4 user pool). Tasks done so far: #1,#2,#3,#5,#10.
- **2026-05-25 — CG checkpoint-selection rule changed (CE → NDCG@10).** `train.py:train_softmax` previously saved 'best' on lowest val cross-entropy; CE is only a surrogate (it keeps falling via confidence calibration after ranks plateau). Now selects on **val NDCG@10** — the ranking metric we ship and report — mirroring `ranker.train`. Logs `val_ndcg@10` + `val_mrr` each step, and adds a final post-loop eval (the loop only scores on log steps, so the last step's weights could be missed). Computed from the same full-corpus score matrix the CE pass already builds (ranks off RAW dot products = inference scoring), so ~free. **UNCOMMITTED** (training-behavior change, held per workflow). CG retrain is cheap (~10 min / 50k steps) so all subsequent CG runs use this rule.
  - **Rule validated — no regression:** retrained the no-text α=0 control under the NDCG@10 rule (`best_triple_full_softmax_popularity_alpha_0_20260525_094445`): R@1 .0279, @10 .1430, @50 .3958, @100 .5572, MRR .0702 — statistically identical to the CE-selected baseline (every Δ ≤ .002, pure noise). CE and NDCG@10 land on equivalent checkpoints here. **This is now the fresh same-rule no-text α=0 baseline** for the clean Stage A text-vs-no-text comparison. Nick's recs (α=0): coherent CRPG/strategy cluster — KOTOR, Mass Effect 1&2, Dark Souls, XCOM, Age of Empires II HD, + Batman Arkham / Tomb Raider / LA Noire.
  - **First V6a α=0 (item text) result — but CE-selected, superseded:** `best_triple_full_softmax_text_popularity_alpha_0_20260525_092413` evaluated **flat** vs the α=0 baseline (R@1 .0275, @50 .3953, @100 .5571, MRR .0703 — every Δ ≤ .002). Exactly the predicted weak/indirect signal for item-only text. ⚠️ This checkpoint was selected under the OLD CE rule, so it carries an asterisk — re-run under the NDCG@10 rule below.
- **2026-05-25 — STAGE A DONE (item-only text, clean NDCG@10-rule comparison at α=0): FLAT.** V6a `…100022` (`use_item_text=True`) vs the same-rule no-text baseline `…094445`:

  | K | Recall no-text | Recall V6a text | NDCG no-text | NDCG V6a text |
  |---|---|---|---|---|
  | 1 | .0279 | .0279 | .0279 | .0279 |
  | 5 | .0868 | .0875 | .0573 | .0576 |
  | 10 | .1430 | .1444 | .0753 | .0758 |
  | 20 | .2308 | .2299 | .0973 | .0972 |
  | 50 | .3958 | .3968 | .1299 | .1302 |
  | 100 | .5572 | .5575 | .1560 | .1562 |

  MRR .0702 → .0704. Every Δ ≤ .0014 — within the same noise band the selection-rule retrain alone produced. **Not a win.** Nick canary near-identical (same CRPG/strategy/action cluster, minor reorders; text dropped Dark Souls/Killing Floor, added LIMBO/Mirror's Edge). Confirms the prediction: item-side-only text has no direct text-vs-text path in a dot-product two-tower.
  - **DECISION: proceed to Stage B at α=0**, defer the α=0.4 run until the winning architecture is locked (α is the last calibration knob — find best features at α=0 first). Item-only text isn't killed (it's harmless and the buffer ships anyway for the ranker), but it doesn't justify itself alone; the lift, if any, must come from the user pool.
- **2026-05-25 — STAGE B WIRED (V6b = item + user text pool), code-only, smoke-clean.** `model.py`: `use_user_text` flag + `user_text_tower` (768→128→32) + masked-MEAN of `game_text_matrix[X_hist_full]` (unit-norm per-game vecs, pad rows drop, divide by non-pad count) → user concat (192→**224**). Shared `game_text_matrix` buffer now registers when item OR user text is on. `train.py`: config `use_user_text`/`user_text_embedding_size`, `build_model` buffer-copy condition, banner variant (`V6b`), checkpoint tag `text_user_`. **No change to `dataset.py`/`evaluate.py`/`offline_eval.py`** — user_embedding signature unchanged, text looked up in-model from `X_hist_full`, so canary/eval/offline get the pool for free. Smoke: V6b forward+backward (grads flow through `user_text_tower`), empty/padded history finite (no NaN), V6a + V5 controls build + load clean. **UNCOMMITTED** (held per workflow).
  - **Next actionable — USER runs Stage B (#13):** `get_config` is set (`use_item_text=True`, `use_user_text=True`, α=0). `python main.py train` → checkpoint `best_triple_full_softmax_text_user_popularity_alpha_0_<ts>.pth` → eval + canary vs **V6a `…100022`** (not the no-text baseline — Stage B isolates the user pool's marginal lift over item-only text).
- **2026-05-25 — STAGE B DONE (user text pool): REGRESSED → DROPPED.** V6b `…101807` (item + user text) vs V6a `…100022` (item only):

  | K | V6a item-only | V6b +user pool | Δ |
  |---|---|---|---|
  | 1 | .0279 | .0268 | −.0011 |
  | 10 | .1444 | .1423 | −.0021 |
  | 50 | .3968 | .3942 | −.0026 |
  | 100 | .5575 | .5567 | −.0008 |

  MRR .0704 → .0694. **Worse on every single metric** (all 6 Recall Ks, all NDCG Ks, MRR — uniform negative = real regression, not noise). Nick canary lateral (one franchise hit: Torchlight entered; still action-heavy, no strategy/4X sharpening). Confirms the mean-centroid failure mode: averaging an eclectic history's text vectors yields a blurry "no man's land" centroid that dilutes the id-history pools, which already encode taste better.
- **2026-05-25 — FINAL VERDICT: CG ships V6a (item text only).** User chose to keep the (flat) item text tower — harmless to CG metrics, and the shared `game_text_matrix` artifact unblocks a future **ranker text bucket** (the stronger use case: a text-similarity cross feature has a direct signal path the CG dot-product lacks). **User text pool code removed** from `model.py` + `train.py` (dropped like the regressing ranker buckets). α=0.4 calibration deferred until any ranker text work is also settled.
  - **Remaining:** #6 wire CG + ranker export to carry the `game_text_matrix` artifact; #14 update CLAUDE.md (Architecture + Offline Eval rows), then commit the V6a wiring + NDCG@10 selection rule + Nick canary + K=100 (all still UNCOMMITTED on `text-feature`).

## Run Stage A now — exact recipe (cold-start ready)

On branch `text-feature` (env: `conda activate pytorch_env`). Two V6a runs, text tower ON, each vs its baseline above. All CG runs now use the **NDCG@10 selection rule** (compare only against same-rule baselines). `_config.json` sidecars record `use_item_text` + `popularity_alpha` per checkpoint, so `eval`/`canary` rebuild the right architecture automatically.

1. **V6a α=0** — offline yardstick. `get_config()` already defaults to `use_item_text=True`, `popularity_alpha=0.0`:
   ```
   python main.py train
   python main.py eval <new_checkpoint>
   ```
   Win = beats the fresh same-rule no-text baseline `…094445` (R@1 .0279 / @50 .3958 / @100 .5572 / MRR .0702).

2. **V6a α=0.4** — canary quality. Set `popularity_alpha=0.4` in `src/train.py:get_config` (keep `use_item_text=True`):
   ```
   python main.py train
   python main.py canary <new_checkpoint>
   ```
   Win = Nick + archetype recs gain 4X/CRPG depth, lose generic-popular picks vs the α=0.4 baseline.

After Stage A: report numbers here; if text helps → Stage B (#4, user-history text pool). Per workflow, commit the V6a wiring (+ `use_item_text` flag, Nick canary, K=100, backward-compat loader — all currently uncommitted on `text-feature`) only after results confirm.

> Note: the "Step sequence" numbers above are narrative; live TaskCreate IDs differ (done: #1,2,3,5,10 · next: #11 Stage A · then #4,#12,#13 Stage B · #6,#14 finalize). Rebuild the task list from this file if a new session starts empty.
