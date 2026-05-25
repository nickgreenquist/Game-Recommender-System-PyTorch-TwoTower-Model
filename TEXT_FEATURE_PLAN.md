# Text Feature Plan — game-description embeddings in the CG two-tower model

Living plan for adding Steam game-description text (`short_description` + `about_the_game`)
as a new content signal to the candidate-generation two-tower model. Multi-session work.
Approach decided 2026-05-24, staged A/B decided 2026-05-25, with user sign-off.

> This is the canonical, human-facing plan. A mirror of the key decisions also lives in
> the agent memory (`project_text_tower_plan.md`) so it auto-loads each session; keep the
> two in sync. Per the repo Git workflow, model changes are code-only until the user runs
> train + canary + eval and reports numbers.

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

Identical offline protocol + 9 canary users each stage; α=0 as the offline yardstick, α=0.4 for canary judgment. Only the winning variant gets wired to the ranker and documented in CLAUDE.md.

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

- **2026-05-25:** Fetch in progress (resumable background job, ~1,250+/5,437 cached after a RemoteDisconnected crash + fix; fix committed). `src/embed_text.py` written + sanity-tested (franchise/genre separation confirmed; uncommitted, to be bundled with the model work). `requirements.txt` += `sentence-transformers` (uncommitted). `bge-base-en-v1.5` pre-downloaded to the HF cache. Nothing trained yet. **Next actionable: finish fetch → run step 1 (embed).**
