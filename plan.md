# Plan: Remove LayerNorm from Sum-Pooling (Match Industry Standard)

## Motivation

YouTube DNN, TikTok, and every major two-tower retrieval system uses raw sum-pooling — no LayerNorm. The projection MLP already learns the right scale. If gradients explode during retraining, only then add it back.

---

## Changes — `src/model.py` only

### `__init__`: delete the norm block (lines 46–49)

```python
# DELETE:
        # ── Normalization for Sum Pooling ──
        self.history_norm      = nn.LayerNorm(item_id_embedding_size)
        self.playtime_pool_norm = nn.LayerNorm(item_id_embedding_size)
        self.tag_norm          = nn.LayerNorm(n_tags)
```

### `user_embedding()`: remove norm calls from all four pools and tag context

- **liked_pool** — `pool_ids(X_hist_liked)`: `self.history_norm(...)` → raw `.sum(dim=1)`
- **disliked_pool** — `pool_ids(X_hist_disliked)`: same
- **full_pool** — `pool_ids(X_hist_full)`: same
- **playtime_pool** — `self.playtime_pool_norm((item_embs * w).sum(dim=1))` → `(item_embs * w).sum(dim=1)`
- **tag context** — `self.user_tag_tower(self.tag_norm(X_tag))` → `self.user_tag_tower(X_tag)`

---

## Verification

1. `python main.py train` — watch first 500 steps for NaN loss or gradient explosion
2. `python main.py canary` — confirm per-genre coherence holds
3. `python main.py eval` — metrics must match or beat V3 PROD (Recall@50 ≥ 0.39, MRR ≥ 0.07)
