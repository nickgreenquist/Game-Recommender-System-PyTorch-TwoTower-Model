"""
Stage 5 (ranker) — export ranker serving artifacts on top of the CG export.

Produces, in serving/:
  model.pth            — α=0 CG weights (raw retrieval stage)        [via src.export.run_export]
  game_embeddings.pt   — α=0 CG per-game embeddings                  [via src.export.run_export]
  feature_store.pt     — CG vocab/metadata + 9 ranker source arrays  [via src.export.run_export]
  ranker.pth           — WideDeepRanker state_dict (params + persistent wide_norm
                         buffers; the non-persistent game_* buffers are NOT saved —
                         the app rebuilds them on load via train._buffers_from_fs from
                         the 9 source arrays in feature_store.pt, same as canary)
  ranker_config.json   — ranker reconstruction config (emb dims / n_cross_features /
                         n_wide_normalized / α + provenance)

Two-stage serving pipeline (architecture decision 2026-05-23): the raw α=0 CG retrieves
the top-100 candidates → the ranker reranks them. So the serving CG is ALWAYS the raw
α=0 CG (the fixed retrieval stage). A ranker-side popularity penalty (α=0.2) was tested
and rejected (2026-05-23), so BOTH stages ship raw α=0 — no popularity penalty anywhere
in the deployed pipeline.

Usage:
    python ranker/main.py export                  # latest ranker + α=0 CG
    python ranker/main.py export <ranker.pth>     # explicit ranker checkpoint
"""
import json
import os

import torch

from ranker.canary import _resolve_cg_checkpoint, _resolve_ranker_checkpoint
from ranker.train import build_ranker, get_config
from src.export import run_export

SERVING_DIR = 'serving'

# Config keys consumed by build_ranker — the sidecar overlay set. Mirrors the canary's
# reconstruction list (ranker/canary.run_canary) so both rebuild the ranker identically.
_RANKER_CONFIG_KEYS = (
    'hidden_dims', 'dropout', 'n_cross_features', 'n_wide_normalized',
    'item_id_emb_dim', 'item_genre_emb_dim', 'item_tag_emb_dim',
    'developer_emb_dim', 'year_emb_dim', 'price_emb_dim',
    'user_genre_emb_dim', 'user_tag_emb_dim', 'text_emb_dim',
    'item_tag_hidden', 'user_tag_hidden', 'user_genre_hidden', 'item_text_hidden',
    'popularity_alpha',
)


def run_ranker_export(ranker_checkpoint: str = None, cg_checkpoint: str = None,
                      data_dir: str = 'data') -> None:
    ranker_checkpoint = ranker_checkpoint or _resolve_ranker_checkpoint()
    # Retrieval CG is ALWAYS the raw α=0 CG — the fixed serving retrieval stage. Same
    # decision as canary; an explicit cg_checkpoint still wins for one-off exports.
    cg_checkpoint = cg_checkpoint or _resolve_cg_checkpoint(0.0)

    print(f"Ranker checkpoint:             {ranker_checkpoint}")
    print(f"CG checkpoint (α=0 retrieval): {cg_checkpoint}\n")

    # ── 1. CG export (model.pth, game_embeddings.pt, feature_store.pt) ───────────
    # run_export augments feature_store.pt with the 9 ranker source arrays and returns
    # the FeatureStore so we can rebuild the ranker without a second load_features pass.
    fs = run_export(data_dir=data_dir, checkpoint_path=cg_checkpoint)

    # ── 2. Build ranker + load trained weights (canary reconstruction recipe) ────
    cfg = get_config()
    cfg_path = os.path.splitext(ranker_checkpoint)[0] + '_config.json'
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k in _RANKER_CONFIG_KEYS:
            if k in saved:
                cfg[k] = saved[k]
    cfg['warm_start_cg_checkpoint'] = None    # loading trained weights, not warm-starting
    ranker = build_ranker(cfg, fs)
    ranker.load_state_dict(
        torch.load(ranker_checkpoint, weights_only=True, map_location='cpu')
    )
    ranker.eval()

    # ── 3. Save ranker.pth ───────────────────────────────────────────────────────
    # state_dict() carries params + the persistent wide_norm_mean/std buffers. The
    # non-persistent game_* buffers are excluded (persistent=False) — the app rebuilds
    # them from feature_store.pt's source arrays on load, exactly as canary does.
    os.makedirs(SERVING_DIR, exist_ok=True)
    ranker_path = os.path.join(SERVING_DIR, 'ranker.pth')
    torch.save(ranker.state_dict(), ranker_path)
    print(f"Saved {ranker_path}  ({os.path.getsize(ranker_path) / 1e6:.1f} MB)")

    # ── 4. Save ranker config sidecar for app reconstruction ─────────────────────
    serving_cfg = {k: cfg[k] for k in _RANKER_CONFIG_KEYS if k in cfg}
    serving_cfg['ranker_checkpoint'] = os.path.basename(ranker_checkpoint)
    serving_cfg['cg_checkpoint']     = os.path.basename(cg_checkpoint)
    cfg_out = os.path.join(SERVING_DIR, 'ranker_config.json')
    with open(cfg_out, 'w') as f:
        json.dump(serving_cfg, f, indent=2)
    print(f"Saved {cfg_out}")
    print("\nDone. Ranker serving artifacts ready. Run: streamlit run streamlit_app.py")
