"""
Stage 1+ — Wide & Deep ranker training loop with STRICT V5 CG parity (Phase A).

Architecture (see ranker/model.py):
  User concat (192): liked(32) + disliked(32) + full(32) + playtime(32) + genre(32) + tag(32)
  Item concat  (96): item_id(32) + item_genre(8) + item_tag(32) + dev(12) + year(8) + price(4)
  Deep MLP   (288 → [256, 128]):  Linear(288→256) → ReLU → Linear(256→128)
                                  — matches CG's projection shape; NO final ReLU.
  Head: cat(deep_out(128), cross_features(n_cross)) → Linear(128+n_cross, 1)
  Score: logit / temperature (=0.1) → softmax CE  (matches CG's logit scaling)

Phase A cross feature: tag_cosine (1 total). DO NOT add more until ranker matches CG.

α plan-of-record (plan §8): ranker α=0, compared against throwaway CG α=0. Warm-start
auto-resolves to the α-matched CG checkpoint via _ALPHA_TO_CG_GLOB. Deep MLP / head /
cross weights are random — they're new (no CG counterpart).

Empirical (2026-05-16): turning warm-start OFF for one Phase A run produced materially
worse NDCG. Keeping it ON as part of the Phase A baseline — the controlled-experiment
purist take ("from-scratch ranker vs from-scratch CG") is correct on paper but loses
to the practical reality that the deep MLP is much harder to optimize without a
content-tower head start. Gate stays for future ablations but the default is True.

Loss: sampled softmax CE over 1099 cands/row (1 label + 99 hard from parquet + 999
random per-step). Closest tractable approximation to CG's full-corpus softmax given
the deep MLP is ~150× more expensive per (user, item) pair than CG's dot product.
N=999 picked per plan §13 experiment A-N1 — Klenitskiy (RecSys'23) shows ML-1M
NDCG@10 keeps climbing through N≈1000 before plateauing. Was N=400 in Try 3.
"""
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from ranker.cross_features import (Bucket5GameBuffers, NicheBuffers,
                                     OverlapBuffers, categorical_overlap_triple,
                                     last_n_history, niche_scalar_triple,
                                     numeric_match_quintuple, weighted_overlap)
from ranker.dataset import (CANDIDATES_PER_ROW, compute_cross_features,
                             load_splits, sample_batch)
from ranker.evaluate import (cg_baseline, compute_label_ranks,
                              evaluate_ndcg_mrr, hit_rates_from_ranks,
                              ndcg_at_k_from_ranks, _ndcg_mrr_from_ranks)
from ranker.model import WideDeepRanker


# ── Config ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# Map of ranker popularity_alpha → CG checkpoint glob to warm-start from.
# The α a CG was trained with is baked into its embeddings — α=0.4 CG learned to
# under-rank popular items in raw dot-product space (compensating for the +α·log(count)
# logit boost added during training); α=0 CG did not. Mismatched warm-start forces
# the deep MLP + head to spend early gradient steps undoing a prior they don't want.
# Auto-resolving here prevents the silent footgun where popularity_alpha changes but
# warm_start_cg_checkpoint doesn't.
# α=0 now points at the V6a item-text CG (the deployed retrieval stage as of 2026-05-25).
# This one glob drives ALL four consumers — warm-start source, precompute retrieval,
# canary CG, and export CG — so flipping it here re-grounds the entire ranker pipeline
# on the served text CG. The glob excludes the dropped V6b user-text CG
# (`..._text_user_popularity_...`) and the pre-text α=0 CG (`..._popularity_alpha_00_...`).
# sorted()[-1] picks the latest matching checkpoint (…100022, the deployed one).
_ALPHA_TO_CG_GLOB = {
    0.0: 'saved_models/best_triple_full_softmax_text_popularity_alpha_0_*.pth',
    0.4: 'saved_models/PROD_best_triple_full_softmax_popularity_alpha_04_*.pth',
}


def _resolve_warm_start_for_alpha(alpha: float) -> str:
    """Return the CG checkpoint glob whose training α matches the ranker's α.

    Add new α values to _ALPHA_TO_CG_GLOB as you train new comparator CGs. Override
    `warm_start_cg_checkpoint` in the config explicitly for one-off experiments —
    that takes precedence over what get_config() pins here.
    """
    if alpha not in _ALPHA_TO_CG_GLOB:
        supported = sorted(_ALPHA_TO_CG_GLOB.keys())
        raise ValueError(
            f"No CG checkpoint convention for ranker α={alpha}. "
            f"Supported: {supported}. Either train a matching CG and add it to "
            f"_ALPHA_TO_CG_GLOB, or set warm_start_cg_checkpoint explicitly."
        )
    return _ALPHA_TO_CG_GLOB[alpha]


def get_config() -> dict:
    cfg = {
        # Optimizer
        'lr':                 1e-3,
        'weight_decay':       0.0,
        'adam_eps':           1e-6,

        # Training — STRICT CG-hyperparam parity. batch_size=512 / training_steps=50_000
        # match src/train.get_config exactly. Sampled softmax over 1 label + n_hard_negs
        # hard + n_random_negs random ≈ closest we can get to CG's full-corpus softmax
        # given the deep MLP is ~150× more expensive per (user, item) pair than CG's
        # dot product. N_random=999 per plan §13 (Klenitskiy RecSys'23 ML-1M curve
        # plateaus around N≈1000). temperature=0.1 matches CG's logit scaling.
        #
        # ACTIVE EXPERIMENT: plan §13 A-N2 — drop hard negs (n_hard_negs=0), pure
        # 1000-cand random pool. Tests whether the 99 parquet hard negs were helping
        # discrimination or over-concentrating gradient on CG-confusables at large N.
        # Flip back to n_hard_negs=99 for the A-N1 baseline (1+99+999=1099 cands).
        'batch_size':         512,
        'n_random_negs':      999,
        'n_hard_negs':        0,
        'training_steps':     50_000,
        'log_every':          500,
        'grad_clip':          1.0,
        'seed':               42,
        'checkpoint_dir':     'saved_models/ranker',
        'temperature':        0.1,

        # Architecture (mirror src/train.get_config V5 defaults)
        'item_id_emb_dim':    32,
        'item_genre_emb_dim': 8,
        'item_tag_emb_dim':   32,
        'developer_emb_dim':  12,
        'year_emb_dim':       8,
        'price_emb_dim':      4,
        'user_genre_emb_dim': 32,
        'user_tag_emb_dim':   32,
        'text_emb_dim':       32,   # V6a item_text_tower output (768→128→32)
        # Hidden dims hardcoded in src/model.py — must match for warm-start parity.
        'item_tag_hidden':    128,
        'user_tag_hidden':    256,
        'user_genre_hidden':  128,
        'item_text_hidden':   128,
        # Deep MLP — matches CG's projection shape [256, 128] exactly (Linear→ReLU→Linear,
        # no final activation). See ranker/model.py docstring.
        'hidden_dims':        [256, 128],
        'dropout':            0.0,

        # Wide bypass — Bucket 6 (23 features total). Column order in compute_cross_features
        # is stable, do not reorder — checkpoints depend on it:
        #    0 tag_cosine (B-0)               ← Phase A
        #    1 genre_overlap (B-1)            ← Bucket 1 (full-history slice)
        #    2 tag_overlap (B-1b)             ← Bucket 1
        #    3 dev_affinity (B-2)             ← Bucket 1
        #    4 genre_overlap_liked (B-3a)     ← Bucket 2A (liked-only slice)
        #    5 tag_overlap_liked (B-3b)       ← Bucket 2A
        #    6 dev_affinity_liked (B-3c)      ← Bucket 2A
        #    7 genre_overlap_recent3 (B-4a)   ← Bucket 2B (last-3-liked window)
        #    8 tag_overlap_recent3 (B-4b)     ← Bucket 2B
        #    9 dev_affinity_recent3 (B-4c)    ← Bucket 2B
        #   10 price_match (B-7a)             ← Bucket 5 (numeric matching, Z-scored)
        #   11 era_gap (B-7b)                 ← Bucket 5
        #   12 playtime_cal_median (B-7c)     ← Bucket 5 (SIGNED)
        #   13 popularity_match (B-7d)        ← Bucket 5
        #   14 sentiment_match (B-7e)         ← Bucket 5
        #   15 tag_overlap_idf_full (B-9a)    ← Bucket 6 (Shape A, IDF overlap)
        #   16 tag_overlap_idf_liked (B-9b)   ← Bucket 6
        #   17 niche_tag_match_full (B-9c)    ← Bucket 6 (Shape B, scalar arithmetic)
        #   18 niche_tag_match_liked (B-9d)   ← Bucket 6
        #   19 max_tag_idf_match_full (B-9e)  ← Bucket 6
        #   20 max_tag_idf_match_liked (B-9f) ← Bucket 6
        #   21 niche_dev_match_full (B-9g)    ← Bucket 6
        #   22 niche_dev_match_liked (B-9h)   ← Bucket 6
        #   23 text_cosine (B-10)             ← item-text CG integration (full slice)
        # Bucket 3 (disliked slice) and Bucket 4 (dev-catalog) were tried + dropped;
        # Bucket 7 (item-intrinsic priors) was dropped in planning and its 2 keep-able
        # features were absorbed into Bucket 6 as proper crosses — see plan §9.
        # Col 23 text_cosine = cosine(user playtime-weighted text centroid, candidate text)
        # — the direct text-vs-text path the CG dot product can't express. Added alongside
        # the deep item_text_tower when the served CG gained item text (V6a).
        'n_cross_features':   24,
        # The trailing 14 cross-feature columns (10-23) get Z-scored by the model's
        # persistent `wide_norm_mean` / `wide_norm_std` buffers; cols 0-9 are already
        # bounded in [-1, 1] / [0, 1] and pass through raw. Populated by
        # populate_wide_norm_buffers below once the model is built and the train dataset
        # is loaded. Was 5 (Bucket 5), 13 (Bucket 6), now +1 for col 23 text_cosine.
        # text_cosine is a bounded cosine like tag_cosine (col 0) but appended at the END,
        # so it joins the trailing normalized block rather than the leading raw block —
        # Z-scoring a bounded cosine is harmless and keeps cols 10-22 contiguous.
        'n_wide_normalized':  14,

        # Menon α — popularity penalty ADDED to TRAINING logits only (Path 2; raw dot
        # products at inference/eval, same as CG). Architecture decision (2026-05-23):
        # both stages ship raw α=0 — the CG retrieves recall-maximizing, and the ranker
        # reorders on its content/cross features alone. A ranker-side penalty (α=0.2) was
        # tested 2026-05-23 and REJECTED (hurt offline ~27%, no meaningfully better
        # canary), so the penalty stays OFF. The plumbing is retained at 0; raise only if
        # a future experiment revisits the decision.
        'popularity_alpha':   0.0,

        # Warm-start from CG state_dict (plan §3 "Warm-start mapping"). Default ON —
        # an off-baseline tested 2026-05-16 was materially worse, so warm-start is part
        # of the Phase A configuration, not a Phase B add-on. Flip to False to disable.
        # To change WHICH CG it inits from, set `warm_start_alpha` below (the live knob);
        # `warm_start_cg_checkpoint` is derived from it and overwritten on every get_config.
        'warm_start':         True,
        # Which CG α to warm-start FROM — DECOUPLED from popularity_alpha. None = match
        # popularity_alpha (historical behavior). Pinned to 0.0 here: the ranker always
        # inits from the RAW α=0 CG (the fixed retrieval stage; no baked-in popularity
        # prior), then learns the penalty from neutral embeddings during training. This
        # is the SAFE warm-start direction (start neutral → learn the bias), unlike
        # α=0-ranker-from-α=0.4-CG which forces the MLP to undo a prior it doesn't want.
        'warm_start_alpha':   0.0,

        # Train-time eval: deterministic sample of N val rows for fast logging.
        # Final eval (evaluate_only) always uses the full val set.
        'n_eval_samples':     20_000,
    }
    if cfg['warm_start']:
        # Warm-start source is chosen by warm_start_alpha (falls back to popularity_alpha
        # when None) — NOT popularity_alpha directly, so an α=0.4 ranker can still init
        # from the raw α=0 CG.
        ws_alpha = cfg['warm_start_alpha'] if cfg['warm_start_alpha'] is not None else cfg['popularity_alpha']
        cfg['warm_start_cg_checkpoint'] = _resolve_warm_start_for_alpha(ws_alpha)
    else:
        cfg['warm_start_cg_checkpoint'] = None
    return cfg


def _config_path(checkpoint_path: str) -> str:
    return os.path.splitext(checkpoint_path)[0] + '_config.json'


def _save_config(config: dict, checkpoint_path: str) -> None:
    with open(_config_path(checkpoint_path), 'w') as f:
        json.dump(config, f, indent=2)


# ── Buffer construction (FeatureStore → ranker buffers, pad row appended) ───

def _buffers_from_fs(fs: dict) -> dict:
    """Build the per-game buffers consumed by WideDeepRanker, each with pad row."""
    n_items   = fs['n_items']
    n_tags    = fs['n_tags']
    n_genres  = fs['n_genres']
    n_devs    = fs['n_developers']

    def _pad_1d_float(arr: np.ndarray) -> np.ndarray:
        """Append a single 0-valued pad row to a (n_items,) float32 array."""
        return np.concatenate([arr.astype(np.float32), np.zeros(1, dtype=np.float32)])

    tag_matrix   = np.vstack([fs['game_tag_matrix'],
                               np.zeros((1, n_tags),  dtype=np.float32)])
    # Pre-L2-normalize once for the tag_cosine cross feature's candidate-side path
    # (see ranker/train._forward_batch, ranker/canary._synthetic_tag_cosine). Avoids
    # re-running F.normalize over (B, n_cand, n_tags) every training step. The RAW
    # `game_tag_matrix` is preserved untouched because user_tag_tower / item_tag_tower
    # are warm-started on raw TF-IDF rows, and the user-side pool needs
    # normalize(Σ wᵢ · rawᵢ), not Σ wᵢ · normalize(rawᵢ).
    tag_norm     = np.linalg.norm(tag_matrix, ord=2, axis=-1, keepdims=True)
    tag_norm     = np.where(tag_norm > 0.0, tag_norm, 1.0)
    tag_matrix_l2 = (tag_matrix / tag_norm).astype(np.float32)
    # ── Bucket 1 binary/count buffers (genre + tag) ──────────────────────
    # The deep towers consume game_tag_matrix (raw TF-IDF) and game_genre_matrix
    # (L1-row-normalized). The overlap cross features want TRUE set-membership
    # semantics, so derive separate binary buffers + safe-divide counts.
    tag_binary   = (tag_matrix > 0).astype(np.float32)
    tag_count    = np.maximum(tag_binary.sum(axis=1), 1.0).astype(np.float32)
    genre_matrix = np.vstack([fs['game_genre_matrix'],
                               np.zeros((1, n_genres), dtype=np.float32)])
    genre_binary = (genre_matrix > 0).astype(np.float32)
    genre_count  = np.maximum(genre_binary.sum(axis=1), 1.0).astype(np.float32)
    year_idx     = np.concatenate([fs['game_year_idx'],
                                    np.zeros((1,), dtype=np.int64)])
    # Padding for developer index: use n_developers (matches Embedding padding_idx).
    dev_idx      = np.concatenate([fs['game_developer_idx'],
                                    np.array([n_devs], dtype=np.int64)])
    price_idx    = np.concatenate([fs['game_price_bucket'],
                                    np.zeros((1,), dtype=np.int64)])

    # Bucket 5 — 4 per-game numeric scalar buffers, each shape (n_items+1,) float32
    # with a 0-valued pad row appended. Same buffer set built device-side in
    # ranker/precompute.py:_score_candidates → both call sites use the same
    # numeric_match_quintuple util so values are bit-identical by construction.
    year_numeric_b     = _pad_1d_float(fs['game_year_numeric'])
    median_log_hours_b = _pad_1d_float(fs['game_median_log_hours'])
    log_count_b        = _pad_1d_float(fs['game_log_count'])
    sentiment_b        = _pad_1d_float(fs['game_sentiment'])

    # Bucket 6 — 5 per-game rarity buffers. game_tag_binary_idf is the (n_items+1, n_tags)
    # matrix companion to game_tag_binary (plays the same role in `weighted_overlap`, just
    # IDF-weighted); the other 4 are scalars analogous to the Bucket 5 quintet. Same
    # pad/zero-row convention. game_tag_count_idf clamps to 1.0 to avoid div-by-zero in
    # `weighted_overlap`'s denominator (some pad/sparse rows may have zero IDF mass).
    tag_binary_idf_b = np.vstack([fs['game_tag_binary_idf'],
                                   np.zeros((1, n_tags), dtype=np.float32)])
    tag_count_idf_b  = np.maximum(tag_binary_idf_b.sum(axis=1), 1.0).astype(np.float32)
    tag_mean_idf_b           = _pad_1d_float(fs['game_tag_mean_idf'])
    tag_max_idf_b            = _pad_1d_float(fs['game_tag_max_idf'])
    dev_log_catalog_size_b   = _pad_1d_float(fs['game_dev_log_catalog_size'])

    # Item text (V6a) — frozen 768-d desc embeddings. RAW (pad row = zeros) feeds the
    # user-side text_cosine pool; the L2-normalized variant feeds the deep item_text_tower
    # (== CG's F.normalize(raw)) and the text_cosine candidate side. Same two-buffer split
    # as the tag matrices above; pre-normalize once so train/precompute don't re-run it.
    text_dim     = fs['game_text_matrix'].shape[1]
    text_matrix  = np.vstack([fs['game_text_matrix'].astype(np.float32),
                              np.zeros((1, text_dim), dtype=np.float32)])
    text_norm    = np.linalg.norm(text_matrix, ord=2, axis=-1, keepdims=True)
    text_norm    = np.where(text_norm > 0.0, text_norm, 1.0)
    text_matrix_l2 = (text_matrix / text_norm).astype(np.float32)

    return {
        'game_tag_matrix':        torch.from_numpy(tag_matrix),
        'game_tag_matrix_l2':     torch.from_numpy(tag_matrix_l2),
        'game_tag_binary':        torch.from_numpy(tag_binary),
        'game_tag_count':         torch.from_numpy(tag_count),
        'game_genre_matrix':      torch.from_numpy(genre_matrix),
        'game_genre_binary':      torch.from_numpy(genre_binary),
        'game_genre_count':       torch.from_numpy(genre_count),
        'game_year_idx':          torch.from_numpy(year_idx),
        'game_dev_idx':           torch.from_numpy(dev_idx),
        'game_price_idx':         torch.from_numpy(price_idx),
        'game_year_numeric':      torch.from_numpy(year_numeric_b),
        'game_median_log_hours':  torch.from_numpy(median_log_hours_b),
        'game_log_count':         torch.from_numpy(log_count_b),
        'game_sentiment':         torch.from_numpy(sentiment_b),
        'game_tag_binary_idf':        torch.from_numpy(tag_binary_idf_b),
        'game_tag_count_idf':         torch.from_numpy(tag_count_idf_b),
        'game_tag_mean_idf':          torch.from_numpy(tag_mean_idf_b),
        'game_tag_max_idf':           torch.from_numpy(tag_max_idf_b),
        'game_dev_log_catalog_size':  torch.from_numpy(dev_log_catalog_size_b),
        'game_text_matrix':           torch.from_numpy(text_matrix),
        'game_text_matrix_l2':        torch.from_numpy(text_matrix_l2),
    }


# ── Warm-start CG → Ranker ──────────────────────────────────────────────────

# CG state_dict keys → ranker state_dict keys. After copy, these become ranker-owned
# parameters that fine-tune during training. See plan §3 "Warm-start mapping".
_CG_TO_RANKER_KEY_MAP = {
    # Lookups
    'item_embedding_lookup.weight':         'item_id_lookup.weight',
    'developer_embedding_lookup.weight':    'developer_lookup.weight',
    'year_embedding_lookup.weight':         'year_lookup.weight',
    'price_embedding_lookup.weight':        'price_lookup.weight',
    # 1-layer item-side towers
    'item_embedding_tower.0.weight':        'item_id_tower.0.weight',
    'item_embedding_tower.0.bias':          'item_id_tower.0.bias',
    'item_genre_tower.0.weight':            'item_genre_tower.0.weight',
    'item_genre_tower.0.bias':              'item_genre_tower.0.bias',
    'developer_tower.0.weight':             'developer_tower.0.weight',
    'developer_tower.0.bias':               'developer_tower.0.bias',
    'year_embedding_tower.0.weight':        'year_tower.0.weight',
    'year_embedding_tower.0.bias':          'year_tower.0.bias',
    'price_embedding_tower.0.weight':       'price_tower.0.weight',
    'price_embedding_tower.0.bias':         'price_tower.0.bias',
    # 2-layer towers (both .0 and .2 Linear layers transfer)
    'item_tag_tower.0.weight':              'item_tag_tower.0.weight',
    'item_tag_tower.0.bias':                'item_tag_tower.0.bias',
    'item_tag_tower.2.weight':              'item_tag_tower.2.weight',
    'item_tag_tower.2.bias':                'item_tag_tower.2.bias',
    'user_tag_tower.0.weight':              'user_tag_tower.0.weight',
    'user_tag_tower.0.bias':                'user_tag_tower.0.bias',
    'user_tag_tower.2.weight':              'user_tag_tower.2.weight',
    'user_tag_tower.2.bias':                'user_tag_tower.2.bias',
    'user_genre_tower.0.weight':            'user_genre_tower.0.weight',
    'user_genre_tower.0.bias':              'user_genre_tower.0.bias',
    'user_genre_tower.2.weight':            'user_genre_tower.2.weight',
    'user_genre_tower.2.bias':              'user_genre_tower.2.bias',
    # Item text tower (V6a CG parity — both .0 and .2 Linear layers transfer)
    'item_text_tower.0.weight':             'item_text_tower.0.weight',
    'item_text_tower.0.bias':               'item_text_tower.0.bias',
    'item_text_tower.2.weight':             'item_text_tower.2.weight',
    'item_text_tower.2.bias':               'item_text_tower.2.bias',
}


def _resolve_cg_checkpoint(spec: str | None) -> str | None:
    """Resolve a glob/path/None to a concrete CG checkpoint path. None disables warm start."""
    if not spec:
        return None
    matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(
            f"warm_start_cg_checkpoint='{spec}' matched no file. "
            "Set to None to skip warm start."
        )
    return matches[-1]


def _warm_start_from_cg(ranker: WideDeepRanker, cg_checkpoint: str) -> None:
    """
    Initialize ranker parameters from a CG checkpoint. Copies parameter VALUES — the
    ranker still owns the params and they update during training. No runtime CG dependency.
    """
    cg_state     = torch.load(cg_checkpoint, weights_only=True, map_location='cpu')
    ranker_state = ranker.state_dict()

    transfer, missing_in_cg, shape_mismatch = {}, [], []
    for cg_key, ranker_key in _CG_TO_RANKER_KEY_MAP.items():
        if cg_key not in cg_state:
            missing_in_cg.append(cg_key)
            continue
        if ranker_key not in ranker_state:
            continue
        cg_t  = cg_state[cg_key]
        rnk_t = ranker_state[ranker_key]
        if cg_t.shape != rnk_t.shape:
            shape_mismatch.append((cg_key, tuple(cg_t.shape), tuple(rnk_t.shape)))
            continue
        transfer[ranker_key] = cg_t

    result = ranker.load_state_dict(transfer, strict=False)
    n_transferred = len(transfer)
    n_total       = len(ranker_state)

    print(f"Warm start from CG checkpoint: {os.path.basename(cg_checkpoint)}")
    print(f"  Transferred {n_transferred} parameter tensors out of {n_total} total ranker params:")
    for cg_key in sorted(_CG_TO_RANKER_KEY_MAP.keys()):
        if _CG_TO_RANKER_KEY_MAP[cg_key] in transfer:
            print(f"    ✓ {cg_key}")
    if missing_in_cg:
        print(f"  Missing from CG state_dict ({len(missing_in_cg)}): "
              f"{missing_in_cg[:3]}{'...' if len(missing_in_cg) > 3 else ''}")
    if shape_mismatch:
        print(f"  Shape mismatch — NOT transferred ({len(shape_mismatch)}):")
        for cg_key, cg_shape, rnk_shape in shape_mismatch:
            print(f"    {cg_key}: CG{cg_shape} vs ranker{rnk_shape}")
    if result.unexpected_keys:
        print(f"  Unexpected keys (mapping bug): {result.unexpected_keys}")


def build_ranker(config: dict, fs: dict) -> WideDeepRanker:
    bufs = _buffers_from_fs(fs)
    ranker = WideDeepRanker(
        n_games=fs['n_items'],
        n_genres=fs['n_genres'],
        n_tags=fs['n_tags'],
        n_developers=fs['n_developers'],
        n_years=fs['n_years'],
        n_price_buckets=fs['n_price_buckets'],
        game_tag_matrix=bufs['game_tag_matrix'],
        game_tag_matrix_l2=bufs['game_tag_matrix_l2'],
        game_tag_binary=bufs['game_tag_binary'],
        game_tag_count=bufs['game_tag_count'],
        game_genre_matrix=bufs['game_genre_matrix'],
        game_genre_binary=bufs['game_genre_binary'],
        game_genre_count=bufs['game_genre_count'],
        game_year_idx=bufs['game_year_idx'],
        game_dev_idx=bufs['game_dev_idx'],
        game_price_idx=bufs['game_price_idx'],
        # Bucket 5 — 4 new per-game scalar buffers (Z-scored at forward time).
        game_year_numeric=bufs['game_year_numeric'],
        game_median_log_hours=bufs['game_median_log_hours'],
        game_log_count=bufs['game_log_count'],
        game_sentiment=bufs['game_sentiment'],
        # Bucket 6 — 1 matrix + 4 scalar rarity buffers (Z-scored at forward time).
        game_tag_binary_idf=bufs['game_tag_binary_idf'],
        game_tag_count_idf=bufs['game_tag_count_idf'],
        game_tag_mean_idf=bufs['game_tag_mean_idf'],
        game_tag_max_idf=bufs['game_tag_max_idf'],
        game_dev_log_catalog_size=bufs['game_dev_log_catalog_size'],
        # Item text (V6a) — 2 buffers + dims for the deep item_text_tower / text_cosine.
        game_text_matrix=bufs['game_text_matrix'],
        game_text_matrix_l2=bufs['game_text_matrix_l2'],
        item_id_emb_dim=config['item_id_emb_dim'],
        item_genre_emb_dim=config['item_genre_emb_dim'],
        item_tag_emb_dim=config['item_tag_emb_dim'],
        developer_emb_dim=config['developer_emb_dim'],
        year_emb_dim=config['year_emb_dim'],
        price_emb_dim=config['price_emb_dim'],
        user_genre_emb_dim=config['user_genre_emb_dim'],
        user_tag_emb_dim=config['user_tag_emb_dim'],
        text_emb_dim=config['text_emb_dim'],
        item_tag_hidden=config['item_tag_hidden'],
        user_tag_hidden=config['user_tag_hidden'],
        user_genre_hidden=config['user_genre_hidden'],
        item_text_hidden=config['item_text_hidden'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        n_cross_features=config['n_cross_features'],
        n_wide_normalized=config.get('n_wide_normalized', 0),
    )

    cg_ckpt = _resolve_cg_checkpoint(config.get('warm_start_cg_checkpoint'))
    if cg_ckpt:
        _warm_start_from_cg(ranker, cg_ckpt)
        config['warm_start_cg_checkpoint_resolved'] = cg_ckpt    # persist for sidecar

    return ranker


# ── Wide-feature Z-score buffer populate (Bucket 5+ scalars) ────────────────

# Normalized cross-feature parquet column names. Order MUST match
# compute_cross_features column slots 10-23 — the model's wide_norm_mean[k] /
# wide_norm_std[k] line up with cross_features[..., 10 + k] at forward time.
# Buckets 5 (cols 10-14, 5 features) + 6 (cols 15-22, 8 features) + text_cosine
# (col 23) = 14 normalized columns. Cols 0-9 are bounded ([-1, 1] / [0, 1]) and
# pass through raw.
_WIDE_NORM_PARQUET_COLS = (
    # Bucket 5 — numeric matching (cols 10-14)
    'price_match_label',
    'era_gap_label',
    'playtime_cal_median_label',
    'popularity_match_label',
    'sentiment_match_label',
    # Bucket 6 — niche feature crosses (cols 15-22)
    'tag_overlap_idf_full_label',
    'tag_overlap_idf_liked_label',
    'niche_tag_match_full_label',
    'niche_tag_match_liked_label',
    'max_tag_idf_match_full_label',
    'max_tag_idf_match_liked_label',
    'niche_dev_match_full_label',
    'niche_dev_match_liked_label',
    # Item-text integration (col 23)
    'text_cosine_label',
)


def populate_wide_norm_buffers(model: WideDeepRanker, train_parquet_path: str) -> None:
    """One-time pre-train pass: populate model.wide_norm_mean / wide_norm_std from
    train-parquet statistics on the Bucket 5 + Bucket 6 columns.

    Computes mean/std across the LABEL column for each of the 13 features (using
    label values rather than neg values is fine — both are drawn from the same
    candidate distribution, and using labels keeps the stats independent of the
    99-hard-neg pool composition).

    Stats are persistent buffers (saved in `state_dict`) so they ride along with
    the checkpoint and don't need recomputation on load.
    """
    if model.n_wide_normalized == 0:
        return
    n = model.n_wide_normalized
    assert len(_WIDE_NORM_PARQUET_COLS) == n, (
        f"wide-norm column list ({len(_WIDE_NORM_PARQUET_COLS)}) "
        f"out of sync with model.n_wide_normalized ({n})"
    )

    import pandas as pd
    print(f"Populating wide-feature Z-score buffers from {train_parquet_path} ...")
    df = pd.read_parquet(train_parquet_path, columns=list(_WIDE_NORM_PARQUET_COLS))

    means = np.zeros(n, dtype=np.float32)
    stds  = np.ones (n, dtype=np.float32)
    for k, col in enumerate(_WIDE_NORM_PARQUET_COLS):
        values   = df[col].values.astype(np.float64)         # float64 for numeric stability
        means[k] = float(values.mean())
        # Guard against zero variance (e.g. all rows have the same value): keep
        # std = 1 in that case so the Z-score reduces to centering. Avoids div-by-zero
        # silently producing NaN cross features on the forward path.
        s        = float(values.std())
        stds[k]  = s if s > 1e-8 else 1.0
        print(f"  col {10 + k} {col:<32}  mean={means[k]:+.4f}  std={stds[k]:.4f}")

    target_dtype = model.wide_norm_mean.dtype
    target_device = model.wide_norm_mean.device
    model.wide_norm_mean.copy_(torch.from_numpy(means).to(target_device, target_dtype))
    model.wide_norm_std .copy_(torch.from_numpy(stds ).to(target_device, target_dtype))


# ── Training loop ────────────────────────────────────────────────────────────

def train(checkpoint_dir: str | None = None) -> str:
    config = get_config()
    if checkpoint_dir:
        config['checkpoint_dir'] = checkpoint_dir

    print("Loading datasets ...")
    # train_ds skips eval-only columns (cross features + neg pool + cg_label_rank,
    # ~40 GB saved) because sample_batch with n_hard_negs=0 only reads label_idx +
    # user history + X_user_b5. The val_ds keeps mode='full' since training-time
    # NDCG/MRR eval (evaluate.compute_label_ranks) reads the cross-feature columns.
    # If the active config wants hard negs, fall back to full mode so sample_batch
    # can read dataset.neg_idx.
    train_mode = 'train_only' if int(config['n_hard_negs']) == 0 else 'full'
    train_ds, val_ds, fs = load_splits('data', train_mode=train_mode)

    device = get_device()
    print(f"\nDevice: {device}")
    train_ds.to(device)
    val_ds.to(device)

    # Deterministic eval sample for training-time logging — fixed seed so the same val
    # rows are scored at every log step → NDCG curves are directly comparable over time.
    n_eval_samples = config.get('n_eval_samples')
    if n_eval_samples is None or n_eval_samples >= val_ds.N:
        eval_indices = None
        print(f"Train-time eval: full val set ({val_ds.N:,} rows)")
    else:
        eval_rng     = np.random.default_rng(config['seed'] + 7)
        eval_indices = eval_rng.choice(val_ds.N, size=n_eval_samples, replace=False).astype(np.int64)
        eval_indices.sort()
        print(f"Train-time eval: deterministic {n_eval_samples:,}-row sample of {val_ds.N:,} val rows")

    cg_ndcg, cg_mrr = cg_baseline(val_ds, eval_indices=eval_indices)
    print(f"CG baseline (sample): NDCG@10={cg_ndcg:.4f}  MRR={cg_mrr:.4f}")
    print(f"Ranker target:        beat NDCG@10={cg_ndcg:.4f}\n")

    n_cross = config['n_cross_features']
    model = build_ranker(config, fs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("WideDeepRanker:")
    print(f"  user_concat({model.user_concat_dim}) + item_concat({model.item_concat_dim}) "
          f"= deep_in({model.deep_in})")
    print(f"  → hidden={config['hidden_dims']} → head({config['hidden_dims'][-1]}+{n_cross}→1)")
    print(f"  cross features ({n_cross}): tag_cosine | "
          f"genre/tag/dev overlap (full | liked | recent-3-liked) | "
          f"price/era/median-playtime/popularity/sentiment match (Z-scored) | "
          f"tag IDF overlap + niche scalar triple × (full | liked) (Z-scored) | "
          f"text_cosine (Z-scored)")
    print(f"  ({n_params:,} trainable params)")

    # Bucket 5 — populate Z-score buffers from train-parquet stats. Single pass over
    # 5 columns (~5-10 sec). Persistent buffers ride along with the checkpoint, so
    # evaluate_only / canary don't need to re-run this.
    populate_wide_norm_buffers(model, os.path.join('data', 'ranker_candidates_train.parquet'))

    # ── Menon α popularity bias (CLOSED in default Phase 1; see plan §8) ────
    pop_alpha = float(config['popularity_alpha'])
    if pop_alpha > 0:
        counts          = torch.from_numpy(fs['game_interaction_counts']).to(device)
        popularity_bias = pop_alpha * torch.log1p(counts)
        print(f"Menon α: alpha={pop_alpha}  bias range="
              f"[{popularity_bias.min():.3f}, {popularity_bias.max():.3f}]  "
              f"mean={popularity_bias.mean():.3f}")
    else:
        popularity_bias = None
        print("Menon α: DISABLED (alpha=0)")

    n_hard_take_for_log = max(0, min(int(config['n_hard_negs']), CANDIDATES_PER_ROW - 1))
    print(f"Loss: sampled softmax CE over (1 label + {n_hard_take_for_log} hard + "
          f"{config['n_random_negs']} random) = "
          f"{1 + n_hard_take_for_log + config['n_random_negs']} cands/row")

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'],
                                  eps=config['adam_eps'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training_steps'], eta_min=1e-4,    # matches CG (src/train.py)
    )

    rng     = np.random.default_rng(config['seed'])
    val_rng = np.random.default_rng(config['seed'] + 1)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_tag = (str(pop_alpha).replace('.', '')
                 if pop_alpha != int(pop_alpha) else str(int(pop_alpha)))
    best_path = os.path.join(config['checkpoint_dir'],
                              f'ranker_wd_alpha_{alpha_tag}_{run_ts}.pth')

    best_ndcg = -1.0
    loss_buf, grad_buf, gap_buf = [], [], []

    n_random_negs = int(config['n_random_negs'])
    n_hard_negs   = max(0, min(int(config['n_hard_negs']), train_ds.n_neg))
    n_total_cands = 1 + n_hard_negs + n_random_negs
    temperature   = float(config['temperature'])

    print(f"\nStarting training loop ({config['training_steps']:,} steps, "
          f"batch={config['batch_size']} × {n_total_cands} cands "
          f"(1 label + {n_hard_negs} hard + {n_random_negs} random), "
          f"lr={config['lr']}) ...\n")
    start = time.time()

    n_items_int   = int(model.game_pad_idx)
    all_item_ids  = torch.arange(n_items_int, device=device)

    pad_idx_int = int(model.game_pad_idx)

    # Bundle the 6 per-item buffers used by every history-slice triple-overlap
    # compute. Built once outside _forward_batch — never changes during training.
    # Same bundle precompute uses, so the train-time and parquet-write call sites
    # are bit-for-bit identical by construction.
    buffers = OverlapBuffers(
        genre_binary=model.game_genre_binary,
        genre_count =model.game_genre_count,
        tag_binary  =model.game_tag_binary,
        tag_count   =model.game_tag_count,
        game_dev_idx=model.game_dev_idx,
        dev_pad_idx =int(model.dev_pad_idx),
    )

    # Bucket 5 — bundle the 5 per-item numeric-match buffers. Same bundle precompute
    # builds locally on device, so on-the-fly compute for sampled random negs matches
    # parquet values bit-exactly. `price_bucket` is cast to float once here (cheap,
    # one tensor per training run) — saves a per-step cast inside numeric_match_quintuple.
    b5_buffers = Bucket5GameBuffers(
        price_bucket    =model.game_price_idx.float(),
        year_numeric    =model.game_year_numeric,
        median_log_hours=model.game_median_log_hours,
        log_count       =model.game_log_count,
        sentiment       =model.game_sentiment,
    )

    # Bucket 6 — bundle the 5 per-item rarity buffers. Same shape precompute builds
    # device-side; identical utils (`weighted_overlap` with IDF buffers + `niche_scalar_triple`)
    # called from both sites so parquet values match on-the-fly compute bit-exactly.
    niche_buffers = NicheBuffers(
        tag_binary_idf       =model.game_tag_binary_idf,
        tag_count_idf        =model.game_tag_count_idf,
        tag_mean_idf         =model.game_tag_mean_idf,
        tag_max_idf          =model.game_tag_max_idf,
        dev_log_catalog_size =model.game_dev_log_catalog_size,
    )

    def _forward_batch(batch):
        """Sampled-softmax forward. Returns (scores (B, n_cand), target (B,), cand_b (B, n_cand))."""
        (x_avg, h_lkd, h_lkd_pw, h_dis, h_full, h_pw, user_b5, cand_b, target_b) = batch
        B, n_cand = cand_b.shape

        # User-side: ONE pass per row.
        user_concat = model.user_forward(x_avg, h_lkd, h_dis, h_full, h_pw)

        # Item-side: embed the ENTIRE corpus once per step, gather for this batch.
        # Math identity (autograd sums grads correctly on repeated indices). The win:
        # 1099 cands × 512 rows = 562k draws from 5,437 items → ~100 duplicates per item
        # on average. Embedding once + gather is ~100× cheaper than embedding each draw,
        # which dominates the per-step compute at large N.
        all_item_concat = model.item_embedding(all_item_ids)                       # (n_items, I)
        item_concat     = all_item_concat[cand_b.reshape(-1)]                      # (B*n_cand, I)

        # Cross features on the fly for the sampled candidates. All 23 match
        # precompute exactly — categorical_overlap_triple + last_n_history +
        # numeric_match_quintuple + weighted_overlap + niche_scalar_triple are the same
        # utils called from precompute, so parquet values and train-time values are
        # bit-identical by construction.
        #
        # Strategy: compute the full (B, n_items+1) user-vs-corpus score matrix once
        # per feature via a dense matmul, then gather per (b, c). Cheaper than 3D
        # gathers over (B, n_cand, n_feat) buffers — MPS handles dense matmuls much
        # better. Pad rows in all buffers are zero → harmless; cand_b never references them.

        # ── tag_cosine (B-0): playtime-weighted user pool over raw TF-IDF, cosine ─
        user_tag_pool = (model.game_tag_matrix[h_full] * h_pw.unsqueeze(-1)).sum(dim=1)  # (B, n_tags)
        user_tag_norm = F.normalize(user_tag_pool, p=2, dim=1)
        full_tag_cos  = user_tag_norm @ model.game_tag_matrix_l2.t()                     # (B, n_items+1)
        tag_cos       = full_tag_cos.gather(1, cand_b)                                   # (B, n_cand)

        # ── text_cosine (B-10): playtime-weighted user TEXT centroid, cosine ─────
        # Identical shape to tag_cosine — sum RAW 768-d rows weighted, L2-normalize the
        # pool, dot with the pre-normalized candidate text rows. Direct text-vs-text path.
        user_text_pool = (model.game_text_matrix[h_full] * h_pw.unsqueeze(-1)).sum(dim=1)  # (B, 768)
        user_text_norm = F.normalize(user_text_pool, p=2, dim=1)
        full_text_cos  = user_text_norm @ model.game_text_matrix_l2.t()                    # (B, n_items+1)
        text_cos       = full_text_cos.gather(1, cand_b)                                   # (B, n_cand)

        # ── Buckets 1+2 — one categorical_overlap_triple call per history slice ───
        # Bucket 1 (full):       (h_full,    h_pw)
        # Bucket 2A (liked):     (h_lkd,     h_lkd_pw)
        # Bucket 2B (recent-3):  last_n_history(h_lkd, h_lkd_pw, n=3) — re-normalized
        genre_ov,    tag_ov,    dev_aff    = categorical_overlap_triple(buffers,
                                                history_indices=h_full,    history_weights=h_pw,        cand_idx=cand_b)
        genre_ov_l,  tag_ov_l,  dev_aff_l  = categorical_overlap_triple(buffers,
                                                history_indices=h_lkd,     history_weights=h_lkd_pw,    cand_idx=cand_b)
        h_recent3, h_recent3_pw            = last_n_history(h_lkd, h_lkd_pw, n=3, pad_idx=pad_idx_int)
        genre_ov_r3, tag_ov_r3, dev_aff_r3 = categorical_overlap_triple(buffers,
                                                history_indices=h_recent3, history_weights=h_recent3_pw, cand_idx=cand_b)

        # ── Bucket 5 — numeric-match quintuple (RAW values; Z-scored inside the model) ──
        price_match, era_gap, ptcal_med, pop_match, sent_match = numeric_match_quintuple(
            b5_buffers, user_b5, cand_b)

        # ── Bucket 6 — niche feature crosses (8 features = 4 concepts × {full, liked}) ──
        # Shape A: IDF-reweighted tag overlap on each slice (reuse `weighted_overlap`
        # with NicheBuffers' IDF-weighted tag matrix + matching count buffer).
        # Shape B: 3-scalar niche cross (`niche_scalar_triple`) per slice. RAW values;
        # the model Z-scores cols 15-22 at forward time.
        tag_ov_idf_full = weighted_overlap(niche_buffers.tag_binary_idf,
                                            niche_buffers.tag_count_idf,
                                            history_indices=h_full, history_weights=h_pw,
                                            cand_idx=cand_b)
        tag_ov_idf_liked = weighted_overlap(niche_buffers.tag_binary_idf,
                                             niche_buffers.tag_count_idf,
                                             history_indices=h_lkd, history_weights=h_lkd_pw,
                                             cand_idx=cand_b)
        niche_tag_f, max_tag_idf_f, niche_dev_f = niche_scalar_triple(niche_buffers,
                                            history_indices=h_full, history_weights=h_pw,
                                            cand_idx=cand_b)
        niche_tag_l, max_tag_idf_l, niche_dev_l = niche_scalar_triple(niche_buffers,
                                            history_indices=h_lkd, history_weights=h_lkd_pw,
                                            cand_idx=cand_b)

        cross = compute_cross_features(
            tag_cos.reshape(-1),
            genre_ov.reshape(-1),    tag_ov.reshape(-1),    dev_aff.reshape(-1),
            genre_ov_l.reshape(-1),  tag_ov_l.reshape(-1),  dev_aff_l.reshape(-1),
            genre_ov_r3.reshape(-1), tag_ov_r3.reshape(-1), dev_aff_r3.reshape(-1),
            price_match.reshape(-1), era_gap.reshape(-1),   ptcal_med.reshape(-1),
            pop_match.reshape(-1),   sent_match.reshape(-1),
            tag_ov_idf_full.reshape(-1),  tag_ov_idf_liked.reshape(-1),
            niche_tag_f.reshape(-1),      niche_tag_l.reshape(-1),
            max_tag_idf_f.reshape(-1),    max_tag_idf_l.reshape(-1),
            niche_dev_f.reshape(-1),      niche_dev_l.reshape(-1),
            text_cos.reshape(-1),
        )
        # score_pairs_batched factorizes the first MLP layer over the (B, n_cand) layout
        # — runs user-side projection on B rows (not B*n_cand) and avoids materializing
        # a (B*n_cand, U) user replica. Math identity; output matches score_pairs to ~1e-6.
        scores = model.score_pairs_batched(user_concat, item_concat, cross, n_cand)
        return scores, target_b, cand_b

    from tqdm import tqdm
    pbar = tqdm(range(config['training_steps']), desc="Training")
    for step in pbar:
        model.train()
        batch = sample_batch(train_ds, config['batch_size'], device, rng,
                              n_random_negs=n_random_negs,
                              n_hard_negs=n_hard_negs)
        scores, target_b, cand_b = _forward_batch(batch)

        # Temperature scaling matches CG: (scores / T) sharpens softmax by 1/T.
        # Apply BEFORE adding popularity_bias (matches src/train.py: scores/T + bias).
        scores = scores / temperature
        if popularity_bias is not None:
            scores = scores + popularity_bias[cand_b]    # (B, n_cand)

        loss = F.cross_entropy(scores, target_b)

        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip']).item()
        optimizer.step()
        scheduler.step()

        loss_buf.append(loss.item())
        grad_buf.append(gn)
        # Diagnostic: mean(label_score - mean_neg_score). Label is at column 0 by construction.
        with torch.no_grad():
            gap = (scores[:, 0] - scores[:, 1:].mean(dim=1)).mean().item()
        gap_buf.append(gap)

        if step > 0 and step % config['log_every'] == 0:
            elapsed = time.time() - start
            start   = time.time()
            ndcg, mrr = evaluate_ndcg_mrr(model, val_ds, device, eval_indices=eval_indices)
            avg_loss = float(np.mean(loss_buf[-config['log_every']:]))
            avg_gn   = float(np.mean(grad_buf[-config['log_every']:]))
            avg_gap  = float(np.mean(gap_buf[-config['log_every']:]))
            cur_lr   = scheduler.get_last_lr()[0]

            # Val loss: sampled-softmax CE over 8 val batches.
            model.eval()
            with torch.no_grad():
                vl_buf = []
                for _ in range(8):
                    vbatch = sample_batch(val_ds, config['batch_size'], device, val_rng,
                                           n_random_negs=n_random_negs,
                                           n_hard_negs=n_hard_negs)
                    vscores, vtarget, _ = _forward_batch(vbatch)
                    vscores = vscores / temperature      # mirror train-side scaling for comparable val_loss
                    vl_buf.append(F.cross_entropy(vscores, vtarget).item())
            val_loss = float(np.mean(vl_buf))

            improved = ndcg > best_ndcg
            tag      = " ← new best" if improved else ""
            print(f"[{step:06d}]  loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"grad={avg_gn:.3f}  label_gap={avg_gap:+.3f}  lr={cur_lr:.5f}  "
                  f"val NDCG@10={ndcg:.4f}  MRR={mrr:.4f}  "
                  f"(CG: {cg_ndcg:.4f}/{cg_mrr:.4f})  ({elapsed:.0f}s){tag}")
            pbar.set_postfix(loss=f"{avg_loss:.4f}", val_loss=f"{val_loss:.4f}",
                             val_ndcg=f"{ndcg:.4f}")
            if improved:
                best_ndcg = ndcg
                torch.save(model.state_dict(), best_path)
                _save_config(config, best_path)

    # Final eval on the same sampled set — last log point on the training curve.
    final_ndcg, final_mrr = evaluate_ndcg_mrr(model, val_ds, device, eval_indices=eval_indices)
    if final_ndcg > best_ndcg:
        best_ndcg = final_ndcg
        torch.save(model.state_dict(), best_path)
        _save_config(config, best_path)
        print(f"[final · sampled]  val NDCG@10={final_ndcg:.4f}  MRR={final_mrr:.4f}  ← new best")

    print("\nTraining complete.")
    print(f"  Best sampled val NDCG@10: {best_ndcg:.4f}  (CG sampled: {cg_ndcg:.4f}, "
          f"delta: {best_ndcg - cg_ndcg:+.4f})")
    print(f"  Checkpoint:               {best_path}")
    print("  Run `python ranker/main.py evaluate` for full-val metrics on the saved checkpoint.")
    return best_path


# ── Eval-only ───────────────────────────────────────────────────────────────

def evaluate_only(checkpoint_path: str | None = None) -> str:
    import io as _io

    if checkpoint_path is None:
        matches = glob.glob('saved_models/ranker/ranker_wd_*.pth')
        if not matches:
            raise FileNotFoundError("No ranker checkpoint found in saved_models/ranker/")
        checkpoint_path = max(matches, key=os.path.getmtime)

    print("Loading datasets ...")
    # evaluate_only touches only val_ds — skip the train parquet entirely (peak
    # ~1.5 GB vs 60+ GB; train_ds in full mode would pull all cross features for
    # 4.3M rows that this path never reads).
    _, val_ds, fs = load_splits('data', train_mode='skip')
    device = get_device()
    print(f"Device: {device}")
    val_ds.to(device)

    config = get_config()
    cfg_path = _config_path(checkpoint_path)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k in ('hidden_dims', 'dropout', 'n_cross_features', 'n_wide_normalized',
                  'item_id_emb_dim', 'item_genre_emb_dim', 'item_tag_emb_dim',
                  'developer_emb_dim', 'year_emb_dim', 'price_emb_dim',
                  'user_genre_emb_dim', 'user_tag_emb_dim',
                  'item_tag_hidden', 'user_tag_hidden', 'user_genre_hidden',
                  'popularity_alpha', 'n_random_negs', 'n_hard_negs',
                  'warm_start_cg_checkpoint',
                  'warm_start_cg_checkpoint_resolved'):
            if k in saved:
                config[k] = saved[k]

    # Disable warm-start during eval-only construction — we're loading the trained checkpoint.
    config_for_build = dict(config)
    config_for_build['warm_start_cg_checkpoint'] = None
    model = build_ranker(config_for_build, fs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  ranker popularity_alpha = {config.get('popularity_alpha', 'unknown')}")
    if 'warm_start_cg_checkpoint_resolved' in config:
        print(f"  warm-started from:       {config['warm_start_cg_checkpoint_resolved']}")

    n_cands = 1 + val_ds.n_neg

    raw_cg_ranks = val_ds.cg_label_rank
    cg_ranks     = np.where(raw_cg_ranks < n_cands, raw_cg_ranks, n_cands + 1)
    ranker_ranks = compute_label_ranks(model, val_ds, device)

    cg_ndcg10, cg_mrr = _ndcg_mrr_from_ranks(cg_ranks)
    rk_ndcg10, rk_mrr = _ndcg_mrr_from_ranks(ranker_ranks)
    cg_hits  = hit_rates_from_ranks(cg_ranks)
    rk_hits  = hit_rates_from_ranks(ranker_ranks)
    cg_ndcgK = ndcg_at_k_from_ranks(cg_ranks)
    rk_ndcgK = ndcg_at_k_from_ranks(ranker_ranks)
    cg_recall_at_pool = float((raw_cg_ranks < n_cands).mean())

    buf = _io.StringIO()

    def emit(line: str = ''):
        print(line)
        buf.write(line + '\n')

    emit(f"Checkpoint: {checkpoint_path}")
    emit(f"Val rows: {len(raw_cg_ranks):,}  |  Pool size: {n_cands}")
    emit(f"Ranker α: {config.get('popularity_alpha', 0.0)}  |  CG comparator: "
          f"{config.get('warm_start_cg_checkpoint_resolved', 'unknown')}")
    emit('')
    emit("══ Production-realistic metrics (E2E ceiling applied to both) ═════════")
    emit(f"{'Metric':<14} {'CG':>10} {'Ranker':>10} {'Delta':>12}")
    emit("─" * 50)
    for k_name in cg_hits:
        emit(f"{k_name:<14} {cg_hits[k_name]:>10.4f} {rk_hits[k_name]:>10.4f} "
              f"{rk_hits[k_name] - cg_hits[k_name]:>+12.4f}")
    emit('')
    for k_name in cg_ndcgK:
        emit(f"{k_name:<14} {cg_ndcgK[k_name]:>10.4f} {rk_ndcgK[k_name]:>10.4f} "
              f"{rk_ndcgK[k_name] - cg_ndcgK[k_name]:>+12.4f}")
    emit('')
    emit(f"{'MRR':<14} {cg_mrr:>10.4f} {rk_mrr:>10.4f} {rk_mrr - cg_mrr:>+12.4f}")
    emit("─" * 50)
    emit(f"{'Recall@'+str(n_cands):<14} {cg_recall_at_pool:>10.4f} {'(ceiling)':>10}   "
          "← max ranker Hit@K in production")

    # Pure reranking quality on the CG-retrieved subset.
    retrieved_mask = raw_cg_ranks < n_cands
    n_retrieved    = int(retrieved_mask.sum())
    if n_retrieved > 0:
        cg_ranks_sub     = raw_cg_ranks[retrieved_mask]
        ranker_ranks_sub = ranker_ranks[retrieved_mask]
        cg_ndcg10_sub, cg_mrr_sub = _ndcg_mrr_from_ranks(cg_ranks_sub)
        rk_ndcg10_sub, rk_mrr_sub = _ndcg_mrr_from_ranks(ranker_ranks_sub)
        cg_hits_sub = hit_rates_from_ranks(cg_ranks_sub)
        rk_hits_sub = hit_rates_from_ranks(ranker_ranks_sub)

        emit('')
        emit(f"══ Pure reranking quality (CG-retrieved subset, n={n_retrieved:,}) ═════")
        emit(f"{'Metric':<14} {'CG':>10} {'Ranker':>10} {'Delta':>12}")
        emit("─" * 50)
        for k_name in cg_hits_sub:
            emit(f"{k_name:<14} {cg_hits_sub[k_name]:>10.4f} {rk_hits_sub[k_name]:>10.4f} "
                  f"{rk_hits_sub[k_name] - cg_hits_sub[k_name]:>+12.4f}")
        emit('')
        emit(f"{'NDCG@10':<14} {cg_ndcg10_sub:>10.4f} {rk_ndcg10_sub:>10.4f} "
              f"{rk_ndcg10_sub - cg_ndcg10_sub:>+12.4f}")
        emit(f"{'MRR':<14} {cg_mrr_sub:>10.4f} {rk_mrr_sub:>10.4f} {rk_mrr_sub - cg_mrr_sub:>+12.4f}")
        emit("─" * 50)
        emit("(E2E ceiling not applied — both CG and ranker scored on labels CG retrieved)")

    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = 'ranker/eval_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{base}.txt')
    with open(out_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nWrote eval results → {out_path}")
    return out_path
