"""
Training: in-batch negatives softmax (YouTube DNN stage-1 objective).

Usage:
    python main.py train
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import pad_history_batch, pad_weights_batch
from src.model import GameRecommender


CACHE_REFRESH_STEPS = 100   # rebuild frozen item embedding cache every N training steps


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_config() -> dict:
    # Sub-embedding sizes — intermediate features, not final representations.
    # Only item_id_embedding_size must match across towers (shared lookup).
    # user concat: item_id + user_genre = 32 + 32 = 64
    # item concat: item_genre + tag + item_id + dev + year + price = 8+16+32+12+8+4 = 80
    # Both project to output_dim via the projection MLP.
    return {
        'item_id_embedding_size':    32,   # shared: user history pool + item tower
        'user_genre_embedding_size': 32,   # user only
        'item_genre_embedding_size': 8,    # item only
        'tag_embedding_size':        16,   # item only (164 tags → MLP handles compression)
        'developer_embedding_size':  12,   # item only
        'item_year_embedding_size':  8,    # item only
        'price_embedding_size':      4,    # item only (9 buckets)
        # Projection MLP — learns cross-feature interactions after sub-embedding concat
        'proj_hidden': 256,
        'output_dim':  128,
        # ipool: pool full item_embedding() output (128-dim) instead of raw item_id (32-dim)
        # user concat: 128 + 32 = 160 → proj → 128  (vs gpool: 32 + 32 = 64 → proj → 128)
        'use_item_pool_for_history': True,
        # Experimental: freeze item tower outputs for history pooling (cache lookup instead of
        # full item tower forward passes). Tried — hurt Recall@10 0.3794→0.2931. Keep False.
        'freeze_item_embeddings': False,
        # Training
        'lr':               0.001,
        'weight_decay':     1e-5,
        'minibatch_size':   512,
        'temperature':      0.05,
        'training_steps':   150_000,
        'log_every':        10_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: dict) -> GameRecommender:
    n_games      = fs['n_items']
    n_tags       = fs['n_tags']
    n_developers = fs['n_developers']

    # game_tag_matrix: (n_games+1, n_tags) — last row = zeros (padding)
    tag_matrix = fs['game_tag_matrix']                           # (n_games, n_tags) float32
    pad_row    = np.zeros((1, n_tags), dtype=np.float32)
    game_tag_matrix = torch.from_numpy(np.vstack([tag_matrix, pad_row]))

    # game_dev_idx: (n_games+1,) — last entry = n_developers (padding index)
    dev_idx_arr  = fs['game_developer_idx']                      # (n_games,) int64
    game_dev_idx = torch.from_numpy(
        np.append(dev_idx_arr, n_developers).astype(np.int64)
    )

    use_ipool = config.get('use_item_pool_for_history', False)
    hist_genre_buf = hist_year_buf = hist_price_buf = None
    if use_ipool:
        # Non-persistent buffers indexed by game embedding index, pad row at index n_games.
        genre_mat      = torch.from_numpy(fs['game_genre_matrix'].astype(np.float32))
        hist_genre_buf = torch.cat([genre_mat,
                                    torch.zeros(1, genre_mat.shape[1])], dim=0)

        year_arr      = torch.from_numpy(fs['game_year_idx'].astype(np.int64))
        hist_year_buf = torch.cat([year_arr, torch.zeros(1, dtype=torch.long)], dim=0)

        price_arr      = torch.from_numpy(fs['game_price_bucket'].astype(np.int64))
        hist_price_buf = torch.cat([price_arr, torch.zeros(1, dtype=torch.long)], dim=0)

    model = GameRecommender(
        n_genres=fs['n_genres'],
        n_tags=n_tags,
        n_games=n_games,
        n_years=fs['n_years'],
        n_developers=n_developers,
        n_price_buckets=fs['n_price_buckets'],
        user_context_size=2 * fs['n_genres'],    # [debiased_avg_log | play_frac] per genre
        game_tag_matrix=game_tag_matrix,
        game_dev_idx=game_dev_idx,
        item_id_embedding_size=config['item_id_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        item_genre_embedding_size=config['item_genre_embedding_size'],
        tag_embedding_size=config['tag_embedding_size'],
        developer_embedding_size=config['developer_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        price_embedding_size=config['price_embedding_size'],
        proj_hidden=config['proj_hidden'],
        output_dim=config['output_dim'],
        use_item_pool_for_history=use_ipool,
        hist_genre_buf=hist_genre_buf,
        hist_year_buf=hist_year_buf,
        hist_price_buf=hist_price_buf,
    )
    return model


@torch.no_grad()
def _build_item_cache(model: GameRecommender, fs: dict, device) -> torch.Tensor:
    """Pre-compute item_embedding() for all corpus items. Returns (n_items+1, output_dim)."""
    n = fs['n_items']
    genre_t = torch.from_numpy(fs['game_genre_matrix'].astype(np.float32)).to(device)
    year_t  = torch.from_numpy(fs['game_year_idx'].astype(np.int64)).to(device)
    dev_t   = torch.from_numpy(fs['game_developer_idx'].astype(np.int64)).to(device)
    price_t = torch.from_numpy(fs['game_price_bucket'].astype(np.int64)).to(device)

    was_training = model.training
    model.eval()
    parts = []
    for s in range(0, n, 512):
        e   = min(s + 512, n)
        idx = torch.arange(s, e, device=device)
        parts.append(model.item_embedding(genre_t[s:e], year_t[s:e], idx, dev_t[s:e], price_t[s:e]))
    model.train(was_training)

    cache = torch.cat(parts, dim=0)                              # (n_items, output_dim)
    pad   = torch.zeros(1, model.output_dim, device=device)
    return torch.cat([cache, pad], dim=0)                        # (n_items+1, output_dim)


def print_model_summary(model: GameRecommender) -> None:
    if model.use_item_pool_for_history:
        history_dim = model.output_dim
        pool_label  = f"item_proj_pool({history_dim})"
    else:
        history_dim = model.item_embedding_lookup.embedding_dim
        pool_label  = f"id_pool({history_dim})"
    genre_dim    = model.user_genre_tower[-2].out_features
    user_concat  = history_dim + genre_dim

    item_genre_d = model.item_genre_tower[-2].out_features
    item_tag_d   = model.item_tag_tower[-2].out_features
    item_game_d  = model.item_embedding_tower[-2].out_features
    item_dev_d   = model.developer_tower[-2].out_features
    year_d       = model.year_embedding_tower[-2].out_features
    price_d      = model.price_embedding_tower[-2].out_features
    item_concat  = item_genre_d + item_tag_d + item_game_d + item_dev_d + year_d + price_d

    proj_hidden  = model.user_projection[0].out_features
    output_dim   = model.output_dim
    n_params     = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  {pool_label} + genre({genre_dim})  =  {user_concat}"
          f"  → proj({proj_hidden})  → {output_dim}")
    print(f"  Item side:  genre({item_genre_d}) + tag({item_tag_d}) + game({item_game_d})"
          f" + dev({item_dev_d}) + year({year_d}) + price({price_d})  =  {item_concat}"
          f"  → proj({proj_hidden})  → {output_dim}")
    print(f"  Parameters: {n_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_softmax(model: GameRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: dict) -> str:
    """
    In-batch negatives softmax training.

    train_data / val_data: 8-tuple from dataset.make_softmax_splits()
      [0] X_genre           (N, 2*n_genres) float32
      [1] X_history         list[list[int]]
      [2] X_history_weights list[list[float]]
      [3] target_item_idx   (N,) int64
      [4] target_genre      (N, n_genres) float32
      [5] target_dev_idx    (N,) int64
      [6] target_year_idx   (N,) int64
      [7] target_price      (N,) int64
    """
    (X_genre_train, X_history_train, X_history_weights_train,
     target_item_idx_train, target_genre_train,
     target_dev_idx_train, target_year_idx_train, target_price_train) = train_data

    (X_genre_val, X_history_val, X_history_weights_val,
     target_item_idx_val, target_genre_val,
     target_dev_idx_val, target_year_idx_val, target_price_val) = val_data

    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available()          else
        torch.device('cpu')
    )
    print(f"  Device: {device}")
    model = model.to(device)

    # Move pre-built tensors to device
    X_genre_train          = X_genre_train.to(device)
    target_item_idx_train  = target_item_idx_train.to(device)
    target_genre_train     = target_genre_train.to(device)
    target_dev_idx_train   = target_dev_idx_train.to(device)
    target_year_idx_train  = target_year_idx_train.to(device)
    target_price_train     = target_price_train.to(device)

    X_genre_val            = X_genre_val.to(device)
    target_item_idx_val    = target_item_idx_val.to(device)
    target_genre_val       = target_genre_val.to(device)
    target_dev_idx_val     = target_dev_idx_val.to(device)
    target_year_idx_val    = target_year_idx_val.to(device)
    target_price_val       = target_price_val.to(device)

    print_model_summary(model)

    item_cache = None
    if config.get('freeze_item_embeddings', False) and config.get('use_item_pool_for_history', False):
        print("  Building initial item embedding cache ...")
        item_cache = _build_item_cache(model, fs, device)
        print(f"  Item cache: {item_cache.shape}  ({item_cache.numel()*4/1e6:.1f} MB)")

    pad_idx          = fs['n_items']
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=0)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    arch_tag  = 'ipool_gpool_softmax' if config.get('use_item_pool_for_history') else 'proj_softmax'
    best_path = os.path.join(checkpoint_dir, f'best_{arch_tag}_{run_timestamp}.pth')

    loss_train = []

    print(f"\nStarting softmax training ({training_steps:,} steps, "
          f"batch={minibatch_size}, temp={temperature}) ...")
    print(f"  Train: {n_train:,} examples  |  Val: {n_val:,} examples")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (softmax)")
    for i in pbar:
        is_log = (i % log_every == 0)

        if is_log:
            if item_cache is not None:
                item_cache = _build_item_cache(model, fs, device)
            model.eval()
            with torch.no_grad():
                vidx = torch.randint(0, n_val, (minibatch_size,)).tolist()
                vhp  = pad_history_batch([X_history_val[j]         for j in vidx], pad_idx).to(device)
                vwp  = pad_weights_batch([X_history_weights_val[j] for j in vidx]).to(device)
                U = model.user_embedding(X_genre_val[vidx], vhp, vwp, item_cache=item_cache)
                V = model.item_embedding(target_genre_val[vidx], target_year_idx_val[vidx],
                                         target_item_idx_val[vidx], target_dev_idx_val[vidx],
                                         target_price_val[vidx])
                scores   = (U @ V.T) / temperature
                labels   = torch.arange(len(vidx), device=device)
                val_loss = F.cross_entropy(scores, labels).item()

                if i == 0:
                    raw = U @ V.T
                    print(f"  [init diagnostics] raw dot products — "
                          f"mean={raw.mean().item():.4f}  std={raw.std().item():.4f}  "
                          f"min={raw.min().item():.4f}  max={raw.max().item():.4f}")
                    print(f"  [init diagnostics] after /temp={temperature} — "
                          f"mean={scores.mean().item():.4f}  std={scores.std().item():.4f}")
                    print(f"  [init diagnostics] random baseline loss = {np.log(minibatch_size):.4f}")

            avg_train  = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train={avg_train:.4f}  val={val_loss:.4f}  lr={current_lr:.6f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'{arch_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            if item_cache is not None and i % CACHE_REFRESH_STEPS == 0:
                item_cache = _build_item_cache(model, fs, device)
            ix  = torch.randint(0, n_train, (minibatch_size,)).tolist()
            hp  = pad_history_batch([X_history_train[j]         for j in ix], pad_idx).to(device)
            wp  = pad_weights_batch([X_history_weights_train[j] for j in ix]).to(device)
            U   = model.user_embedding(X_genre_train[ix], hp, wp, item_cache=item_cache)
            V   = model.item_embedding(target_genre_train[ix], target_year_idx_train[ix],
                                       target_item_idx_train[ix], target_dev_idx_train[ix],
                                       target_price_train[ix])
            scores = (U @ V.T) / temperature
            labels = torch.arange(len(ix), device=device)
            loss   = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())

    print(f"\nSoftmax training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
