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


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_config() -> dict:
    item_id_embedding_size    = 40
    user_genre_embedding_size = 65   # sized so user_dim == item_dim (no timestamp tower)
    item_genre_embedding_size = 10
    tag_embedding_size        = 25
    developer_embedding_size  = 15
    item_year_embedding_size  = 10
    price_embedding_size      = 5

    user_dim = item_id_embedding_size + user_genre_embedding_size
    item_dim = (item_genre_embedding_size + tag_embedding_size
                + item_id_embedding_size + developer_embedding_size
                + item_year_embedding_size + price_embedding_size)
    assert user_dim == item_dim, (
        f"Tower size mismatch — user={user_dim} "
        f"(history={item_id_embedding_size} + genre={user_genre_embedding_size}), "
        f"item={item_dim} "
        f"(genre={item_genre_embedding_size} + tag={tag_embedding_size} "
        f"+ game={item_id_embedding_size} + dev={developer_embedding_size} "
        f"+ year={item_year_embedding_size} + price={price_embedding_size})"
    )

    return {
        'item_id_embedding_size':    item_id_embedding_size,
        'user_genre_embedding_size': user_genre_embedding_size,
        'item_genre_embedding_size': item_genre_embedding_size,
        'tag_embedding_size':        tag_embedding_size,
        'developer_embedding_size':  developer_embedding_size,
        'item_year_embedding_size':  item_year_embedding_size,
        'price_embedding_size':      price_embedding_size,
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
    dev_idx_arr = fs['game_developer_idx']                       # (n_games,) int64
    game_dev_idx = torch.from_numpy(
        np.append(dev_idx_arr, n_developers).astype(np.int64)
    )

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
    )
    return model


def print_model_summary(model: GameRecommender) -> None:
    history_dim  = model.item_embedding_lookup.embedding_dim
    genre_dim    = model.user_genre_tower[-2].out_features
    user_total   = history_dim + genre_dim

    item_genre_d = model.item_genre_tower[-2].out_features
    item_tag_d   = model.item_tag_tower[-2].out_features
    item_game_d  = model.item_embedding_tower[-2].out_features
    item_dev_d   = model.developer_tower[-2].out_features
    year_d       = model.year_embedding_tower[-2].out_features
    price_d      = model.price_embedding_tower[-2].out_features
    item_total   = item_genre_d + item_tag_d + item_game_d + item_dev_d + year_d + price_d

    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  history({history_dim}) + genre({genre_dim})  =  {user_total}")
    print(f"  Item side:  genre({item_genre_d}) + tag({item_tag_d}) + game({item_game_d})"
          f" + dev({item_dev_d}) + year({year_d}) + price({price_d})  =  {item_total}")
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
    best_path     = os.path.join(checkpoint_dir, f'best_softmax_{run_timestamp}.pth')

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
            model.eval()
            with torch.no_grad():
                vidx = torch.randint(0, n_val, (minibatch_size,)).tolist()
                vhp  = pad_history_batch([X_history_val[j]         for j in vidx], pad_idx).to(device)
                vwp  = pad_weights_batch([X_history_weights_val[j] for j in vidx]).to(device)
                U = model.user_embedding(X_genre_val[vidx], vhp, vwp)
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
                                        f'softmax_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            ix  = torch.randint(0, n_train, (minibatch_size,)).tolist()
            hp  = pad_history_batch([X_history_train[j]         for j in ix], pad_idx).to(device)
            wp  = pad_weights_batch([X_history_weights_train[j] for j in ix]).to(device)
            U   = model.user_embedding(X_genre_train[ix], hp, wp)
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
