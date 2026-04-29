"""
Training: Full softmax over the entire corpus.

Usage:
    python main.py train
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import pad_history_batch
from src.model import GameRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_config() -> dict:
    return {
        'item_id_embedding_size':    32,   # shared across all pools
        'user_genre_embedding_size': 32,
        'user_tag_embedding_size':   32,
        'item_genre_embedding_size': 8,
        'tag_embedding_size':        32,
        'developer_embedding_size':  12,
        'item_year_embedding_size':  8,
        'price_embedding_size':      4,
        'proj_hidden':               256,
        'output_dim':                128,
        # Training
        'lr':               0.001,
        'weight_decay':     1e-5,
        'minibatch_size':   512,
        'temperature':      0.05,
        'training_steps':   50_000,
        'log_every':        1_000,
        'checkpoint_every': 10_000,
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

    model = GameRecommender(
        n_genres=fs['n_genres'],
        n_tags=n_tags,
        n_games=n_games,
        n_years=fs['n_years'],
        n_developers=n_developers,
        n_price_buckets=fs['n_price_buckets'],
        item_id_embedding_size=config['item_id_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        user_tag_embedding_size=config['user_tag_embedding_size'],
        item_genre_embedding_size=config['item_genre_embedding_size'],
        tag_embedding_size=config['tag_embedding_size'],
        developer_embedding_size=config['developer_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        price_embedding_size=config['price_embedding_size'],
        proj_hidden=config['proj_hidden'],
        output_dim=config['output_dim'],
    )
    # Load tag matrix buffer
    model.game_tag_matrix.copy_(game_tag_matrix)
    return model


def print_model_summary(model: GameRecommender) -> None:
    # User side components
    item_id_dim = model.item_embedding_lookup.embedding_dim
    genre_dim   = model.user_genre_tower[-2].out_features
    tag_dim     = model.user_tag_tower[-2].out_features
    user_total  = (3 * item_id_dim) + genre_dim + tag_dim
    user_desc   = f"liked({item_id_dim}) + disliked({item_id_dim}) + full({item_id_dim}) + genre({genre_dim}) + tag({tag_dim})"

    # Item side components
    item_genre_dim = model.item_genre_tower[-2].out_features
    item_tag_dim   = model.item_tag_tower[-2].out_features
    item_id_tower  = model.item_embedding_tower[0].out_features
    dev_dim        = model.developer_tower[-2].out_features
    year_dim       = model.year_embedding_tower[-2].out_features
    price_dim      = model.price_embedding_tower[-2].out_features
    item_total     = item_genre_dim + item_tag_dim + item_id_tower + dev_dim + year_dim + price_dim

    proj_h   = model.user_projection[0].out_features
    out_dim  = model.output_dim
    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  {user_desc}  =  {user_total}")
    print(f"  Item side:  genre({item_genre_dim}) + tag({item_tag_dim}) + item_id({item_id_tower}) + dev({dev_dim}) + year({year_dim}) + price({price_dim})  =  {item_total}")
    print(f"  Projection: Linear({proj_h}) → ReLU → Linear({out_dim})  [both towers]")
    print(f"  Parameters: {n_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_softmax(model: GameRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: dict) -> str:
    """
    Full softmax training.
    """
    (X_genre_train, X_tag_train, X_hist_liked_train, X_hist_disliked_train, X_hist_full_train,
     target_item_idx_train, target_genre_train,
     target_dev_idx_train, target_year_idx_train, target_price_train) = train_data

    (X_genre_val, X_tag_val, X_hist_liked_val, X_hist_disliked_val, X_hist_full_val,
     target_item_idx_val, target_genre_val,
     target_dev_idx_val, target_year_idx_val, target_price_val) = val_data

    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available()          else
        torch.device('cpu')
    )
    print(f"  Device: {device}")
    model = model.to(device)

    # Move tensors to device
    X_genre_train         = X_genre_train.to(device)
    X_tag_train           = X_tag_train.to(device)
    target_item_idx_train = target_item_idx_train.to(device)

    X_genre_val           = X_genre_val.to(device)
    X_tag_val             = X_tag_val.to(device)
    target_item_idx_val   = target_item_idx_val.to(device)

    # Pre-compute all item metadata for full softmax
    print("Preparing full corpus item metadata ...")
    all_game_idxs = torch.arange(fs['n_items'], device=device)
    all_genres    = torch.from_numpy(fs['game_genre_matrix']).to(device)
    all_years     = torch.from_numpy(fs['game_year_idx']).to(device)
    all_devs      = torch.from_numpy(fs['game_developer_idx']).to(device)
    all_prices    = torch.from_numpy(fs['game_price_bucket']).to(device)

    print_model_summary(model)

    pad_idx          = fs['n_items']
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=1e-4)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    # Fixed val index set — sampled once so val_loss is comparable across steps
    val_eval_size = min(8_192, n_val)
    rng_val = torch.Generator()
    rng_val.manual_seed(0)
    val_eval_idx = torch.randperm(n_val, generator=rng_val)[:val_eval_size].tolist()

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    arch_tag  = 'triple_full_softmax'
    best_path = os.path.join(checkpoint_dir, f'best_{arch_tag}_{run_timestamp}.pth')

    loss_train = []
    grad_norms = []
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (V2 Softmax)")
    for i in pbar:
        is_log = (i % log_every == 0)

        if is_log:
            model.eval()
            with torch.no_grad():
                V_all = model.item_embedding(all_genres, all_years, all_game_idxs, all_devs, all_prices)
                val_losses = []
                for vs in range(0, val_eval_size, minibatch_size):
                    ve   = min(vs + minibatch_size, val_eval_size)
                    vidx = val_eval_idx[vs:ve]

                    v_liked    = pad_history_batch([X_hist_liked_val[j]    for j in vidx], pad_idx).to(device)
                    v_disliked = pad_history_batch([X_hist_disliked_val[j] for j in vidx], pad_idx).to(device)
                    v_full     = pad_history_batch([X_hist_full_val[j]     for j in vidx], pad_idx).to(device)

                    U      = model.user_embedding(X_genre_val[vidx], X_tag_val[vidx], v_liked, v_disliked, v_full)
                    scores = (U @ V_all.T) / temperature
                    val_losses.append(F.cross_entropy(scores, target_item_idx_val[vidx]).item())
                val_loss = float(np.mean(val_losses))

            avg_train     = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            avg_grad_norm = np.mean(grad_norms[i - log_every:i]) if i >= log_every else (grad_norms[-1] if grad_norms else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.6f}  grad_norm={avg_grad_norm:.2f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir, f'{arch_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            ix  = torch.randint(0, n_train, (minibatch_size,)).tolist()

            liked    = pad_history_batch([X_hist_liked_train[j]    for j in ix], pad_idx).to(device)
            disliked = pad_history_batch([X_hist_disliked_train[j] for j in ix], pad_idx).to(device)
            full     = pad_history_batch([X_hist_full_train[j]     for j in ix], pad_idx).to(device)

            U = model.user_embedding(X_genre_train[ix], X_tag_train[ix], liked, disliked, full)
            V_all = model.item_embedding(all_genres, all_years, all_game_idxs, all_devs, all_prices)

            scores = (U @ V_all.T) / temperature
            loss   = F.cross_entropy(scores, target_item_idx_train[ix])

            optimizer.zero_grad()
            loss.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())
            grad_norms.append(raw_norm)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
