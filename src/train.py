"""
Training: Full softmax over the entire corpus.

Usage:
    python main.py train
"""
import json
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
        'weight_decay':     0.0,
        'adam_eps':         1e-6,
        'minibatch_size':   512,
        'popularity_alpha': 0.4,    # logit-space adjustment; 0=off. Uses log1p(count)
        'training_steps':   50_000,
        'log_every':        1_000,
        'checkpoint_every': 10_000,
        'checkpoint_dir':   'saved_models',
    }


def _config_path(checkpoint_path: str) -> str:
    return os.path.splitext(checkpoint_path)[0] + '_config.json'

def _save_config(config: dict, checkpoint_path: str) -> None:
    with open(_config_path(checkpoint_path), 'w') as f:
        json.dump(config, f, indent=2)

def load_config_for_checkpoint(checkpoint_path: str) -> dict:
    path = _config_path(checkpoint_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    cfg = get_config()
    cfg['popularity_alpha'] = 0.0  # safe: never apply unknown bias to an untagged checkpoint
    return cfg


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: dict) -> GameRecommender:
    n_games      = fs['n_items']
    n_tags       = fs['n_tags']
    n_genres     = fs['n_genres']
    n_developers = fs['n_developers']

    # 1. game_tag_matrix: (n_games+1, n_tags) — last row = zeros (padding)
    tag_matrix = fs['game_tag_matrix']                           # (n_games, n_tags) float32
    pad_tag    = np.zeros((1, n_tags), dtype=np.float32)
    game_tag_matrix = torch.from_numpy(np.vstack([tag_matrix, pad_tag]))

    # 2. game_genre_matrix: (n_games+1, n_genres) — last row = zeros (padding)
    genre_matrix = fs['game_genre_matrix']                        # (n_games, n_genres) float32
    pad_genre    = np.zeros((1, n_genres), dtype=np.float32)
    game_genre_matrix = torch.from_numpy(np.vstack([genre_matrix, pad_genre]))

    model = GameRecommender(
        n_genres=n_genres,
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
    # Load buffers
    model.game_tag_matrix.copy_(game_tag_matrix)
    model.game_genre_matrix.copy_(game_genre_matrix)
    return model


def print_model_summary(model: GameRecommender) -> None:
    # User side components
    item_id_dim = model.item_embedding_lookup.embedding_dim
    genre_dim   = model.user_genre_tower[-2].out_features
    tag_dim     = model.user_tag_tower[-2].out_features
    user_total  = (4 * item_id_dim) + genre_dim + tag_dim
    user_desc   = f"liked({item_id_dim}) + disliked({item_id_dim}) + full({item_id_dim}) + playtime({item_id_dim}) + genre({genre_dim}) + tag({tag_dim})"

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
    (X_avg_log_train, X_hist_liked_train, X_hist_disliked_train, X_hist_full_train,
     X_hist_playtime_weights_train,
     target_item_idx_train,
     target_dev_idx_train, target_year_idx_train, target_price_train) = train_data

    (X_avg_log_val, X_hist_liked_val, X_hist_disliked_val, X_hist_full_val,
     X_hist_playtime_weights_val,
     target_item_idx_val,
     target_dev_idx_val, target_year_idx_val, target_price_val) = val_data

    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available()          else
        torch.device('cpu')
    )
    print(f"  Device: {device}")
    model = model.to(device)

    # Move tensors to device
    X_avg_log_train       = X_avg_log_train.to(device)
    target_item_idx_train = target_item_idx_train.to(device)

    X_avg_log_val         = X_avg_log_val.to(device)
    target_item_idx_val   = target_item_idx_val.to(device)

    # Pre-compute all item metadata for full softmax
    print("Preparing full corpus item metadata ...")
    all_game_idxs = torch.arange(fs['n_items'], device=device)
    all_years     = torch.from_numpy(fs['game_year_idx']).to(device)
    all_devs      = torch.from_numpy(fs['game_developer_idx']).to(device)
    all_prices    = torch.from_numpy(fs['game_price_bucket']).to(device)

    print_model_summary(model)

    # ── Popularity logit adjustment (Menon et al. 2021) ───────────────────────
    # Subtract alpha * log1p(count_i) from item i's logit before softmax.
    item_counts    = torch.from_numpy(fs['game_interaction_counts'])
    popularity_bias = (config['popularity_alpha'] * torch.log1p(item_counts)).to(device)
    print(f"  Popularity bias: alpha={config['popularity_alpha']}  "
          f"max_adj={popularity_bias.max():.3f}  min_adj={popularity_bias.min():.3f}")

    pad_idx          = fs['n_items']
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'],
                                        eps=config['adam_eps'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=1e-4)
    minibatch_size   = config['minibatch_size']
    temperature      = 0.5 / minibatch_size
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_avg_log_train.shape[0]
    n_val   = X_avg_log_val.shape[0]

    # Fixed val index set — sampled once so val_loss is comparable across steps
    val_eval_size = min(8_192, n_val)
    rng_val = torch.Generator()
    rng_val.manual_seed(0)
    val_eval_idx = torch.randperm(n_val, generator=rng_val)[:val_eval_size].tolist()

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    arch_tag  = 'triple_full_softmax_popularity_alpha'
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
                V_all = model.item_embedding(all_years, all_game_idxs, all_devs, all_prices)
                val_losses = []
                for vs in range(0, val_eval_size, minibatch_size):
                    ve   = min(vs + minibatch_size, val_eval_size)
                    vidx = val_eval_idx[vs:ve]

                    # Histories are already tensors (padded in dataset.py)
                    v_liked    = X_hist_liked_val[vidx].to(device)
                    v_disliked = X_hist_disliked_val[vidx].to(device)
                    v_full     = X_hist_full_val[vidx].to(device)
                    v_pw       = X_hist_playtime_weights_val[vidx].to(device)

                    U      = model.user_embedding(X_avg_log_val[vidx], v_liked, v_disliked, v_full, v_pw)
                    scores = (U @ V_all.T) / temperature - popularity_bias
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
                _save_config(config, best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir, f'{arch_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                _save_config(config, periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            ix  = torch.randint(0, n_train, (minibatch_size,))

            # Histories are already tensors (padded in dataset.py)
            liked    = X_hist_liked_train[ix].to(device)
            disliked = X_hist_disliked_train[ix].to(device)
            full     = X_hist_full_train[ix].to(device)
            pw       = X_hist_playtime_weights_train[ix].to(device)

            # user_embedding(avg_log, liked, disliked, full, weights)
            U = model.user_embedding(X_avg_log_train[ix], liked, disliked, full, pw)
            
            # item_embedding(years, game_idxs, devs, prices)
            V_all = model.item_embedding(all_years, all_game_idxs, all_devs, all_prices)

            scores = (U @ V_all.T) / temperature - popularity_bias
            loss   = F.cross_entropy(scores, target_item_idx_train[ix].to(device))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            grad_norms.append(grad_norm)
            loss_train.append(loss.item())
            
            optimizer.step()
            scheduler.step()

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
