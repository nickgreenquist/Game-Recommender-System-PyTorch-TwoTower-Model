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
        'text_input_dim':            768,  # frozen bge-base-en-v1.5 dim — must match embed_text encoder
        'text_embedding_size':       32,   # item_text_tower output into the item concat
        'use_item_text':             True, # False = V5 control (no text tower); flip for the A/B baseline
        'proj_hidden':               256,
        'output_dim':                128,
        # Training
        'lr':               0.001,
        'weight_decay':     0.0,
        'adam_eps':         1e-6,
        'minibatch_size':   512,
        # Temperature is a pure hyperparameter for full softmax — do NOT compute from batch size.
        # (0.5 / minibatch_size = 0.000977 was used in PROD V4; lower = sharper distribution)
        'temperature':      0.1,
        'popularity_alpha': 0.0,    # logit-space adjustment; 0=off. Uses log1p(count)
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

    # 3. game_text_matrix: (n_games+1, text_dim) — last row = zeros (padding)
    text_matrix = fs['game_text_matrix']                          # (n_games, text_dim) float32
    pad_text    = np.zeros((1, text_matrix.shape[1]), dtype=np.float32)
    game_text_matrix = torch.from_numpy(np.vstack([text_matrix, pad_text]))

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
        text_input_dim=config.get('text_input_dim', 768),
        text_embedding_size=config.get('text_embedding_size', 32),
        # Default False so legacy checkpoints (no flag in their sidecar) build the exact
        # V5 tower; new training passes use_item_text explicitly via get_config().
        use_item_text=config.get('use_item_text', False),
        proj_hidden=config['proj_hidden'],
        output_dim=config['output_dim'],
    )
    # Load buffers (game_text_matrix only exists when the text tower is active)
    model.game_tag_matrix.copy_(game_tag_matrix)
    model.game_genre_matrix.copy_(game_genre_matrix)
    if model.use_item_text:
        model.game_text_matrix.copy_(game_text_matrix)
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
    item_desc      = (f"genre({item_genre_dim}) + tag({item_tag_dim}) + item_id({item_id_tower}) + "
                      f"dev({dev_dim}) + year({year_dim}) + price({price_dim})")
    item_total     = item_genre_dim + item_tag_dim + item_id_tower + dev_dim + year_dim + price_dim
    # Text tower is the V6a delta — fold it into the printed concat so the summary tells you
    # at a glance whether this run carries item-text embeddings (V6a) or not (V5 control).
    if model.use_item_text:
        text_dim    = model.item_text_tower[-2].out_features
        item_desc  += f" + text({text_dim})"
        item_total += text_dim

    proj_h   = model.user_projection[0].out_features
    out_dim  = model.output_dim
    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    variant = "V6a (item text tower ON)" if model.use_item_text else "V5 control (no text tower)"
    print(f"\n── Model dimensions ──  [{variant}]")
    print(f"  User side:  {user_desc}  =  {user_total}")
    print(f"  Item side:  {item_desc}  =  {item_total}")
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
    # Add alpha * log1p(count_i) to item i's logit before softmax (Menon Eq. 4).
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
    temperature     = config['temperature']
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
    best_ndcg10 = -1.0   # selection metric (see evaluate_val) — replaces val-CE selection
    alpha       = config['popularity_alpha']
    alpha_tag   = str(alpha).replace('.', '') if alpha != int(alpha) else str(int(alpha))
    # 'text_' marks the V6a item-text tower so its checkpoints are distinguishable from the
    # V5 / V5-control ones (which keep the legacy name). Removable with the use_item_text flag.
    text_tag    = 'text_' if config.get('use_item_text') else ''
    arch_tag    = f'triple_full_softmax_{text_tag}popularity_alpha_{alpha_tag}'
    best_path = os.path.join(checkpoint_dir, f'best_{arch_tag}_{run_timestamp}.pth')

    loss_train = []
    grad_norms = []
    start = time.time()

    def evaluate_val():
        """
        Score the fixed val set and return (val_loss, val_ndcg@10, val_mrr).

        We pick the 'best' checkpoint on val NDCG@10 — the ranking metric we actually
        ship and report — not on cross-entropy, which is only a surrogate (CE keeps
        falling via confidence calibration after the ranks have plateaued). Same idea
        as ranker.train. Both metrics come from the same full-corpus score matrix that
        the CE pass already builds, so this is nearly free.

        CE uses temperature-scaled scores (matches the train-side scaling so val_loss is
        comparable across steps); ranks use the RAW dot products — exactly the inference
        scoring (Menon Path 2, no popularity correction). Temperature is monotonic, so it
        would not change ranks anyway.
        """
        model.eval()
        with torch.no_grad():
            V_all      = model.item_embedding(all_years, all_game_idxs, all_devs, all_prices)
            val_losses = []
            ndcg10_sum = 0.0
            mrr_sum    = 0.0
            n_eval     = 0
            for vs in range(0, val_eval_size, minibatch_size):
                ve   = min(vs + minibatch_size, val_eval_size)
                vidx = val_eval_idx[vs:ve]

                v_liked    = X_hist_liked_val[vidx].to(device)
                v_disliked = X_hist_disliked_val[vidx].to(device)
                v_full     = X_hist_full_val[vidx].to(device)
                v_pw       = X_hist_playtime_weights_val[vidx].to(device)

                U      = model.user_embedding(X_avg_log_val[vidx], v_liked, v_disliked, v_full, v_pw)
                scores = U @ V_all.T                                   # (B, n_items) raw dot products
                tgt    = target_item_idx_val[vidx]                    # (B,)

                val_losses.append(F.cross_entropy(scores / temperature, tgt).item())

                tgt_scores  = scores.gather(1, tgt.unsqueeze(1))       # (B, 1)
                ranks       = (scores > tgt_scores).sum(dim=1) + 1     # (B,) 1-based rank of target
                mrr_sum    += (1.0 / ranks.float()).sum().item()
                in_top10    = ranks <= 10
                ndcg10_sum += (1.0 / torch.log2(ranks.float() + 1.0))[in_top10].sum().item()
                n_eval     += tgt.shape[0]

        return float(np.mean(val_losses)), ndcg10_sum / n_eval, mrr_sum / n_eval

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (full softmax)")
    for i in pbar:
        is_log = (i % log_every == 0)

        if is_log:
            val_loss, val_ndcg10, val_mrr = evaluate_val()

            avg_train     = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            avg_grad_norm = np.mean(grad_norms[i - log_every:i]) if i >= log_every else (grad_norms[-1] if grad_norms else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}", ndcg=f"{val_ndcg10:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
                  f"val_ndcg@10={val_ndcg10:.4f}  val_mrr={val_mrr:.4f}  "
                  f"lr={current_lr:.6f}  grad_norm={avg_grad_norm:.2f}  ({elapsed:.0f}s)")

            if val_ndcg10 > best_ndcg10:
                best_ndcg10 = val_ndcg10
                torch.save(model.state_dict(), best_path)
                _save_config(config, best_path)
                print(f"  → new best NDCG@10={best_ndcg10:.4f}  (val_loss={val_loss:.4f}) → {best_path}")

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

            scores = (U @ V_all.T) / temperature + popularity_bias
            loss   = F.cross_entropy(scores, target_item_idx_train[ix].to(device))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            grad_norms.append(grad_norm)
            loss_train.append(loss.item())
            
            optimizer.step()
            scheduler.step()

    # Final eval — the loop only scores on log steps, so the last training step's weights
    # may never be evaluated. Catch a final-step best (mirrors ranker.train).
    final_loss, final_ndcg10, final_mrr = evaluate_val()
    if final_ndcg10 > best_ndcg10:
        best_ndcg10 = final_ndcg10
        torch.save(model.state_dict(), best_path)
        _save_config(config, best_path)
        print(f"[final]  val NDCG@10={final_ndcg10:.4f}  val_mrr={final_mrr:.4f}  ← new best")

    print(f"\nTraining complete. Best Val NDCG@10: {best_ndcg10:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
