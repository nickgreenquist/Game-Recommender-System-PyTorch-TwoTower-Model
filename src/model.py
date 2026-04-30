"""
Two-Tower GameRecommender model V2.

Key changes:
- ReLU everywhere (replaces Tanh).
- Sum pooling for history (replaces weighted avg).
- Triple history pools: Liked, Disliked, Full (shared ID embedding).
- Shallow history pooling: sum raw 32-dim ID embeddings directly.
- User tag context tower.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GameRecommender(nn.Module):
    def __init__(self,
                 n_genres,
                 n_tags,
                 n_games,
                 n_years,
                 n_developers,
                 n_price_buckets,
                 item_id_embedding_size=32,
                 user_genre_embedding_size=32,
                 user_tag_embedding_size=32,
                 item_genre_embedding_size=8,
                 tag_embedding_size=16,
                 developer_embedding_size=12,
                 item_year_embedding_size=8,
                 price_embedding_size=4,
                 proj_hidden=256,
                 output_dim=128,
                 ):
        super().__init__()

        self.game_pad_idx = n_games
        self.dev_pad_idx  = n_developers
        self.output_dim   = output_dim

        # ── Shared item embedding lookup (used by item tower and all 4 user history pools) ──
        self.item_embedding_lookup = nn.Embedding(
            n_games + 1, item_id_embedding_size, padding_idx=n_games
        )

        # ── Item Tower sub-networks ──────────────────────────────────────────
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_id_embedding_size, item_id_embedding_size),
            nn.ReLU()
        )
        self.developer_embedding_lookup = nn.Embedding(
            n_developers + 1, developer_embedding_size, padding_idx=n_developers
        )
        self.developer_tower = nn.Sequential(
            nn.Linear(developer_embedding_size, developer_embedding_size),
            nn.ReLU()
        )
        tag_hidden = 128
        self.item_tag_tower = nn.Sequential(
            nn.Linear(n_tags, tag_hidden),
            nn.ReLU(),
            nn.Linear(tag_hidden, tag_embedding_size),
            nn.ReLU()
        )
        self.item_genre_tower = nn.Sequential(
            nn.Linear(n_genres, item_genre_embedding_size),
            nn.ReLU()
        )
        self.year_embedding_lookup = nn.Embedding(n_years, item_year_embedding_size)
        self.year_embedding_tower = nn.Sequential(
            nn.Linear(item_year_embedding_size, item_year_embedding_size),
            nn.ReLU()
        )
        self.price_embedding_lookup = nn.Embedding(n_price_buckets, price_embedding_size)
        self.price_embedding_tower = nn.Sequential(
            nn.Linear(price_embedding_size, price_embedding_size),
            nn.ReLU()
        )

        # ── User Tower context sub-networks ──────────────────────────────────
        genre_hidden = 128
        self.user_genre_tower = nn.Sequential(
            nn.Linear(2 * n_genres, genre_hidden),
            nn.ReLU(),
            nn.Linear(genre_hidden, user_genre_embedding_size),
            nn.ReLU()
        )
        tag_ctx_hidden = 256
        self.user_tag_tower = nn.Sequential(
            nn.Linear(n_tags, tag_ctx_hidden),
            nn.ReLU(),
            nn.Linear(tag_ctx_hidden, user_tag_embedding_size),
            nn.ReLU()
        )

        # ── Projection MLPs ──────────────────────────────────────────────────
        # User Concat:
        #   4 pools * 32-dim + genre context 32-dim + tag context 32-dim = 192-dim
        #   pools: liked, disliked, full (equal-weight), playtime-weighted full
        user_concat_dim = (4 * item_id_embedding_size +
                           user_genre_embedding_size +
                           user_tag_embedding_size)
        
        # Item Concat:
        #   genre(8)+tag(16)+id(32)+dev(12)+year(8)+price(4) = 80-dim
        item_concat_dim = (item_genre_embedding_size + tag_embedding_size + 
                           item_id_embedding_size + developer_embedding_size + 
                           item_year_embedding_size + price_embedding_size)

        self.user_projection = nn.Sequential(
            nn.Linear(user_concat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, output_dim),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(item_concat_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, output_dim),
        )

        # Buffers (stored in model but not trained)
        self.register_buffer('game_tag_matrix', torch.zeros(n_games + 1, n_tags))
        self.register_buffer('game_genre_matrix', torch.zeros(n_games + 1, n_genres))

        self.apply(self._init_weights)
        # Final projection layers need standard gain=1.0 to prevent signal vanishing
        for proj in [self.user_projection, self.item_projection]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Sub-tower gain=0.1 to keep activations small
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def user_embedding(self, X_user_avg_log, X_hist_liked, X_hist_disliked, X_hist_full,
                       X_hist_playtime_weights):
        """
        X_user_avg_log           (B, 1)   — per-user avg_log_playtime for debiasing
        X_hist_liked             (B, L)   — equal-weight sum pool
        X_hist_disliked          (B, D)   — equal-weight sum pool
        X_hist_full              (B, F)   — equal-weight sum pool
        X_hist_playtime_weights  (B, F)   — log(1+hours) weights normalized per user,
                                            same indices as X_hist_full
        """
        # Shallow equal-weight sum pooling (padding_idx zeros out pad entries)
        def pool_ids(ids):
            return self.item_embedding_lookup(ids).sum(dim=1)

        liked_emb    = pool_ids(X_hist_liked)
        disliked_emb = pool_ids(X_hist_disliked)
        full_emb     = pool_ids(X_hist_full)

        # Playtime-weighted sum pool: sum(item_emb * w) where w = log(1+h) / sum(log(1+h))
        # Weights are pre-normalized in dataset.py; padding positions have weight=0
        item_embs = self.item_embedding_lookup(X_hist_full)          # (B, F, D)
        w = X_hist_playtime_weights.unsqueeze(-1)                     # (B, F, 1)
        playtime_emb = (item_embs * w).sum(dim=1)                           # (B, D)

        # ── Dynamic Genre Context (In-Model Contextual Pooling) ───────────
        # Gather genre rows for history: (B, F, n_genres)
        hist_genres = self.game_genre_matrix[X_hist_full]
        
        # mask = (X_hist_full != self.game_pad_idx).float().unsqueeze(-1)
        running_genre_count = (hist_genres > 0).float().sum(dim=1)      # (B, n_genres)
        running_genre_sum   = (hist_genres * w).sum(dim=1)              # (B, n_genres)
        
        # Debiased Affinity: (B, n_genres)
        # Avoid division by zero: if count=0, affinity=0
        safe_count = torch.where(running_genre_count > 0, running_genre_count, torch.ones_like(running_genre_count))
        genre_affinity = torch.where(
            running_genre_count > 0,
            (running_sum_weighted := (hist_genres * w * 10.0).sum(dim=1)) / safe_count - X_user_avg_log, 
            torch.zeros_like(running_genre_count)
        )
        # NOTE: The dataset.py used 'running_genre_sum' which was raw_logs. 
        # Our 'w' here is normalized weights: w = raw_log / sum(raw_logs).
        # So sum(hist_genres * w) = sum(hist_genres * raw_log) / sum(raw_log).
        # To match the original logic: genre_ctx = (running_genre_sum / running_genre_count) - avg_log
        # We need the ACTUAL raw logs. But wait, we can just pass the raw logs instead of normalized weights!
        # Actually, let's keep the weights and just use them. 
        # Original: (running_genre_sum / running_genre_count) - avg_log
        # If we use normalized weights w_i = raw_log_i / total_log, then:
        # running_genre_sum = sum(raw_log_i * is_genre) = total_log * sum(w_i * is_genre)
        # So: (total_log * sum(w_i * is_genre) / running_genre_count) - avg_log
        # Since total_log = N * avg_log:
        # (N * avg_log * sum(w_i * is_genre) / running_genre_count) - avg_log
        # = avg_log * [ (N * sum(w_i * is_genre) / running_genre_count) - 1 ]

        N = (X_hist_full != self.game_pad_idx).float().sum(dim=1, keepdim=True) # (B, 1)
        safe_N = torch.where(N > 0, N, torch.ones_like(N))
        
        genre_sum_w = (hist_genres * w).sum(dim=1) # (B, n_genres)
        genre_affinity = torch.where(
            running_genre_count > 0,
            X_user_avg_log * ((N * genre_sum_w / safe_count) - 1.0),
            torch.zeros_like(running_genre_count)
        )

        total_assign = running_genre_count.sum(dim=1, keepdim=True) # (B, 1)
        safe_assign  = torch.where(total_assign > 0, total_assign, torch.ones_like(total_assign))
        genre_fractions = running_genre_count / safe_assign
        
        X_genre = torch.cat([genre_affinity, genre_fractions], dim=1) # (B, 2*n_genres)
        genre_emb = self.user_genre_tower(X_genre)

        # ── Dynamic Tag Context ───────────────────────────────────────────
        # Simple sum of tag vectors in history
        X_tag = (self.game_tag_matrix[X_hist_full]).sum(dim=1) # (B, n_tags)
        tag_emb   = self.user_tag_tower(X_tag)

        concat = torch.cat([liked_emb, disliked_emb, full_emb, playtime_emb, genre_emb, tag_emb], dim=1)
        return F.normalize(self.user_projection(concat), dim=-1)

    def item_embedding(self, target_year_idx, target_game_idx,
                       target_dev_idx, target_price):
        target_genre   = self.game_genre_matrix[target_game_idx]
        item_genre_emb = self.item_genre_tower(target_genre)
        item_tag_vec   = self.game_tag_matrix[target_game_idx]
        item_tag_emb   = self.item_tag_tower(item_tag_vec)
        item_emb       = self.item_embedding_tower(self.item_embedding_lookup(target_game_idx))
        dev_emb        = self.developer_tower(self.developer_embedding_lookup(target_dev_idx))
        year_emb       = self.year_embedding_tower(self.year_embedding_lookup(target_year_idx))
        price_emb      = self.price_embedding_tower(self.price_embedding_lookup(target_price))
        
        concat = torch.cat([item_genre_emb, item_tag_emb, item_emb,
                            dev_emb, year_emb, price_emb], dim=1)
        return F.normalize(self.item_projection(concat), dim=-1)

    def forward(self, X_user_avg_log, X_hist_liked, X_hist_disliked, X_hist_full,
                X_hist_playtime_weights,
                target_year_idx, target_game_idx,
                target_dev_idx, target_price):
        user_emb = self.user_embedding(X_user_avg_log, X_hist_liked, X_hist_disliked, X_hist_full,
                                       X_hist_playtime_weights)
        item_emb = self.item_embedding(target_year_idx, target_game_idx,
                                       target_dev_idx, target_price)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
