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

        # ── Shared item embedding lookup (used by item tower and all 3 user history pools) ──
        self.item_embedding_lookup = nn.Embedding(
            n_games + 1, item_id_embedding_size, padding_idx=n_games
        )

        # ── Normalization for Sum Pooling ──
        self.history_norm = nn.LayerNorm(item_id_embedding_size)
        self.tag_norm     = nn.LayerNorm(n_tags)

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
        #   3 pools * 32-dim + genre context 32-dim + tag context 32-dim = 160-dim
        user_concat_dim = (3 * item_id_embedding_size + 
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

    def user_embedding(self, X_genre, X_tag, X_hist_liked, X_hist_disliked, X_hist_full):
        """
        X_genre         (B, 2*n_genres)
        X_tag           (B, n_tags)
        X_hist_liked    (B, L)
        X_hist_disliked (B, D)
        X_hist_full     (B, F)
        """
        # Shallow sum pooling over shared item embeddings
        def pool(ids):
            # Mask out padding index (n_games) so its embedding contributes nothing to the sum
            # Actually nn.Embedding(..., padding_idx=...) already sets pad embedding to zeros.
            raw_sum = self.item_embedding_lookup(ids).sum(dim=1)
            return self.history_norm(raw_sum)

        liked_emb    = pool(X_hist_liked)
        disliked_emb = pool(X_hist_disliked)
        full_emb     = pool(X_hist_full)
        
        genre_emb = self.user_genre_tower(X_genre)
        tag_emb   = self.user_tag_tower(self.tag_norm(X_tag))
        
        concat = torch.cat([liked_emb, disliked_emb, full_emb, genre_emb, tag_emb], dim=1)
        return self.user_projection(concat)

    def item_embedding(self, target_genre, target_year_idx, target_game_idx,
                       target_dev_idx, target_price):
        item_genre_emb = self.item_genre_tower(target_genre)
        item_tag_vec   = self.game_tag_matrix[target_game_idx]
        item_tag_emb   = self.item_tag_tower(item_tag_vec)
        item_emb       = self.item_embedding_tower(self.item_embedding_lookup(target_game_idx))
        dev_emb        = self.developer_tower(self.developer_embedding_lookup(target_dev_idx))
        year_emb       = self.year_embedding_tower(self.year_embedding_lookup(target_year_idx))
        price_emb      = self.price_embedding_tower(self.price_embedding_lookup(target_price))
        
        concat = torch.cat([item_genre_emb, item_tag_emb, item_emb,
                            dev_emb, year_emb, price_emb], dim=1)
        return self.item_projection(concat)

    def forward(self, X_genre, X_tag, X_hist_liked, X_hist_disliked, X_hist_full,
                target_genre, target_year_idx, target_game_idx,
                target_dev_idx, target_price):
        user_emb = self.user_embedding(X_genre, X_tag, X_hist_liked, X_hist_disliked, X_hist_full)
        item_emb = self.item_embedding(target_genre, target_year_idx, target_game_idx,
                                       target_dev_idx, target_price)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
