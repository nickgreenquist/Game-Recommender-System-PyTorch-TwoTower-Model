"""
Two-Tower GameRecommender model.

No timestamp tower — Steam has no buy date / first-play timestamp in australian_users_items.json.

User tower: play-history playtime-weighted avg pool + genre context
Item tower: genre + tag + game_id + developer + year + price

Embedding sizes (must satisfy user_dim == item_dim):
  item_id_embedding_size    = 40   shared: user history pool + item tower
  user_genre_embedding_size = 65   user only (sized to match item_dim with no timestamp)
  item_genre_embedding_size = 10   item only
  tag_embedding_size        = 25   item only
  developer_embedding_size  = 15   item only
  item_year_embedding_size  = 10   item only
  price_embedding_size      = 5    item only

  user: 40 + 65 = 105
  item: 10 + 25 + 40 + 15 + 10 + 5 = 105  ✓
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
                 user_context_size,
                 game_tag_matrix,
                 game_dev_idx,
                 item_id_embedding_size=40,
                 user_genre_embedding_size=65,
                 item_genre_embedding_size=10,
                 tag_embedding_size=25,
                 developer_embedding_size=15,
                 item_year_embedding_size=10,
                 price_embedding_size=5,
                 ):
        """
        game_tag_matrix : float32 tensor (n_games+1, n_tags)
            Row i = TF-IDF tag scores for game at embedding index i.
            Last row (index n_games) = zeros — padding.
        game_dev_idx    : int64 tensor (n_games+1,)
            Entry i = developer vocab index for game at embedding index i.
            Last entry (index n_games) = n_developers — padding.
        """
        super().__init__()

        self.game_pad_idx = n_games
        self.dev_pad_idx  = n_developers

        self.register_buffer('game_tag_matrix', game_tag_matrix)
        self.register_buffer('game_dev_idx',    game_dev_idx)

        # ── Shared item embedding ─────────────────────────────────────────────
        self.item_embedding_lookup = nn.Embedding(
            n_games + 1, item_id_embedding_size, padding_idx=n_games
        )
        self.item_embedding_tower = nn.Sequential(
            nn.Linear(item_id_embedding_size, item_id_embedding_size),
            nn.Tanh()
        )

        # ── Item-only developer tower ─────────────────────────────────────────
        self.developer_embedding_lookup = nn.Embedding(
            n_developers + 1, developer_embedding_size, padding_idx=n_developers
        )
        self.developer_tower = nn.Sequential(
            nn.Linear(developer_embedding_size, developer_embedding_size),
            nn.Tanh()
        )

        # ── Item-only tag tower ───────────────────────────────────────────────
        tag_hidden = 128
        self.item_tag_tower = nn.Sequential(
            nn.Linear(n_tags, tag_hidden),
            nn.Tanh(),
            nn.Linear(tag_hidden, tag_embedding_size),
            nn.Tanh()
        )

        # ── Item-only genre tower ─────────────────────────────────────────────
        self.item_genre_tower = nn.Sequential(
            nn.Linear(n_genres, item_genre_embedding_size),
            nn.Tanh()
        )

        # ── Item-only year tower ──────────────────────────────────────────────
        self.year_embedding_lookup = nn.Embedding(n_years, item_year_embedding_size)
        self.year_embedding_tower = nn.Sequential(
            nn.Linear(item_year_embedding_size, item_year_embedding_size),
            nn.Tanh()
        )

        # ── Item-only price tower ─────────────────────────────────────────────
        self.price_embedding_lookup = nn.Embedding(n_price_buckets, price_embedding_size)
        self.price_embedding_tower = nn.Sequential(
            nn.Linear(price_embedding_size, price_embedding_size),
            nn.Tanh()
        )

        # ── User-only genre tower ─────────────────────────────────────────────
        genre_hidden = 128
        self.user_genre_tower = nn.Sequential(
            nn.Linear(user_context_size, genre_hidden),
            nn.Tanh(),
            nn.Linear(genre_hidden, user_genre_embedding_size),
            nn.Tanh()
        )

        # ── Dimension check ───────────────────────────────────────────────────
        user_dim = item_id_embedding_size + user_genre_embedding_size
        item_dim = (item_genre_embedding_size + tag_embedding_size
                    + item_id_embedding_size + developer_embedding_size
                    + item_year_embedding_size + price_embedding_size)
        if user_dim != item_dim:
            raise ValueError(
                f"User dim ({user_dim}) != item dim ({item_dim}). "
                f"user: history {item_id_embedding_size} + genre {user_genre_embedding_size}. "
                f"item: genre {item_genre_embedding_size} + tag {tag_embedding_size} "
                f"+ game {item_id_embedding_size} + dev {developer_embedding_size} "
                f"+ year {item_year_embedding_size} + price {price_embedding_size}."
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def user_embedding(self, X_genre, X_history, X_history_weights):
        """
        X_genre           (B, user_context_size) float
        X_history         (B, max_hist_len)       long  — padded game indices
        X_history_weights (B, max_hist_len)       float — log(1+h) weights; 0 at padding
        """
        pad_mask       = (X_history != self.game_pad_idx).float().unsqueeze(-1)
        w              = X_history_weights.unsqueeze(-1) * pad_mask
        weight_sum     = w.sum(dim=1).clamp(min=1e-6)
        history_embs   = self.item_embedding_lookup(X_history)
        history_emb    = (history_embs * w).sum(dim=1) / weight_sum
        genre_emb      = self.user_genre_tower(X_genre)
        return torch.cat([history_emb, genre_emb], dim=1)

    def item_embedding(self, target_genre, target_year_idx, target_game_idx,
                       target_dev_idx, target_price):
        """
        target_genre    (B, n_genres) float
        target_year_idx (B,)          long
        target_game_idx (B,)          long
        target_dev_idx  (B,)          long
        target_price    (B,)          long
        """
        item_genre_emb = self.item_genre_tower(target_genre)
        item_tag_vec   = self.game_tag_matrix[target_game_idx]
        item_tag_emb   = self.item_tag_tower(item_tag_vec)
        item_emb       = self.item_embedding_tower(
                             self.item_embedding_lookup(target_game_idx))
        dev_emb        = self.developer_tower(
                             self.developer_embedding_lookup(target_dev_idx))
        year_emb       = self.year_embedding_tower(
                             self.year_embedding_lookup(target_year_idx))
        price_emb      = self.price_embedding_tower(
                             self.price_embedding_lookup(target_price))
        return torch.cat([item_genre_emb, item_tag_emb, item_emb,
                          dev_emb, year_emb, price_emb], dim=1)

    def forward(self, X_genre, X_history, X_history_weights,
                target_genre, target_year_idx, target_game_idx,
                target_dev_idx, target_price):
        """Dot-product score for a (user, item) pair."""
        user_emb = self.user_embedding(X_genre, X_history, X_history_weights)
        item_emb = self.item_embedding(target_genre, target_year_idx, target_game_idx,
                                       target_dev_idx, target_price)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
