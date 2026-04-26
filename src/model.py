"""
Two-Tower GameRecommender model.

No timestamp tower — Steam has no buy date / first-play timestamp in australian_users_items.json.

User tower: play-history playtime-weighted avg pool + genre context → projection MLP
Item tower: genre + tag + game_id + developer + year + price → projection MLP

Both towers project to output_dim via a shared-size MLP so they can be dot-producted.
The projection MLP learns cross-feature interactions (e.g. genre × tag, price × history)
that a plain concat + dot-product cannot express.

Sub-embedding sizes (inputs to the projection MLPs):
  item_id_embedding_size    = 32   shared: user history pool + item tower
  user_genre_embedding_size = 32   user only
  item_genre_embedding_size = 8    item only
  tag_embedding_size        = 16   item only
  developer_embedding_size  = 12   item only
  item_year_embedding_size  = 8    item only
  price_embedding_size      = 4    item only

  gpool:  user concat = item_id(32) + user_genre(32)        =  64 → proj → output_dim
  ipool:  user concat = output_dim(128) + user_genre(32)    = 160 → proj → output_dim
  item:   item concat = genre(8)+tag(16)+id(32)+dev(12)+year(8)+price(4) = 80 → proj → output_dim
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
                 item_id_embedding_size=32,
                 user_genre_embedding_size=32,
                 item_genre_embedding_size=8,
                 tag_embedding_size=16,
                 developer_embedding_size=12,
                 item_year_embedding_size=8,
                 price_embedding_size=4,
                 proj_hidden=256,
                 output_dim=128,
                 use_item_pool_for_history=False,
                 hist_genre_buf=None,
                 hist_year_buf=None,
                 hist_price_buf=None,
                 ):
        """
        game_tag_matrix : float32 tensor (n_games+1, n_tags)
            Row i = TF-IDF tag scores for game at embedding index i.
            Last row (index n_games) = zeros — padding.
        game_dev_idx    : int64 tensor (n_games+1,)
            Entry i = developer vocab index for game at embedding index i.
            Last entry (index n_games) = n_developers — padding.
        proj_hidden     : hidden size of the projection MLP in both towers.
        output_dim      : final embedding size for dot-product comparison.
        use_item_pool_for_history : when True, pool full item_embedding() output (output_dim)
            instead of raw item_id embeddings (item_id_embedding_size) in the user tower.
            Requires hist_genre_buf, hist_year_buf, hist_price_buf (non-persistent buffers
            indexed by embedding index, with a pad row at index n_games).
        """
        super().__init__()
        self.use_item_pool_for_history = use_item_pool_for_history

        self.game_pad_idx = n_games
        self.dev_pad_idx  = n_developers
        self.output_dim   = output_dim

        self.register_buffer('game_tag_matrix', game_tag_matrix)
        self.register_buffer('game_dev_idx',    game_dev_idx)

        if use_item_pool_for_history:
            assert hist_genre_buf is not None and hist_year_buf is not None \
                   and hist_price_buf is not None, \
                   "hist_genre_buf, hist_year_buf, hist_price_buf required when use_item_pool_for_history=True"
            self.register_buffer('hist_genre_buf', hist_genre_buf, persistent=False)
            self.register_buffer('hist_year_buf',  hist_year_buf,  persistent=False)
            self.register_buffer('hist_price_buf', hist_price_buf, persistent=False)

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

        # ── Projection MLPs (learn cross-feature interactions) ────────────────
        if use_item_pool_for_history:
            user_concat_dim = output_dim + user_genre_embedding_size
        else:
            user_concat_dim = item_id_embedding_size + user_genre_embedding_size
        item_concat_dim = (item_genre_embedding_size + tag_embedding_size
                           + item_id_embedding_size + developer_embedding_size
                           + item_year_embedding_size + price_embedding_size)

        # No activation on the final linear — feeds directly into dot product.
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

        self.apply(self._init_weights)
        # Projection layers need standard gain — gain=0.01 compounds across multiple
        # layers and causes vanishing gradients when sub-tower outputs are also small.
        for proj in [self.user_projection, self.item_projection]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    def user_embedding(self, X_genre, X_history, X_history_weights, item_cache=None):
        """
        X_genre           (B, user_context_size) float
        X_history         (B, max_hist_len)       long  — padded game indices
        X_history_weights (B, max_hist_len)       float — log(1+h) weights; 0 at padding
        item_cache        (n_items+1, output_dim) float — pre-computed frozen item embeddings,
                          or None to call item_embedding() for each history entry (slow path).
        """
        pad_mask   = (X_history != self.game_pad_idx).float().unsqueeze(-1)
        w          = X_history_weights.unsqueeze(-1) * pad_mask
        weight_sum = w.sum(dim=1).clamp(min=1e-6)

        if self.use_item_pool_for_history:
            if item_cache is not None:
                # Fast path: O(B*H) table lookup, no grad flows to item tower.
                history_emb = (item_cache[X_history] * w).sum(dim=1) / weight_sum
            else:
                # Slow path: B*H item tower forward passes — used by evaluate.py / export.
                B, H      = X_history.shape
                flat      = X_history.view(-1)
                flat_embs = self.item_embedding(
                    self.hist_genre_buf[flat],
                    self.hist_year_buf[flat],
                    flat,
                    self.game_dev_idx[flat],
                    self.hist_price_buf[flat],
                )
                history_emb = (flat_embs.view(B, H, self.output_dim) * w).sum(dim=1) / weight_sum
        else:
            history_embs = self.item_embedding_lookup(X_history)
            history_emb  = (history_embs * w).sum(dim=1) / weight_sum

        genre_emb = self.user_genre_tower(X_genre)
        concat    = torch.cat([history_emb, genre_emb], dim=1)
        return self.user_projection(concat)

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
        concat         = torch.cat([item_genre_emb, item_tag_emb, item_emb,
                                    dev_emb, year_emb, price_emb], dim=1)
        return self.item_projection(concat)

    def forward(self, X_genre, X_history, X_history_weights,
                target_genre, target_year_idx, target_game_idx,
                target_dev_idx, target_price):
        """Dot-product score for a (user, item) pair."""
        user_emb = self.user_embedding(X_genre, X_history, X_history_weights)
        item_emb = self.item_embedding(target_genre, target_year_idx, target_game_idx,
                                       target_dev_idx, target_price)
        return torch.einsum('ij,ij->i', user_emb, item_emb)
