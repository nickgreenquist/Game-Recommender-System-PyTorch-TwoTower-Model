"""
Stage 1 — Wide & Deep ranker with STRICT V5 CG feature parity.

Mirrors src/model.GameRecommender (V5) exactly:
  - 4 user history pools (liked / disliked / full / playtime-weighted full)
  - Shared item_id_lookup across pools and item-side tower
  - In-model genre debiasing context (uses game_genre_matrix, playtime weights, avg_log
    — the avg_log scalar is used internally for debiasing but NOT concatenated)
  - In-model tag sum context (uses game_tag_matrix)
  - 2-layer item_tag / user_tag / user_genre towers (hidden dims match CG: 128, 256, 128)
  - 1-layer item_id / item_genre / developer / year / price towers
  - NO LayerNorm on pools (V5 removed it)
  - NO F.normalize anywhere — ranker scores softmax CE on raw logits (temperature scaled
    in train.py to match CG's logit range)

User concat (192):  liked(32) + disliked(32) + full(32) + playtime(32) + genre(32) + tag(32)
Item concat  (96):  item_id(32) + item_genre(8) + item_tag(32) + dev(12) + year(8) + price(4)
Deep MLP   (288 → [256, 128]):  Linear(288→256) → ReLU → Linear(256→128)
                                — matches CG's projection shape exactly; NO final ReLU
                                  (would clamp deep_out ≥ 0 and break parity with CG's
                                  dot-product geometry).
Head: cat(deep_out(128), cross_features(n_cross)) → Linear(128+n_cross, 1)

The ONLY architectural difference vs CG is "joint MLP on concat" vs "per-tower projection
then dot product." Everything else (per-feature towers, dims, init, dataset, labels, loss)
matches CG. This isolates the experiment to "can an MLP match a two-tower dot product?"

ZERO src/ imports — buffers built externally from FeatureStore in train.py / canary.py
and passed in. Warm-start from a CG checkpoint is a one-time copy at construction;
the ranker owns all params after that and trains them freely (see train._warm_start_from_cg).
"""
from collections import namedtuple

import torch
import torch.nn as nn


# Named slice offsets into user_concat (callers use these to pull pool_disliked etc.
# without hardcoding numbers). Keep in sync with the order in user_forward().
USER_CONCAT_LAYOUT = {
    'pool_liked':    (0,   32),
    'pool_disliked': (32,  64),
    'pool_full':     (64,  96),
    'pool_playtime': (96,  128),
    'genre_emb':     (128, 160),
    'tag_emb':       (160, 192),
}


# Bundled result of WideDeepRanker.user_forward() — shared user-side compute that
# future cross features can read without re-running the user tower.
UserForwardResult = namedtuple('UserForwardResult', [
    'user_concat',         # (B, user_concat_dim)
    'tag_profile',         # (B, n_tags)            — sum of game_tag_matrix[X_hist_full]
    'last_item_emb',       # (B, item_id_emb_dim)   — last real item in shuffled context
])


class WideDeepRanker(nn.Module):
    def __init__(self,
                 # Vocab / corpus sizes
                 n_games: int,
                 n_genres: int,
                 n_tags: int,
                 n_developers: int,
                 n_years: int,
                 n_price_buckets: int,
                 # Per-game static buffers (each shaped (n_games+1, ...) — pad row appended)
                 game_tag_matrix:    torch.Tensor,  # (n_games+1, n_tags)   float32 — RAW TF-IDF
                 game_tag_matrix_l2: torch.Tensor,  # (n_games+1, n_tags)   float32 — L2-normalized rows
                 game_tag_binary:    torch.Tensor,  # (n_games+1, n_tags)   float32 — binary one-hot   (B-1b cross feature)
                 game_tag_count:     torch.Tensor,  # (n_games+1,)          float32 — per-item tag count, clamp(min=1)
                 game_genre_matrix:  torch.Tensor,  # (n_games+1, n_genres) float32 — L1-row-normalized (1/k per genre)
                 game_genre_binary:  torch.Tensor,  # (n_games+1, n_genres) float32 — binary one-hot   (B-1  cross feature)
                 game_genre_count:   torch.Tensor,  # (n_games+1,)          float32 — per-item genre count, clamp(min=1)
                 game_year_idx:      torch.Tensor,  # (n_games+1,)          int64
                 game_dev_idx:       torch.Tensor,  # (n_games+1,)          int64
                 game_price_idx:     torch.Tensor,  # (n_games+1,)          int64
                 # Sub-tower output dims (V5 CG defaults)
                 item_id_emb_dim:        int = 32,
                 item_genre_emb_dim:     int = 8,
                 item_tag_emb_dim:       int = 32,
                 developer_emb_dim:      int = 12,
                 year_emb_dim:           int = 8,
                 price_emb_dim:          int = 4,
                 user_genre_emb_dim:     int = 32,
                 user_tag_emb_dim:       int = 32,
                 # Hardcoded hidden dims of CG's 2-layer towers (must match for warm-start)
                 item_tag_hidden:        int = 128,
                 user_tag_hidden:        int = 256,
                 user_genre_hidden:      int = 128,
                 # Deep MLP
                 hidden_dims: list = None,
                 dropout:    float = 0.0,
                 # Wide bypass
                 n_cross_features: int = 0,
                 ):
        super().__init__()
        if hidden_dims is None:
            # Match CG's user_projection / item_projection shape: [256, 128]. CG's projection
            # is exactly 2 Linear layers — Linear(N→256) → ReLU → Linear(256→128) — with
            # NO final activation. This deep MLP replicates that shape on the joint
            # (user+item) concat so the only architectural diff vs CG is "joint MLP" vs
            # "per-tower projection + cosine."
            hidden_dims = [256, 128]

        self.game_pad_idx     = n_games
        self.dev_pad_idx      = n_developers
        self.n_cross_features = n_cross_features

        # ── Buffers (non-persistent: rebuilt from FeatureStore on every load) ─
        # game_tag_matrix (RAW) feeds user_tag_tower and item_tag_tower — both warm-started
        # from CG on raw TF-IDF, so MUST NOT be mutated. game_tag_matrix_l2 is the pre-
        # L2-normalized variant used ONLY by the tag_cosine cross feature's cand-side
        # path (see train._forward_batch). Two buffers, two consumers — no overlap.
        self.register_buffer('game_tag_matrix',    game_tag_matrix,    persistent=False)
        self.register_buffer('game_tag_matrix_l2', game_tag_matrix_l2, persistent=False)
        # Binary one-hot + per-item tag count — fed into the tag_overlap cross feature
        # (B-1b). True set-membership semantics, distinct from the raw TF-IDF matrix
        # the deep towers consume.
        self.register_buffer('game_tag_binary',    game_tag_binary,    persistent=False)
        self.register_buffer('game_tag_count',     game_tag_count,     persistent=False)
        # L1-row-normalized — fed into item_genre_tower / user_genre_tower (CG parity).
        self.register_buffer('game_genre_matrix',  game_genre_matrix,  persistent=False)
        # Binary one-hot + per-item count — fed into the genre_overlap cross feature
        # (B-1). True set-membership semantics, distinct from the L1-normalized matrix
        # the deep towers consume.
        self.register_buffer('game_genre_binary',  game_genre_binary,  persistent=False)
        self.register_buffer('game_genre_count',   game_genre_count,   persistent=False)
        self.register_buffer('game_year_idx',      game_year_idx,      persistent=False)
        # game_dev_idx is also consumed by the dev_affinity cross feature (B-2) via
        # ranker.cross_features.dev_affinity — see train._forward_batch.
        self.register_buffer('game_dev_idx',       game_dev_idx,       persistent=False)
        self.register_buffer('game_price_idx',     game_price_idx,     persistent=False)

        # ── Embedding lookups (all CG-warm-startable) ───────────────────────
        self.item_id_lookup    = nn.Embedding(n_games + 1,       item_id_emb_dim,
                                              padding_idx=n_games)
        self.developer_lookup  = nn.Embedding(n_developers + 1,  developer_emb_dim,
                                              padding_idx=n_developers)
        self.year_lookup       = nn.Embedding(n_years,           year_emb_dim)
        self.price_lookup      = nn.Embedding(n_price_buckets,   price_emb_dim)

        # ── 1-layer towers (Linear → ReLU) — match CG names + shapes ───────
        self.item_id_tower    = nn.Sequential(nn.Linear(item_id_emb_dim,    item_id_emb_dim),    nn.ReLU())
        self.item_genre_tower = nn.Sequential(nn.Linear(n_genres,           item_genre_emb_dim), nn.ReLU())
        self.developer_tower  = nn.Sequential(nn.Linear(developer_emb_dim,  developer_emb_dim),  nn.ReLU())
        self.year_tower       = nn.Sequential(nn.Linear(year_emb_dim,       year_emb_dim),       nn.ReLU())
        self.price_tower      = nn.Sequential(nn.Linear(price_emb_dim,      price_emb_dim),      nn.ReLU())

        # ── 2-layer towers — hidden dims must match CG for warm-start parity ─
        self.item_tag_tower   = nn.Sequential(
            nn.Linear(n_tags, item_tag_hidden), nn.ReLU(),
            nn.Linear(item_tag_hidden, item_tag_emb_dim), nn.ReLU(),
        )
        self.user_tag_tower   = nn.Sequential(
            nn.Linear(n_tags, user_tag_hidden), nn.ReLU(),
            nn.Linear(user_tag_hidden, user_tag_emb_dim), nn.ReLU(),
        )
        self.user_genre_tower = nn.Sequential(
            nn.Linear(2 * n_genres, user_genre_hidden), nn.ReLU(),
            nn.Linear(user_genre_hidden, user_genre_emb_dim), nn.ReLU(),
        )

        # ── Deep MLP ─────────────────────────────────────────────────────────
        # User concat dim matches CG exactly (192): 4 pools + genre + tag. The
        # X_user_avg_log scalar is used internally for genre debiasing but NOT
        # concatenated — keeping the +1 was "parity+1" which we drop for strict
        # CG-parity baseline.
        user_concat_dim = (4 * item_id_emb_dim
                           + user_genre_emb_dim
                           + user_tag_emb_dim)
        item_concat_dim = (item_id_emb_dim + item_genre_emb_dim + item_tag_emb_dim
                           + developer_emb_dim + year_emb_dim + price_emb_dim)
        deep_in = user_concat_dim + item_concat_dim

        # NO final ReLU after the last hidden layer — mirrors CG's projection
        # (Linear→ReLU→Linear, ends on raw Linear). A final ReLU would clamp
        # deep_out ≥ 0, compressing the head's effective input geometry and
        # making it structurally different from CG's dot product (which freely
        # spans negative values).
        layers = []
        in_dim = deep_in
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            is_last = (i == len(hidden_dims) - 1)
            if not is_last:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = h
        self.deep = nn.Sequential(*layers)

        # ── Head: deep representation + cross-feature wide bypass ────────────
        self.head = nn.Linear(hidden_dims[-1] + n_cross_features, 1)

        # Public dims (used by train.py for the summary line)
        self.user_concat_dim = user_concat_dim
        self.item_concat_dim = item_concat_dim
        self.deep_in         = deep_in

        # ── Init: gain=0.1 on sub-towers, gain=1.0 on deep + head, gain=0.01 on Embeddings ─
        self.apply(self._init_weights)
        for module in [self.deep, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight, gain=0.01)

    # ── User tower ───────────────────────────────────────────────────────────

    def user_forward(self,
                     X_user_avg_log:          torch.Tensor,    # (B, 1)
                     X_hist_liked:            torch.Tensor,    # (B, H) int64, pad=game_pad_idx
                     X_hist_disliked:         torch.Tensor,    # (B, H)
                     X_hist_full:             torch.Tensor,    # (B, H)
                     X_hist_playtime_weights: torch.Tensor,    # (B, H) float32, pre-normalized
                     ) -> UserForwardResult:
        """
        Unified user-side pass. Returns user_concat plus auxiliary signals that
        cross features may need (tag_profile, last_item_emb). All look up the same
        item_id_lookup table — keeping this in one place avoids 2x recompute when
        evaluate / canary expand user_concat across hundreds of candidates.
        """
        # ── 4 pools (no LayerNorm — V5 removed it) ──────────────────────────
        # padding_idx in item_id_lookup zeroes pad rows automatically.
        pool_liked     = self.item_id_lookup(X_hist_liked).sum(dim=1)
        pool_disliked  = self.item_id_lookup(X_hist_disliked).sum(dim=1)
        full_embs      = self.item_id_lookup(X_hist_full)                          # (B, H, id_dim)
        pool_full      = full_embs.sum(dim=1)
        w              = X_hist_playtime_weights.unsqueeze(-1)                     # (B, H, 1)
        pool_playtime  = (full_embs * w).sum(dim=1)

        # ── In-model genre debiasing (mirror src/model.user_embedding) ──────
        hist_genres = self.game_genre_matrix[X_hist_full]                          # (B, H, n_genres)
        running_genre_count = (hist_genres > 0).float().sum(dim=1)                 # (B, n_genres)
        N        = (X_hist_full != self.game_pad_idx).float().sum(dim=1, keepdim=True)  # (B, 1)
        safe_cnt = torch.where(running_genre_count > 0, running_genre_count,
                               torch.ones_like(running_genre_count))
        genre_sum_w = (hist_genres * w).sum(dim=1)                                 # (B, n_genres)
        genre_affinity = torch.where(
            running_genre_count > 0,
            X_user_avg_log * ((N * genre_sum_w / safe_cnt) - 1.0),
            torch.zeros_like(running_genre_count),
        )
        total_assign  = running_genre_count.sum(dim=1, keepdim=True)
        safe_assign   = torch.where(total_assign > 0, total_assign,
                                    torch.ones_like(total_assign))
        genre_fractions = running_genre_count / safe_assign
        X_genre   = torch.cat([genre_affinity, genre_fractions], dim=1)            # (B, 2*n_genres)
        genre_emb = self.user_genre_tower(X_genre)

        # ── In-model tag context: sum of tag vectors over full history ──────
        tag_profile = self.game_tag_matrix[X_hist_full].sum(dim=1)                 # (B, n_tags)
        tag_emb     = self.user_tag_tower(tag_profile)

        user_concat = torch.cat([
            pool_liked, pool_disliked, pool_full, pool_playtime,
            genre_emb, tag_emb,
        ], dim=1)

        # ── Auxiliary signals for cross features ────────────────────────────
        # Last real item in shuffled context = position of the last non-pad index.
        # X_hist_full is left-aligned (real first, pads last), so last real = N-1.
        last_pos       = (N.squeeze(-1).long() - 1).clamp(min=0)                   # (B,)
        b_idx          = torch.arange(X_hist_full.shape[0], device=X_hist_full.device)
        last_idx       = X_hist_full[b_idx, last_pos]                              # (B,)
        last_real_mask = (N.squeeze(-1) > 0).float().unsqueeze(-1)                 # (B, 1)
        last_item_emb  = self.item_id_lookup(last_idx) * last_real_mask            # (B, id_dim)

        return UserForwardResult(user_concat, tag_profile, last_item_emb)

    def user_embedding(self,
                       X_user_avg_log, X_hist_liked, X_hist_disliked,
                       X_hist_full, X_hist_playtime_weights) -> torch.Tensor:
        """Convenience wrapper: returns user_concat only."""
        return self.user_forward(X_user_avg_log, X_hist_liked, X_hist_disliked,
                                  X_hist_full, X_hist_playtime_weights).user_concat

    # ── Item tower ───────────────────────────────────────────────────────────

    def item_embedding(self, cand_idx: torch.Tensor) -> torch.Tensor:
        """
        cand_idx: (B,) int64 — corpus item index. Per-game metadata (year, dev, price,
        genre row, tag row) is fetched from registered buffers — no separate args.
        """
        item_id_emb    = self.item_id_tower(self.item_id_lookup(cand_idx))
        item_genre_emb = self.item_genre_tower(self.game_genre_matrix[cand_idx])
        item_tag_emb   = self.item_tag_tower(self.game_tag_matrix[cand_idx])
        dev_emb        = self.developer_tower(self.developer_lookup(self.game_dev_idx[cand_idx]))
        year_emb       = self.year_tower(self.year_lookup(self.game_year_idx[cand_idx]))
        price_emb      = self.price_tower(self.price_lookup(self.game_price_idx[cand_idx]))
        return torch.cat([item_id_emb, item_genre_emb, item_tag_emb,
                          dev_emb, year_emb, price_emb], dim=1)

    # ── Score pairs ──────────────────────────────────────────────────────────

    def score_pairs(self,
                    user_concat:    torch.Tensor,
                    item_concat:    torch.Tensor,
                    cross_features: torch.Tensor) -> torch.Tensor:
        """
        Score pre-computed user / item / cross feature blocks. Returns (B,) raw logits.

        Generic 1:1 entry point — caller must pre-align user_concat / item_concat to
        the same outer batch dim (typically B*n_cand). For the (B users × n_cand items)
        shape used in training / eval, prefer `score_pairs_batched` instead — it
        factorizes the first MLP layer and avoids materializing a (B*n_cand, U) user
        replica. This entry point is kept for single-user paths (canary) and any
        future caller that has a true 1:1 user-item pairing.
        """
        deep_out = self.deep(torch.cat([user_concat, item_concat], dim=1))
        if self.n_cross_features > 0:
            combined = torch.cat([deep_out, cross_features], dim=1)
        else:
            combined = deep_out
        return self.head(combined).squeeze(-1)

    def score_pairs_batched(self,
                            user_concat:    torch.Tensor,    # (B, U)
                            item_concat:    torch.Tensor,    # (B * n_cand, I)
                            cross_features: torch.Tensor,    # (B * n_cand, n_cross) or empty
                            n_cand:         int,
                            ) -> torch.Tensor:               # (B, n_cand)
        """
        Compute-/memory-efficient form of score_pairs for the (B × n_cand) layout.

        Mathematically identical (up to FP summation order, ~1e-6) to:
            user_exp = user_concat.unsqueeze(1).expand(-1, n_cand, -1).reshape(B*n_cand, U)
            score_pairs(user_exp, item_concat, cross_features).view(B, n_cand)

        The optimization is a math identity on the first MLP layer:
            cat([u, i]) @ Wᵀ + b  ≡  u @ Wᵤᵀ + i @ Wᵢᵀ + b
        Splitting the matmul lets us run the user-side projection on only B rows
        instead of B*n_cand, and skip materializing a (B*n_cand, U) tensor that's
        just n_cand copies of each user.

        Layer 2 onward runs on (B*n_cand, H) as before — after the ReLU, user/item
        dims entangle and can't be factorized further. The win is concentrated on
        layer 1 (the fattest layer: 288→256 → ~70% of MLP FLOPs at the current dims).
        """
        B, U = user_concat.shape
        first = self.deep[0]                     # nn.Linear(deep_in, H1)
        W     = first.weight                     # (H1, U + I)
        b     = first.bias                       # (H1,)
        H1    = W.shape[0]

        u_pre = user_concat @ W[:, :U].t()       # (B, H1)
        i_pre = item_concat @ W[:, U:].t()       # (B*n_cand, H1)

        # Broadcast-add user/item halves + bias, then collapse to (B*n_cand, H1).
        pre = u_pre.unsqueeze(1) + i_pre.view(B, n_cand, H1) + b
        x   = pre.view(B * n_cand, H1)

        # Apply remaining MLP layers (ReLU, optional Dropout, Linear(H1→H2), ...).
        for layer in self.deep[1:]:
            x = layer(x)

        if self.n_cross_features > 0:
            combined = torch.cat([x, cross_features], dim=1)
        else:
            combined = x
        scores = self.head(combined).squeeze(-1)  # (B*n_cand,)
        return scores.view(B, n_cand)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self,
                X_user_avg_log, X_hist_liked, X_hist_disliked,
                X_hist_full, X_hist_playtime_weights,
                cand_idx, cross_features):
        user_concat = self.user_embedding(X_user_avg_log, X_hist_liked, X_hist_disliked,
                                          X_hist_full, X_hist_playtime_weights)
        item_concat = self.item_embedding(cand_idx)
        return self.score_pairs(user_concat, item_concat, cross_features)
