"""
Data acquisition — game description text → frozen sentence embeddings
Reads the cached Storefront payloads in data/appdetails/, builds one text string
per corpus game, encodes it with a frozen pretrained sentence encoder, and writes
data/base_game_text_emb.parquet (keyed by item_id, mirroring base_game_tags).

This is the offline precompute behind the planned item-side text tower (and its
user-side history-pool parity). The encoder is FROZEN — only a small projection
adapter trains later inside the model. Encode once here; the model never runs the
encoder at train or serve time. features.py assembles the (n_games+1, dim)
game_text_matrix buffer from this parquet, exactly as it does for the tag matrix.

Text string (high-density tokens first, so 512-token truncation only ever clips
the long-form About flavor text at the tail):

    "Title: {title}. Short Description: {short}. About the Game: {about}"

Encoder prefix convention is model-family-specific and MUST match pretraining:
  - BGE (bge-*-en-v1.5): documents get NO prefix; only short search queries get
    an instruction. We embed game text AND (later) user-history text as documents,
    so both sides use the empty passage prefix — symmetric.
  - E5 (e5-*-v2): documents get "passage: ", queries get "query: ". If you swap
    ENCODER_MODEL to an E5 variant, set PASSAGE_PREFIX = "passage: ".
QUERY_INSTRUCTION is recorded for a future external text-search feature only; it
is not used in this precompute.

Usage:
    python -m src.embed_text          # encode all corpus games, write parquet
    python -m src.embed_text --force  # overwrite an existing parquet
"""
import argparse
import html
import json
import os
import re

import pandas as pd
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_GAMES_PARQUET   = "data/base_games.parquet"
APPDETAILS_DIR       = "data/appdetails"
OUTPUT_PARQUET       = "data/base_game_text_emb.parquet"
OUTPUT_META_JSON     = "data/base_game_text_emb_meta.json"

ENCODER_MODEL        = "BAAI/bge-base-en-v1.5"   # 768-d; strong MTEB / cost balance
PASSAGE_PREFIX       = ""                          # BGE docs: no prefix (see header)
QUERY_INSTRUCTION    = "Represent this sentence for searching relevant passages: "  # future search only
EMBED_BATCH_SIZE     = 64
NORMALIZE            = True    # L2-normalize; adapter sees unit vectors

MAX_ABOUT_CHARS      = 4000   # hard cap on About text before encoding (encoder truncates
                              # at 512 tokens anyway; this just bounds string build cost)


# ── Text extraction ───────────────────────────────────────────────────────────

def _strip_html(s):
    """
    Drop Steam BBCode-derived markup: remove <img ...> tags whole (they carry a
    junk asset URL), strip remaining tags, unescape entities, collapse whitespace.
    """
    if not s:
        return ""
    s = re.sub(r"<img[^>]*>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _load_descriptions(appid):
    """
    Read data/appdetails/{appid}.json. Returns (short, about) cleaned strings,
    each "" when the field is absent or the game was delisted (success=False).
    """
    path = os.path.join(APPDETAILS_DIR, f"{appid}.json")
    if not os.path.exists(path):
        return "", ""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(appid, {})
    if not entry.get("success"):
        return "", ""
    d = entry.get("data", {})
    return _strip_html(d.get("short_description", "")), _strip_html(d.get("about_the_game", ""))


def _build_text(title, short, about):
    """
    Assemble the per-game string, high-density tokens first. Title always leads
    (present for every corpus game), so a delisted game with no store text still
    yields a non-degenerate embedding from its title rather than a zero vector.
    """
    parts = [f"Title: {title}."]
    if short:
        parts.append(f"Short Description: {short}")
    if about:
        parts.append(f"About the Game: {about[:MAX_ABOUT_CHARS]}")
    return PASSAGE_PREFIX + " ".join(parts)


# ── Encode ────────────────────────────────────────────────────────────────────

def embed_all(force=False):
    """
    Build the text string for every corpus game, batch-encode with the frozen
    encoder, write base_game_text_emb.parquet (item_id, embedding, has_description)
    plus a provenance sidecar. Order of rows matches base_games.parquet.
    """
    if os.path.exists(OUTPUT_PARQUET) and not force:
        raise SystemExit(f"{OUTPUT_PARQUET} exists — pass --force to overwrite.")

    from sentence_transformers import SentenceTransformer   # local: heavy import

    games = pd.read_parquet(BASE_GAMES_PARQUET, columns=["item_id", "title"])
    appids = [str(x) for x in games["item_id"].tolist()]
    titles = games["title"].fillna("").tolist()

    texts, has_desc = [], []
    for appid, title in zip(appids, tqdm(titles, desc="build text")):
        short, about = _load_descriptions(appid)
        texts.append(_build_text(title, short, about))
        has_desc.append(bool(short or about))

    n_missing = len(has_desc) - sum(has_desc)
    print(f"text built for {len(texts)} games — {sum(has_desc)} with store text, "
          f"{n_missing} title-only (delisted / not yet fetched)")

    model = SentenceTransformer(ENCODER_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    dim = int(embeddings.shape[1])

    out = pd.DataFrame({
        "item_id":          appids,
        "embedding":        list(embeddings.astype("float32")),
        "has_description":  has_desc,
    })
    out.to_parquet(OUTPUT_PARQUET, index=False)

    meta = {
        "encoder_model":        ENCODER_MODEL,
        "embedding_dim":        dim,
        "passage_prefix":       PASSAGE_PREFIX,
        "query_instruction":    QUERY_INSTRUCTION,
        "normalize":            NORMALIZE,
        "n_games":              len(appids),
        "n_with_description":   int(sum(has_desc)),
    }
    with open(OUTPUT_META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {OUTPUT_PARQUET} ({len(out)} rows, dim={dim}) and {OUTPUT_META_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode game descriptions to frozen embeddings.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing parquet")
    args = parser.parse_args()
    embed_all(force=args.force)
