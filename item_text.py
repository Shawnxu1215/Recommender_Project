import gzip
import json
import os

import numpy as np
import pandas as pd

from cleaning import item_ids

if not os.path.exists("data/meta_Electronics_1M.jsonl"):
    # Item IDs that appear in the subset
    remaining = set(pd.read_csv("data/Electronics_1M.csv", usecols=["parent_asin"])["parent_asin"])
    print("Items in subset:", len(remaining))

    # Stream the full metadata line by line and copy only the subset's items to a new file
    # Write to a temp file and rename it at the end, so an interruption never leaves a partial file
    with gzip.open("data/meta_Electronics.jsonl.gz", "rt", encoding="utf-8") as fin, \
            open("data/meta_Electronics_1M.jsonl.tmp", "w", encoding="utf-8") as fout:
        for line in fin:
            asin = json.loads(line)["parent_asin"]
            if asin in remaining:
                remaining.discard(asin)  # keep only the first record of each item
                fout.write(line)
                if not remaining:
                    break

    os.replace("data/meta_Electronics_1M.jsonl.tmp", "data/meta_Electronics_1M.jsonl")
    print("Items missing from metadata:", len(remaining))

# Read the filtered item metadata
meta = pd.read_json("data/meta_Electronics_1M.jsonl", lines=True)
print("Metadata shape:", meta.shape)
print(meta[["parent_asin", "title"]].head())


def has_text(v):
    return isinstance(v, str) and v.strip() != ""


def build_text(item):
    """Join an item's title, brand, category, price and features into one text; missing fields are skipped"""
    parts = [f"Title: {item['title'].strip()}"]
    if has_text(item["store"]):
        parts.append(f"Brand: {item['store'].strip()}")

    # Almost every category path starts with Electronics, so drop it; fall back to main_category when empty
    cats = [c for c in item["categories"] if c != "Electronics"]
    if cats:
        parts.append("Category: " + " > ".join(cats))
    elif has_text(item["main_category"]):
        parts.append(f"Category: {item['main_category']}")

    # Prices are mostly numbers; a few are strings such as "from 135.51" or "—"
    price = item["price"]
    if isinstance(price, float) and not np.isnan(price):
        parts.append(f"Price: {price:.2f}")
    elif has_text(price) and price != "—":
        parts.append(f"Price: {price}")

    # Features go last and are truncated, so long text does not drown out the title
    if item["features"]:
        parts.append("Features: " + " ".join(item["features"])[:300])
    return ". ".join(parts)


if os.path.exists("data/item_emb.npy"):
    item_emb = np.load("data/item_emb.npy")
else:
    from sentence_transformers import SentenceTransformer

    # Order items by item_idx, so text k belongs to item_idx = k
    records = meta.set_index("parent_asin").loc[item_ids].reset_index().to_dict("records")
    texts = [build_text(item) for item in records]
    print("Sample texts:")
    for t in texts[:3]:
        print("  ", t)

    # sentence-t5-base: T5 encoder → mean pooling → Dense → L2 normalization, one 768-d vector per item
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")
    print("Encoding device:", model.device, "| max length:", model.max_seq_length)
    item_emb = model.encode(texts, batch_size=256, show_progress_bar=True).astype(np.float16)

    np.save("data/item_emb.npy", item_emb)

print("Item embedding shape:", item_emb.shape)
