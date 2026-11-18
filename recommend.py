import argparse
import os

import numpy as np
import pandas as pd
import torch

import tiger
from ncf_mlp import NCF_MLP

# Usage: python recommend.py <user_id> [--n 10] [--model both|ncf|tiger]
# Without a user_id, a random user is picked
CATALOG_PATH = "data/item_catalog.csv"
META_PATH = "data/meta_Electronics_1M.jsonl"
# TIGER generates 200 candidate items per user, drops the ones already bought, then keeps the top N
# Checked on 20 users against exact scoring of all 209,379 items: beam 50 matched the top-10 for 14/20 users, beam 200 for 20/20
BEAM_SIZE = 200

device = tiger.device
split = tiger.split
item_ids = split["item_ids"]
num_users, num_items = int(split["num_users"]), int(split["num_items"])
train_item_pool = np.unique(split["train_item"])  # items that appear in train

# user_id → index: as in cleaning.py, users are numbered in order of first appearance in Electronics_1M.csv
user_ids = pd.factorize(pd.read_csv("data/Electronics_1M.csv", usecols=["user_id"])["user_id"])[1]
user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)

# Each user's full purchase history (oldest to newest): train + val + test; recommendations use all of it
hist_user = np.concatenate([split["train_user"], split["val_user"], split["test_user"]])
hist_item = np.concatenate([split["train_item"], split["val_item"], split["test_item"]])
order = np.argsort(hist_user, kind="stable")
hist_user, hist_item = hist_user[order], hist_item[order]


def clean_price(p):
    """Prices are mostly numbers; a few are strings such as "from 135.51" or "—" """
    if isinstance(p, (int, float)) and not np.isnan(p):
        return f"{p:.2f}"
    if isinstance(p, str) and p.strip() not in ("", "—"):
        return p.strip()
    return ""


def build_catalog():
    """Extract the display fields from the item metadata, ordered by item_idx, into a small table (only needed once)"""
    print("First run: building the item catalog from the metadata, this takes a minute or two...")
    meta = pd.read_json(META_PATH, lines=True)
    meta = meta.drop_duplicates("parent_asin").set_index("parent_asin").reindex(item_ids)
    catalog = pd.DataFrame({
        "item_idx": np.arange(num_items),
        "parent_asin": item_ids,
        "title": meta["title"].to_numpy(),
        "store": meta["store"].to_numpy(),
        "price": meta["price"].map(clean_price).to_numpy(),
        "category": meta["categories"].map(
            lambda c: " > ".join(x for x in c if x != "Electronics") if isinstance(c, list) else "").to_numpy(),
        "average_rating": meta["average_rating"].to_numpy(),
        "rating_number": meta["rating_number"].to_numpy(),
    })
    catalog.to_csv(CATALOG_PATH, index=False)


if not os.path.exists(CATALOG_PATH):
    build_catalog()
catalog = pd.read_csv(CATALOG_PATH, dtype={"price": str})
catalog[["title", "store", "price", "category"]] = catalog[["title", "store", "price", "category"]].fillna("")

# Load both models once
ncf = NCF_MLP(num_users, num_items).to(device)
ncf.load_state_dict(torch.load("models/ncf_mlp_best.pt", map_location=device))
ncf.eval()

tiger_model = tiger.build_model().to(device)
tiger_model.load_state_dict(torch.load(f"{tiger.MODEL_DIR}/tiger_best.pt", map_location=device))
tiger_model.eval()

# Prefix tree: tokens generated so far → tokens allowed next. Guarantees that the 4 generated tokens form a real item
allowed_next = {}
for row in tiger.item_tokens.tolist():
    for k in range(4):
        allowed_next.setdefault(tuple(row[:k]), set()).add(row[k])
allowed_next = {prefix: sorted(tokens) for prefix, tokens in allowed_next.items()}
token_to_item = {tuple(row): i for i, row in enumerate(tiger.item_tokens.tolist())}


def find_user(user_id):
    """user_id → (index, start and end of this user's history in hist_item)"""
    if user_id not in user_index.index:
        raise KeyError(f"User {user_id} not found: recommendations need a user with purchase history in the data")
    u = int(user_index[user_id])
    return u, np.searchsorted(hist_user, u, side="left"), np.searchsorted(hist_user, u, side="right")


def make_table(items, scores):
    """Item indices + scores → recommendation table with title, brand and price"""
    table = catalog.iloc[items][["title", "store", "price", "parent_asin"]].reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(items) + 1))
    table["score"] = np.round(scores, 4)
    return table


@torch.no_grad()
def recommend_ncf(user_id, n=10):
    """NCF: score every item that appears in train, drop the ones already bought, keep the n highest"""
    u, start, end = find_user(user_id)
    items = torch.as_tensor(train_item_pool, dtype=torch.long, device=device)
    scores = ncf(torch.full_like(items, u), items)
    seen = torch.as_tensor(np.isin(train_item_pool, hist_item[start:end]), device=device)
    scores[seen] = -torch.inf
    top = torch.topk(scores, n)
    return make_table(train_item_pool[top.indices.cpu().numpy()], top.values.cpu().numpy())


@torch.no_grad()
def recommend_tiger(user_id, n=10):
    """TIGER: feed the 20 most recent items, generate semantic IDs with beam search, drop items already bought, keep the top n"""
    u, start, end = find_user(user_id)
    x = tiger.build_inputs(hist_item, np.array([u]), np.array([start]), np.array([end])).to(device).long()
    out = tiger_model.generate(
        input_ids=x, attention_mask=x != tiger.PAD,
        num_beams=BEAM_SIZE, num_return_sequences=BEAM_SIZE, max_new_tokens=4,
        prefix_allowed_tokens_fn=lambda _, prefix: allowed_next[tuple(prefix[1:].tolist())],
        length_penalty=0.0,  # score = sum of the 4 tokens' log-probs, the same as in evaluation
        output_scores=True, return_dict_in_generate=True,
    )
    seen = set(hist_item[start:end].tolist())
    items, scores = [], []
    for seq, score in zip(out.sequences[:, 1:].tolist(), out.sequences_scores.tolist()):
        item = token_to_item[tuple(seq)]
        if item not in seen:
            items.append(item)
            scores.append(score)
    table = make_table(np.array(items[:n]), np.array(scores[:n]))
    table["new_item"] = ~np.isin(items[:n], train_item_pool)  # items never seen in train, which NCF cannot recommend
    return table


def short(text, width=90):
    return text if len(text) <= width else text[:width - 3] + "..."


def show(name, table):
    print(f"\n{name} top-{len(table)} recommendations:")
    for r in table.itertuples():
        info = ", ".join(x for x in [r.store, f"${r.price}" if r.price else ""] if x)
        new = "  [new item]" if getattr(r, "new_item", False) else ""
        print(f"  {r.rank:2d}. {short(r.title)}  ({info}){new}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend items for a user ID")
    parser.add_argument("user_id", nargs="?", help="Amazon user_id; a random user is picked if omitted")
    parser.add_argument("--n", type=int, default=10, help="number of items to recommend")
    parser.add_argument("--model", choices=["both", "ncf", "tiger"], default="both")
    args = parser.parse_args()

    user_id = args.user_id or user_ids[np.random.default_rng().integers(num_users)]
    u, start, end = find_user(user_id)
    print(f"User: {user_id} ({end - start} purchases)")
    print("Recent purchases (newest first):")
    for item in hist_item[start:end][::-1][:5]:
        print(f"  - {short(catalog.at[item, 'title'])}")

    if args.model in ("both", "ncf"):
        show("NCF", recommend_ncf(user_id, args.n))
    if args.model in ("both", "tiger"):
        show("TIGER", recommend_tiger(user_id, args.n))
