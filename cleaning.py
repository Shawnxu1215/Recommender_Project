import os

import numpy as np
import pandas as pd

if os.path.exists("data/Electronics_1M.csv"):
    # Already subset: read the 1M-interaction file directly
    electronics_1m = pd.read_csv("data/Electronics_1M.csv")
else:
    # Read the full Electronics ratings
    electronics = pd.read_csv("data/Electronics.csv")

    # Each user's most recent interaction time, newest first
    last_time = electronics.groupby("user_id")["timestamp"].max().sort_values(ascending=False)

    # Each user's number of interactions, in the same user order
    counts = electronics.groupby("user_id").size().reindex(last_time.index)

    # Add users from the most recently active down, keeping each user's full history, up to 1M interactions in total
    selected_users = counts.index[counts.cumsum() <= 1_000_000]
    electronics_1m = electronics[electronics["user_id"].isin(selected_users)]
    electronics_1m = electronics_1m.sort_values(["user_id", "timestamp"])

    # Save the subset
    electronics_1m.to_csv("data/Electronics_1M.csv", index=False)

print("Users:", electronics_1m["user_id"].nunique())
print("Data shape:", electronics_1m.shape)
print(electronics_1m[["user_id", "parent_asin"]].head())

# Index users and items from 0
electronics_1m["user_idx"], user_ids = pd.factorize(electronics_1m["user_id"])
electronics_1m["item_idx"], item_ids = pd.factorize(electronics_1m["parent_asin"])

num_users = len(user_ids)
num_items = len(item_ids)

print("Users:", num_users)
print("Items:", num_items)
print(electronics_1m[["user_id", "user_idx", "parent_asin", "item_idx"]].head())

electronics_1m = electronics_1m.sort_values(
    ["user_idx", "timestamp"]
).copy()

position = electronics_1m.groupby("user_idx").cumcount()
count = electronics_1m.groupby("user_idx")["item_idx"].transform("size")

val_mask = (count >= 3) & (position == count - 2)
test_mask = (count >= 3) & (position == count - 1)

val = electronics_1m[val_mask].copy()
test = electronics_1m[test_mask].copy()
train = electronics_1m[~(val_mask | test_mask)].copy()

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

rng = np.random.default_rng(42)

train_users = train["user_idx"].to_numpy()
train_items = train["item_idx"].to_numpy()

# Encode each (user, item) pair as one integer, to check quickly whether a user interacted with an item in train
train_pairs = np.unique(train_users.astype(np.int64) * num_items + train_items)

# Items that appear in train; negative sampling and evaluation candidates only use these
train_item_pool = np.unique(train_items)


def sample_negatives(num_neg=4):
    """Called once per epoch: pair each positive with num_neg items the user has not interacted with in train"""
    neg_users = np.repeat(train_users, num_neg)
    neg_items = rng.choice(train_item_pool, size=len(neg_users))

    # Redraw any item the user has interacted with, until all are true negatives
    bad = np.isin(neg_users.astype(np.int64) * num_items + neg_items, train_pairs)
    while bad.any():
        neg_items[bad] = rng.choice(train_item_pool, size=bad.sum())
        bad[bad] = np.isin(neg_users[bad].astype(np.int64) * num_items + neg_items[bad], train_pairs)

    # Positives are labeled 1 and negatives 0; concatenate and shuffle
    users = np.concatenate([train_users, neg_users])
    items = np.concatenate([train_items, neg_items])
    labels = np.concatenate([np.ones(len(train_users)), np.zeros(len(neg_users))]).astype(np.float32)

    order = rng.permutation(len(users))
    return users[order], items[order], labels[order]


train_item_set = set(train["item_idx"])

for name, df in [("Val", val), ("Test", test)]:
    unseen = ~df["item_idx"].isin(train_item_set)
    print(f"{name} interactions with items unseen in train: {unseen.sum()} ({unseen.mean():.2%})")

val_eval = val[
    val["item_idx"].isin(train_item_pool)
].copy()

test_eval = test[
    test["item_idx"].isin(train_item_pool)
].copy()

print("Val users evaluated:", len(val_eval))
print("Test users evaluated:", len(test_eval))


def sample_eval_candidates(eval_df, history_df, num_neg=99, seed=0):
    """Draw a fixed set of num_neg unseen items per evaluation user; returns (users, candidates) with the target in column 0"""
    eval_rng = np.random.default_rng(seed)
    eval_df = eval_df.sort_values("user_idx")
    users = eval_df["user_idx"].to_numpy()
    targets = eval_df["item_idx"].to_numpy()

    # Neither the user's history nor the target can be a negative
    seen = np.unique(np.concatenate([
        history_df["user_idx"].to_numpy().astype(np.int64) * num_items + history_df["item_idx"].to_numpy(),
        users.astype(np.int64) * num_items + targets,
    ]))

    neg_users = np.repeat(users, num_neg)
    neg_items = eval_rng.choice(train_item_pool, size=len(neg_users))
    while True:
        keys = neg_users.astype(np.int64) * num_items + neg_items

        # Redraw items the user has interacted with and items drawn twice for the same user
        _, first = np.unique(keys, return_index=True)
        bad = np.ones(len(keys), dtype=bool)
        bad[first] = False
        bad |= np.isin(keys, seen)
        if not bad.any():
            break
        neg_items[bad] = eval_rng.choice(train_item_pool, size=bad.sum())

    candidates = np.column_stack([targets, neg_items.reshape(-1, num_neg)])
    return users, candidates


# val excludes only the train history; test excludes train and val history. Fixed seeds, so every model gets the same candidates
val_users, val_candidates = sample_eval_candidates(val_eval, train, seed=1)
test_users, test_candidates = sample_eval_candidates(test_eval, pd.concat([train, val]), seed=2)

# Save the split and evaluation candidates. TIGER (including training on Colab) reads this file, so it uses exactly the same data as NCF
# train is sorted by (user_idx, timestamp); val/test have one row per user, sorted by user_idx
if not os.path.exists("data/split.npz"):
    np.savez_compressed(
        "data/split.npz",
        train_user=train["user_idx"].to_numpy(np.int32), train_item=train["item_idx"].to_numpy(np.int32),
        val_user=val["user_idx"].to_numpy(np.int32), val_item=val["item_idx"].to_numpy(np.int32),
        test_user=test["user_idx"].to_numpy(np.int32), test_item=test["item_idx"].to_numpy(np.int32),
        val_users=val_users.astype(np.int32), val_candidates=val_candidates.astype(np.int32),
        test_users=test_users.astype(np.int32), test_candidates=test_candidates.astype(np.int32),
        num_users=num_users, num_items=num_items,
        item_ids=np.asarray(item_ids, dtype=str),
    )
