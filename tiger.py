import math
import os
import sys
import time

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# Run locally from the project root: data is read from data/ and models are saved to models/
# On Colab all files sit in one folder (no data/ or models/), so the current folder is used
# python tiger.py --smoke: 1 epoch on a few users, to check the code for bugs
SMOKE = "--smoke" in sys.argv
# python tiger.py --test: no training, evaluate models/tiger_best.pt on the test set
TEST_ONLY = "--test" in sys.argv
PREFIX = "smoke_" if SMOKE else ""
SMOKE_USERS = 2000

MAX_HIST = 20  # use each user's 20 most recent items
NUM_USER_TOKENS = 2000  # hash user IDs into 2000 tokens (as in TIGER)
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
MAX_EPOCHS = 0 if TEST_ONLY else (1 if SMOKE else 30)
PATIENCE = 3
TOP_K = 10
EVAL_USER_BATCH = 64

DATA_DIR = "data" if os.path.isdir("data") else "."
MODEL_DIR = "models" if os.path.isdir("models") else "."

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("logs", exist_ok=True)

# Mixed precision for training: bf16 on GPUs such as the 3060 / A100 / L4; Colab's free T4 has no bf16, so fp16 + GradScaler
# Evaluation stays in fp32, so low precision cannot create score ties that change the ranks
USE_AMP = device.type == "cuda"
AMP_DTYPE = torch.bfloat16 if USE_AMP and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def log(msg):
    """Print and append to logs/tiger.log (on Colab the log stays in Drive)"""
    print(msg, flush=True)
    with open(f"logs/{PREFIX}tiger.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


split = np.load(f"{DATA_DIR}/split.npz")
semantic_ids = np.load(f"{DATA_DIR}/semantic_ids.npy")

# Vocabulary: 0 = PAD (also the decoder start token), 1 = EOS (unused), then the tokens of the 4 code levels and the user tokens
# Each level has its own token range, so code 17 at level 1 and code 17 at level 2 are different tokens
PAD, EOS = 0, 1
LEVEL_SIZES = [256, 256, 256, int(semantic_ids[:, 3].max()) + 1]
LEVEL_OFFSETS = 2 + np.cumsum([0] + LEVEL_SIZES[:-1])
USER_OFFSET = 2 + sum(LEVEL_SIZES)
VOCAB_SIZE = USER_OFFSET + NUM_USER_TOKENS

item_tokens = (semantic_ids + LEVEL_OFFSETS).astype(np.int16)  # 4 tokens per item
item_tokens_gpu = torch.as_tensor(item_tokens, dtype=torch.long, device=device)


def build_inputs(seq_items, users, starts, ends):
    """Each row: [user token] + tokens of the last MAX_HIST items of seq_items[starts:ends] (oldest to newest), padded with PAD at the end"""
    pos = ends[:, None] - MAX_HIST + np.arange(MAX_HIST)
    valid = pos >= starts[:, None]
    items = seq_items[np.maximum(pos, 0)]

    # Move the valid items to the front and the empty slots to the back as PAD
    order = np.argsort(~valid, axis=1, kind="stable")
    valid = np.take_along_axis(valid, order, axis=1)
    items = np.take_along_axis(items, order, axis=1)

    hist = np.where(valid[..., None], item_tokens[items], PAD).reshape(len(users), -1)
    user_tok = (USER_OFFSET + users % NUM_USER_TOKENS).astype(np.int16)
    return torch.as_tensor(np.column_stack([user_tok, hist]).astype(np.int16))


def load_samples():
    """Build the training samples and the val/test inputs. Only called for training and testing, so importing tiger (e.g. from recommend.py) builds nothing"""
    # Training samples (sliding window): every train interaction except each user's first is a target, with the earlier ones as history
    train_user, train_item = split["train_user"], split["train_item"]
    user_start = np.searchsorted(train_user, train_user, side="left")
    sample_pos = np.arange(len(train_user))
    sample_pos = sample_pos[sample_pos > user_start]
    if SMOKE:
        sample_pos = sample_pos[train_user[sample_pos] < SMOKE_USERS]
    train_x = build_inputs(train_item, train_user[sample_pos], user_start[sample_pos], sample_pos)
    train_y = torch.as_tensor(item_tokens[train_item[sample_pos]])

    # val: the history is the train part
    val_users, val_candidates = split["val_users"], split["val_candidates"]
    if SMOKE:
        val_users, val_candidates = val_users[:SMOKE_USERS], val_candidates[:SMOKE_USERS]
    val_x = build_inputs(train_item, val_users,
                         np.searchsorted(train_user, val_users, side="left"),
                         np.searchsorted(train_user, val_users, side="right"))

    # test: the history is train plus the val interaction (val comes after train in time)
    hist_user = np.concatenate([train_user, split["val_user"]])
    hist_item = np.concatenate([train_item, split["val_item"]])
    order = np.argsort(hist_user, kind="stable")
    hist_user, hist_item = hist_user[order], hist_item[order]
    test_users, test_candidates = split["test_users"], split["test_candidates"]
    if SMOKE:
        test_users, test_candidates = test_users[:SMOKE_USERS], test_candidates[:SMOKE_USERS]
    test_x = build_inputs(hist_item, test_users,
                          np.searchsorted(hist_user, test_users, side="left"),
                          np.searchsorted(hist_user, test_users, side="right"))

    return train_x, train_y, val_x, val_candidates, test_x, test_candidates


def build_model():
    # Following TIGER: 4 encoder and 4 decoder layers, d_model 128, 6 attention heads, FFN 1024, trained from scratch
    # The sequences are short (81 tokens), so eager attention is faster than the default sdpa
    config = T5Config(
        vocab_size=VOCAB_SIZE, d_model=128, d_kv=64, d_ff=1024,
        num_layers=4, num_decoder_layers=4, num_heads=6,
        dropout_rate=0.1, feed_forward_proj="relu",
        pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD,
        attn_implementation="eager",
    )
    return T5ForConditionalGeneration(config)


@torch.no_grad()
def evaluate(model, enc_x, candidates):
    """Rank each user's target among 1 target + 99 negatives; score = sum of the log-probs of generating the item's 4 tokens"""
    model.eval()
    num_cand = candidates.shape[1]
    ranks = []
    for start in range(0, len(enc_x), EVAL_USER_BATCH):
        x = enc_x[start:start + EVAL_USER_BATCH].to(device).long()
        mask = x != PAD

        # Run the encoder once per user and share the output across the 100 candidates
        enc = model.get_encoder()(input_ids=x, attention_mask=mask).last_hidden_state
        enc = enc.repeat_interleave(num_cand, dim=0)
        mask = mask.repeat_interleave(num_cand, dim=0)

        cand = torch.as_tensor(candidates[start:start + EVAL_USER_BATCH], device=device).long()
        labels = item_tokens_gpu[cand.reshape(-1)]
        logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                       attention_mask=mask, labels=labels).logits
        scores = torch.log_softmax(logits, dim=-1).gather(-1, labels[..., None]).squeeze(-1).sum(-1)
        scores = scores.view(-1, num_cand)

        # Rank = number of negatives scoring strictly higher than the target (column 0), starting at 0
        ranks.append((scores[:, 1:] > scores[:, :1]).sum(dim=1))

    ranks = torch.cat(ranks).float()
    hit = ranks < TOP_K
    hr = hit.float().mean().item()
    ndcg = torch.where(hit, 1.0 / torch.log2(ranks + 2), torch.zeros_like(ranks)).mean().item()
    return hr, ndcg


def train_one_epoch(model, optimizer, scheduler, scaler, epoch, train_x, train_y):
    model.train()
    torch.manual_seed(42 + epoch)  # fixed shuffle order and dropout per epoch, so resumed training is reproducible
    perm = torch.randperm(len(train_x))

    total_loss = 0.0
    for start in range(0, len(perm), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        x = train_x[idx].to(device).long()
        y = train_y[idx].to(device).long()
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            loss = model(input_ids=x, attention_mask=x != PAD, labels=y).loss  # cross-entropy over the 4 tokens

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(idx)

    return total_loss / len(perm)


if __name__ == "__main__":
    train_x, train_y, val_x, val_candidates, test_x, test_candidates = load_samples()
    log(f"Device: {device} | training precision: {AMP_DTYPE if USE_AMP else torch.float32} | "
        f"vocab size: {VOCAB_SIZE} | level-4 codes: {LEVEL_SIZES[3]}")
    log(f"Training samples: {len(train_x)} | val users: {len(val_x)} | test users: {len(test_x)} | input length: {train_x.shape[1]}")

    model = build_model().to(device)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Warmup, then inverse square-root decay by step (as in T5 / TIGER)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / WARMUP_STEPS, math.sqrt(WARMUP_STEPS / (step + 1))))
    scaler = torch.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE == torch.float16)  # only fp16 needs it

    # Resume: if a checkpoint exists, continue where the last run stopped
    last_path, best_path = f"{MODEL_DIR}/{PREFIX}tiger_last.pt", f"{MODEL_DIR}/{PREFIX}tiger_best.pt"
    start_epoch, best_ndcg, bad_epochs = 1, -1.0, 0
    if os.path.exists(last_path) and not TEST_ONLY:
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_ndcg, bad_epochs = ckpt["epoch"] + 1, ckpt["best_ndcg"], ckpt["bad_epochs"]
        log(f"Resuming from the epoch {ckpt['epoch']} checkpoint")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if bad_epochs >= PATIENCE:
            break
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, scheduler, scaler, epoch, train_x, train_y)
        t1 = time.time()
        hr, ndcg = evaluate(model, val_x, val_candidates)
        log(f"Epoch {epoch:2d} | loss {loss:.4f} | Val HR@{TOP_K} {hr:.4f} | Val NDCG@{TOP_K} {ndcg:.4f} | "
            f"train {t1 - t0:.0f}s, eval {time.time() - t1:.0f}s")

        # Early stopping: stop when val NDCG has not improved for PATIENCE epochs; keep the best model
        if ndcg > best_ndcg:
            best_ndcg, bad_epochs = ndcg, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_ndcg": best_ndcg, "bad_epochs": bad_epochs}, last_path)
        if bad_epochs >= PATIENCE:
            log("Early stopping")

    model.load_state_dict(torch.load(best_path, map_location=device))
    hr, ndcg = evaluate(model, test_x, test_candidates)
    log(f"Test HR@{TOP_K} {hr:.4f} | Test NDCG@{TOP_K} {ndcg:.4f}")
