import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_DIM = 32
NUM_NEG = 4
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 20
PATIENCE = 3
TOP_K = 10
EVAL_USER_BATCH = 4096


class NCF_MLP(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=EMB_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # Tower: 64 → 32 → 16 → 8 → 1
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.out = nn.Linear(8, 1)

    def forward(self, users, items):
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=1)
        return self.out(self.mlp(x)).squeeze(-1)  # returns a logit; the loss applies the sigmoid


@torch.no_grad()
def evaluate(model, users, candidates):
    """Rank each user's target among 1 target + 99 negatives; returns HR@K and NDCG@K"""
    model.eval()
    ranks = []
    for start in range(0, len(users), EVAL_USER_BATCH):
        u = torch.as_tensor(users[start:start + EVAL_USER_BATCH], device=device)
        c = torch.as_tensor(candidates[start:start + EVAL_USER_BATCH], device=device)

        scores = model(u[:, None].expand_as(c).reshape(-1), c.reshape(-1)).view(c.shape)

        # Rank = number of negatives scoring strictly higher than the target (column 0), starting at 0
        ranks.append((scores[:, 1:] > scores[:, :1]).sum(dim=1))

    ranks = torch.cat(ranks).float()
    hit = ranks < TOP_K
    hr = hit.float().mean().item()
    ndcg = torch.where(hit, 1.0 / torch.log2(ranks + 2), torch.zeros_like(ranks)).mean().item()
    return hr, ndcg


def train_one_epoch(model, optimizer, loss_fn):
    model.train()
    users, items, labels = data.sample_negatives(NUM_NEG)
    users = torch.as_tensor(users, device=device)
    items = torch.as_tensor(items, device=device)
    labels = torch.as_tensor(labels, device=device)

    total_loss = 0.0
    for start in range(0, len(labels), BATCH_SIZE):
        end = start + BATCH_SIZE
        logits = model(users[start:end], items[start:end])
        loss = loss_fn(logits, labels[start:end])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(logits)

    return total_loss / len(labels)


if __name__ == "__main__":
    # Data processing lives here, so importing ncf_mlp elsewhere (e.g. recommend.py) only loads the model and never runs cleaning.py
    import cleaning as data

    print("Device:", device)
    model = NCF_MLP(data.num_users, data.num_items).to(device)

    # Weight decay on the embeddings only
    optimizer = torch.optim.Adam([
        {"params": list(model.user_emb.parameters()) + list(model.item_emb.parameters()),
         "weight_decay": WEIGHT_DECAY},
        {"params": list(model.mlp.parameters()) + list(model.out.parameters()),
         "weight_decay": 0.0},
    ], lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    best_ndcg, bad_epochs = -1.0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, loss_fn)
        hr, ndcg = evaluate(model, data.val_users, data.val_candidates)
        print(f"Epoch {epoch:2d} | loss {loss:.4f} | Val HR@{TOP_K} {hr:.4f} | Val NDCG@{TOP_K} {ndcg:.4f}")

        # Early stopping: stop when val NDCG has not improved for PATIENCE epochs; keep the best model
        if ndcg > best_ndcg:
            best_ndcg, bad_epochs = ndcg, 0
            torch.save(model.state_dict(), "models/ncf_mlp_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load("models/ncf_mlp_best.pt"))
    hr, ndcg = evaluate(model, data.test_users, data.test_candidates)
    print(f"Test HR@{TOP_K} {hr:.4f} | Test NDCG@{TOP_K} {ndcg:.4f}")
