import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 32
NUM_LEVELS = 3
CODEBOOK_SIZE = 256
BETA = 0.25
BATCH_SIZE = 1024
LR = 1e-3
EPOCHS = 300
FREEZE_RESET_EPOCHS = 50  # no code resets in the last 50 epochs, so the codebooks can settle
LOG_EVERY = 25


def kmeans(x, k, iters=20):
    """Simple k-means; returns k centers"""
    centers = x[torch.randperm(len(x), device=x.device)[:k]].clone()
    for _ in range(iters):
        idx = torch.cdist(x, centers).argmin(dim=1)
        sums = torch.zeros_like(centers).index_add_(0, idx, x)
        counts = torch.bincount(idx, minlength=k)
        filled = counts > 0
        centers[filled] = sums[filled] / counts[filled, None]
    return centers


class RQVAE(nn.Module):
    def __init__(self, in_dim=768, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder 768 → 512 → 256 → 128 → 32; the decoder mirrors it
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, in_dim),
        )
        self.codebooks = nn.ModuleList([nn.Embedding(CODEBOOK_SIZE, latent_dim) for _ in range(NUM_LEVELS)])

    def quantize(self, z):
        """Residual quantization: pick the nearest code at each level, subtract it, and pass the residual to the next level"""
        residual = z
        z_q = torch.zeros_like(z)
        codes, residuals = [], []
        quant_loss = 0.0
        for codebook in self.codebooks:
            residuals.append(residual)
            idx = torch.cdist(residual, codebook.weight).argmin(dim=1)
            e = codebook(idx)

            # The codebook loss pulls codes toward the residuals; the commitment loss keeps the encoder output near its codes
            quant_loss = quant_loss + ((residual.detach() - e) ** 2).sum(-1).mean() \
                + BETA * ((residual - e.detach()) ** 2).sum(-1).mean()

            z_q = z_q + e
            residual = residual - e.detach()
            codes.append(idx)

        # Straight-through estimator: the forward pass uses the quantized vector, the backward pass sends gradients straight to the encoder
        z_q = z + (z_q - z).detach()
        return z_q, torch.stack(codes, dim=1), quant_loss, residuals

    def forward(self, x):
        z_q, codes, quant_loss, _ = self.quantize(self.encoder(x))
        recon_loss = ((self.decoder(z_q) - x) ** 2).sum(-1).mean()
        return recon_loss, quant_loss, codes

    @torch.no_grad()
    def init_codebooks(self, x):
        """Initialize each level's codebook with k-means (as in TIGER), so few codes start out unused"""
        residual = self.encoder(x)
        for codebook in self.codebooks:
            centers = kmeans(residual, CODEBOOK_SIZE)
            codebook.weight.copy_(centers)
            residual = residual - centers[torch.cdist(residual, centers).argmin(dim=1)]

    @torch.no_grad()
    def reset_dead_codes(self, x, usage):
        """Replace codes unused this epoch with random items' residuals at that level; returns how many were reset"""
        _, _, _, residuals = self.quantize(self.encoder(x))
        num_reset = 0
        for level, codebook in enumerate(self.codebooks):
            dead = (usage[level] == 0).nonzero().squeeze(1)
            if len(dead) > 0:
                pick = torch.randint(len(x), (len(dead),), device=x.device)
                codebook.weight[dead] = residuals[level][pick]
                num_reset += len(dead)
        return num_reset


@torch.no_grad()
def get_codes(model, x, batch_size=8192):
    model.eval()
    codes = [model.quantize(model.encoder(x[s:s + batch_size]))[1] for s in range(0, len(x), batch_size)]
    return torch.cat(codes)


def train(model, x):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        usage = torch.zeros(NUM_LEVELS, CODEBOOK_SIZE, device=device)
        perm = torch.randperm(len(x), device=device)
        total_recon = total_quant = 0.0

        for start in range(0, len(x), BATCH_SIZE):
            batch = x[perm[start:start + BATCH_SIZE]]
            recon_loss, quant_loss, codes = model(batch)
            loss = recon_loss + quant_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for level in range(NUM_LEVELS):
                usage[level] += torch.bincount(codes[:, level], minlength=CODEBOOK_SIZE)
            total_recon += recon_loss.item() * len(batch)
            total_quant += quant_loss.item() * len(batch)

        num_reset = 0
        if epoch <= EPOCHS - FREEZE_RESET_EPOCHS:
            sample = x[torch.randint(len(x), (8192,), device=device)]
            num_reset = model.reset_dead_codes(sample, usage)

        if epoch == 1 or epoch % LOG_EVERY == 0:
            all_codes = get_codes(model, x)
            used = [len(torch.unique(all_codes[:, level])) for level in range(NUM_LEVELS)]
            unique_ids = len(torch.unique(all_codes, dim=0))
            print(f"Epoch {epoch:3d} | recon {total_recon / len(x):.4f} | quant {total_quant / len(x):.4f} | "
                  f"codes used {used} | unique IDs {unique_ids} ({unique_ids / len(x):.1%}) | reset {num_reset}")


if __name__ == "__main__":
    print("Device:", device)
    x = torch.as_tensor(np.load("data/item_emb.npy"), dtype=torch.float32, device=device)
    print("Item embedding shape:", tuple(x.shape))

    model = RQVAE(in_dim=x.shape[1]).to(device)
    model.init_codebooks(x)
    train(model, x)
    torch.save(model.state_dict(), "models/rqvae.pt")

    # Each item's 3-level code (c1, c2, c3)
    codes = get_codes(model, x).cpu().numpy()

    # Items with identical 3-level codes get a 4th code c4 = 0, 1, 2, ..., so every Semantic ID is unique
    c4 = pd.DataFrame(codes).groupby([0, 1, 2]).cumcount().to_numpy()
    semantic_ids = np.column_stack([codes, c4])

    group_size = pd.DataFrame(codes).groupby([0, 1, 2]).size()
    print("Distinct (c1, c2, c3):", len(group_size))
    print(f"Items sharing their code with another item: {(group_size[group_size > 1]).sum()} "
          f"({(group_size[group_size > 1]).sum() / len(codes):.1%})")
    print("Largest group of colliding items:", group_size.max(), "items")
    print("Codes used per level:", [len(np.unique(codes[:, level])) for level in range(NUM_LEVELS)])

    np.save("data/semantic_ids.npy", semantic_ids)
    print("Semantic ID shape:", semantic_ids.shape)
    print(semantic_ids[:5])
