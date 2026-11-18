# Shopping Recommender: NCF vs. TIGER on Amazon Electronics

DSC200 course project by Shawn (Weiyang) Xu, Micah and Serena (Yueying) Tao.

Given a shopper's purchase history, the system recommends the next Top-N products. We compare three models on 1M Amazon Electronics interactions:

- a **popularity** baseline;
- **NCF**, neural collaborative filtering with an MLP;
- **TIGER**, a generative retriever that predicts the next item's *semantic ID*.

TIGER performs best. A conversational shopping agent built on top of the recommender is the next milestone.

## Results

Each user's target item is ranked against 99 sampled negatives. Every model scores the same candidates: 73,594 validation users and 74,087 test users.

| Model | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
| --- | --- | --- | --- | --- |
| Popularity | 0.5290 | 0.3320 | 0.5133 | 0.3123 |
| NCF (MLP) | 0.5674 | 0.3585 | 0.5279 | 0.3236 |
| **TIGER** | **0.6040** | **0.4067** | **0.5939** | **0.3881** |

On test, TIGER beats NCF by +12.5% HR@10 and +19.9% NDCG@10. Each user has exactly one relevant item, so Recall@10 equals HR@10. Per-epoch logs, derived metrics and caveats are in [results.json](results.json). All settings are in [config.json](config.json).

## How it works

1. **Data** ([cleaning.py](cleaning.py)): keeps the most recently active users with their full histories, up to 1M interactions (999,984 rows, 81,062 users, 209,379 items).
   - Splits each user's history by time: the last item goes to test, the second-to-last to validation.
   - Samples 99 fixed negatives per validation and test user.
2. **NCF** ([ncf_mlp.py](ncf_mlp.py)): user and item embeddings (32-d) are fed to an MLP (64→32→16→8→1).
   - Trained with 4 fresh negatives per positive each epoch.
   - Early stopping on validation NDCG@10, patience 3.
3. **Item text** ([item_text.py](item_text.py)): turns each item's title, brand, category, price and features into a 768-d Sentence-T5 vector.
4. **Semantic IDs** ([rqvae.py](rqvae.py)): an RQ-VAE quantizes each vector into 3 codes (256 per level). A 4th code makes every ID unique, so similar products share ID prefixes.
5. **TIGER** ([tiger.py](tiger.py)): a T5 encoder-decoder (4+4 layers, d_model 128, 4.8M parameters) trained from scratch.
   - Input: a user token plus the last 20 items as semantic-ID tokens (81 tokens).
   - Output: the next item's 4 tokens.
6. **Recommendation** ([recommend.py](recommend.py)): takes a user_id and prints the top N items from both models, leaving out items the user already bought.
   - NCF scores every item seen in train.
   - TIGER runs beam search (200 beams) restricted to valid semantic IDs. In our check on 20 users, this matched exact scoring of all 209,379 items for every user.

## Repository layout

```
cleaning.py       data subset, time-based split, evaluation candidates
ncf_mlp.py        NCF model, training and evaluation
item_text.py      item metadata -> Sentence-T5 embeddings
rqvae.py          RQ-VAE -> semantic IDs
tiger.py          TIGER training and evaluation (--smoke, --test)
recommend.py      Top-N recommendations for a user_id
colab_run.ipynb   runs tiger.py on a Colab GPU with checkpoints on Google Drive
config.json       all settings used in the reported runs
results.json      all metrics (per epoch, validation and test)
logs/             training and check logs
docs/             project proposal
data/             split, semantic IDs, 1M-interaction subset, item catalog
models/           trained NCF and TIGER weights
```

## Setup

Python 3.11 with a CUDA GPU. We used an RTX 3060 12 GB locally and a Colab T4 for TIGER.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Included data and weights (about 210 MB).** The repository includes everything needed to run the [Top-N demo](#top-n-demo) and `python tiger.py --test` without retraining:
- `data/split.npz`, `data/semantic_ids.npy`, `data/Electronics_1M.csv` and `data/item_catalog.csv`;
- `models/ncf_mlp_best.pt` and `models/tiger_best.pt`.

**Raw data (only to rebuild from scratch).** These files are too large for GitHub. Download them from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html) into `data/`:

- `Electronics.csv`: the ratings-only file, with columns user_id, parent_asin, rating, timestamp. `cleaning.py` needs it only if `data/Electronics_1M.csv` is missing.
- `meta_Electronics.jsonl.gz`: the item metadata, needed by `item_text.py`.

## Reproduce

To retrain, run the scripts from the repository root, in this order. `cleaning.py` and `item_text.py` reuse their output files if they already exist. `ncf_mlp.py` and `rqvae.py` retrain and overwrite theirs, and `tiger.py` resumes from its last checkpoint.

```bash
python cleaning.py        # -> data/Electronics_1M.csv, data/split.npz
python ncf_mlp.py         # -> models/ncf_mlp_best.pt
python item_text.py       # -> data/meta_Electronics_1M.jsonl, data/item_emb.npy
python rqvae.py           # -> models/rqvae.pt, data/semantic_ids.npy
python tiger.py --smoke   # optional 1-epoch check on 2,000 users
python tiger.py           # -> models/tiger_best.pt, models/tiger_last.pt, logs/tiger.log
python tiger.py --test    # score the saved TIGER weights on test only
```

TIGER takes about 13 minutes per epoch on a T4: 6 minutes to train and 7 to evaluate. It saves a checkpoint after every epoch and resumes from it when restarted. [colab_run.ipynb](colab_run.ipynb) runs it on Colab and keeps the checkpoints on Google Drive.

The popularity baseline ranks items by their number of train interactions. It has no separate script yet; its numbers are in [results.json](results.json).

## Top-N demo

```bash
python recommend.py AHQ4LW3VLOY44LFZJJNRINUW7DYA --n 5
python recommend.py --n 10 --model tiger    # random user, TIGER only
```

Example output (titles shortened):

```
User: AHQ4LW3VLOY44LFZJJNRINUW7DYA (11 purchases)
Recent purchases (newest first):
  - Cowyawn Toddler Headphones Ultra Light Comfort Kids Headphones ...
  - Statik Light Up Charger Cable - GloBright Braided LED Charging Cable ...
  ...

NCF top-5 recommendations:
   1. TOZO T10 Bluetooth 5.3 Wireless Earbuds ...  (TOZO, $21.99)
   2. Fire TV Stick 4K streaming device with Alexa Voice Remote ...  (Amazon)
   ...

TIGER top-5 recommendations:
   1. Cleaner Kit for Airpods Pro 1 2 3 Multi-Function Cleaning Pen ...  (Dotbox, $6.99)
   2. KEEPXYZ Genuine Leather Airtag Holder ...  (KEEPXYZ, $12.95)
   ...
```

NCF tends to recommend popular items. TIGER follows the user's recent purchases (AirPods and iPad accessories in this example).

## Limitations

- Evaluation is sampled (1 target + 99 negatives), not a full ranking over all items. It favours popular items: the popularity baseline alone reaches 0.51 test HR@10.
- Validation and test targets that never appear in train are left out of evaluation (9.2% and 8.6%).
- One category (Electronics) and one configuration per model; no hyperparameter search.
- TIGER stopped at epoch 22, when Colab compute ran out, while validation NDCG was still improving.
- NCF was selected on validation but not retrained on train + validation before testing.
