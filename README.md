# DeFence: Robust Node Classification under Joint Label-Structure Noise

This is the official implementation of **DeFence** (*DEcoupled FEature aNChors for robust nodE classification*), submitted to CIKM 2026.

## 📦 Requirements

We recommend using the following versions:

```bash
python==3.9
torch==2.7.0+cu118
torch-geometric==2.6.1
torchvision==0.22.0+cu118
torchaudio==2.7.0+cu118
scikit-learn==1.6.1
scipy==1.13.1
pandas==2.2.3
tqdm==4.67.1
```

---

## 📁 Code Structure

* `main.py`: Entry point for training and evaluation.
* `src/models.py`: Model definitions.
* `src/train.py`: Training/evaluation loops and losses supports instance-reweighting via `--use-meta-align-weight`.
* `src/data.py`: Dataset loaders; mask normalization; noise utilities `add_edge_noise`, `drop_edge_random`, and optional split helpers.
* `src/utils.py`: Utilities for seeding, metrics, hidden embedding extraction, and class–cluster alignment.

---

## 🧪 How to Run a Demo

You can run a demo on Cora dataset with the following command:

```bash
python main.py --dataset Cora --use-meta-align-weight
```

### Main Arguments

| Argument                  | Description                                        |
| ------------------------- |----------------------------------------------------|
| `--dataset`               | Dataset name (e.g., `Cora`, `CiteSeer`, `PubMed`.) |
| `--seed`                  | Random seed.                                       |
| `--label-noise`           | Fraction of training labels to randomly flip.      |
| `--edge-noise`            | Fraction of edges to add as random spurious edges  |
| `--pair-loss-weight`      | Weight of the pairwise/consistency loss.           |
| `--align-prob-weight`     | Weight of the prediction–anchor alignment loss.    |
| `--use-meta-align-weight` | Enable meta-reweighted alignment (boolean flag).   |
