# UCB-Based Feature Engineering for Cold-Start in Recommenders

This repository contains implementations and experiments for two bandit-based algorithms:

1. **UCB-FE (Upper Confidence Bound Feature Engineering)**: An approach for cold-start recommendation systems
2. **AuctionUCB-PBM**: A bandit-based algorithm for ranking with a fixed fee in advertising platforms

## Repository Structure

### UCB-FE
- [UCB-FE/UCB-FE-exp.ipynb](./UCB-FE/UCB-FE-exp.ipynb): Main experiment notebook for UCB-FE
- [UCB-FE/DatasetDescribe.pdf](./UCB-FE/DatasetDescribe.pdf): Detailed description of the dataset used in UCB-FE experiments
- [UCB-FE/weights/](./UCB-FE/weights/): Directory for pre-trained model weights
- [UCB-FE/data/](./UCB-FE/data/): Directory for pre-trained model weights


### AuctionUCB-PBM
- [AuctionUCB-PBM/pbm_ucb.py](./AuctionUCB-PBM/pbm_ucb.py): Implementation of the PBM-UCB algorithm
- [AuctionUCB-PBM/PBM_UCB.ipynb](./AuctionUCB-PBM/PBM_UCB.ipynb): Notebook with running results and analysis

## Data and Weights

The pre-trained weights and data for UCB-FE experiments can be downloaded from:
[Link to be provided]

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
conda env create -n ucbBasedFE -f environment.yml
```

## Running Experiments

### UCB-FE
To run the UCB-FE experiments:

```bash
python UCB-FE/utils_ucb_fe.py
```
