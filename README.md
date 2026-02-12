# UCB-Based Feature Engineering for Cold-Start in Recommenders

This repository contains implementations and experiments for UCB-based algorithm:

 **UCB-FE (Upper Confidence Bound Feature Engineering)**: An approach for cold-start recommendation systems

## Repository Structure

### UCB-FE
- [UCB-FE/src/](./UCB-FE/src/): Main code for the experiments from the original paper
- [UCB-FE/DatasetDescribe.pdf](./UCB-FE/DatasetDescribe.pdf): Detailed description of the dataset used in UCB-FE experiments
- [UCB-FE/weights/](./UCB-FE/weights/): Directory for pre-trained model weights
- [UCB-FE/data/](./UCB-FE/data/): Directory for the presented results in the article (absolute values)


### Theory
- [Theory/pbm_ucb.py](./Theory/pbm_ucb.py): Implementation of the theoretical algorithm
- [Theory/PBM_UCB.ipynb](./Theory/PBM_UCB.ipynb): Notebook with running results and analysis

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
To run the UCB-FE experiments you can use script [UCB-FE/run.sh](./UCB-FE/src/run.sh).