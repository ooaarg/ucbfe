# UCB-Based Feature Engineering for Cold-Start in Recommenders

This repository contains implementations and experiments for UCB-based algorithm:

**UCB-FE (Upper Confidence Bound Feature Engineering)**: An approach for cold-start recommendation systems

The link for the paper:
[Link to be provided]

## Repository Structure

### UCB-FE
- [UCB-FE/src/](./UCB-FE/src/): Main code for the experiments from the original paper
- [UCB-FE/DatasetDescribe.pdf](./UCB-FE/DatasetDescribe.pdf): Detailed description of the dataset used in UCB-FE experiments
- [UCB-FE/weights/](./UCB-FE/weights/): Directory for pre-trained model weights
- [UCB-FE/data/](./UCB-FE/data/): Directory for the presented results in the article (absolute values)


## Data

The data for UCB-FE experiments can be downloaded from:
[Link to be provided]

## Installation

1. Clone the repository:
```bash
git clone git@github.com:ooaarg/ucbfe.git
```

2. Install dependencies:
```bash
conda env create -n ucbBasedFE -f environment.yml
```

## Running Experiments

### UCB-FE
To run the UCB-FE experiments you can use script [UCB-FE/run.sh](./UCB-FE/src/run.sh).
