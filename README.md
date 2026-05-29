# UCB-Based Feature Engineering for Cold-Start in Recommenders

This repository contains implementations and experiments for UCB-based algorithm:

**UCB-FE (Upper Confidence Bound Feature Engineering)**: An approach for cold-start recommendation systems

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20433675">
    <img alt="Data: Zenodo DOI 10.5281/zenodo.20433675" src="https://img.shields.io/badge/Data-Zenodo%20DOI-1682D4?style=for-the-badge&logo=zenodo&logoColor=white">
  </a>
  <img alt="Main paper: To Be Done" src="https://img.shields.io/badge/Main%20Paper-To%20Be%20Done-9CA3AF?style=for-the-badge">
</p>

## Repository Structure

### UCB-FE
- [UCB-FE/src/](./UCB-FE/src/): Main code for the experiments from the original paper
- [UCB-FE/DatasetDescribe.pdf](./UCB-FE/DatasetDescribe.pdf): Detailed description of the dataset used in UCB-FE experiments
- [UCB-FE/weights/](./UCB-FE/weights/): Directory for pre-trained model weights
- [UCB-FE/data/](./UCB-FE/data/): Directory for the presented results in the article (absolute values)


## Data

The data for UCB-FE experiments can be downloaded from Zenodo:
[https://doi.org/10.5281/zenodo.20433675](https://doi.org/10.5281/zenodo.20433675)

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
