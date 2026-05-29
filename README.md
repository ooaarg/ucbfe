# UCB-Based Feature Engineering for Cold-Start in Recommenders - [![Paper][paper-image]][paper] [![Data][data-image]][data] [![Python][python-image]][python]

*A model-agnostic feature engineering method for improving cold-start item ranking in CTR-based recommender systems.*

This repository contains the source code and experiment setup for the KDD 2026 paper [UCB-Based Feature Engineering for Cold-Start in Recommenders][paper]. UCB-FE transforms behavioral features at inference time using position-aware upper confidence bounds, improving cold-item exploration while preserving the quality of existing CTR models.

## What is UCB-FE?

Industrial ranking systems often sort items by predicted Click-Through Rate (CTR). These models rely heavily on behavioral features such as historical clicks, impressions, and CTR. For new or rarely shown items, those signals are sparse, so otherwise relevant cold items can be underestimated and pushed into the long tail of search results.

UCB-FE addresses this by applying Upper Confidence Bound (UCB) feature engineering to behavioral features during inference. Instead of changing model weights or architecture, UCB-FE replaces uncertain behavioral feature values for cold items with position-aware optimistic estimates and then reuses the original trained CTR model. The repository also includes the `coldNDCG` evaluation setup for measuring ranking quality when cold-item performance matters.

## Repository Structure

- [UCB-FE/src/](./UCB-FE/src/): Main experiment code for UCB-FE.
- [UCB-FE/src/run.sh](./UCB-FE/src/run.sh): End-to-end script for reproducing the UCB-FE experiments.
- [UCB-FE/src/models/](./UCB-FE/src/models/): CTR model implementations and configurations.
- [UCB-FE/src/run/](./UCB-FE/src/run/): Dataset building, prediction, and metric preparation pipeline.
- [UCB-FE/DatasetDescribe.pdf](./UCB-FE/DatasetDescribe.pdf): Dataset description for the released UCB-FE data.
- [UCB-FE/data/](./UCB-FE/data/): Data/results directory used by the experiment pipeline.
- [UCB-FE/weights/](./UCB-FE/weights/): Directory for trained model weights.

## Data

The data for UCB-FE experiments can be downloaded from Zenodo:

[https://doi.org/10.5281/zenodo.20433675][data]

After downloading the data, place the files according to the paths expected by the model configuration files under [UCB-FE/src/models/](./UCB-FE/src/models/).

## Installation

Clone the repository and create the Conda environment:

```bash
git clone git@github.com:ooaarg/ucbfe.git
cd ucbfe
conda env create -f environment.yml
conda activate ucbBasedFE
```

## Running Experiments

Run the full UCB-FE experiment script:

```bash
cd UCB-FE/src
bash run.sh
```

You can also run an individual pipeline configuration directly:

```bash
python run/pipeline.py --dataset mq2007 --cold-periods 0 10 100 200 500 --experiment FinalNet_mq2007
```

[paper-image]: https://img.shields.io/badge/Paper-ACM%20DOI-blue.svg
[paper]: https://doi.org/10.1145/3770855.3817886
[data-image]: https://zenodo.org/badge/DOI/10.5281/zenodo.20433675.svg
[data]: https://doi.org/10.5281/zenodo.20433675
[python-image]: https://img.shields.io/badge/python-3.10-blue.svg
[python]: ./environment.yml
