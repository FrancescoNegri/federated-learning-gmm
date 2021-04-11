# Federated Learning GMM
### _An implementation of the Gaussian Mixture Model according to federated learning paradigm_

The aim of this project is demonstrating an effective implementation of the Gaussian Mixture Model (GMM) with Expectation-Maximization (EM) algorithm according to the vanilla federated learning paradigm as decribed in the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

The Gaussian Mixture Model is employed in unsupervised learning problems, especially in clustering tasks. This repository allows to execute alternatively a baseline local version of GMM and a federated distributed implementation of the same model, in order to compare their performance.

## Parameters
| Name | Description | Default | Baseline | Federated |
|:----:|:-----------:|:-------:|:--------:|:---------:|
| `--dataset` | Name of the dataset. | _iris_ | X | X |
| `--components` | Number of Gaussians to fit. | _3_ | X | X |
| `--random_seed` | Number to have random consistent results across executions. | _None_| X | X |
| `--samples` | Number of samples to generate. | _10000_ | X | X |
| `--features` | Number of features for each generated sample. | _2_ | X | X |
| `--soft_clustering` | Specifies if cluster bounds are soft or hard. | _True_ | X | X |
| `--plots_3d` | Specifies if plots are to be done in 3D or 2D. | _False_ | X | X |
| `--epochs` | Number of epochs of training. | _100_ | X |  |
| `--rounds` | Number of rounds of training. | _10_ |  | X |
| `--K` | Total number of clients. | _100_ |  | X |
| `--C` | Fraction of clients to employ in each round. From 0 to 1. | _0.1_ |  | X |

## Datasets

## Commands

## Requirements

## References

## License

MIT