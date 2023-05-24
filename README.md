# Group Robust Classification Without Any Group Information
This PyTorch repository contains code for the reproduction of experiments presented at the NeurIPS 2023 submission.

## Datasets
Please find instructions for the benchmarks we consider in the paper below:  
- For the `sMPI3D` task, which is based on the [MPI3D robotics dataset](https://github.com/rr-learning/disentanglement_dataset) for disentanglement
, please follow the instructions provided in `/ula/data/mpi3d.py` Python file.
The same file can be used to generate the systematic splits.  
- For the colored MNIST (`cMNIST`) and corrupted CIFAR10 (`cCIFAR10`) tasks, please follow the instructions provided by the [DFA](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled) code repository.  
- For the CelebA and Waterbirds tasks, instructions can be found at the [Just-Train-Twice](https://github.com/anniesch/jtt) code repository.

## Execute
In `/scripts` directory, you can find two example scripts to reproduce the `uLA` training and validation methodology.
First, execute the bash script `mocov2plus.py` in order to pretrain a base network on the selected dataset.
Please find instructions inside.
Training will output checkpoints for the pretrained network in an experiment logging directory.
Second, use a checkpoint for the pretrained base network to execute the `ula.py` script.
This script will instantiate the base network,
train a proxy for the bias variable by fitting a linear layer on top of the extracted representation,
and then use the bias proxy to train a debiased network with logit adjustment and
to perform bias-unsupervised validation on the debiased network, as it trains, at each epoch.
