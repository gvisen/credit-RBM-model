# Credit RBM model

This repository contains the code for the paper "Universal approximation of credit portfolio losses using Restricted Boltzmann Machines" by Genovese, Nikeghbali, Serra and Visentin.

The dataset used in the paper comes from the Bloomberg Corporate Default Risk Model and is therefore proprietary. Nevertheless, this repository contains a toy dataset of default probabilities for 30 US listed companies from January 2000 to August 2020 estimated using the Merton model.

To get started, you can follow the brief tutorial in the Jupyter notebook `demo.ipynb`, where we guide you through the following tasks:
* simulation and calibration of multi-factor Gaussian copula,
* training of the credit RBM model on the toy dataset (requires a CUDA-enabled GPU).

## Code overview

The main modules are as follows:

* RBM: module implementing the RBM model in PyTorch
* copulas: module with implementation of multi-factor Gaussian and t copula models (requires TensorFlow).
* utils: module with auxiliary routines and toy dataset implementation

## Dependencies
The code has originally run with the following package versions:
* torch = 1.6.0
* tensorflow = 2.11.0
* tensorflow_probability = 0.19.0
* pandas = 1.3.5
* scipy = 1.7.3
* sklearn = 1.0.2
* matplotlib = 3.5.3
* numpy = 1.21.5