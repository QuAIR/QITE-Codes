# QITE-codes Documentation

Source code for the paper [Quantum Imaginary-Time Evolution with Polynomial Resources in Time](http://arxiv.org/abs/2507.00908).

Preparing the normalized imaginary-time evolution state (ITE state) is a useful tool for problems in quantum many-body system, while current methods that suffer from exponentially growing classical resources or deteriorating success probabilities in long evolution time.
This repository gives the simulation code for a quantum algorithm that introduces a normalization scheme that ensures state preparation for large imaginary-time durations with stable success probability, and a quantum algorithm using the idea of ITE to solve ground-state-related problems.

| Visualization / Protocols in the paper      | Location in this repository                                           | Extra hardware requirement |
|--------------------|------------------------------------------------------------------|:-----------------:|
| [Tutorial for the ITE circuit](./code/tutorial.ipynb)           | `tutorial.ipynb`                              | \ |
| [Figure 1](./code/normalization.ipynb)           | `./code/normalization.ipynb`                              | \ |
| [Figure 2(a)](./code/long%20evolution.ipynb)           | `./code/long evolution.ipynb`                              | \ |
| [Figure 2(b, c)](./code/ground%20state.ipynb)           | `./code/ground state.ipynb`                              | \ |

## Repository Structure

```plaintext
QITE-CODES/
├── code/                    # code for reproducible experiment for the paper
│   ├── data                 # QSP angles and Fourier coefficients
├── qite/                    # source code for the ITE algorithm in the paper
└── qsp/                     # source code for the QSP-related functional
```

## How to Run These Files?

We recommend running these files by creating a virtual environment using `conda` and install Jupyter Notebook. We recommend using Python `3.10` for compatibility.

```bash
conda create -n qite python=3.10
conda activate qite
conda install jupyter notebook
```

These codes are highly dependent on the [QuAIRKit](https://github.com/QuAIR/QuAIRKit) package no lower than v0.4.0. This package is featured for batch and qudit computations in quantum information science and quantum machine learning. The minimum Python version required for QuAIRKit is `3.8`.

To install QuAIRKit, run the following commands:

```bash
pip install quairkit
```

## System and Package Versions

It is recommended to run these files on a server with high performance. Below are our environment settings:

**Package Versions**:

- quairkit: 0.4.3
- torch: 2.6.0+cu124
- torch cuda: 12.4
- numpy: 1.26.4
- scipy: 1.15.2
- matplotlib: 3.10.1

**System Information**:

- Python version: 3.10.16
- OS: Linux, Ubuntu (version: #63-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 15 19:04:15 UTC 2025)
- CPU: AMD EPYC 9654 96-Core Processor
- GPU: NVIDIA GeForce RTX 4090

These settings ensure compatibility and optimal performance when running the codes.
