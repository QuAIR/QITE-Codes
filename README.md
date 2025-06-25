# QITE-Codes

Source code for the paper *Quantum Imaginary-Time Evolution with Polynomial Resources in Time*.

QITE is a novel quantum algorithm framework designed to simulate normalized imaginary-time evolution with rigorous performance guarantees. Unlike conventional methods that suffer from exponentially growing classical resources or deteriorating success probabilities in long evolution times, QITE introduces an adaptive normalization scheme that ensures stable and highly probable state preparation even for large imaginary-time durations.

By leveraging polynomially-scaling quantum resources in imaginary evolution time, QITE achieves high precision in approximating ground state of quantum many-body systems. It utilizes only a single ancilla qubit and a moderate number of elementary quantum gates, making it particularly suitable for near-term quantum devices.

| Visualization / Protocols in the paper      | Location in this repository                                           | Extra hardware requirement |
|--------------------|------------------------------------------------------------------|:-----------------:|
| [Tutorial for the ITE circuit](./qite.ipynb)           | `tutorial.ipynb`                              | \ |
| [Figures in the paper](./qite.ipynb)      | `qite.ipynb`                              | GPU for efficient simulation |

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

- quairkit: 0.4.0
- torch: 2.6.0+cu124
- torch cuda: 12.4
- numpy: 1.26.4
- scipy: 1.15.2
- matplotlib: 3.10.1

**System Information**:

- Python version: 3.10.16
- OS: Linux, Ubuntu (version: #63-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 15 19:04:15 UTC 2025)
- CPU: AMD EPYC 9654 96-Core Processor
- GPU: (0) NVIDIA GeForce RTX 4090

These settings ensure compatibility and optimal performance when running the qite codes.
