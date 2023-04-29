# MSci Project: Classification of Brain Scans with Quantum Machine Learning

## Introduction

In this project, we created quantum machine learning models that classified images in the SMIR BRATS 2015 dataset of brain scans based on the presence of a tumour.
We used the Quanvolution algorithm for feature extraction, and created hybrid quantum-classical neural networks for classification. 
Initially, we analysed these algorithms separately to understand what factors affected their performance and identify any avenues in which they achieve a quantum
advantage.
We later formed complete models that used quantum algorithms at each step of the pipeline and compared them to their classical counterparts. We created a fully quantum
model with a final accuracy of 85.92% and a loss of 0.329. It's classical counterpart had a final accuracy of 81.81% and a loss of 0.429. Both models had a similar 
structure and size, only the method of computation was different. Here, we present the code from our project.

## Dependencies

This project was written completely in Python 3.7, it should be noted that Python 3.9 is **not** compatible with some of the libraries used. Below are the libraries that
we used:
- Numpy
- PyTorch and TorchVision
- Pandas
- TQDM
- Matplotlib
- TorchQuantum (can be found [here](https://github.com/mit-han-lab/torchquantum))
- Qiskit

