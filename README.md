

## 📖 Overview

This repository documents a progressive journey into Deep Learning using **PyTorch**. It contains two main projects exploring the fundamental architectures of neural networks: moving from **Multi-Layer Perceptrons (MLP)** for basic classification to **Convolutional Neural Networks (CNN)** for advanced image processing.

The experiments focus on understanding the mathematical impact of hyperparameters, regularization techniques, and training strategies on datasets like **Fashion-MNIST** and **CIFAR-10**.

---

## 📂 Project Structure

### 1. Fundamentals & Optimization (MLP)
*File: `Quairel_Arthur_DL_debut.py`*

This module focuses on building Feed-Forward Neural Networks (FNN) from scratch to classify flattened images. It emphasizes the "science" behind training a network.

**Key Concepts Implemented:**
* **Architectures:** Designing linear models and Deep Neural Networks (DNN) with `nn.Sequential`.
* **Activation Functions:** Comparative analysis of `Sigmoid` vs. `ReLU` performance.
* **Training Strategies:**
    * **Online Training:** Stochastic Gradient Descent (SGD) updating weights one sample at a time.
    * **Mini-Batch Training:** Balancing computational efficiency and gradient stability.
    * **Full-Batch Training:** Gradient computation on the entire dataset.
* **Regularization:** Implementation of **Dropout** layers to prevent overfitting and **Batch Normalization** to accelerate convergence.
* **Optimization:** Tuning Learning Rates and SGD Momentum.

### 2. Computer Vision with CNNs
*File: `Quairel_Arthur_DL_Conv.py`*

This module transitions to 2D image processing, leveraging spatial hierarchies in data.

**Key Concepts Implemented:**
* **Data Pipeline:** Advanced use of `DataLoader` and `TensorDataset` for efficient batch processing.
* **Convolution Mechanics:** Manual experimentation with Kernels, Stride, and Padding to understand feature extraction.
* **Pooling:** Implementation of Max-Pooling for dimensionality reduction.
* **Custom Architecture:** Building a modular `FashionCNN` class inheriting from `nn.Module`.
* **CIFAR-10 Expansion:** extending the architecture to handle color images (3 input channels).

---

## 🛠 Technologies & Libraries

* **Core:** Python 3.x
* **Deep Learning:** [PyTorch](https://pytorch.org/) (`torch`, `torch.nn`, `torchvision`)
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib (Used for Loss/Accuracy curves and filter visualization)

## To run the MLP and Optimization experiments
python Quairel_Arthur_DL_debut.py

## To run the CNN and Vision experiments
python Quairel_Arthur_DL_Conv.py
