# ISVAE - Interpretable Spectral Variational Autoencoder

This repository contains the implementation of an Interpretable Spectral Variational Autoencoder (ISVAE) for unsupervised clustering of time series data. The ISVAE model is designed to learn meaningful representations of time series data in the frequency domain, enabling effective clustering and interpretation of the learned features.

## Key Features

* **Unsupervised Clustering:** The ISVAE model performs unsupervised clustering of time series data, discovering natural groupings within the data without the need for labeled examples.
* **Frequency Domain Representation:** The model operates on data converted to frequency domain by means of a DCT (Discrete Cosine Transform)  of time series data, capturing important spectral characteristics for clustering. Nevertheless the model can also learn in time domain. 
* **Interpretability:** The ISVAE incorporates interpretability mechanisms, allowing users to understand the learned features and their relationship to the clustering results.
* **Flexibility:** The code provides options for different model configurations and hyperparameters, enabling customization for a synthetic data example.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/Oscarm96/Interprertable-Spectral-VAE.git](https://github.com/Oscarm96/Interprertable-Spectral-VAE.git)
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt

3. **Hyperarameter settings**
Modify the main.py to set hyperparameters (ISVAE version, latent dimension, number of filters, filter's width and epochs) at      will.
4. **Run the code**

   ```bash
   python main.py

5. **License**
This project is licensed under the MIT License. See the 1  LICENSE file for details.
