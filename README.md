# Front-Running Attack Detection using Transformer Models

This repository contains the implementation, datasets, and experiments for our research on detecting front-running attacks in Ethereum transactions using transformer-based models.

## 📌 Overview
Front-running attacks (displacement, insertion, suppression) represent a critical threat in decentralized blockchain systems.  
This project explores transformer-based models (Llama, Gemma) for detecting and classifying such attacks.

## 🛠 Features
- Data preprocessing pipeline using **Alchemy** and **Chainstack RPC**.  
- Automatic labeling of transactions by attack type.  
- Training and evaluation of transformer models with varying input lengths (64, 128, 256).  
- Metrics: Accuracy, Precision, Recall, and F1-score.  
- Reproducible experiments with baseline comparisons.  

## 📂 Repository Structure
├── data/ # Datasets (or scripts to fetch them)
├── models/ # Pretrained and trained model checkpoints
├── src/ # Core source code
│ ├── preprocessing/ # Data cleaning and labeling scripts
│ ├── training/ # Training and evaluation code
│ └── utils/ # Helper functions
├── results/ # Output metrics, logs, and plots
└── README.md # Project documentation
