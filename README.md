# Front-Running Attack Detection using Transformer Models

This repository contains the implementation, datasets, and experiments for our published research on detecting front-running attacks in Ethereum transactions using transformer-based models. For more information, you may check the paper through this link: https://ieeexplore.ieee.org/abstract/document/11418824/metrics#metrics.

## 📌 Overview
Front-running attacks (displacement, insertion, suppression) represent a critical threat in decentralized blockchain systems.  
This project explores transformer-based models (Llama, Gemma) for detecting and classifying such attacks.

## 🛠 Features
- Data preprocessing pipeline using **Alchemy** and **Chainstack RPC**.  
- Automatic labeling of transactions by attack type.  
- Training and evaluation of transformer models with varying input lengths (64, 128, 256).  
- Metrics: Accuracy, Precision, Recall, and F1-score.  
- Reproducible experiments with baseline comparisons.  
