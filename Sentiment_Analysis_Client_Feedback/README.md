# Sentiment Analysis on Client Feedback

## Project Overview

This project predicts **positive or negative feedback** from client service orders using **Word2Vec** embeddings and a classification model.

## Features

- **Word2Vec embeddings** are used to represent text feedback.
- **Multi-model approach** for sentiment classification.
- **Trained on 50,000 feedback entries** (70% positive, 30% negative).
- **Useful for customer excellence teams** to analyze client sentiment.

## Project Structure

```
Sentiment_Analysis_Client_Feedback/
│── data/                 # Raw and processed data
│── notebooks/            # Jupyter notebooks for EDA and modeling
│── src/                  # Source code
│   ├── features/         # Feature engineering scripts (Word2Vec embeddings)
│   ├── models/           # Model training and selection scripts
│   ├── utils/            # Utility functions
│── reports/              # Model performance reports
│── README.md             # Documentation
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run feature extraction (Word2Vec embeddings):
   ```bash
   python src/features/create_embeddings.py
   ```
3. Train the model:
   ```bash
   python src/models/train_model.py
   ```
