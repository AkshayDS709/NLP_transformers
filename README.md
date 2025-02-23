
# cybersecurity_classification-malicious-benign_websites.ipynb
### Web Traffic Transaction Classification

This project ingests web transaction logs, processes the data, and classifies transactions as either malicious or benign using a Transformer-based deep learning model.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Project Structure](#project-structure)
- [License](#license)

## Overview
The project fetches transaction logs from Google BigQuery, preprocesses the data, and trains a Transformer-based neural network model to classify transactions. The model uses PyTorch and sklearn for processing and evaluation.

## Requirements
Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- Scikit-learn
- Pandas
- Google Cloud SDK
- pandas-gbq
- Google Authentication Library

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/web-traffic-classifier.git
   cd web-traffic-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Authenticate with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```

## Usage
1. Update the SQL query in `load_data(query, credentials)` to fetch relevant data.
2. Run the script to train the model:
   ```bash
   python web_traffic_classifier.py
   ```
3. The model will be trained and evaluated on the dataset, displaying accuracy and performance metrics.

## Model Training and Evaluation
- The model is trained using a Transformer encoder architecture.
- The dataset is preprocessed with standardization and label encoding.
- Training loss is monitored during each epoch.
- The final model accuracy is computed on test data.

## Project Structure
```
web-traffic-classifier/
│── web_traffic_classifier.py  # Main script
│── requirements.txt  # Required dependencies
│── README.md  # Documentation
│── service_account.json  # Google Cloud authentication (add this manually)
```


