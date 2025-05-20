# 🛡️ Fraud Detection System

## 🔍 Overview
Fraudulent transactions are a major threat to financial institutions, e-commerce platforms, and consumers. The aim of this project is to develop a fraud detection system adopting machine learning techniques that can accurately recognize fraudulent transactions while minimizing false positives. This model is trained on an open source dataset and deployed as a live API for credit card fraud detection.

## 🛠 Tools Used

- Python (pandas, scikit-learn, imbalanced-learn)
- Machine Learning (Random Forest, SMOTE)
- Flask API for deployment

## Setup & Installation
- Creating the virtual environment in IDE terminal & Installing Dependencies
```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```
- Training the model
```
    python src/train.py
```
- Running the Flask API
```
    python app.py
```

## API Usage
End point : ```/predict```
- Method : 'POST'
- Request: (You're welcome to input your request in JSON format)
```
    {
  "features": [0.1, -0.5, 0.3, 1.2, -0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```
- Response: ```True/False```
