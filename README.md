# ML Experiment Tracking with MLflow 🚀

## Project Overview
This project demonstrates a hands-on implementation of the **Machine Learning Life Cycle**, specifically focusing on **Experiment Tracking**. I used **MLflow** to log parameters, metrics, and models during the training of a Random Forest Regressor.

## 📊 Dataset Information
For this implementation, a **Synthetic Linear Dataset** was used to simulate a regression problem. 

* **Type:** Generated using Python's `range` function and list comprehensions.
* **Structure:** * **Features ($X$):** A single independent variable representing a sequence from 0 to 99.
    * **Target ($y$):** A dependent variable calculated using the linear relationship $y = 2x$.
* **Pre-processing:** The data was split into **80% Training** and **20% Testing** sets using `train_test_split` to evaluate the model's ability to generalize to unseen data.

## Key Features
* **Automated Logging:** Tracks hyperparameters like `n_estimators` and `max_depth`.
* **Metric Tracking:** Logs the Mean Squared Error (MSE) to evaluate model performance.
* **Model Registry:** Saves the trained model as an artifact for future deployment.
* **Interactive Dashboard:** Uses the MLflow UI to compare different runs visually.

## Tech Stack
* **Language:** Python 3.12
* **ML Library:** Scikit-Learn
* **Experiment Tracking:** MLflow
* **Data Handling:** Pandas

## How to Run This Project

### 1. Prerequisites
Ensure you have Python installed. You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
Execute the script to train the model and log the experiment:
```bash
python train.py

```

### 3. Launch the MLflow UI
To view your "lab notebook" and compare results:

```bash
python -m mlflow ui --host 127.0.0.1 --workers 1
```

Then, open your browser and go to http://127.0.0.1:5000.

## What I Learned
How to set up a professional ML development environment using virtual environments.

The importance of logging "metadata" (parameters and metrics) instead of just writing them down.

How to use a tracking server to compare model versions and pick the best performing one.

Standardizing project dependencies with a requirements.txt file.