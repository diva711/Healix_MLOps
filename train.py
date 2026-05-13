import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. Connect to the 'Tracking Server'
mlflow.set_experiment("Patient_Satisfaction")

def train_model():
    with mlflow.start_run():
        # Load data (In a real pipeline, DVC would pull this)
        df = pd.read_csv("data.csv")
        X = df[['wait_time_mins']]
        y = df['satisfied']
        
        # Train
        params = {"n_estimators": 100, "max_depth": 3}
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        
        # 2. Log "Breadcrumbs" (Experiment Tracking)
        # Why: So you can compare this run with others later.
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", model.score(X, y))
        
        # 3. Model Registry
        # Why: This 'tags' the model so the Serving API knows which one to pick up.
        mlflow.sklearn.log_model(model, "model", registered_model_name="Healix_Model")

if __name__ == "__main__":
    train_model()