import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Start the Experiment
mlflow.set_experiment("Healix_Patient_Satisfaction")

def run_training():
    with mlflow.start_run() as run:
        # Load the data tracked by DVC
        df = pd.read_csv("data.csv")
        X = df[['wait_time_mins']]
        y = df['satisfied']
        
        # Train logic
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        # 2. Log Metrics (Tracking)
        accuracy = model.score(X, y)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model trained with accuracy: {accuracy}")

        # 3. THE KEY STEP: Model Registry
        # This saves the model AND gives it a name in the central 'Store'
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="Healix_Sentiment_Model"
        )
        print("Model registered successfully!")

if __name__ == "__main__":
    run_training()