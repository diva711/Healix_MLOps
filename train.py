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

        # FIX: Actually use train_test_split so accuracy is measured on unseen data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Log hyperparameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)

        # Train
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 2. Log Metrics — evaluated on TEST set, not training set
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model trained. Test accuracy: {accuracy:.4f}")

        # 3. Model Registry — register and set alias so serve.py can find it
        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Healix_Sentiment_Model"
)

        # FIX: set a named alias on version 1 so serve.py can use @champion URI
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions("Healix_Sentiment_Model")
        latest_version = latest[-1].version
        client.set_registered_model_alias(
        name="Healix_Sentiment_Model",
        alias="champion",
        version=latest_version
)
        print("Model registered and aliased as @champion!")

if __name__ == "__main__":
    run_training()