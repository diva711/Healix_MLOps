import csv
import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException

app = FastAPI()

# ── Configuration ─────────────────────────────────────────────────────────────

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set. Run: set GEMINI_API_KEY=your_key_here")

genai.configure(api_key=gemini_api_key)
llm_model = genai.GenerativeModel('gemini-1.5-flash')

# Point to the same database train.py used
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model from Registry using @champion alias

MODEL_NAME = "Healix_Sentiment_Model"
try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@champion")
except Exception:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/1")


# ── Guardrails ────────────────────────────────────────────────────────────────

def health_guardrail(bot_output):
    """Prevent the bot from discussing off-topic financial subjects."""
    prohibited_words = ["buy", "crypto", "investment"]
    if any(word in bot_output.lower() for word in prohibited_words):
        return "I am a medical assistant and cannot discuss financial topics."
    return bot_output

def input_guardrail(user_query):
    """Block dangerous or crisis-level input."""
    forbidden = ["kill", "suicide", "bomb"]
    if any(word in user_query.lower() for word in forbidden):
        return False
    return True

def output_guardrail(llm_response):
    """Block specific medical dosage advice from the LLM."""
    if "mg" in llm_response or "dosage" in llm_response.lower():
        return "I cannot provide specific medical dosages. Please consult a doctor."
    return llm_response


# ── Logging & LLMOps ──────────────────────────────────────────────────────────

def log_to_production_file(wait_time, prediction):
    with open('production_logs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([wait_time, prediction])

def track_llm_metrics(usage_metadata):
    """LLMOps: Track token usage and costs in MLflow."""
    tokens = usage_metadata.total_token_count
    cost_per_token = 0.00002
    total_cost = tokens * cost_per_token
    with mlflow.start_run(run_name="LLM_Inference_Log", nested=True):
        mlflow.log_metric("tokens_used", tokens)
        mlflow.log_metric("llm_cost_usd", total_cost)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Healix Sentiment API is Online"}

@app.post("/predict")
def predict(wait_time: int):
    data = pd.DataFrame([[wait_time]], columns=['wait_time_mins'])
    prediction = model.predict(data)
    log_to_production_file(wait_time, int(prediction[0]))
    sentiment = "Satisfied" if prediction[0] == 1 else "Unsatisfied"
    return {
        "wait_time_input": wait_time,
        "prediction": sentiment,
        "satisfied": int(prediction[0])
    }

@app.post("/chat")
def chat(query: str):
    # 1. Input Guard
    if not input_guardrail(query):
        return {"response": "I cannot answer this. Please contact emergency services."}

    try:
        # 2. Real LLM Call
        system_prompt = "You are Healix, a helpful and empathetic healthcare assistant. "
        response = llm_model.generate_content(system_prompt + query)
        raw_response = response.text

        # LLMOps: Track cost and tokens
        track_llm_metrics(response.usage_metadata)

        # 3. Output Guardrails
        safe_response = output_guardrail(raw_response)
        final_response = health_guardrail(safe_response)

        return {"response": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Service Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)