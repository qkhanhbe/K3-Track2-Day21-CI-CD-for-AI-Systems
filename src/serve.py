import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

model_path = os.getenv("MODEL_PATH", "models/model.pkl")

# Load model globally on startup
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    pred = model.predict([req.features])[0]
    
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": label_map.get(int(pred), "unknown")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
