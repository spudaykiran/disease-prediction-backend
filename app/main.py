from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model_manager import ModelManager
from app.preprocess import preprocess_input
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="Disease Prediction API")

# Allow frontend requests
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://predict-prevent.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model_manager = ModelManager()

# Request schema
class PredictionRequest(BaseModel):
    disease: str
    input: dict

@app.get("/health")
def health_check():
    return {"status": "ok", "models": list(model_manager.models.keys())}

@app.post("/predict")
def predict(req: PredictionRequest):
    disease = req.disease
    if disease not in model_manager.models:
        raise HTTPException(status_code=400, detail="Invalid disease name")

    model = model_manager.models[disease]
    features = preprocess_input(disease, req.input)

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    result = {
        "prediction": "Present" if pred == 1 else "Not Present",
        "probability": float(proba),
    }
    logging.info(f"Prediction Request | Disease: {disease} | Inputs: {req.input} | Result: {result}")
    return result
