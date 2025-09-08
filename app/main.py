import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from .model_manager import ModelManager
from .preprocess import preprocess_input

# ----------------------------------------------------
# Setup logging
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Disease Prediction API")

# Allowed origins (local + Netlify)
allowed_origins = [
    "http://localhost:5173",   # Vite local dev
    "http://localhost:3000",   # CRA local dev
    "https://predict-prevent.netlify.app",  # ✅ your deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Load models
# ----------------------------------------------------
model_manager = ModelManager()

# ----------------------------------------------------
# Request schema
# ----------------------------------------------------
class PredictionRequest(BaseModel):
    disease: str
    input: Dict[str, Any]

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "models": list(model_manager.models.keys())}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Make prediction for a given disease"""
    disease = request.disease
    inputs = request.input

    if disease not in model_manager.models:
        raise HTTPException(status_code=400, detail=f"Model for {disease} not found")

    try:
        # Preprocess input
        features = preprocess_input(disease, inputs)

        # Predict
        model = model_manager.models[disease]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = {
            "prediction": "Present" if prediction == 1 else "Not Present",
            "probability": float(probability),
        }

        logger.info(
            f"Prediction Request | Disease: {disease} | Inputs: {inputs} | Result: {result}"
        )
        return result

    except Exception as e:
        logger.error(f"Prediction error for {disease}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
