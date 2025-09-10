import joblib
from pathlib import Path
from loguru import logger


class ModelManager:
    def __init__(self):
        # Models will be stored in app/models/
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        """Load all trained models from models_dir"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        for model_file in self.models_dir.glob("*Model.pkl"):
            try:
                disease = model_file.stem.replace("Model", "")
                self.models[disease] = joblib.load(model_file)
                logger.info(f"✅ Loaded {disease} model from {model_file.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file.name}: {e}")

    def get_models(self):
        """Return list of available models"""
        return list(self.models.keys())

    def predict(self, disease: str, features: dict):
        """Make prediction for given disease and input features"""
        model = self.models.get(disease)
        if not model:
            raise ValueError(f"Model for {disease} not found")

        # Ensure features match model input order
        X = [list(features.values())]

        try:
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            return {
                "prediction": "Present" if prediction == 1 else "Not Present",
                "probability": float(probability),
            }
        except Exception as e:
            logger.error(f"Prediction failed for {disease}: {e}")
            raise
