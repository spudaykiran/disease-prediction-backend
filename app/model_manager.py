import joblib
from pathlib import Path
from .config import MODELS_DIR

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        """Load all trained models from the models directory."""
        for path in MODELS_DIR.glob("*.pkl"):
            try:
                # Example: HeartModel.pkl → Heart
                name = path.stem.replace("Model", "")
                self.models[name] = joblib.load(path)
                print(f"✅ Loaded model: {name} from {path.name}")
            except Exception as e:
                print(f"❌ Failed to load model {path.name}: {e}")

    def get(self, disease: str):
        """Retrieve model by disease name."""
        return self.models.get(disease)

    def list_models(self):
        """List all available disease models."""
        return list(self.models.keys())
