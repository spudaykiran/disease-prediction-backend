import os, pickle

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        models_path = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for file in os.listdir(models_path):
            if file.endswith(".pkl"):
                name = file.replace("Model.pkl", "")
                with open(os.path.join(models_path, file), "rb") as f:
                    self.models[name] = pickle.load(f)
        print(f"✅ Loaded models: {list(self.models.keys())}")
