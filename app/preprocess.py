import numpy as np

def preprocess_input(disease: str, inputs: dict):
    features = []
    for value in inputs.values():
        if isinstance(value, str) and value.lower() in ["yes", "no"]:
            features.append(1 if value.lower() == "yes" else 0)
        elif isinstance(value, str) and value.lower() in ["male", "female"]:
            features.append(1 if value.lower() == "male" else 0)
        else:
            try:
                features.append(float(value))
            except:
                features.append(0.0)
    return np.array([features])
