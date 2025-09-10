import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from pathlib import Path

# --------------------------
# Setup: Models directory
# --------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "app" / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------
# Synthetic Data Generators
# --------------------------

def generate_heart_data(n=1000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(30, 80, n)
    gender = np.random.randint(0, 2, n)
    cholesterol = np.random.normal(220, 40, n).clip(150, 350)
    blood_pressure = np.random.normal(130, 20, n).clip(90, 200)
    smoking = np.random.randint(0, 2, n)
    angina = np.random.randint(0, 2, n)

    risk = (
        (cholesterol > 240).astype(int)
        + (blood_pressure > 140).astype(int)
        + smoking
        + angina
    )
    target = (risk + np.random.binomial(1, 0.2, n) > 1).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "cholesterol": cholesterol,
        "bloodPressure": blood_pressure,
        "smoking": smoking,
        "exerciseInducedAngina": angina,
        "target": target,
    })


def generate_diabetes_data(n=1000, seed=43):
    np.random.seed(seed)
    age = np.random.randint(20, 80, n)
    gender = np.random.randint(0, 2, n)
    glucose = np.random.normal(120, 30, n).clip(70, 250)
    bmi = np.random.normal(27, 5, n).clip(15, 45)
    family_history = np.random.randint(0, 2, n)
    physical_activity = np.random.normal(3, 2, n).clip(0, 15)

    risk = (
        (glucose > 140).astype(int)
        + (bmi > 30).astype(int)
        + family_history
        - (physical_activity > 5).astype(int)
    )
    target = (risk + np.random.binomial(1, 0.1, n) > 1).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "glucose": glucose,
        "bmi": bmi,
        "familyHistory": family_history,
        "physicalActivity": physical_activity,
        "target": target,
    })


def generate_lung_data(n=1000, seed=44):
    np.random.seed(seed)
    age = np.random.randint(20, 80, n)
    gender = np.random.randint(0, 2, n)
    smoking = np.random.randint(0, 2, n)
    cough_duration = np.random.randint(0, 30, n)
    breathlessness = np.random.randint(0, 2, n)
    pollution = np.random.randint(0, 2, n)

    risk = smoking + (cough_duration > 10).astype(int) + breathlessness + pollution
    target = (risk + np.random.binomial(1, 0.1, n) > 2).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "coughDuration": cough_duration,
        "breathlessness": breathlessness,
        "exposureToPollution": pollution,
        "target": target,
    })


def generate_kidney_data(n=1000, seed=45):
    np.random.seed(seed)
    age = np.random.randint(20, 80, n)
    gender = np.random.randint(0, 2, n)
    blood_urea = np.random.normal(30, 10, n).clip(10, 100)
    creatinine = np.random.normal(1.2, 0.5, n).clip(0.5, 5.0)
    hypertension = np.random.randint(0, 2, n)
    diabetes = np.random.randint(0, 2, n)

    risk = (blood_urea > 45).astype(int) + (creatinine > 1.5).astype(int) + hypertension + diabetes
    target = (risk + np.random.binomial(1, 0.1, n) > 1).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "bloodUrea": blood_urea,
        "serumCreatinine": creatinine,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "target": target,
    })

# --------------------------
# Train and Save Function
# --------------------------

def train_and_save(df, disease, features):
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ {disease} model trained. Accuracy: {acc:.2f}")

    model_path = MODELS_DIR / f"{disease}Model.pkl"
    joblib.dump(model, model_path)


if __name__ == "__main__":
    train_and_save(generate_heart_data(), "Heart",
        ["age", "gender", "cholesterol", "bloodPressure", "smoking", "exerciseInducedAngina"])
    train_and_save(generate_diabetes_data(), "Diabetes",
        ["age", "gender", "glucose", "bmi", "familyHistory", "physicalActivity"])
    train_and_save(generate_lung_data(), "Lung",
        ["age", "gender", "smoking", "coughDuration", "breathlessness", "exposureToPollution"])
    train_and_save(generate_kidney_data(), "Kidney",
        ["age", "gender", "bloodUrea", "serumCreatinine", "hypertension", "diabetes"])

    print("🎉 All models trained and saved in app/models/")
