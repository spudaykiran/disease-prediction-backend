import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# Where to save models
MODELS_DIR = Path(__file__).resolve().parent / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Synthetic Data Generators
# --------------------------

def generate_heart_data(n=1000):
    np.random.seed(42)
    age = np.random.randint(30, 80, n)
    gender = np.random.randint(0, 2, n)
    cholesterol = np.random.normal(220, 40, n).clip(150, 350)
    blood_pressure = np.random.normal(130, 20, n).clip(90, 200)
    smoking = np.random.randint(0, 2, n)
    angina = np.random.randint(0, 2, n)

    # Risk rule: high cholesterol + bp + smoking increases heart disease chance
    risk = (
        (cholesterol > 240).astype(int)
        + (blood_pressure > 140).astype(int)
        + smoking
        + angina
    )
    target = (risk + np.random.binomial(1, 0.2, n) > 1).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "cholesterol": cholesterol,
            "bloodPressure": blood_pressure,
            "smoking": smoking,
            "exerciseInducedAngina": angina,
            "target": target,
        }
    )
    return df


def generate_diabetes_data(n=1000):
    np.random.seed(43)
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

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "glucose": glucose,
            "bmi": bmi,
            "familyHistory": family_history,
            "physicalActivity": physical_activity,
            "target": target,
        }
    )
    return df


def generate_lung_data(n=1000):
    np.random.seed(44)
    age = np.random.randint(20, 80, n)
    gender = np.random.randint(0, 2, n)
    smoking = np.random.randint(0, 2, n)
    cough_duration = np.random.randint(0, 30, n)
    breathlessness = np.random.randint(0, 2, n)
    pollution = np.random.randint(0, 2, n)

    risk = smoking + (cough_duration > 10).astype(int) + breathlessness + pollution
    target = (risk + np.random.binomial(1, 0.1, n) > 2).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "smoking": smoking,
            "coughDuration": cough_duration,
            "breathlessness": breathlessness,
            "exposureToPollution": pollution,
            "target": target,
        }
    )
    return df


def generate_kidney_data(n=1000):
    np.random.seed(45)
    age = np.random.randint(20, 80, n)
    gender = np.random.randint(0, 2, n)
    blood_urea = np.random.normal(30, 10, n).clip(10, 100)
    creatinine = np.random.normal(1.2, 0.5, n).clip(0.5, 5.0)
    hypertension = np.random.randint(0, 2, n)
    diabetes = np.random.randint(0, 2, n)

    risk = (blood_urea > 45).astype(int) + (creatinine > 1.5).astype(int) + hypertension + diabetes
    target = (risk + np.random.binomial(1, 0.1, n) > 1).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "bloodUrea": blood_urea,
            "serumCreatinine": creatinine,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "target": target,
        }
    )
    return df


# --------------------------
# Train and Save Models
# --------------------------

def train_and_save(df, disease, features):
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{disease} model trained. Accuracy: {acc:.2f}")

    joblib.dump(model, MODELS_DIR / f"{disease}Model.pkl")


if __name__ == "__main__":
    # Heart
    heart_df = generate_heart_data()
    train_and_save(
        heart_df,
        "Heart",
        ["age", "gender", "cholesterol", "bloodPressure", "smoking", "exerciseInducedAngina"],
    )

    # Diabetes
    diabetes_df = generate_diabetes_data()
    train_and_save(
        diabetes_df,
        "Diabetes",
        ["age", "gender", "glucose", "bmi", "familyHistory", "physicalActivity"],
    )

    # Lung
    lung_df = generate_lung_data()
    train_and_save(
        lung_df,
        "Lung",
        ["age", "gender", "smoking", "coughDuration", "breathlessness", "exposureToPollution"],
    )

    # Kidney
    kidney_df = generate_kidney_data()
    train_and_save(
        kidney_df,
        "Kidney",
        ["age", "gender", "bloodUrea", "serumCreatinine", "hypertension", "diabetes"],
    )

    print("✅ All models trained and saved in app/models/")
