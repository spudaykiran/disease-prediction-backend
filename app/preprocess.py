import numpy as np
import pandas as pd
from .disease_schemas import DISEASE_SCHEMAS


def normalize_value(field, value):
    """Convert user input into numeric format (safe fallback)."""
    if value is None or value == "":
        return 0.0

    # Gender
    if field == "gender":
        return 1.0 if str(value).lower().startswith("m") else 0.0

    # Yes/No categorical fields
    yes_no_fields = {
        "smoking",
        "exerciseInducedAngina",
        "familyHistory",
        "breathlessness",
        "exposureToPollution",
        "hypertension",
        "diabetes",
    }
    if field in yes_no_fields:
        return 1.0 if str(value).lower().startswith("y") else 0.0

    # Numeric fallback
    try:
        return float(value)
    except ValueError:
        return 0.0


def map_inputs(disease: str, input_dict: dict):
    """Convert raw user inputs into a DataFrame with correct feature names."""
    fields = DISEASE_SCHEMAS[disease]
    row = {}

    for f in fields:
        val = input_dict.get(f)
        row[f] = normalize_value(f, val)

    # Return a DataFrame with correct column names
    return pd.DataFrame([row], columns=fields)
