import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("app/models")

def predict_values(input_dict):

    # Load model, scaler, encoder
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    encoder_path = MODEL_DIR / "encoder.pkl"
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])

    # --- Encode categorical ---
    cat_cols = df.select_dtypes(include=["object"]).columns
    if encoder and len(cat_cols) > 0:
        encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(columns=cat_cols)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # --- Scale numeric ---
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]

    return {"prediction": float(pred)}
