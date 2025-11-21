import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

MODEL_DIR = Path("app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_model(file_path, target, test_size):

    df = pd.read_csv(file_path)

    # Split X & y
    X = df.drop(columns=[target])
    y = df[target]

    # --- Categorical Encoding (OneHotEncoder) ---
    cat_cols = X.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    if len(cat_cols) > 0:
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = X.drop(columns=cat_cols)
        X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    else:
        encoder = None

    # ⭐ Save FINAL feature order BEFORE SCALING
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

    # --- Scaling (StandardScaler) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save encoder & scaler
    if encoder:
        joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_DIR / "model.pkl")

    score = model.score(X_test, y_test)

    return {"r2_score": score}
