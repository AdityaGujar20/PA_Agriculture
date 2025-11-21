import matplotlib
matplotlib.use("Agg")

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import base64

MODEL_DIR = Path("app/models")
PROCESSED_DIR = Path("data/processed")


def _load_latest_processed():
    files = sorted(PROCESSED_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No processed data available.")
    return pd.read_csv(files[0])


def _load_model_components():
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    feature_order = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, scaler, encoder, feature_order


def _prepare_data_for_shap():
    df = _load_latest_processed()

    if "yield_kg_per_ha" in df.columns:
        df = df.drop(columns=["yield_kg_per_ha"])

    model, scaler, encoder, feature_order = _load_model_components()

    # Encode categorical
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(cat_cols) > 0 and encoder:
        encoded = encoder.transform(df[cat_cols])
        enc_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(columns=cat_cols).reset_index(drop=True)
        df = pd.concat([df, enc_df], axis=1)

    # reorder
    df = df.reindex(columns=feature_order, fill_value=0)

    df_scaled = scaler.transform(df)

    return df, df_scaled, model


def _fig_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode()
    plt.close()
    return img


def shap_summary_plot():
    _, df_scaled, model = _prepare_data_for_shap()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scaled)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, df_scaled, show=False)

    return _fig_to_base64()


def shap_bar_plot():
    _, df_scaled, model = _prepare_data_for_shap()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_scaled)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, df_scaled, plot_type="bar", show=False)

    return _fig_to_base64()
