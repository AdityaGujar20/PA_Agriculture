import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("app/models")


# ----------------------------------------------------------------------
# Safe loader for feature order (called inside the optimizer, NOT at import)
# ----------------------------------------------------------------------
def load_feature_order():
    path = MODEL_DIR / "feature_names.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "feature_names.pkl not found. Train the model first before optimizing."
        )
    return joblib.load(path)


# ----------------------------------------------------------------------
# Learned cost & environmental formulas
# ----------------------------------------------------------------------
def compute_cost(fert, irr, pest):
    return (
        0.50165 * fert +
        0.20026 * irr +
        0.14837 * pest +
        20.19
    )


def compute_env(fert, irr, pest):
    return (
        0.60055 * fert +
        0.30009 * irr +
        0.04946 * pest +
        6.73
    )


# ----------------------------------------------------------------------
# Optimization Logic
# ----------------------------------------------------------------------
def optimize_inputs(base_features):
    """
    base_features → dict containing all NON-optimizable inputs
    """

    # Load model artifacts
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    encoder_path = MODEL_DIR / "encoder.pkl"
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    # Load feature order ONLY when optimizer is called
    feature_order = load_feature_order()

    best = {
        "yield": -1,
        "fertilizer_kg_per_ha": None,
        "irrigation_mm": None,
        "pesticide_ml": None,
        "input_cost_total": None,
        "environmental_score": None
    }

    # Search ranges
    fert_range = np.arange(0, 400, 10)
    irr_range = np.arange(0, 400, 10)
    pest_range = np.arange(0, 300, 10)

    for fert in fert_range:
        for irr in irr_range:
            for pest in pest_range:

                # Compute cost & environmental score
                cost = compute_cost(fert, irr, pest)
                env = compute_env(fert, irr, pest)

                # Constraints
                if cost > 12000:
                    continue
                if env > 10000:
                    continue

                # Build candidate row
                row = base_features.copy()
                row.update({
                    "fertilizer_kg_per_ha": fert,
                    "irrigation_mm": irr,
                    "pesticide_ml": pest,
                    "input_cost_total": cost,
                    "environmental_score": env
                })

                df = pd.DataFrame([row])

                # One-hot encode crop_type
                cat_cols = df.select_dtypes(include=["object"]).columns
                if encoder and len(cat_cols) > 0:
                    encoded = encoder.transform(df[cat_cols])
                    encoded_df = pd.DataFrame(
                        encoded, columns=encoder.get_feature_names_out(cat_cols)
                    )
                    df = df.drop(columns=cat_cols)
                    df = pd.concat(
                        [df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
                        axis=1
                    )

                # Enforce EXACT column order
                df = df.reindex(columns=feature_order, fill_value=0)

                # Scale
                df_scaled = scaler.transform(df)

                # Predict
                pred_yield = model.predict(df_scaled)[0]

                # Track best result
                if pred_yield > best["yield"]:
                    best.update({
                        "yield": float(pred_yield),
                        "fertilizer_kg_per_ha": float(fert),
                        "irrigation_mm": float(irr),
                        "pesticide_ml": float(pest),
                        "input_cost_total": float(cost),
                        "environmental_score": float(env)
                    })

    return best
