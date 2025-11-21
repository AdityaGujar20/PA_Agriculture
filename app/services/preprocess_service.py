import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def handle_missing_values(file_path, strategy):
    df = pd.read_csv(file_path)

    # Convert date → datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Extract date features
    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df = df.drop(columns=["date"])  # Drop original

    # Missing value handling
    if strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        df = df.fillna(df.mode().iloc[0])

    # Save processed file (NO ENCODING OR SCALING HERE)
    out_path = PROCESSED_DIR / (Path(file_path).stem + f"_{strategy}.csv")
    df.to_csv(out_path, index=False)
    return str(out_path)
