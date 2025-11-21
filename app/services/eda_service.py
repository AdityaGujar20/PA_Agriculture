import matplotlib
matplotlib.use("Agg")   # IMPORTANT FIX

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def generate_eda_summary(file_path):
    df = pd.read_csv(file_path)
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "describe": df.describe().to_dict()
    }
    return summary


def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return encoded

def generate_eda_plots(file_path):
    df = pd.read_csv(file_path)
    images = {}

    # --- 1) Missing values plot ---
    plt.figure(figsize=(6,4))
    df.isna().sum().plot(kind="bar", color="teal")
    plt.title("Missing Values per Column")
    images["missing_values"] = fig_to_base64()

    # --- 2) Correlation heatmap ---
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="viridis")
    plt.title("Correlation Heatmap")
    images["correlation_heatmap"] = fig_to_base64()

    # --- 3) Yield distribution ---
    if "yield_kg_per_ha" in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df["yield_kg_per_ha"], kde=True, color="orange")
        plt.title("Yield Distribution")
        images["yield_distribution"] = fig_to_base64()

    # --- 4) Scatter (rainfall vs yield) ---
    if "yield_kg_per_ha" in df.columns and "rainfall_mm" in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df["rainfall_mm"], y=df["yield_kg_per_ha"])
        plt.title("Rainfall vs Yield")
        images["scatter_rainfall_yield"] = fig_to_base64()

    # --- 5) Boxplot for numerical features ---
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df.select_dtypes(include='number'))
    plt.xticks(rotation=45)
    plt.title("Boxplots of Numerical Features")
    images["boxplot_numeric"] = fig_to_base64()

    # --- 6) Pairplot (sample of 300 rows to avoid heavy rendering) ---
    sample_df = df.sample(n=min(300, len(df)))
    sns.pairplot(sample_df.select_dtypes(include='number'))
    images["pairplot"] = fig_to_base64()

    # --- 7) Outlier detection using IQR ---
    outlier_info = {}

    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_pct = (len(outliers) / len(df)) * 100

        outlier_info[col] = round(outlier_pct, 2)

    # Represent as a bar chart
    plt.figure(figsize=(8,5))
    plt.bar(outlier_info.keys(), outlier_info.values(), color="crimson")
    plt.xticks(rotation=45)
    plt.title("Outlier Percentage per Feature")
    images["outlier_percentages"] = fig_to_base64()

    return images
