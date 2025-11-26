import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_PATH = "data/processed/telco_churn_clean.csv"


def load_data(path=RAW_PATH):
    print("Cargando dataset RAW...")
    df = pd.read_csv(path)
    print(f"Shape inicial: {df.shape}")
    return df


def clean_data(df):
    print("Limpiando dataset...")

    df.columns = df.columns.str.strip()
    df.replace(" ", np.nan, inplace=True)

    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df = df.dropna(subset=["total_charges"])

    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(int)

    print(f"Shape después de limpiar: {df.shape}")
    return df


def save_processed(df, path=PROCESSED_PATH):
    print("Guardando dataset limpio...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Archivo guardado en: {path}")


def main():
    df = load_data()
    df_clean = clean_data(df)
    save_processed(df_clean)
    print("✔ Limpieza completada.")


if __name__ == "__main__":
    main()
