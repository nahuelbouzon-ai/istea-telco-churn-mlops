import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

PROCESSED_PATH = "data/processed/telco_churn_clean.csv"
MODEL_PATH = "models/model_baseline.pkl"


def load_data(path=PROCESSED_PATH):
    print("Cargando dataset procesado...")
    df = pd.read_csv(path)
    print(f"Shape dataset limpio: {df.shape}")
    return df


def split_data(df):
    print("Preparando features (X) y target (y)...")

    # 1) Sacamos columna ID si existe
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # 2) Separar target
    y = df["churn"]
    X = df.drop(columns=["churn"])

    # 3) Quedarnos solo con columnas numéricas para este baseline
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X = X[numeric_cols]

    print(f"Cantidad de columnas numéricas usadas: {len(numeric_cols)}")
    print(f"Columnas numéricas: {list(numeric_cols)}")

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    print("Escalando features numéricas...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Entrenando modelo Logistic Regression (baseline)...")
    model = LogisticRegression(max_iter=200)

    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    print("Evaluando modelo...")

    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n📊 Métricas baseline:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


def save_model(model, scaler, path=MODEL_PATH):
    print("Guardando modelo baseline...")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, path)

    print(f"Modelo guardado en: {path}")


def run_all():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    model, scaler = train_model(X_train, y_train)

    evaluate(model, scaler, X_test, y_test)

    save_model(model, scaler)

    print("✔ Entrenamiento baseline completado.")


if __name__ == "__main__":
    run_all()
