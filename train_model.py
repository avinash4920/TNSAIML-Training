"""Train and compare regression models for house price estimation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "house_prices.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "floors", "age_years", "parking", "location"]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "floors", "age_years", "parking"]
CATEGORICAL = ["location"]

def main():
    if not DATA.exists():
        raise FileNotFoundError("Dataset missing. Run: python generate_dataset.py")
    df = pd.read_csv(DATA)
    required = FEATURES + ["price_lakhs"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df[required].dropna(subset=["price_lakhs"])
    X, y = df[FEATURES], df["price_lakhs"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", categorical_pipe, CATEGORICAL)])
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42, n_estimators=180, learning_rate=0.05, max_depth=2)
    }
    results = {}
    best_name, best_model, best_rmse = None, None, float("inf")
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", prep), ("regressor", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results[name] = {"mae": float(mean_absolute_error(y_test, pred)), "rmse": rmse,
                         "r2": float(r2_score(y_test, pred))}
        if rmse < best_rmse:
            best_name, best_model, best_rmse = name, model, rmse
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS / "house_price_model.pkl")
    payload = {"best_model": best_name, "target_unit": "INR lakhs", "metrics": results,
               "test_rows": int(len(X_test)), "feature_columns": FEATURES,
               "dataset_note": "Use real, verified local property data for meaningful market estimates."}
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2))
    print("Model comparison (lower MAE/RMSE and higher R² are better):")
    for name, score in results.items():
        print(f"{name}: MAE={score['mae']:.2f}, RMSE={score['rmse']:.2f}, R²={score['r2']:.3f}")
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODELS / 'house_price_model.pkl'}")

if __name__ == "__main__":
    main()
