from __future__ import annotations
import argparse, json, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from src.forecasting import make_supervised, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

def load_sample():
    df = pd.read_csv(DATA / "sample_hourly_load.csv", parse_dates=["timestamp"])
    return df.set_index("timestamp")["Global_active_power"].sort_index()

def load_uci(local_file=None):
    DATA.mkdir(exist_ok=True)
    if local_file:
        txt_path = Path(local_file)
    else:
        txt_path = DATA / "household_power_consumption.txt"
        if not txt_path.exists():
            zip_path = DATA / "uci_household_power.zip"
            print("Downloading official UCI dataset (about 20 MB)...")
            urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                candidates = [n for n in zf.namelist() if n.endswith("household_power_consumption.txt")]
                if not candidates:
                    raise FileNotFoundError("Could not find household_power_consumption.txt inside downloaded ZIP.")
                with zf.open(candidates[0]) as src, open(txt_path, "wb") as dst:
                    dst.write(src.read())
    print("Reading UCI file; this may take a minute...")
    df = pd.read_csv(txt_path, sep=";", na_values=["?"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        dayfirst=True, errors="coerce"
    )
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    # Hourly mean reduces 1-minute readings to a manageable forecast dataset.
    hourly = df.set_index("timestamp")["Global_active_power"].resample("h").mean()
    hourly.name = "Global_active_power"
    return hourly.dropna()

def make_graphs(hourly, test, predictions, model, metrics):
    OUT.mkdir(exist_ok=True)
    # Whole-series load profile: daily average
    plt.figure(figsize=(10, 4))
    hourly.resample("D").mean().plot()
    plt.title("Daily Average Electrical Load")
    plt.xlabel("Date"); plt.ylabel("Active power (kW)"); plt.tight_layout()
    plt.savefig(OUT / "load_profile.png", dpi=150); plt.close()

    plt.figure(figsize=(11, 4))
    n = min(24*14, len(test))
    pd.DataFrame({"Actual": test[TARGET].iloc[:n], "Predicted": predictions[:n]},
                 index=test.index[:n]).plot(ax=plt.gca())
    plt.title("Actual vs Predicted Load (first 14 test days)")
    plt.xlabel("Forecast origin"); plt.ylabel("Next-hour active power (kW)")
    plt.tight_layout(); plt.savefig(OUT / "actual_vs_predicted.png", dpi=150); plt.close()

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5)); importances.plot(kind="barh")
    plt.title("Random Forest Feature Importance"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train the electrical load forecasting model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample", action="store_true", help="Use included offline synthetic demo data")
    group.add_argument("--uci", action="store_true", help="Use public UCI household power dataset")
    parser.add_argument("--local-file", help="Path to extracted UCI household_power_consumption.txt")
    args = parser.parse_args()
    if args.local_file:
        hourly = load_uci(args.local_file)
        source = "UCI public dataset (local file)"
    elif args.uci:
        hourly = load_uci()
        source = "UCI Individual Household Electric Power Consumption"
    else:
        hourly = load_sample()
        source = "Synthetic demo data (not real meter readings)"

    supervised = make_supervised(hourly)
    if len(supervised) < 500:
        raise ValueError(f"Not enough usable rows after feature engineering: {len(supervised)}")
    split = int(len(supervised) * 0.8)
    train, test = supervised.iloc[:split], supervised.iloc[split:]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = test["lag_1"].to_numpy()  # persist the latest known load
    metrics = {
        "data_source": source,
        "target": "next-hour average active power (kW)",
        "raw_hourly_rows": int(len(hourly)),
        "supervised_rows": int(len(supervised)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "MAE_kW": float(mean_absolute_error(y_test, pred)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "persistence_baseline_MAE_kW": float(mean_absolute_error(y_test, baseline)),
        "note": "Scores are valid only for the stated data source and chronological test period."
    }
    MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": TARGET, "data_source": source},
                MODELS / "load_forecasting_model.joblib")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    predictions = pd.DataFrame({"actual_next_hour_kw": y_test, "predicted_next_hour_kw": pred,
                                "persistence_baseline_kw": baseline}, index=test.index)
    predictions.index.name = "forecast_origin"
    predictions.to_csv(OUT / "test_predictions.csv")
    make_graphs(hourly, test, pred, model, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {MODELS / 'load_forecasting_model.joblib'}")
    print(f"Saved charts and predictions to {OUT}")

if __name__ == "__main__":
    main()
