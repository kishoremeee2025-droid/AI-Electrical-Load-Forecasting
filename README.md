# AI-Based Electrical Load Forecasting System
Beginner-friendly EEE capstone | Python + Random Forest + Streamlit

## What this project does
It uses historical electricity consumption to forecast **the next hour's average active power (kW)**. It creates time/calendar features and lag features, trains a Random Forest regressor, evaluates it on a later chronological test period, saves graphs and a model, and displays results in a Streamlit dashboard.

> **Important data note:** `data/sample_hourly_load.csv` is synthetic demo data included so the project can run offline immediately. The included `models/load_forecasting_model.joblib` and initial graphs are trained from this demo data, not from real meter measurements. For your final submission, download the public UCI dataset and run the training command below; that replaces the model, metrics, and graphs with results from the real public dataset.

## Public dataset
UCI Machine Learning Repository, **Individual Household Electric Power Consumption**:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- About 2.07 million one-minute readings (Dec 2006–Nov 2010).
- `Global_active_power` is household average active power in kW.
- Missing numeric readings are represented by `?`.
- Dataset license: CC BY 4.0. Cite: Hebrail, G. & Berard, A. (2006), UCI Machine Learning Repository, DOI: 10.24432/C58K54.

## Folder structure
- `app.py` — interactive Streamlit dashboard
- `train_model.py` — data loading, cleaning, hourly aggregation, feature engineering, chronological split, training, metrics and graphs
- `src/forecasting.py` — reusable helper functions
- `data/sample_hourly_load.csv` — synthetic offline demo data
- `models/load_forecasting_model.joblib` — demo model artifact
- `outputs/` — metric JSON, test predictions CSV and PNG charts
- `reports/` — project report PDF and report Markdown
- `requirements.txt` — Python dependencies

## Setup (Windows / macOS / Linux)
1. Install Python 3.10 or newer.
2. Unzip this project folder.
3. Open a terminal/Command Prompt inside the folder.
4. Create an environment (recommended):
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
5. Install packages: `pip install -r requirements.txt`

## Run the offline demo
```bash
python train_model.py --sample
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).

## Train using the real public UCI dataset (recommended before submission)
Internet is required for the first download:
```bash
python train_model.py --uci
```
The script downloads the official UCI ZIP, extracts the text file into `data/`, parses timestamps, turns `?` into missing values, resamples minute readings to hourly mean kW, drops hours with no usable target, creates lag features, trains and evaluates the model, and updates the saved model and graphs. Then run:
```bash
streamlit run app.py
```
If automatic download is blocked, download the dataset ZIP from the UCI page above, extract `household_power_consumption.txt` into `data/`, then run:
```bash
python train_model.py --uci --local-file data/household_power_consumption.txt
```

## How the prediction works
- **Target:** next hour's average active power in kW.
- **Features available at forecast time:** current/previous known hourly readings (`lag_1`, `lag_24`, `lag_168`), recent rolling means calculated only from prior readings, hour, day of week, month, weekend flag.
- The target is shifted one hour forward. The lag and rolling features are shifted so the model cannot see the future target.
- The first 80% of time-ordered rows are training data; the last 20% are test data. There is no random shuffle.
- Model: `RandomForestRegressor` with 150 trees.
- Metrics: MAE, RMSE and R². Lower MAE/RMSE are better; R² closer to 1 is generally better.
- A persistence baseline (predict next hour equals the latest observed hour) is also reported for comparison.

## Outputs
`outputs/metrics.json`, `outputs/test_predictions.csv`, `outputs/actual_vs_predicted.png`, `outputs/load_profile.png`, and `outputs/feature_importance.png`.

## Limitations to discuss in viva
- This is a short-horizon one-hour-ahead forecast; it is not a long-term building design model.
- UCI data represents one household in France, not an Indian campus/building. Results may not transfer directly.
- No weather, occupancy, tariff, holiday, or equipment-state data are included.
- Random Forest does not naturally extrapolate trends far beyond the range seen in training.
- Model scores depend on dataset period and cleaning decisions. Do not quote demo metrics as real-world performance.
- A forecast is decision support, not a substitute for electrical protection or safety systems.

## Viva quick answers
1. **Why Random Forest?** It models nonlinear patterns and interactions and usually works well on tabular features without heavy scaling.
2. **Why chronological split?** Future readings must not be used to train a model that predicts the past.
3. **What is lag_24?** The value observed 24 hours before the forecast time, approximately the same hour on the previous day.
4. **What is MAE?** Average absolute prediction error, in kW here.
5. **What is RMSE?** Square root of average squared error; it penalizes large errors more strongly.
6. **What is R²?** A comparison against predicting the test-set mean; it can be negative if predictions are poor.
7. **What is leakage?** Accidentally using information unavailable at prediction time, making evaluation unrealistically optimistic.
