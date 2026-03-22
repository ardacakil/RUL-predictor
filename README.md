# RUL Predictor — NASA CMAPSS Turbofan Engine Dataset

Predicts the **Remaining Useful Life (RUL)** of jet engines from sensor data.  
Built as a portfolio project targeting aerospace ML roles.

---

## What this project does

A jet engine degrades over time. If we can predict how many cycles it has left before it needs maintenance, airlines can plan ahead instead of reacting to failures. This project builds models that read 30 cycles of live sensor data from an engine and output that number — its Remaining Useful Life.

The project covers the full ML pipeline: raw data exploration, feature engineering, model training, and a production-ready API endpoint.

---

## Dataset

**NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)**  
Four subsets of increasing difficulty:

| Subset | Train engines | Operating conditions | Fault modes |
|--------|:---:|:---:|:---:|
| FD001  | 100 | 1 | 1 |
| FD002  | 260 | 6 | 1 |
| FD003  | 100 | 1 | 2 |
| FD004  | 249 | 6 | 2 |

Each engine runs from healthy to failure. 21 sensors are recorded every cycle.  
The task is to predict how many cycles remain at any given point.

---

## Project structure

```
rul-predictor/
├── data/
│   ├── raw/                    ← CMAPSS .txt files
│   ├── X_train_multi.npy       ← combined (139798, 30, 14)
│   ├── X_test_FD00{1-4}.npy
│   └── y_test_FD00{1-4}.npy
├── notebooks/
│   ├── 01_eda.ipynb            ← FD001 exploration, sensor selection
│   ├── 02_preprocessing.ipynb  ← RUL labelling, normalisation, scaler
│   ├── 03_features.ipynb       ← sliding window construction
│   ├── 04_baseline_model.ipynb ← XGBoost on FD001
│   ├── 05_lstm_model.ipynb     ← LSTM v1 and v2 on FD001
│   ├── 06_eda_multi.ipynb      ← multi-condition EDA
│   ├── 07_pipeline_multi.ipynb ← cluster normalisation pipeline
│   ├── 08_multi_baseline.ipynb ← XGBoost on all 4 subsets
│   └── 09_multi_lstm.ipynb     ← LSTM on all 4 subsets
├── models/
│   ├── scaler.pkl              ← FD001 MinMaxScaler
│   ├── multi_scalers.pkl       ← per-cluster scalers + KMeans objects
│   ├── xgb_baseline.pkl        ← XGBoost FD001
│   ├── xgb_multi.pkl           ← XGBoost all subsets
│   └── lstm_multi_best.pt      ← best LSTM checkpoint
├── api/                        ← FastAPI endpoint
├── DECISIONS.md                ← full engineering decision log
└── requirements.txt
```

---

## Key engineering decisions

### 1 — Sensor selection
21 sensors are recorded but 7 are completely flat across all operating conditions (zero variance). Flat sensors carry no degradation signal and were dropped. The remaining 14 sensors were validated across all four subsets using within-cluster standard deviation — not global std, which would give false signal on multi-condition data.

**Kept:** `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`  
**Dropped:** `s1, s5, s6, s10, s16, s18, s19`

### 2 — RUL capping at 125 cycles
A brand new engine with 300 cycles remaining looks identical to one with 280. The difference between 20 and 0 is critical. Capping RUL at 125 tells the model to ignore healthy-vs-very-healthy and direct all precision toward the danger zone.

### 3 — Sliding window of 30 cycles
A single sensor reading tells you the current state. Thirty consecutive readings tell you the trend. An engine where temperature has climbed steadily for 30 cycles is very different from one where it spiked once and recovered — but both look identical from a single reading.

### 4 — Per-cluster normalisation for FD002/FD004
This is the core challenge of multi-condition data. FD002 and FD004 have 6 operating conditions. Sensor s4 (turbine exit temperature) reads ~1100 at low power and ~1420 at high power — both perfectly healthy readings. A global scaler would interpret 1100 as degraded when it's just a different flight phase.

Solution: KMeans (k=6) on the three setting columns clusters each row into its operating regime. MinMaxScaler is then fit separately within each cluster on training data only, then applied to test data using the same cluster assignment. This removes operating-point bias before the model sees any values.

### 5 — XGBoost before LSTM
XGBoost was built first for three reasons: it trains in seconds, it produces feature importances that inform the LSTM design, and it might be good enough. It turned out competitive with the LSTM on most subsets, making it the production candidate for latency-sensitive deployments.

### 6 — Data leakage prevention
Scalers are always fit on training data only, then applied to test data. This is enforced consistently across both the FD001 pipeline and the per-cluster multi-condition pipeline. Fitting on combined data would give the model indirect knowledge of the test distribution — an illegal advantage that would inflate reported RMSE.

---

## Results

### FD001 only

| Model | RMSE |
|-------|-----:|
| XGBoost | 14.26 |
| LSTM v1 | 15.11 |
| LSTM v2 | 14.63 |

### All subsets — multi-condition pipeline

| Model | FD001 | FD002 | FD003 | FD004 |
|-------|------:|------:|------:|------:|
| XGBoost multi | 13.86 | 16.95 | 15.30 | 23.23 |
| LSTM multi    | **11.79** | 18.19 | **13.10** | **22.14** |

**Reading the results:**  
RMSE is in cycles. An RMSE of 11.79 means the model is off by ~12 cycles on average. If the true RUL is 50 cycles, the model predicts roughly 38–62.

The LSTM outperforms XGBoost on FD001, FD003, FD004 — subsets where sequence context matters. XGBoost wins on FD002 (6 conditions, 1 fault mode) where the degradation signal is clean enough after per-cluster normalisation that the last few sensor readings are sufficient.

FD004 is the hardest subset (6 operating conditions + 2 fault modes simultaneously). Both models struggle — the model must learn two distinct failure signatures while filtering regime noise. A fault-mode classifier as a preprocessing step is the natural next improvement.

### Compared to published benchmarks
State-of-the-art deep learning results on CMAPSS FD001 range from 11–13 RMSE (Li et al. 2018, attention-based LSTMs). The LSTM multi-condition model at 11.79 on FD001 sits within that range, trained on a combined dataset without per-subset tuning.

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/

# Start the API
uvicorn api.main:app --reload
```

---

## Tech stack

Python · pandas · numpy · scikit-learn · XGBoost · PyTorch · FastAPI

---

## What's next

- [ ] FastAPI endpoint with request validation
- [ ] Docker container
- [ ] Streamlit dashboard for visual RUL monitoring
- [ ] Fault-mode classifier for FD003/FD004 preprocessing
- [ ] Attention mechanism on LSTM to learn which timesteps matter most