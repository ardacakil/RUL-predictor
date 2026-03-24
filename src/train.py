"""
train.py
--------
Train the XGBoost or LSTM model on the pre-built multi-condition CMAPSS arrays.

Run preprocess.py first to generate the required .npy files.

Usage
~~~~~
    # XGBoost (fast baseline)
    python src/train.py --model xgb

    # LSTM
    python src/train.py --model lstm

    # Override common settings
    python src/train.py --model lstm --epochs 150 --lr 0.0005 --batch-size 256
    python src/train.py --model xgb  --n-estimators 500 --data-dir data --models-dir models
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Constants shared by both models
# ---------------------------------------------------------------------------

SUBSETS        = ["FD001", "FD002", "FD003", "FD004"]
USEFUL_SENSORS = [
    "s2",  "s3",  "s4",  "s7",  "s8",  "s9",  "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
WINDOW_SIZE    = 30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_arrays(data_dir: Path):
    """Load combined train arrays and per-subset test arrays."""
    X_train = np.load(data_dir / "X_train_multi.npy")
    y_train = np.load(data_dir / "y_train_multi.npy")

    test_data = {}
    for subset in SUBSETS:
        X = np.load(data_dir / f"X_test_{subset}.npy")
        y = np.load(data_dir / f"y_test_{subset}.npy")
        test_data[subset] = (X, y)

    return X_train, y_train, test_data


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model_fn, test_data: dict, label: str):
    """
    Call model_fn(X) -> y_pred for each subset and print a results table.
    Returns a dict subset -> rmse.
    """
    results = {}
    all_true, all_pred = [], []

    for subset in SUBSETS:
        X, y_true = test_data[subset]
        y_pred = model_fn(X)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        results[subset] = rmse
        all_true.append(y_true)
        all_pred.append(y_pred)

    combined = np.sqrt(mean_squared_error(
        np.concatenate(all_true), np.concatenate(all_pred)
    ))

    print(f"\n{'='*50}")
    print(f"  {label} Results")
    print(f"{'='*50}")
    for subset, rmse in results.items():
        print(f"  {subset}      RMSE: {rmse:.2f} cycles")
    print(f"  Combined  RMSE: {combined:.2f} cycles")
    print(f"{'='*50}")

    return results


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgb(X_train, y_train, test_data, models_dir: Path, args):
    from xgboost import XGBRegressor

    print("\nTraining XGBoost …")

    # Use FD001 as the early-stopping validation set (same as notebook 08)
    X_eval, y_eval = test_data["FD001"]
    X_train_flat   = X_train.reshape(X_train.shape[0], -1)
    X_eval_flat    = X_eval.reshape(X_eval.shape[0], -1)
    test_flat      = {s: X.reshape(X.shape[0], -1) for s, (X, _) in test_data.items()}

    model = XGBRegressor(
        n_estimators          = args.n_estimators,
        learning_rate         = args.lr,
        max_depth             = args.max_depth,
        subsample             = 0.8,
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 50,
    )

    model.fit(
        X_train_flat, y_train,
        eval_set = [(X_train_flat, y_train), (X_eval_flat, y_eval)],
        verbose  = 100,
    )

    print(f"Best round : {model.best_iteration}")

    evaluate(lambda X: model.predict(X), test_flat, "XGBoost multi-condition")

    out_path = models_dir / "xgb_multi.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {out_path}")


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class RULPredictor:
    """
    Thin wrapper so the class definition stays importable without torch
    being required at module level when running --model xgb.
    Instantiated inside train_lstm only.
    """
    pass


def train_lstm(X_train, y_train, test_data, models_dir: Path, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print("Training LSTM …\n")

    # ── Model definition ────────────────────────────────────────────────────
    class _RULPredictor(nn.Module):
        def __init__(self, input_size=14, hidden_size=256, num_layers=3, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = input_size,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0,
            )
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.regressor(lstm_out[:, -1, :]).squeeze(1)

    # ── Data loaders ────────────────────────────────────────────────────────
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size = args.batch_size,
        shuffle    = True,
    )

    test_tensors = {
        s: (torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device))
        for s, (X, y) in test_data.items()
    }
    X_eval, y_eval = test_tensors["FD001"]

    # ── Training setup ──────────────────────────────────────────────────────
    model     = _RULPredictor(input_size=len(USEFUL_SENSORS)).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=15
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")
    print(f"Batches/epoch : {len(train_loader)}\n")

    # ── Early stopping ──────────────────────────────────────────────────────
    patience_counter = 0
    best_rmse        = float("inf")
    best_weights     = None

    out_path = models_dir / "lstm_multi_best.pt"

    for epoch in range(args.epochs):
        # Train
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        train_rmse = float(np.sqrt(np.mean(batch_losses)))

        # Eval on FD001
        model.eval()
        with torch.no_grad():
            preds_eval = model(X_eval)
            test_rmse  = float(np.sqrt(criterion(preds_eval, y_eval).item()))

        scheduler.step(test_rmse)

        # Checkpoint best weights
        if test_rmse < best_rmse - 0.1:
            best_rmse    = test_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), out_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:>4} | "
                f"Train RMSE: {train_rmse:.2f} | "
                f"Test RMSE: {test_rmse:.2f} | "
                f"Best: {best_rmse:.2f}"
            )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # ── Final evaluation with best weights ──────────────────────────────────
    model.load_state_dict(best_weights)
    model.eval()

    def predict(X_np):
        with torch.no_grad():
            t = torch.FloatTensor(X_np).to(device)
            return model(t).cpu().numpy()

    evaluate(predict, test_data, "LSTM multi-condition")

    print(f"\nBest weights → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train XGBoost or LSTM on the multi-condition CMAPSS arrays."
    )

    # Required
    p.add_argument(
        "--model",
        required = True,
        choices  = ["xgb", "lstm"],
        help     = "Which model to train.",
    )

    # Paths
    p.add_argument("--data-dir",   type=Path, default=Path("data"),   help="Directory with .npy arrays. (default: data)")
    p.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory to save trained models. (default: models)")

    # XGBoost hyperparameters
    xgb_g = p.add_argument_group("XGBoost hyperparameters")
    xgb_g.add_argument("--n-estimators", type=int,   default=1000, help="Max boosting rounds. (default: 1000)")
    xgb_g.add_argument("--max-depth",    type=int,   default=6,    help="Max tree depth. (default: 6)")

    # LSTM hyperparameters
    lstm_g = p.add_argument_group("LSTM hyperparameters")
    lstm_g.add_argument("--epochs",     type=int,   default=200,   help="Max training epochs. (default: 200)")
    lstm_g.add_argument("--batch-size", type=int,   default=512,   help="Mini-batch size. (default: 512)")
    lstm_g.add_argument("--patience",   type=int,   default=40,    help="Early-stopping patience in epochs. (default: 40)")

    # Shared
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. XGB default: 0.05 | LSTM default: 0.001")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply per-model lr defaults when the user didn't specify
    if args.lr is None:
        args.lr = 0.05 if args.model == "xgb" else 0.001

    args.models_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, test_data = load_arrays(args.data_dir)

    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    for s, (X, y) in test_data.items():
        print(f"{s} test  : {X.shape}")

    if args.model == "xgb":
        train_xgb(X_train, y_train, test_data, args.models_dir, args)
    else:
        train_lstm(X_train, y_train, test_data, args.models_dir, args)