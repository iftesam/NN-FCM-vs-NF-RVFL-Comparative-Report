import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import optuna
import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
import argparse
from time import time
import numpy as np
from sklearn.metrics import r2_score


class _TorchFCM(nn.Module):
    """Internal PyTorch module for Neural-Fuzzy Cognitive Map with BatchNorm."""
    def __init__(self, n_features, n_rules, mlp_units, dropout):
        super().__init__()
        self.n_rules = n_rules
        self.attn_centers = nn.Parameter(torch.randn(n_rules, n_features) * 0.5)
        self.attn_sigmas  = nn.Parameter(torch.ones(n_rules, n_features) * 0.5)

        self.fc1 = nn.Linear(n_rules, mlp_units); self.bn1 = nn.BatchNorm1d(mlp_units)
        self.fc2 = nn.Linear(mlp_units, mlp_units); self.bn2 = nn.BatchNorm1d(mlp_units)
        self.fc3 = nn.Linear(mlp_units, mlp_units); self.bn3 = nn.BatchNorm1d(mlp_units)
        self.fc4 = nn.Linear(mlp_units, mlp_units); self.bn4 = nn.BatchNorm1d(mlp_units)

        self.heads = nn.ModuleList([nn.Linear(mlp_units, 1) for _ in range(3)])
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # RBF memberships over rules
        diff = X.unsqueeze(1) - self.attn_centers.unsqueeze(0)
        rbf = torch.exp(-0.5 * (diff**2) / (self.attn_sigmas.unsqueeze(0)**2 + 1e-8))
        mu = rbf.mean(dim=-1)  # [B, R]

        # Residual MLP with 4 blocks
        h = self.dropout(self.relu(self.bn1(self.fc1(mu))))
        h = self.dropout(self.relu(self.bn2(self.fc2(h))) + h)
        h = self.dropout(self.relu(self.bn3(self.fc3(h))) + h)
        h = self.dropout(self.relu(self.bn4(self.fc4(h))) + h)

        out_heads = torch.stack([head(h) for head in self.heads], dim=2)  # [B, 1, 3]
        return out_heads.mean(dim=2).squeeze(1)  # [B]


class NNFuzzyCognitiveMap(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_rules=10,
        lr=1e-3,
        mlp_units=20,
        dropout=0.2,
        l2=1e-4,
        batch_size=32,
        epochs=100,
        patience=10,
        random_state=42
    ):
        self.n_rules = n_rules
        self.lr = lr
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.l2 = l2
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self._model = None
        self._scaler = StandardScaler()

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        X = self._scaler.fit_transform(X)  # 🟢 Standardize features
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n_samples, n_features = X.shape

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=0.4,  # 60/40 split
            random_state=self.random_state
        )

        tX_tr = torch.tensor(X_tr, dtype=torch.float32)
        ty_tr = torch.tensor(y_tr, dtype=torch.float32)
        tX_val = torch.tensor(X_val, dtype=torch.float32)
        ty_val = torch.tensor(y_val, dtype=torch.float32)

        train_ds = TensorDataset(tX_tr, ty_tr)
        val_ds   = TensorDataset(tX_val, ty_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size)

        self._model = _TorchFCM(
            n_features=n_features,
            n_rules=self.n_rules,
            mlp_units=self.mlp_units,
            dropout=self.dropout
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        opt = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=20
        )

        loss_fn = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = self._model(xb)
                loss = loss_fn(out.unsqueeze(1), yb)
                loss.backward()
                opt.step()

            val_losses = []
            self._model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = self._model(xb)
                    val_losses.append(loss_fn(out.unsqueeze(1), yb).item())

            avg_val = np.mean(val_losses)
            print(f"Epoch {epoch:3d} | val_loss={avg_val:.4f}")

            # learning rate adjustment
            scheduler.step(avg_val)

            if avg_val < best_val - 1e-6:
                best_val = avg_val
                best_state = self._model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping @ epoch {epoch}")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X):
        X = self._scaler.transform(np.asarray(X, dtype=float))  # 🟢 Standardize input before prediction
        X = torch.tensor(X, dtype=torch.float32)
        device = next(self._model.parameters()).device
        X = X.to(device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X).cpu().numpy()
        return out

    def score(self, X, y):
        yhat = self.predict(X)
        return r2_score(y, yhat)


def optimize_hyperparams(X, y, n_trials=30, random_state=42):
    def objective(trial):
        params = {
            'n_rules': trial.suggest_categorical('n_rules', [75, 100, 150, 200, 300, 400]),
            'lr':           trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'mlp_units':    trial.suggest_int('mlp_units', 50, 120),
            'dropout':      trial.suggest_float('dropout', 0.0, 0.5),
            'l2':           trial.suggest_float('l2', 1e-6, 1e-2, log=True),
            'epochs':       50,
            'batch_size':   32,
            'patience':     5,
            'random_state': random_state
        }
        model = NNFuzzyCognitiveMap(**params)
        scores = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=random_state),
                                 scoring='r2', n_jobs=1)
        return -np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparams:", study.best_params)
    return study.best_params


def load_csv(path):
    # 1) Try comma-separated first
    df = pd.read_csv(path)

    # 2) If it came in as one big column (or too few numeric cols), re-try whitespace delim
    num_numeric = df.select_dtypes(include=[np.number]).shape[1]
    if df.shape[1] < 2 or num_numeric < 2:
        df = pd.read_csv(path, header=None, delim_whitespace=True)

    # 3) Drop any incomplete rows
    df = df.dropna()

    # 4) Pull off the last column as raw labels (could be strings)
    y_raw = df.iloc[:, -1]

    # 5) If it’s not already numeric, turn every distinct value into an integer code
    if not np.issubdtype(y_raw.dtype, np.number):
        y = pd.factorize(y_raw)[0]
    else:
        y = y_raw.values

    # 6) From the remaining columns, keep only the numeric ones as X
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    return X, y


def main():
    parser = argparse.ArgumentParser("nn-fcm-auto")
    parser.add_argument(
        "data",
        nargs='?',
        help="Path to CSV (or whitespace-delimited) file where the last column is the target"
    )
    args = parser.parse_args()

    if not args.data:
        app = QApplication(sys.argv)
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select data file",
            "",
            "CSV (*.csv);;Data (*.data);;All (*)"
        )
        if not path:
            sys.exit("No file selected.")
        args.data = path

    X, y = load_csv(args.data)

    print("Running Optuna hyperparameter search...")
    best_params = optimize_hyperparams(X, y, n_trials=100)
    print("Using best hyperparameters:", best_params)

    # Final training settings
    best_params.update({
        'epochs':       2000,
        'batch_size':   70,
        'patience':     400,
        'random_state': 50
    })

    # Ensemble training
    n_members = 5
    models = []
    start_time = time()
    for i in range(n_members):
        params = best_params.copy()
        params['random_state'] = best_params['random_state'] + i
        model = NNFuzzyCognitiveMap(**params).fit(X, y)
        models.append(model)
    elapsed = time() - start_time

    # Announce which file we are processing
    print(f"Processing data file: {args.data}")
    print(f"Ensemble training took {elapsed:.1f} seconds")

    # Ensemble prediction & scoring
    all_preds = np.stack([m.predict(X) for m in models], axis=0)
    yhat = all_preds.mean(axis=0)
    score = r2_score(y, yhat) * 100
    print(f"Ensemble R² = {score:.2f}%")

    # Show first few predictions
    print("First 5 ensemble predictions:", yhat[:5])


if __name__ == "__main__":
    main()
