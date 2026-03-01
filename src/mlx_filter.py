"""
Apple MLX 기반 경량 신경망 필터.
M4의 통합 GPU를 자동으로 활용한다.
"""
import numpy as np
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path

from src.ml_features import FEATURE_COLS


class _Net(nn.Module):
    """3층 MLP 이진 분류기."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.2)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class MLXFilter:
    """
    scikit-learn 호환 인터페이스를 제공하는 MLX 신경망 필터.
    M4 통합 GPU(Metal)를 자동으로 사용한다.
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = _Net(input_dim, hidden_dim)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLXFilter":
        X_np = X[FEATURE_COLS].values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        self._mean = X_np.mean(axis=0)
        self._std = X_np.std(axis=0) + 1e-8
        X_np = (X_np - self._mean) / self._std

        optimizer = optim.Adam(learning_rate=self.lr)

        def loss_fn(model: _Net, x: mx.array, y: mx.array) -> mx.array:
            logits = model(x)
            return nn.losses.binary_cross_entropy(logits, y, with_logits=True)

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)

        n = len(X_np)
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                x_batch = mx.array(X_np[batch_idx])
                y_batch = mx.array(y_np[batch_idx])
                loss, grads = loss_and_grad(self._model, x_batch, y_batch)
                optimizer.update(self._model, grads)
                mx.eval(self._model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                steps += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}  loss={epoch_loss / steps:.4f}")

        self._trained = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X[FEATURE_COLS].values.astype(np.float32)
        if self._trained and self._mean is not None:
            X_np = (X_np - self._mean) / self._std
        x = mx.array(X_np)
        self._model.eval()
        logits = self._model(x)
        proba = mx.sigmoid(logits)
        mx.eval(proba)
        self._model.train()
        return np.array(proba)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        weights_path = path.with_suffix(".npz")
        self._model.save_weights(str(weights_path))
        meta_path = path.with_suffix(".meta.npz")
        np.savez(
            meta_path,
            mean=self._mean,
            std=self._std,
            input_dim=np.array(self.input_dim),
            hidden_dim=np.array(self.hidden_dim),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLXFilter":
        path = Path(path)
        meta = np.load(path.with_suffix(".meta.npz"))
        obj = cls(
            input_dim=int(meta["input_dim"]),
            hidden_dim=int(meta["hidden_dim"]),
        )
        obj._mean = meta["mean"]
        obj._std = meta["std"]
        obj._model.load_weights(str(path.with_suffix(".npz")))
        obj._trained = True
        return obj
