"""
Apple MLX 기반 경량 신경망 필터.
M4의 통합 GPU를 자동으로 활용한다.
학습 후 ONNX로 export해 Linux 서버에서 onnxruntime으로 추론한다.
"""
import numpy as np
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path

from src.ml_features import FEATURE_COLS


def _export_onnx(
    weights_npz: Path,
    meta_npz: Path,
    onnx_path: Path,
) -> None:
    """
    MLX 가중치(.npz)를 읽어 ONNX 그래프로 변환한다.
    네트워크 구조: fc1(ReLU) → dropout(추론 시 비활성) → fc2(ReLU) → fc3 → sigmoid
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    meta = np.load(meta_npz)
    mean: np.ndarray = meta["mean"].astype(np.float32)
    std: np.ndarray  = meta["std"].astype(np.float32)
    input_dim  = int(meta["input_dim"])
    hidden_dim = int(meta["hidden_dim"])

    w = np.load(weights_npz)
    # MLX save_weights 키 패턴: fc1.weight, fc1.bias, ...
    fc1_w = w["fc1.weight"].astype(np.float32)   # (hidden, input)
    fc1_b = w["fc1.bias"].astype(np.float32)
    fc2_w = w["fc2.weight"].astype(np.float32)   # (hidden//2, hidden)
    fc2_b = w["fc2.bias"].astype(np.float32)
    fc3_w = w["fc3.weight"].astype(np.float32)   # (1, hidden//2)
    fc3_b = w["fc3.bias"].astype(np.float32)

    def _t(name: str, arr: np.ndarray) -> onnx.TensorProto:
        return numpy_helper.from_array(arr, name=name)

    initializers = [
        _t("mean",  mean),
        _t("std",   std),
        _t("fc1_w", fc1_w),
        _t("fc1_b", fc1_b),
        _t("fc2_w", fc2_w),
        _t("fc2_b", fc2_b),
        _t("fc3_w", fc3_w),
        _t("fc3_b", fc3_b),
    ]

    nodes = [
        # 정규화: (x - mean) / std
        helper.make_node("Sub",     ["X", "mean"],      ["x_sub"]),
        helper.make_node("Div",     ["x_sub", "std"],   ["x_norm"]),
        # fc1: x_norm @ fc1_w.T + fc1_b
        helper.make_node("Gemm",    ["x_norm", "fc1_w", "fc1_b"], ["fc1_out"],
                         transB=1),
        helper.make_node("Relu",    ["fc1_out"],         ["relu1"]),
        # fc2: relu1 @ fc2_w.T + fc2_b
        helper.make_node("Gemm",    ["relu1",  "fc2_w", "fc2_b"], ["fc2_out"],
                         transB=1),
        helper.make_node("Relu",    ["fc2_out"],         ["relu2"]),
        # fc3: relu2 @ fc3_w.T + fc3_b  → (N, 1)
        helper.make_node("Gemm",    ["relu2",  "fc3_w", "fc3_b"], ["logits"],
                         transB=1),
        # sigmoid → (N, 1)
        helper.make_node("Sigmoid", ["logits"],          ["proba_2d"]),
        # squeeze: (N, 1) → (N,)
        helper.make_node("Flatten", ["proba_2d"],        ["proba"], axis=0),
    ]

    graph = helper.make_graph(
        nodes,
        "mlx_filter",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, input_dim])],
        outputs=[helper.make_tensor_value_info("proba", TensorProto.FLOAT, [None])],
        initializer=initializers,
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto)
    onnx_path.parent.mkdir(exist_ok=True)
    onnx.save(model_proto, str(onnx_path))
    print(f"  ONNX export 완료: {onnx_path}")


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

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "MLXFilter":
        X_np = X[FEATURE_COLS].values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        # nan-safe 정규화: nanmean/nanstd로 통계 계산 후 nan → 0.0 대치
        # (z-score 후 0.0 = 평균값, 신경망에 줄 수 있는 가장 무난한 결측 대치값)
        mean_vals  = np.nanmean(X_np, axis=0)
        self._mean = np.nan_to_num(mean_vals, nan=0.0)   # 전체-NaN 컬럼 → 평균 0.0
        std_vals   = np.nanstd(X_np, axis=0)
        self._std  = np.nan_to_num(std_vals, nan=1.0) + 1e-8  # 전체-NaN 컬럼 → std 1.0
        X_np = (X_np - self._mean) / self._std
        X_np = np.nan_to_num(X_np, nan=0.0)

        w_np = sample_weight.astype(np.float32) if sample_weight is not None else None

        optimizer = optim.Adam(learning_rate=self.lr)

        def loss_fn(
            model: _Net, x: mx.array, y: mx.array, w: mx.array | None
        ) -> mx.array:
            logits = model(x)
            per_sample = nn.losses.binary_cross_entropy(
                logits, y, with_logits=True, reduction="none"
            )
            if w is not None:
                return (per_sample * w).sum() / w.sum()
            return per_sample.mean()

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
                w_batch = mx.array(w_np[batch_idx]) if w_np is not None else None
                loss, grads = loss_and_grad(self._model, x_batch, y_batch, w_batch)
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
            X_np = np.nan_to_num(X_np, nan=0.0)
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
        # ONNX export: Linux 서버에서 onnxruntime으로 추론하기 위해 변환
        try:
            onnx_path = path.with_suffix(".onnx")
            _export_onnx(weights_path, meta_path, onnx_path)
        except ImportError:
            print("  [경고] onnx 패키지 없음 → ONNX export 생략 (pip install onnx)")

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
