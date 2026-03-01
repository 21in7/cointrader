"""
MLXFilter 단위 테스트.
Apple Silicon GPU(Metal)가 없는 환경에서는 스킵한다.
"""
import numpy as np
import pandas as pd
import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX 미설치")


def _make_X(n: int = 4) -> pd.DataFrame:
    from src.ml_features import FEATURE_COLS
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.uniform(-1.0, 1.0, (n, len(FEATURE_COLS))).astype(np.float32),
        columns=FEATURE_COLS,
    )


def test_mlx_gpu_device():
    """MLX가 GPU 디바이스를 기본으로 사용해야 한다."""
    import mlx.core as mx

    device = mx.default_device()
    assert "gpu" in str(device)


def test_mlx_filter_predict_shape_untrained():
    """학습 전에도 predict_proba가 (N,) 형태를 반환해야 한다."""
    from src.mlx_filter import MLXFilter
    from src.ml_features import FEATURE_COLS

    X = _make_X(4)
    model = MLXFilter(input_dim=len(FEATURE_COLS), hidden_dim=32)
    proba = model.predict_proba(X)
    assert proba.shape == (4,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_mlx_filter_fit_and_predict():
    """학습 후 predict_proba가 유효한 확률값을 반환해야 한다."""
    from src.mlx_filter import MLXFilter
    from src.ml_features import FEATURE_COLS

    n = 100
    X = _make_X(n)
    y = pd.Series(np.random.randint(0, 2, n))

    model = MLXFilter(input_dim=len(FEATURE_COLS), hidden_dim=32, epochs=5, batch_size=32)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (n,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_fit_with_nan_features():
    """oi_change 피처에 nan이 포함된 경우 학습이 정상 완료되어야 한다."""
    import numpy as np
    import pandas as pd
    from src.mlx_filter import MLXFilter
    from src.ml_features import FEATURE_COLS

    n = 300
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, len(FEATURE_COLS)).astype(np.float32),
        columns=FEATURE_COLS,
    )
    # oi_change 앞 절반을 nan으로
    X["oi_change"] = np.where(np.arange(n) < n // 2, np.nan, X["oi_change"])
    y = pd.Series((np.random.rand(n) > 0.5).astype(np.float32))

    model = MLXFilter(input_dim=len(FEATURE_COLS), hidden_dim=32, epochs=3)
    model.fit(X, y)  # nan 있어도 예외 없이 완료되어야 함

    proba = model.predict_proba(X)
    assert not np.any(np.isnan(proba)), "예측 확률에 nan이 없어야 함"
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_mlx_filter_save_load(tmp_path):
    """저장 후 로드한 모델이 동일한 예측값을 반환해야 한다."""
    from src.mlx_filter import MLXFilter
    from src.ml_features import FEATURE_COLS

    n = 50
    X = _make_X(n)
    y = pd.Series(np.random.randint(0, 2, n))

    model = MLXFilter(input_dim=len(FEATURE_COLS), hidden_dim=32, epochs=3, batch_size=32)
    model.fit(X, y)
    proba_before = model.predict_proba(X)

    save_path = tmp_path / "mlx_filter.weights"
    model.save(save_path)

    loaded = MLXFilter.load(save_path)
    proba_after = loaded.predict_proba(X)

    np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)
