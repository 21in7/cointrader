import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.ml_filter import MLFilter
from src.ml_features import FEATURE_COLS


def make_features(side="LONG") -> pd.Series:
    return pd.Series({col: 0.5 for col in FEATURE_COLS} | {"side": 1.0 if side == "LONG" else 0.0})


def test_no_model_file_is_not_loaded(tmp_path):
    f = MLFilter(
        onnx_path=str(tmp_path / "nonexistent.onnx"),
        lgbm_path=str(tmp_path / "nonexistent.pkl"),
    )
    assert not f.is_model_loaded()


def test_no_model_should_enter_returns_true(tmp_path):
    """모델 없으면 항상 진입 허용 (폴백)"""
    f = MLFilter(
        onnx_path=str(tmp_path / "nonexistent.onnx"),
        lgbm_path=str(tmp_path / "nonexistent.pkl"),
    )
    features = make_features()
    assert f.should_enter(features) is True


def test_should_enter_above_threshold():
    """확률 >= 0.60 이면 True"""
    f = MLFilter(threshold=0.60)
    f._onnx_session = None  # ONNX 비활성화, LightGBM만 테스트
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.35, 0.65]])
    f._lgbm_model = mock_model
    features = make_features()
    assert f.should_enter(features) is True


def test_should_enter_below_threshold():
    """확률 < 0.60 이면 False"""
    f = MLFilter(threshold=0.60)
    f._onnx_session = None  # ONNX 비활성화, LightGBM만 테스트
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.55, 0.45]])
    f._lgbm_model = mock_model
    features = make_features()
    assert f.should_enter(features) is False


def test_reload_model(tmp_path):
    """reload_model 호출 후 모델 로드 상태 변경"""
    import joblib

    # 모델 파일이 없는 상태에서 시작
    f = MLFilter(
        onnx_path=str(tmp_path / "nonexistent.onnx"),
        lgbm_path=str(tmp_path / "lgbm_filter.pkl"),
    )
    assert not f.is_model_loaded()

    # _lgbm_model을 직접 주입해서 is_model_loaded가 True인지 확인
    mock_model = MagicMock()
    f._lgbm_model = mock_model
    assert f.is_model_loaded()

    # reload_model은 항상 _lgbm_model/_onnx_session을 초기화 후 재로드한다.
    # 파일이 없으면 None으로 리셋되어 폴백 상태가 된다.
    f.reload_model()
    assert not f.is_model_loaded()  # 파일 없으므로 폴백 상태
