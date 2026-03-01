from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.ml_features import FEATURE_COLS

ONNX_MODEL_PATH = Path("models/mlx_filter.weights.onnx")
LGBM_MODEL_PATH = Path("models/lgbm_filter.pkl")


class MLFilter:
    """
    ML 필터. ONNX(MLX 신경망) 우선 로드, 없으면 LightGBM으로 폴백한다.
    둘 다 없으면 항상 진입을 허용한다.

    우선순위: ONNX > LightGBM > 폴백(항상 허용)
    """

    def __init__(
        self,
        onnx_path: str = str(ONNX_MODEL_PATH),
        lgbm_path: str = str(LGBM_MODEL_PATH),
        threshold: float = 0.60,
    ):
        self._onnx_path = Path(onnx_path)
        self._lgbm_path = Path(lgbm_path)
        self._threshold = threshold
        self._onnx_session = None
        self._lgbm_model = None
        self._try_load()

    def _try_load(self):
        # ONNX 우선 시도
        if self._onnx_path.exists():
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(
                    str(self._onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self._lgbm_model = None
                logger.info(f"ML 필터 ONNX 모델 로드 완료: {self._onnx_path}")
                return
            except Exception as e:
                logger.warning(f"ONNX 모델 로드 실패: {e}")
                self._onnx_session = None

        # LightGBM 폴백
        if self._lgbm_path.exists():
            try:
                self._lgbm_model = joblib.load(self._lgbm_path)
                logger.info(f"ML 필터 LightGBM 모델 로드 완료: {self._lgbm_path}")
            except Exception as e:
                logger.warning(f"LightGBM 모델 로드 실패: {e}")
                self._lgbm_model = None

    def is_model_loaded(self) -> bool:
        return self._onnx_session is not None or self._lgbm_model is not None

    def should_enter(self, features: pd.Series) -> bool:
        """
        확률 >= threshold 이면 True (진입 허용).
        모델 없으면 True 반환 (폴백).
        """
        if not self.is_model_loaded():
            return True
        try:
            if self._onnx_session is not None:
                input_name = self._onnx_session.get_inputs()[0].name
                X = features[FEATURE_COLS].values.astype(np.float32).reshape(1, -1)
                proba = float(self._onnx_session.run(None, {input_name: X})[0][0])
            else:
                X = features.to_frame().T
                proba = float(self._lgbm_model.predict_proba(X)[0][1])
            logger.debug(f"ML 필터 확률: {proba:.3f} (임계값: {self._threshold})")
            return bool(proba >= self._threshold)
        except Exception as e:
            logger.warning(f"ML 필터 예측 오류 (폴백 허용): {e}")
            return True

    def reload_model(self):
        """재학습 후 모델을 핫 리로드한다."""
        self._onnx_session = None
        self._lgbm_model = None
        self._try_load()
        logger.info("ML 필터 모델 리로드 완료")
