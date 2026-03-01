from pathlib import Path
import joblib
import pandas as pd
from loguru import logger


class MLFilter:
    """
    LightGBM 모델을 로드하고 진입 여부를 판단한다.
    모델 파일이 없으면 항상 진입을 허용한다 (폴백).
    """

    def __init__(self, model_path: str = "models/lgbm_filter.pkl", threshold: float = 0.60):
        self._model_path = Path(model_path)
        self._threshold = threshold
        self._model = None
        self._try_load()

    def _try_load(self):
        if self._model_path.exists():
            try:
                self._model = joblib.load(self._model_path)
                logger.info(f"ML 필터 모델 로드 완료: {self._model_path}")
            except Exception as e:
                logger.warning(f"ML 필터 모델 로드 실패: {e}")
                self._model = None

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def should_enter(self, features: pd.Series) -> bool:
        """
        확률 >= threshold 이면 True (진입 허용).
        모델 없으면 True 반환 (폴백).
        """
        if not self.is_model_loaded():
            return True
        try:
            X = features.to_frame().T
            proba = self._model.predict_proba(X)[0][1]
            logger.debug(f"ML 필터 확률: {proba:.3f} (임계값: {self._threshold})")
            return bool(proba >= self._threshold)
        except Exception as e:
            logger.warning(f"ML 필터 예측 오류 (폴백 허용): {e}")
            return True

    def reload_model(self):
        """재학습 후 모델을 핫 리로드한다."""
        self._try_load()
        logger.info("ML 필터 모델 리로드 완료")
