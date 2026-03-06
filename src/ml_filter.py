import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.ml_features import FEATURE_COLS

ONNX_MODEL_PATH = Path("models/mlx_filter.weights.onnx")
LGBM_MODEL_PATH = Path("models/lgbm_filter.pkl")


def _mtime(path: Path) -> float:
    """파일이 없으면 0.0 반환."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


class MLFilter:
    """
    ML 필터. ONNX(MLX 신경망) 우선 로드, 없으면 LightGBM으로 폴백한다.
    둘 다 없으면 항상 진입을 허용한다.

    우선순위: ONNX > LightGBM > 폴백(항상 허용)

    check_and_reload()를 주기적으로 호출하면 모델 파일 변경 시 자동 리로드된다.
    """

    def __init__(
        self,
        onnx_path: str = str(ONNX_MODEL_PATH),
        lgbm_path: str = str(LGBM_MODEL_PATH),
        threshold: float = 0.55,
    ):
        self._disabled = os.environ.get("NO_ML_FILTER", "").lower() in ("1", "true", "yes")
        self._onnx_path = Path(onnx_path)
        self._lgbm_path = Path(lgbm_path)
        self._threshold = threshold
        self._onnx_session = None
        self._lgbm_model = None
        self._loaded_onnx_mtime: float = 0.0
        self._loaded_lgbm_mtime: float = 0.0

        if self._disabled:
            logger.info("ML 필터 비활성화 모드 (NO_ML_FILTER=true) → 모든 신호 허용")
        else:
            self._try_load()

    def _try_load(self):
        # 로드 여부와 무관하게 두 파일의 현재 mtime을 항상 기록한다.
        # 이렇게 해야 로드하지 않은 쪽 파일이 나중에 변경됐을 때만 리로드가 트리거된다.
        self._loaded_onnx_mtime = _mtime(self._onnx_path)
        self._loaded_lgbm_mtime = _mtime(self._lgbm_path)

        # ONNX 우선 시도
        if self._onnx_path.exists():
            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1
                self._onnx_session = ort.InferenceSession(
                    str(self._onnx_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._lgbm_model = None
                logger.info(
                    f"ML 필터 로드: ONNX ({self._onnx_path}) "
                    f"| 임계값={self._threshold}"
                )
                return
            except Exception as e:
                logger.warning(f"ONNX 모델 로드 실패: {e}")
                self._onnx_session = None

        # LightGBM 폴백
        if self._lgbm_path.exists():
            try:
                self._lgbm_model = joblib.load(self._lgbm_path)
                logger.info(
                    f"ML 필터 로드: LightGBM ({self._lgbm_path}) "
                    f"| 임계값={self._threshold}"
                )
            except Exception as e:
                logger.warning(f"LightGBM 모델 로드 실패: {e}")
                self._lgbm_model = None
        else:
            logger.warning("ML 필터: 모델 파일 없음 → 모든 신호 허용 (폴백)")

    def is_model_loaded(self) -> bool:
        return self._onnx_session is not None or self._lgbm_model is not None

    @property
    def active_backend(self) -> str:
        if self._onnx_session is not None:
            return "ONNX"
        if self._lgbm_model is not None:
            return "LightGBM"
        return "폴백(없음)"

    def check_and_reload(self) -> bool:
        """
        모델 파일의 mtime을 확인해 변경됐으면 리로드한다.
        실제로 리로드가 일어났으면 True 반환.
        """
        if self._disabled: return False
        onnx_changed = _mtime(self._onnx_path) != self._loaded_onnx_mtime
        lgbm_changed = _mtime(self._lgbm_path) != self._loaded_lgbm_mtime

        if onnx_changed or lgbm_changed:
            changed_files = []
            if onnx_changed:
                changed_files.append(str(self._onnx_path))
            if lgbm_changed:
                changed_files.append(str(self._lgbm_path))
            logger.info(f"ML 필터: 모델 파일 변경 감지 → 리로드 ({', '.join(changed_files)})")
            self._onnx_session = None
            self._lgbm_model = None
            self._try_load()
            logger.info(f"ML 필터 핫리로드 완료: 백엔드={self.active_backend}")
            return True
        return False

    def should_enter(self, features: pd.Series) -> bool:
        """
        확률 >= threshold 이면 True (진입 허용).
        NO_ML_FILTER=true 이거나 모델 없으면 True 반환 (폴백).
        """
        if self._disabled:
            logger.debug("ML 필터 비활성화 모드 → 진입 허용")
            return True
        if not self.is_model_loaded():
            return True
        try:
            if self._onnx_session is not None:
                input_name = self._onnx_session.get_inputs()[0].name
                X = features[FEATURE_COLS].values.astype(np.float32).reshape(1, -1)
                proba = float(self._onnx_session.run(None, {input_name: X})[0][0])
            else:
                available = [c for c in FEATURE_COLS if c in features.index]
                X = pd.DataFrame([features[available].values.astype(np.float64)], columns=available)
                proba = float(self._lgbm_model.predict_proba(X)[0][1])
            logger.debug(
                f"ML 필터 [{self.active_backend}] 확률: {proba:.3f} "
                f"(임계값: {self._threshold})"
            )
            return bool(proba >= self._threshold)
        except Exception as e:
            logger.warning(f"ML 필터 예측 오류 (진입 차단): {e}")
            return False

    def reload_model(self):
        """외부에서 강제 리로드할 때 사용 (하위 호환)."""
        prev_backend = self.active_backend
        self._onnx_session = None
        self._lgbm_model = None
        self._try_load()
        logger.info(
            f"ML 필터 강제 리로드 완료: {prev_backend} → {self.active_backend}"
        )
