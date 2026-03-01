import asyncio
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.ml_filter import MLFilter

MODEL_PATH      = Path("models/lgbm_filter.pkl")
PREV_MODEL_PATH = Path("models/lgbm_filter_prev.pkl")
LOG_PATH        = Path("models/training_log.json")


def get_current_auc() -> float:
    """training_log.json에서 가장 최근 AUC를 읽는다."""
    if not LOG_PATH.exists():
        return 0.0
    with open(LOG_PATH) as f:
        log = json.load(f)
    return log[-1]["auc"] if log else 0.0


def rollback_model():
    """이전 모델로 롤백한다."""
    if PREV_MODEL_PATH.exists():
        import shutil
        shutil.copy(PREV_MODEL_PATH, MODEL_PATH)
        logger.warning("ML 모델 롤백 완료")
    else:
        logger.warning("롤백할 이전 모델 없음")


async def fetch_and_save(data_path: str):
    """증분 데이터 수집 (fetch_history.py 로직 재사용)."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/fetch_history.py", "--output", data_path, "--days", "90"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"데이터 수집 실패: {result.stderr}")
    logger.info(f"데이터 수집 완료: {data_path}")


def run_training(data_path: str) -> float:
    """train_model.py를 실행하고 새 AUC를 반환한다."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/train_model.py", "--data", data_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"학습 실패: {result.stderr}")
    new_auc = get_current_auc()
    return new_auc


class Retrainer:
    def __init__(self, ml_filter: MLFilter, data_path: str = "data/xrpusdt_1m.parquet"):
        self._ml_filter = ml_filter
        self._data_path = data_path

    async def retrain(self):
        logger.info("자동 재학습 시작")
        old_auc = get_current_auc()
        try:
            await fetch_and_save(self._data_path)
            new_auc = run_training(self._data_path)
            logger.info(f"재학습 완료: 이전 AUC={old_auc:.4f} → 새 AUC={new_auc:.4f}")

            if new_auc < old_auc - 0.01:
                logger.warning(f"새 모델 성능 저하 ({new_auc:.4f} < {old_auc:.4f}), 롤백")
                rollback_model()
            else:
                self._ml_filter.reload_model()
                logger.success("새 ML 모델 적용 완료")
        except Exception as e:
            logger.error(f"재학습 실패: {e}")

    async def schedule_daily(self, hour: int = 3):
        """매일 지정 시각(컨테이너 로컬 시간 기준)에 재학습을 실행한다."""
        from datetime import timedelta
        while True:
            now = datetime.now()
            next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            wait_secs = (next_run - now).total_seconds()
            logger.info(f"다음 재학습까지 {wait_secs/3600:.1f}시간 대기")
            await asyncio.sleep(wait_secs)
            await self.retrain()
