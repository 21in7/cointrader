import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from src.retrainer import Retrainer


@pytest.mark.asyncio
async def test_retrain_calls_train(tmp_path):
    """재학습 시 train 함수가 호출되는지 확인"""
    ml_filter = MagicMock()
    r = Retrainer(ml_filter=ml_filter, data_path=str(tmp_path / "data.parquet"))

    with patch("src.retrainer.fetch_and_save", new_callable=AsyncMock) as mock_fetch, \
         patch("src.retrainer.run_training", return_value=0.72) as mock_train, \
         patch("src.retrainer.get_current_auc", return_value=0.65):
        await r.retrain()

    mock_fetch.assert_called_once()
    mock_train.assert_called_once()


@pytest.mark.asyncio
async def test_retrain_rollback_when_worse(tmp_path):
    """새 모델이 기존보다 나쁘면 롤백"""
    ml_filter = MagicMock()
    r = Retrainer(ml_filter=ml_filter, data_path=str(tmp_path / "data.parquet"))

    with patch("src.retrainer.fetch_and_save", new_callable=AsyncMock), \
         patch("src.retrainer.run_training", return_value=0.55), \
         patch("src.retrainer.get_current_auc", return_value=0.70), \
         patch("src.retrainer.rollback_model") as mock_rollback:
        await r.retrain()

    mock_rollback.assert_called_once()
