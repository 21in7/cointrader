import os
import pytest
from src.config import Config


def test_config_loads_symbol():
    os.environ["SYMBOL"] = "XRPUSDT"
    os.environ["LEVERAGE"] = "10"
    cfg = Config()
    assert cfg.symbol == "XRPUSDT"
    assert cfg.leverage == 10


def test_config_dynamic_margin_params():
    os.environ["MARGIN_MAX_RATIO"] = "0.50"
    os.environ["MARGIN_MIN_RATIO"] = "0.20"
    os.environ["MARGIN_DECAY_RATE"] = "0.0006"
    cfg = Config()
    assert cfg.margin_max_ratio == 0.50
    assert cfg.margin_min_ratio == 0.20
    assert cfg.margin_decay_rate == 0.0006


def test_config_loads_symbols_list():
    """SYMBOLS 환경변수로 쉼표 구분 리스트를 로드한다."""
    os.environ["SYMBOLS"] = "XRPUSDT,TRXUSDT,DOGEUSDT"
    os.environ.pop("SYMBOL", None)
    cfg = Config()
    assert cfg.symbols == ["XRPUSDT", "TRXUSDT", "DOGEUSDT"]


def test_config_fallback_to_symbol():
    """SYMBOLS 미설정 시 SYMBOL에서 1개짜리 리스트로 변환한다."""
    os.environ.pop("SYMBOLS", None)
    os.environ["SYMBOL"] = "XRPUSDT"
    cfg = Config()
    assert cfg.symbols == ["XRPUSDT"]


def test_config_correlation_symbols():
    """상관관계 심볼 로드."""
    os.environ["CORRELATION_SYMBOLS"] = "BTCUSDT,ETHUSDT"
    cfg = Config()
    assert cfg.correlation_symbols == ["BTCUSDT", "ETHUSDT"]


def test_config_max_same_direction_default():
    """동일 방향 최대 수 기본값 2."""
    cfg = Config()
    assert cfg.max_same_direction == 2
