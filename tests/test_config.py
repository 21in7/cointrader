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
