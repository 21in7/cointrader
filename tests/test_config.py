import os
import pytest
from src.config import Config


def test_config_loads_symbol():
    os.environ["SYMBOL"] = "XRPUSDT"
    os.environ["LEVERAGE"] = "10"
    os.environ["RISK_PER_TRADE"] = "0.02"
    cfg = Config()
    assert cfg.symbol == "XRPUSDT"
    assert cfg.leverage == 10
    assert cfg.risk_per_trade == 0.02


def test_config_notion_keys():
    os.environ["NOTION_TOKEN"] = "secret_test"
    os.environ["NOTION_DATABASE_ID"] = "db_test_id"
    cfg = Config()
    assert cfg.notion_token == "secret_test"
    assert cfg.notion_database_id == "db_test_id"
