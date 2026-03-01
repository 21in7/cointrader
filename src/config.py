import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "XRPUSDT"
    leverage: int = 10
    risk_per_trade: float = 0.02
    max_positions: int = 3
    stop_loss_pct: float = 0.015    # 1.5%
    take_profit_pct: float = 0.045  # 4.5% (3:1 RR)
    trailing_stop_pct: float = 0.01  # 1%
    discord_webhook_url: str = ""

    def __post_init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.02"))
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
