import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "XRPUSDT"
    symbols: list = None
    correlation_symbols: list = None
    leverage: int = 10
    max_positions: int = 3
    max_same_direction: int = 2
    stop_loss_pct: float = 0.015    # 1.5%
    take_profit_pct: float = 0.045  # 4.5% (3:1 RR)
    trailing_stop_pct: float = 0.01  # 1%
    discord_webhook_url: str = ""
    margin_max_ratio: float = 0.50
    margin_min_ratio: float = 0.20
    margin_decay_rate: float = 0.0006
    ml_threshold: float = 0.55

    def __post_init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.symbol = os.getenv("SYMBOL", "XRPUSDT")
        self.leverage = int(os.getenv("LEVERAGE", "10"))
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.margin_max_ratio = float(os.getenv("MARGIN_MAX_RATIO", "0.50"))
        self.margin_min_ratio = float(os.getenv("MARGIN_MIN_RATIO", "0.20"))
        self.margin_decay_rate = float(os.getenv("MARGIN_DECAY_RATE", "0.0006"))
        self.ml_threshold = float(os.getenv("ML_THRESHOLD", "0.55"))
        self.max_same_direction = int(os.getenv("MAX_SAME_DIRECTION", "2"))

        # symbols: SYMBOLS 환경변수 우선, 없으면 SYMBOL에서 변환
        symbols_env = os.getenv("SYMBOLS", "")
        if symbols_env:
            self.symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]
        else:
            self.symbols = [self.symbol]

        # correlation_symbols
        corr_env = os.getenv("CORRELATION_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.correlation_symbols = [s.strip() for s in corr_env.split(",") if s.strip()]
