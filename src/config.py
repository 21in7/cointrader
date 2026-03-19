import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SymbolStrategyParams:
    """Per-symbol strategy parameters (from sweep optimization)."""
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 2.0
    signal_threshold: int = 3
    adx_threshold: float = 25.0
    volume_multiplier: float = 2.5


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
    discord_webhook_url: str = ""
    margin_max_ratio: float = 0.50
    margin_min_ratio: float = 0.20
    margin_decay_rate: float = 0.0006
    ml_threshold: float = 0.55
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 2.0
    signal_threshold: int = 3
    adx_threshold: float = 25.0
    volume_multiplier: float = 2.5

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
        self.atr_sl_mult = float(os.getenv("ATR_SL_MULT", "2.0"))
        self.atr_tp_mult = float(os.getenv("ATR_TP_MULT", "2.0"))
        self.signal_threshold = int(os.getenv("SIGNAL_THRESHOLD", "3"))
        self.adx_threshold = float(os.getenv("ADX_THRESHOLD", "25"))
        self.volume_multiplier = float(os.getenv("VOL_MULTIPLIER", "2.5"))

        # symbols: SYMBOLS 환경변수 우선, 없으면 SYMBOL에서 변환
        symbols_env = os.getenv("SYMBOLS", "")
        if symbols_env:
            self.symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]
        else:
            self.symbols = [self.symbol]

        # correlation_symbols
        corr_env = os.getenv("CORRELATION_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.correlation_symbols = [s.strip() for s in corr_env.split(",") if s.strip()]

        # Per-symbol strategy params: {symbol: SymbolStrategyParams}
        self._symbol_params: dict[str, SymbolStrategyParams] = {}
        for sym in self.symbols:
            self._symbol_params[sym] = SymbolStrategyParams(
                atr_sl_mult=float(os.getenv(f"ATR_SL_MULT_{sym}", str(self.atr_sl_mult))),
                atr_tp_mult=float(os.getenv(f"ATR_TP_MULT_{sym}", str(self.atr_tp_mult))),
                signal_threshold=int(os.getenv(f"SIGNAL_THRESHOLD_{sym}", str(self.signal_threshold))),
                adx_threshold=float(os.getenv(f"ADX_THRESHOLD_{sym}", str(self.adx_threshold))),
                volume_multiplier=float(os.getenv(f"VOL_MULTIPLIER_{sym}", str(self.volume_multiplier))),
            )

    def get_symbol_params(self, symbol: str) -> SymbolStrategyParams:
        """Get strategy params for a symbol. Falls back to global defaults."""
        return self._symbol_params.get(symbol, SymbolStrategyParams(
            atr_sl_mult=self.atr_sl_mult,
            atr_tp_mult=self.atr_tp_mult,
            signal_threshold=self.signal_threshold,
            adx_threshold=self.adx_threshold,
            volume_multiplier=self.volume_multiplier,
        ))

