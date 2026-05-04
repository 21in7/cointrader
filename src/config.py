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
    kline_interval: str = "15m"
    testnet: bool = False

    def __post_init__(self):
        self.testnet = os.getenv("BINANCE_TESTNET", "").lower() in ("true", "1", "yes")

        if self.testnet:
            self.api_key = os.getenv("BINANCE_DEMO_API_KEY", "")
            self.api_secret = os.getenv("BINANCE_DEMO_API_SECRET", "")
        else:
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
        self.kline_interval = os.getenv("KLINE_INTERVAL", "15m")

        # symbols: SYMBOLS 환경변수 우선, 없으면 SYMBOL에서 변환
        symbols_env = os.getenv("SYMBOLS", "")
        if symbols_env:
            self.symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]
        else:
            self.symbols = [self.symbol]

        # correlation_symbols
        corr_env = os.getenv("CORRELATION_SYMBOLS", "BTCUSDT,ETHUSDT")
        self.correlation_symbols = [s.strip() for s in corr_env.split(",") if s.strip()]

        # 입력 검증
        if self.leverage < 1:
            raise ValueError(f"LEVERAGE는 1 이상이어야 합니다: {self.leverage}")
        if not (0.0 < self.margin_max_ratio <= 1.0):
            raise ValueError(f"MARGIN_MAX_RATIO는 (0, 1] 범위여야 합니다: {self.margin_max_ratio}")
        if not (0.0 < self.margin_min_ratio <= 1.0):
            raise ValueError(f"MARGIN_MIN_RATIO는 (0, 1] 범위여야 합니다: {self.margin_min_ratio}")
        if self.margin_min_ratio > self.margin_max_ratio:
            raise ValueError(f"MARGIN_MIN_RATIO({self.margin_min_ratio}) > MARGIN_MAX_RATIO({self.margin_max_ratio})")
        if not (0.0 < self.ml_threshold <= 1.0):
            raise ValueError(f"ML_THRESHOLD는 (0, 1] 범위여야 합니다: {self.ml_threshold}")

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


# ── OOS 사후보정용 비용 모델 ──────────────────────────────────────
COST_MODEL = {
    "taker_fee_bps": 4.0,       # Binance USDⓈ-M Futures VIP 0
    "maker_fee_bps": 2.0,       # 향후 limit TP 도입 대비
    # MTF bot 주문 타입 (현재 전부 MARKET = taker)
    "entry_order_type": "taker",
    "sl_order_type": "taker",
    "tp_order_type": "taker",
}

# 3개 프리셋 시나리오 (확장 금지, 이 셋으로 고정)
COST_SCENARIOS = {
    "fees_only":   {"slippage_bps_per_side": 0.0, "funding_bps_per_8h": 0.0},
    "realistic":   {"slippage_bps_per_side": 1.0, "funding_bps_per_8h": 1.0},
    "pessimistic": {"slippage_bps_per_side": 3.0, "funding_bps_per_8h": 2.0},
}

