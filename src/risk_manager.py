import asyncio
from loguru import logger
from src.config import Config


class RiskManager:
    def __init__(self, config: Config, max_daily_loss_pct: float = 0.05):
        self.config = config
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_pnl: float = 0.0
        self.initial_balance: float = 0.0
        self.open_positions: dict[str, str] = {}  # {symbol: side}
        self._lock = asyncio.Lock()
        self._entry_lock = asyncio.Lock()  # 동시 진입 시 잔고 레이스 방지

    async def is_trading_allowed(self) -> bool:
        """일일 최대 손실 초과 시 거래 중단"""
        async with self._lock:
            if self.initial_balance <= 0:
                return True
            loss_pct = abs(self.daily_pnl) / self.initial_balance
            if self.daily_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
                logger.warning(
                    f"일일 손실 한도 초과: {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}"
                )
                return False
            return True

    async def can_open_new_position(self, symbol: str, side: str) -> bool:
        """포지션 오픈 가능 여부 (전체 한도 + 중복 진입 + 동일 방향 제한)"""
        async with self._lock:
            if len(self.open_positions) >= self.config.max_positions:
                logger.info(f"최대 포지션 수 도달: {len(self.open_positions)}/{self.config.max_positions}")
                return False
            if symbol in self.open_positions:
                logger.info(f"{symbol} 이미 포지션 보유 중")
                return False
            same_dir = sum(1 for s in self.open_positions.values() if s == side)
            if same_dir >= self.config.max_same_direction:
                logger.info(f"동일 방향({side}) 한도 도달: {same_dir}/{self.config.max_same_direction}")
                return False
            return True

    async def register_position(self, symbol: str, side: str):
        """포지션 등록"""
        async with self._lock:
            self.open_positions[symbol] = side
            logger.info(f"포지션 등록: {symbol} {side} (현재 {len(self.open_positions)}개)")

    async def close_position(self, symbol: str, pnl: float):
        """포지션 닫기 + PnL 기록"""
        async with self._lock:
            self.open_positions.pop(symbol, None)
            self.daily_pnl += pnl
            logger.info(f"포지션 종료: {symbol}, PnL={pnl:+.4f}, 누적={self.daily_pnl:+.4f}")

    async def record_pnl(self, pnl: float):
        async with self._lock:
            self.daily_pnl += pnl
            logger.info(f"오늘 누적 PnL: {self.daily_pnl:.4f} USDT")

    async def reset_daily(self):
        """매일 자정 초기화"""
        async with self._lock:
            self.daily_pnl = 0.0
            logger.info("일일 PnL 초기화")

    def set_base_balance(self, balance: float) -> None:
        """봇 시작 시 기준 잔고 설정"""
        self.initial_balance = balance

    def get_dynamic_margin_ratio(self, balance: float) -> float:
        """잔고에 따라 선형 감소하는 증거금 비율 반환"""
        ratio = self.config.margin_max_ratio - (
            (balance - self.initial_balance) * self.config.margin_decay_rate
        )
        return max(self.config.margin_min_ratio, min(self.config.margin_max_ratio, ratio))
