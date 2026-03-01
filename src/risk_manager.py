from loguru import logger
from src.config import Config


class RiskManager:
    def __init__(self, config: Config, max_daily_loss_pct: float = 0.05):
        self.config = config
        self.max_daily_loss_pct = max_daily_loss_pct  # 일일 최대 손실 5%
        self.daily_pnl: float = 0.0
        self.initial_balance: float = 0.0
        self.open_positions: list = []

    def is_trading_allowed(self) -> bool:
        """일일 최대 손실 초과 시 거래 중단"""
        if self.initial_balance <= 0:
            return True
        loss_pct = abs(self.daily_pnl) / self.initial_balance
        if self.daily_pnl < 0 and loss_pct >= self.max_daily_loss_pct:
            logger.warning(
                f"일일 손실 한도 초과: {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}"
            )
            return False
        return True

    def can_open_new_position(self) -> bool:
        """최대 동시 포지션 수 체크"""
        return len(self.open_positions) < self.config.max_positions

    def record_pnl(self, pnl: float):
        self.daily_pnl += pnl
        logger.info(f"오늘 누적 PnL: {self.daily_pnl:.4f} USDT")

    def reset_daily(self):
        """매일 자정 초기화"""
        self.daily_pnl = 0.0
        logger.info("일일 PnL 초기화")
