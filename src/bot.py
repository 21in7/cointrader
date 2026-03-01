import asyncio
import os
from loguru import logger
from src.config import Config
from src.exchange import BinanceFuturesClient
from src.indicators import Indicators
from src.data_stream import KlineStream
from src.database import TradeRepository
from src.risk_manager import RiskManager


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceFuturesClient(config)
        self.db = TradeRepository(
            token=config.notion_token,
            database_id=config.notion_database_id,
        )
        self.risk = RiskManager(config)
        self.current_trade_id: str | None = None
        self.stream = KlineStream(
            symbol=config.symbol,
            interval="1m",
            on_candle=self._on_candle_closed,
        )

    def _on_candle_closed(self, candle: dict):
        df = self.stream.get_dataframe()
        if df is not None:
            asyncio.create_task(self.process_candle(df))

    async def process_candle(self, df):
        if not self.risk.is_trading_allowed():
            logger.warning("리스크 한도 초과 - 거래 중단")
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        signal = ind.get_signal(df_with_indicators)
        logger.info(f"신호: {signal}")

        position = await self.exchange.get_position()

        if position is None and signal != "HOLD":
            if not self.risk.can_open_new_position():
                logger.info("최대 포지션 수 도달")
                return
            await self._open_position(signal, df_with_indicators)

        elif position is not None:
            pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
            if (pos_side == "LONG" and signal == "SHORT") or \
               (pos_side == "SHORT" and signal == "LONG"):
                await self._close_position(position)

    async def _open_position(self, signal: str, df):
        balance = await self.exchange.get_balance()
        price = df["close"].iloc[-1]
        quantity = self.exchange.calculate_quantity(
            balance=balance, price=price, leverage=self.config.leverage
        )
        stop_loss, take_profit = Indicators(df).get_atr_stop(df, signal, price)

        side = "BUY" if signal == "LONG" else "SELL"
        await self.exchange.set_leverage(self.config.leverage)
        await self.exchange.place_order(side=side, quantity=quantity)

        last_row = df.iloc[-1]
        signal_snapshot = {
            "rsi":       float(last_row.get("rsi", 0)),
            "macd_hist": float(last_row.get("macd_hist", 0)),
            "atr":       float(last_row.get("atr", 0)),
        }
        trade = self.db.save_trade(
            symbol=self.config.symbol,
            side=signal,
            entry_price=price,
            quantity=quantity,
            leverage=self.config.leverage,
            signal_data=signal_snapshot,
        )
        self.current_trade_id = trade["id"]
        logger.success(
            f"{signal} 진입: 가격={price}, 수량={quantity}, "
            f"SL={stop_loss:.4f}, TP={take_profit:.4f}"
        )

        sl_side = "SELL" if signal == "LONG" else "BUY"
        await self.exchange.place_order(
            side=sl_side,
            quantity=quantity,
            order_type="STOP_MARKET",
            stop_price=round(stop_loss, 4),
            reduce_only=True,
        )
        await self.exchange.place_order(
            side=sl_side,
            quantity=quantity,
            order_type="TAKE_PROFIT_MARKET",
            stop_price=round(take_profit, 4),
            reduce_only=True,
        )

    async def _close_position(self, position: dict):
        amt = abs(float(position["positionAmt"]))
        side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
        await self.exchange.cancel_all_orders()
        await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)

        entry = float(position["entryPrice"])
        mark  = float(position["markPrice"])
        pnl   = (mark - entry) * amt if side == "SELL" else (entry - mark) * amt

        if self.current_trade_id:
            self.db.close_trade(self.current_trade_id, exit_price=mark, pnl=pnl)
        self.risk.record_pnl(pnl)
        self.current_trade_id = None
        logger.success(f"포지션 청산: PnL={pnl:.4f} USDT")

    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )
