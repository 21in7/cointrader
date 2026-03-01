import asyncio
import pandas as pd
from loguru import logger
from src.config import Config
from src.exchange import BinanceFuturesClient
from src.indicators import Indicators
from src.data_stream import MultiSymbolStream
from src.notifier import DiscordNotifier
from src.risk_manager import RiskManager
from src.ml_filter import MLFilter
from src.ml_features import build_features


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceFuturesClient(config)
        self.notifier = DiscordNotifier(config.discord_webhook_url)
        self.risk = RiskManager(config)
        self.ml_filter = MLFilter()
        self.current_trade_side: str | None = None  # "LONG" | "SHORT"
        self.stream = MultiSymbolStream(
            symbols=[config.symbol, "BTCUSDT", "ETHUSDT"],
            interval="15m",
            on_candle=self._on_candle_closed,
        )

    def _on_candle_closed(self, candle: dict):
        xrp_df = self.stream.get_dataframe(self.config.symbol)
        btc_df = self.stream.get_dataframe("BTCUSDT")
        eth_df = self.stream.get_dataframe("ETHUSDT")
        if xrp_df is not None:
            asyncio.create_task(self.process_candle(xrp_df, btc_df=btc_df, eth_df=eth_df))

    async def _recover_position(self) -> None:
        """재시작 시 바이낸스에서 현재 포지션을 조회하여 상태 복구."""
        position = await self.exchange.get_position()
        if position is not None:
            amt = float(position["positionAmt"])
            self.current_trade_side = "LONG" if amt > 0 else "SHORT"
            entry = float(position["entryPrice"])
            logger.info(
                f"기존 포지션 복구: {self.current_trade_side} | "
                f"진입가={entry:.4f} | 수량={abs(amt)}"
            )
            self.notifier.notify_info(
                f"봇 재시작 - 기존 포지션 감지: {self.current_trade_side} "
                f"진입가={entry:.4f} 수량={abs(amt)}"
            )
        else:
            logger.info("기존 포지션 없음 - 신규 진입 대기")

    async def process_candle(self, df, btc_df=None, eth_df=None):
        self.ml_filter.check_and_reload()

        if not self.risk.is_trading_allowed():
            logger.warning("리스크 한도 초과 - 거래 중단")
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        signal = ind.get_signal(df_with_indicators)

        if signal != "HOLD" and self.ml_filter.is_model_loaded():
            features = build_features(df_with_indicators, signal, btc_df=btc_df, eth_df=eth_df)
            if not self.ml_filter.should_enter(features):
                logger.info(f"ML 필터 차단: {signal} 신호 무시")
                signal = "HOLD"

        current_price = df_with_indicators["close"].iloc[-1]
        logger.info(f"신호: {signal} | 현재가: {current_price:.4f} USDT")

        position = await self.exchange.get_position()

        if position is None and signal != "HOLD":
            self.current_trade_side = None
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
        margin_ratio = self.risk.get_dynamic_margin_ratio(balance)
        quantity = self.exchange.calculate_quantity(
            balance=balance, price=price, leverage=self.config.leverage, margin_ratio=margin_ratio
        )
        logger.info(f"포지션 크기: 잔고={balance:.2f} USDT, 증거금비율={margin_ratio:.1%}, 수량={quantity}")
        stop_loss, take_profit = Indicators(df).get_atr_stop(df, signal, price)

        notional = quantity * price
        if quantity <= 0 or notional < self.exchange.MIN_NOTIONAL:
            logger.warning(
                f"주문 건너뜀: 명목금액 {notional:.2f} USDT < 최소 {self.exchange.MIN_NOTIONAL} USDT "
                f"(잔고={balance:.2f}, 수량={quantity})"
            )
            return

        side = "BUY" if signal == "LONG" else "SELL"
        await self.exchange.set_leverage(self.config.leverage)
        await self.exchange.place_order(side=side, quantity=quantity)

        last_row = df.iloc[-1]
        signal_snapshot = {
            "rsi":       float(last_row["rsi"])       if "rsi"       in last_row.index and pd.notna(last_row["rsi"])       else 0.0,
            "macd_hist": float(last_row["macd_hist"]) if "macd_hist" in last_row.index and pd.notna(last_row["macd_hist"]) else 0.0,
            "atr":       float(last_row["atr"])       if "atr"       in last_row.index and pd.notna(last_row["atr"])       else 0.0,
        }

        self.current_trade_side = signal
        self.notifier.notify_open(
            symbol=self.config.symbol,
            side=signal,
            entry_price=price,
            quantity=quantity,
            leverage=self.config.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_data=signal_snapshot,
        )
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
        pos_side = "LONG" if side == "SELL" else "SHORT"
        await self.exchange.cancel_all_orders()
        await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)

        entry = float(position["entryPrice"])
        mark  = float(position["markPrice"])
        pnl   = (mark - entry) * amt if side == "SELL" else (entry - mark) * amt

        self.notifier.notify_close(
            symbol=self.config.symbol,
            side=pos_side,
            exit_price=mark,
            pnl=pnl,
        )
        self.risk.record_pnl(pnl)
        self.current_trade_side = None
        logger.success(f"포지션 청산: PnL={pnl:.4f} USDT")

    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self._recover_position()
        balance = await self.exchange.get_balance()
        self.risk.set_base_balance(balance)
        logger.info(f"기준 잔고 설정: {balance:.2f} USDT (동적 증거금 비율 기준점)")
        await self.stream.start(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )
