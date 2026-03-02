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
from src.user_data_stream import UserDataStream


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = BinanceFuturesClient(config)
        self.notifier = DiscordNotifier(config.discord_webhook_url)
        self.risk = RiskManager(config)
        self.ml_filter = MLFilter()
        self.current_trade_side: str | None = None  # "LONG" | "SHORT"
        self._entry_price: float | None = None
        self._entry_quantity: float | None = None
        self._is_reentering: bool = False  # _close_and_reenter 중 콜백 상태 초기화 방지
        self._prev_oi: float | None = None  # OI 변화율 계산용 이전 값
        self.stream = MultiSymbolStream(
            symbols=[config.symbol, "BTCUSDT", "ETHUSDT"],
            interval="15m",
            on_candle=self._on_candle_closed,
        )

    async def _on_candle_closed(self, candle: dict):
        xrp_df = self.stream.get_dataframe(self.config.symbol)
        btc_df = self.stream.get_dataframe("BTCUSDT")
        eth_df = self.stream.get_dataframe("ETHUSDT")
        if xrp_df is not None:
            await self.process_candle(xrp_df, btc_df=btc_df, eth_df=eth_df)

    async def _recover_position(self) -> None:
        """재시작 시 바이낸스에서 현재 포지션을 조회하여 상태 복구."""
        position = await self.exchange.get_position()
        if position is not None:
            amt = float(position["positionAmt"])
            self.current_trade_side = "LONG" if amt > 0 else "SHORT"
            self._entry_price = float(position["entryPrice"])
            self._entry_quantity = abs(amt)
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

    async def _fetch_market_microstructure(self) -> tuple[float, float]:
        """OI 변화율과 펀딩비를 실시간으로 조회한다. 실패 시 0.0으로 폴백."""
        oi_val, fr_val = await asyncio.gather(
            self.exchange.get_open_interest(),
            self.exchange.get_funding_rate(),
            return_exceptions=True,
        )
        # None(API 실패) 또는 Exception이면 _calc_oi_change를 호출하지 않고 0.0 반환
        if isinstance(oi_val, (int, float)) and oi_val > 0:
            oi_change = self._calc_oi_change(float(oi_val))
        else:
            oi_change = 0.0
        fr_float = float(fr_val) if isinstance(fr_val, (int, float)) else 0.0
        logger.debug(f"OI={oi_val}, OI변화율={oi_change:.6f}, 펀딩비={fr_float:.6f}")
        return oi_change, fr_float

    def _calc_oi_change(self, current_oi: float) -> float:
        """이전 OI 대비 변화율을 계산한다. 첫 캔들은 0.0 반환."""
        if self._prev_oi is None or self._prev_oi == 0.0:
            self._prev_oi = current_oi
            return 0.0
        change = (current_oi - self._prev_oi) / self._prev_oi
        self._prev_oi = current_oi
        return change

    async def process_candle(self, df, btc_df=None, eth_df=None):
        self.ml_filter.check_and_reload()

        # 캔들 마감 시 OI/펀딩비 실시간 조회 (실패해도 0으로 폴백)
        oi_change, funding_rate = await self._fetch_market_microstructure()

        if not self.risk.is_trading_allowed():
            logger.warning("리스크 한도 초과 - 거래 중단")
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        raw_signal = ind.get_signal(df_with_indicators)

        current_price = df_with_indicators["close"].iloc[-1]
        logger.info(f"신호: {raw_signal} | 현재가: {current_price:.4f} USDT")

        position = await self.exchange.get_position()

        if position is None and raw_signal != "HOLD":
            self.current_trade_side = None
            if not self.risk.can_open_new_position():
                logger.info("최대 포지션 수 도달")
                return
            signal = raw_signal
            features = build_features(
                df_with_indicators, signal,
                btc_df=btc_df, eth_df=eth_df,
                oi_change=oi_change, funding_rate=funding_rate,
            )
            if self.ml_filter.is_model_loaded():
                if not self.ml_filter.should_enter(features):
                    logger.info(f"ML 필터 차단: {signal} 신호 무시")
                    return
            await self._open_position(signal, df_with_indicators)

        elif position is not None:
            pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
            if (pos_side == "LONG" and raw_signal == "SHORT") or \
               (pos_side == "SHORT" and raw_signal == "LONG"):
                await self._close_and_reenter(
                    position, raw_signal, df_with_indicators,
                    btc_df=btc_df, eth_df=eth_df,
                    oi_change=oi_change, funding_rate=funding_rate,
                )

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
        self._entry_price = price
        self._entry_quantity = quantity
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

    def _calc_estimated_pnl(self, exit_price: float) -> float:
        """진입가·수량 기반 예상 PnL 계산 (수수료 미반영)."""
        if self._entry_price is None or self._entry_quantity is None or self.current_trade_side is None:
            return 0.0
        if self.current_trade_side == "LONG":
            return (exit_price - self._entry_price) * self._entry_quantity
        return (self._entry_price - exit_price) * self._entry_quantity

    async def _on_position_closed(
        self,
        net_pnl: float,
        close_reason: str,
        exit_price: float,
    ) -> None:
        """User Data Stream에서 청산 감지 시 호출되는 콜백."""
        estimated_pnl = self._calc_estimated_pnl(exit_price)
        diff = net_pnl - estimated_pnl

        self.risk.record_pnl(net_pnl)

        self.notifier.notify_close(
            symbol=self.config.symbol,
            side=self.current_trade_side or "UNKNOWN",
            close_reason=close_reason,
            exit_price=exit_price,
            estimated_pnl=estimated_pnl,
            net_pnl=net_pnl,
            diff=diff,
        )

        logger.success(
            f"포지션 청산({close_reason}): 예상={estimated_pnl:+.4f}, "
            f"순수익={net_pnl:+.4f}, 차이={diff:+.4f} USDT"
        )

        # _close_and_reenter 중이면 신규 포지션 상태를 덮어쓰지 않는다
        if self._is_reentering:
            return

        # Flat 상태로 초기화
        self.current_trade_side = None
        self._entry_price = None
        self._entry_quantity = None

    async def _close_position(self, position: dict):
        """포지션 청산 주문만 실행한다. PnL 기록/알림은 _on_position_closed 콜백이 담당."""
        amt = abs(float(position["positionAmt"]))
        side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
        await self.exchange.cancel_all_orders()
        await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)
        logger.info(f"청산 주문 전송 완료 (side={side}, qty={amt})")

    async def _close_and_reenter(
        self,
        position: dict,
        signal: str,
        df,
        btc_df=None,
        eth_df=None,
        oi_change: float = 0.0,
        funding_rate: float = 0.0,
    ) -> None:
        """기존 포지션을 청산하고, ML 필터 통과 시 반대 방향으로 즉시 재진입한다."""
        # 재진입 플래그: User Data Stream 콜백이 신규 포지션 상태를 초기화하지 않도록 보호
        self._is_reentering = True
        try:
            await self._close_position(position)

            if not self.risk.can_open_new_position():
                logger.info("최대 포지션 수 도달 — 재진입 건너뜀")
                return

            if self.ml_filter.is_model_loaded():
                features = build_features(
                    df, signal,
                    btc_df=btc_df, eth_df=eth_df,
                    oi_change=oi_change, funding_rate=funding_rate,
                )
                if not self.ml_filter.should_enter(features):
                    logger.info(f"ML 필터 차단: {signal} 재진입 무시")
                    return

            await self._open_position(signal, df)
        finally:
            self._is_reentering = False

    async def run(self):
        logger.info(f"봇 시작: {self.config.symbol}, 레버리지 {self.config.leverage}x")
        await self._recover_position()
        balance = await self.exchange.get_balance()
        self.risk.set_base_balance(balance)
        logger.info(f"기준 잔고 설정: {balance:.2f} USDT (동적 증거금 비율 기준점)")

        user_stream = UserDataStream(
            symbol=self.config.symbol,
            on_order_filled=self._on_position_closed,
        )

        await asyncio.gather(
            self.stream.start(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
            ),
            user_stream.start(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
            ),
        )
