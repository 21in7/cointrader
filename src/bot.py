import asyncio
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from loguru import logger
from src.config import Config
from src.exchange import BinanceFuturesClient
from src.indicators import Indicators
from src.data_stream import MultiSymbolStream
from src.notifier import DiscordNotifier
from src.risk_manager import RiskManager
from src.ml_filter import MLFilter
from src.ml_features import build_features_aligned
from src.user_data_stream import UserDataStream

# ── 킬스위치 상수 ──────────────────────────────────────────────────
_FAST_KILL_STREAK = 8       # 연속 손실 N회 → 즉시 중단
_SLOW_KILL_WINDOW = 15      # 최근 N거래 PF 산출
_SLOW_KILL_PF_THRESHOLD = 0.75  # PF < 이 값이면 중단
_TRADE_HISTORY_DIR = Path("data/trade_history")


def _tail_lines(path: Path, n: int) -> list[str]:
    """파일 끝에서 최대 n줄을 효율적으로 읽는다 (전체 파일 로드 없이)."""
    with open(path, "rb") as f:
        f.seek(0, 2)  # EOF
        fsize = f.tell()
        if fsize == 0:
            return []
        # 뒤에서부터 청크 단위로 읽기
        chunk_size = min(4096, fsize)
        lines: list[str] = []
        pos = fsize
        remaining = b""
        while pos > 0 and len(lines) < n + 1:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size) + remaining
            remaining = b""
            split = chunk.split(b"\n")
            # 첫 조각은 이전 청크와 이어질 수 있으므로 따로 보관
            remaining = split[0]
            lines = [s.decode() for s in split[1:] if s.strip()] + lines
        # 남은 조각 처리
        if remaining.strip():
            lines = [remaining.decode()] + lines
        return lines[-n:]


class TradingBot:
    def __init__(self, config: Config, symbol: str = None, risk: RiskManager = None):
        self.config = config
        self.symbol = symbol or config.symbol
        self.strategy = config.get_symbol_params(self.symbol)
        self.exchange = BinanceFuturesClient(config, symbol=self.symbol)
        self.notifier = DiscordNotifier(config.discord_webhook_url)
        self.risk = risk or RiskManager(config)
        # 심볼별 모델 디렉토리. 없으면 기존 models/ 루트로 폴백
        symbol_model_dir = Path(f"models/{self.symbol.lower()}")
        if symbol_model_dir.exists():
            onnx_path = str(symbol_model_dir / "mlx_filter.weights.onnx")
            lgbm_path = str(symbol_model_dir / "lgbm_filter.pkl")
        else:
            onnx_path = "models/mlx_filter.weights.onnx"
            lgbm_path = "models/lgbm_filter.pkl"
        self.ml_filter = MLFilter(
            onnx_path=onnx_path,
            lgbm_path=lgbm_path,
            threshold=config.ml_threshold,
        )
        self.current_trade_side: str | None = None  # "LONG" | "SHORT"
        self._entry_price: float | None = None
        self._entry_quantity: float | None = None
        self._is_reentering: bool = False  # _close_and_reenter 중 콜백 상태 초기화 방지
        self._entry_time_ms: int | None = None  # 포지션 진입 시각 (ms, SYNC PnL 범위 제한용)
        self._close_event = asyncio.Event()  # 콜백 청산 완료 대기용
        self._close_lock = asyncio.Lock()  # 청산 처리 원자성 보장 (C3 fix)
        self._prev_oi: float | None = None  # OI 변화율 계산용 이전 값
        self._oi_history: deque = deque(maxlen=96)  # z-score 윈도우(96=1일분 15분봉)
        self._funding_history: deque = deque(maxlen=96)
        self._latest_ret_1: float = 0.0
        self._killed: bool = False  # 킬스위치 발동 상태
        self._trade_history: list[dict] = []  # 최근 거래 이력 (net_pnl 기록)
        self.stream = MultiSymbolStream(
            symbols=[self.symbol] + config.correlation_symbols,
            interval="15m",
            on_candle=self._on_candle_closed,
        )
        # 부팅 시 거래 이력 복원 및 킬스위치 소급 검증
        self._restore_trade_history()
        self._restore_kill_switch()

    # ── 킬스위치 ──────────────────────────────────────────────────────

    def _trade_history_path(self) -> Path:
        return _TRADE_HISTORY_DIR / f"{self.symbol.lower()}.jsonl"

    def _restore_trade_history(self) -> None:
        """부팅 시 파일 마지막 N줄만 읽어 거래 이력을 복원한다.
        킬스위치 판단에 필요한 최대 윈도우(_SLOW_KILL_WINDOW)만큼만 유지."""
        path = self._trade_history_path()
        if not path.exists():
            return
        try:
            tail_n = max(_FAST_KILL_STREAK, _SLOW_KILL_WINDOW)
            lines = _tail_lines(path, tail_n)
            for line in lines:
                line = line.strip()
                if line:
                    self._trade_history.append(json.loads(line))
            logger.info(f"[{self.symbol}] 거래 이력 복원: {len(self._trade_history)}건 (최근 {tail_n}건)")
        except Exception as e:
            logger.warning(f"[{self.symbol}] 거래 이력 복원 실패: {e}")

    def _restore_kill_switch(self) -> None:
        """부팅 시 .env 리셋 플래그 확인 후, 이력 기반으로 킬스위치 소급 검증."""
        reset_key = f"RESET_KILL_SWITCH_{self.symbol}"
        if os.environ.get(reset_key, "").lower() == "true":
            logger.info(f"[{self.symbol}] 킬스위치 수동 해제 감지 ({reset_key}=True)")
            self._killed = False
            return
        # 소급 검증
        if self._check_kill_switch(silent=True):
            logger.warning(f"[{self.symbol}] 부팅 시 킬스위치 조건 충족 — 신규 진입 차단")

    def _append_trade(self, net_pnl: float, close_reason: str) -> None:
        """거래 기록을 메모리 + 파일에 추가한다."""
        record = {
            "net_pnl": round(net_pnl, 4),
            "reason": close_reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(record)
        # 메모리에는 킬스위치 윈도우만큼만 유지
        max_window = max(_FAST_KILL_STREAK, _SLOW_KILL_WINDOW)
        if len(self._trade_history) > max_window * 2:
            self._trade_history = self._trade_history[-max_window:]
        # 파일에 append (JSONL)
        try:
            _TRADE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._trade_history_path(), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"[{self.symbol}] 거래 기록 저장 실패: {e}")

    def _check_kill_switch(self, silent: bool = False) -> bool:
        """킬스위치 조건을 검사하고, 발동 시 True를 반환한다.

        Fast Kill: 최근 8연속 순손실
        Slow Kill: 최근 15거래 PF < 0.75
        """
        trades = self._trade_history
        if not trades:
            return False

        # Fast Kill: 8연속 순손실
        if len(trades) >= _FAST_KILL_STREAK:
            recent = trades[-_FAST_KILL_STREAK:]
            if all(t["net_pnl"] < 0 for t in recent):
                reason = f"Fast Kill ({_FAST_KILL_STREAK}연속 순손실)"
                self._trigger_kill_switch(reason, silent)
                return True

        # Slow Kill: 최근 15거래 PF < 0.75
        if len(trades) >= _SLOW_KILL_WINDOW:
            recent = trades[-_SLOW_KILL_WINDOW:]
            gross_profit = sum(t["net_pnl"] for t in recent if t["net_pnl"] > 0)
            gross_loss = abs(sum(t["net_pnl"] for t in recent if t["net_pnl"] < 0))
            if gross_loss > 0:
                pf = gross_profit / gross_loss
                if pf < _SLOW_KILL_PF_THRESHOLD:
                    reason = f"Slow Kill (최근 {_SLOW_KILL_WINDOW}거래 PF={pf:.2f})"
                    self._trigger_kill_switch(reason, silent)
                    return True

        return False

    def _trigger_kill_switch(self, reason: str, silent: bool = False) -> None:
        """킬스위치 발동: 상태 변경 + 알림."""
        self._killed = True
        msg = (
            f"🚨 [KILL SWITCH] {self.symbol} 신규 진입 중단\n"
            f"사유: {reason}\n"
            f"기존 포지션 SL/TP는 정상 작동합니다.\n"
            f"해제: RESET_KILL_SWITCH_{self.symbol}=True 후 봇 재시작"
        )
        logger.error(msg)
        if not silent:
            self.notifier.notify_info(msg)

    async def _on_candle_closed(self, candle: dict):
        primary_df = self.stream.get_dataframe(self.symbol)
        corr = self.config.correlation_symbols
        corr_dfs = {s: self.stream.get_dataframe(s) for s in corr}
        btc_df = corr_dfs.get("BTCUSDT")
        eth_df = corr_dfs.get("ETHUSDT")
        if primary_df is not None:
            await self.process_candle(primary_df, btc_df=btc_df, eth_df=eth_df)

    async def _recover_position(self) -> None:
        """재시작 시 바이낸스에서 현재 포지션을 조회하여 상태 복구.
        SL/TP 주문이 누락된 경우 ATR 기반으로 재배치한다."""
        position = await self.exchange.get_position()
        if position is not None:
            amt = float(position["positionAmt"])
            self.current_trade_side = "LONG" if amt > 0 else "SHORT"
            self._entry_price = float(position["entryPrice"])
            self._entry_quantity = abs(amt)
            self._entry_time_ms = int(float(position.get("updateTime", time.time() * 1000)))
            entry = float(position["entryPrice"])
            logger.info(
                f"[{self.symbol}] 기존 포지션 복구: {self.current_trade_side} | "
                f"진입가={entry:.4f} | 수량={abs(amt)}"
            )
            # SL/TP 주문 존재 여부 확인 후 누락 시 재배치
            await self._ensure_sl_tp_orders(position)
            self.notifier.notify_info(
                f"봇 재시작 - 기존 포지션 감지: {self.current_trade_side} "
                f"진입가={entry:.4f} 수량={abs(amt)}"
            )
        else:
            logger.info(f"[{self.symbol}] 기존 포지션 없음 - 신규 진입 대기")

    async def _ensure_sl_tp_orders(self, position: dict) -> None:
        """포지션에 SL/TP 주문이 없으면 ATR 기반으로 재배치한다."""
        try:
            open_orders = await self.exchange.get_open_orders()
            has_sl = any(o.get("type") == "STOP_MARKET" for o in open_orders)
            has_tp = any(o.get("type") == "TAKE_PROFIT_MARKET" for o in open_orders)
            if has_sl and has_tp:
                return
            missing = []
            if not has_sl:
                missing.append("SL")
            if not has_tp:
                missing.append("TP")
            logger.warning(f"[{self.symbol}] {'/'.join(missing)} 주문 누락 감지 — 재배치")

            # 캔들 데이터로 ATR 기반 SL/TP 계산
            primary_df = self.stream.get_dataframe(self.symbol)
            if primary_df is None:
                logger.warning(f"[{self.symbol}] 캔들 데이터 부족 — SL/TP 재배치 건너뜀")
                return
            ind = Indicators(primary_df)
            df_ind = ind.calculate_all()
            entry = self._entry_price
            qty = self._entry_quantity
            sl, tp = ind.get_atr_stop(
                df_ind, self.current_trade_side, entry,
                atr_sl_mult=self.strategy.atr_sl_mult,
                atr_tp_mult=self.strategy.atr_tp_mult,
            )
            sl_side = "SELL" if self.current_trade_side == "LONG" else "BUY"
            if not has_sl:
                await self.exchange.place_order(
                    side=sl_side, quantity=qty,
                    order_type="STOP_MARKET",
                    stop_price=self.exchange._round_price(sl),
                    reduce_only=True,
                )
                logger.info(f"[{self.symbol}] SL 재배치: {sl:.4f}")
            if not has_tp:
                await self.exchange.place_order(
                    side=sl_side, quantity=qty,
                    order_type="TAKE_PROFIT_MARKET",
                    stop_price=self.exchange._round_price(tp),
                    reduce_only=True,
                )
                logger.info(f"[{self.symbol}] TP 재배치: {tp:.4f}")
        except Exception as e:
            logger.warning(f"[{self.symbol}] SL/TP 재배치 실패: {e}")

    async def _init_oi_history(self) -> None:
        """봇 시작 시 최근 OI 변화율 히스토리를 조회하여 deque를 채운다."""
        try:
            changes = await self.exchange.get_oi_history(limit=5)
            for c in changes:
                self._oi_history.append(c)
            if changes:
                self._prev_oi = None
            logger.info(f"[{self.symbol}] OI 히스토리 초기화: {len(self._oi_history)}개")
        except Exception as e:
            logger.warning(f"OI 히스토리 초기화 실패 (무시): {e}")

    async def _fetch_market_microstructure(self) -> tuple[float, float, float, float]:
        """OI 변화율, 펀딩비, OI MA5, OI-가격 스프레드를 실시간으로 조회한다."""
        oi_val, fr_val = await asyncio.gather(
            self.exchange.get_open_interest(),
            self.exchange.get_funding_rate(),
            return_exceptions=True,
        )
        if isinstance(oi_val, (int, float)) and oi_val > 0:
            oi_change = self._calc_oi_change(float(oi_val))
        else:
            oi_change = 0.0
        fr_float = float(fr_val) if isinstance(fr_val, (int, float)) else 0.0

        # 히스토리 업데이트 (z-score 계산용)
        self._oi_history.append(oi_change)
        self._funding_history.append(fr_float)

        # OI MA5 계산
        recent_5 = list(self._oi_history)[-5:]
        oi_ma5 = sum(recent_5) / len(recent_5) if recent_5 else 0.0

        # OI-가격 스프레드
        oi_price_spread = oi_change - self._latest_ret_1

        logger.debug(
            f"[{self.symbol}] OI={oi_val}, OI변화율={oi_change:.6f}, 펀딩비={fr_float:.6f}, "
            f"OI_MA5={oi_ma5:.6f}, OI_Price_Spread={oi_price_spread:.6f}"
        )
        return oi_change, fr_float, oi_ma5, oi_price_spread

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

        # 가격 수익률 계산 (oi_price_spread용)
        if len(df) >= 2:
            prev_close = df["close"].iloc[-2]
            curr_close = df["close"].iloc[-1]
            self._latest_ret_1 = (curr_close - prev_close) / prev_close if prev_close != 0 else 0.0

        # 캔들 마감 시 OI/펀딩비 실시간 조회 (실패해도 0으로 폴백)
        oi_change, funding_rate, oi_ma5, oi_price_spread = await self._fetch_market_microstructure()

        if not await self.risk.is_trading_allowed():
            logger.warning(f"[{self.symbol}] 리스크 한도 초과 - 거래 중단")
            return

        # 킬스위치: 신규 진입만 차단, 기존 포지션 모니터링은 계속
        if self._killed:
            return

        ind = Indicators(df)
        df_with_indicators = ind.calculate_all()
        raw_signal, signal_detail = ind.get_signal(
            df_with_indicators,
            signal_threshold=self.strategy.signal_threshold,
            adx_threshold=self.strategy.adx_threshold,
            volume_multiplier=self.strategy.volume_multiplier,
        )

        current_price = df_with_indicators["close"].iloc[-1]
        adx_str = f"ADX={signal_detail['adx']:.1f}" if signal_detail['adx'] is not None else "ADX=N/A"
        vol_str = "Vol급증" if signal_detail['vol_surge'] else "Vol정상"
        score_str = f"L={signal_detail['long']} S={signal_detail['short']}"
        if raw_signal == "HOLD" and signal_detail['hold_reason']:
            logger.info(f"[{self.symbol}] 신호: HOLD | {score_str} | {adx_str} | {vol_str} | 사유: {signal_detail['hold_reason']} | 현재가: {current_price:.4f}")
        else:
            logger.info(f"[{self.symbol}] 신호: {raw_signal} | {score_str} | {adx_str} | {vol_str} | 현재가: {current_price:.4f}")

        position = await self.exchange.get_position()

        if position is None and raw_signal != "HOLD":
            # Binance에 포지션이 없는데 로컬에 남아있으면 risk manager 동기화
            if self.current_trade_side is not None:
                logger.warning(
                    f"[{self.symbol}] 포지션 불일치: 로컬={self.current_trade_side}, "
                    f"바이낸스=없음 — risk manager 동기화"
                )
                await self.risk.close_position(self.symbol, 0.0)
                self.current_trade_side = None
                self._entry_price = None
                self._entry_quantity = None
            if not await self.risk.can_open_new_position(self.symbol, raw_signal):
                logger.info(f"[{self.symbol}] 포지션 오픈 불가")
                return
            signal = raw_signal
            features = build_features_aligned(
                df_with_indicators, signal,
                btc_df=btc_df, eth_df=eth_df,
                oi_change=oi_change, funding_rate=funding_rate,
                oi_change_ma5=oi_ma5, oi_price_spread=oi_price_spread,
                oi_history=list(self._oi_history),
                funding_history=list(self._funding_history),
            )
            if self.ml_filter.is_model_loaded():
                if not self.ml_filter.should_enter(features):
                    logger.info(f"[{self.symbol}] ML 필터 차단: {signal} 신호 무시")
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
                    oi_change_ma5=oi_ma5, oi_price_spread=oi_price_spread,
                )

    async def _open_position(self, signal: str, df):
        # 동시 진입 시 잔고 레이스 방지: entry_lock으로 잔고 조회→주문→등록을 직렬화
        async with self.risk._entry_lock:
            balance = await self.exchange.get_balance()
            num_symbols = len(self.config.symbols)
            per_symbol_balance = balance / num_symbols
            price = df["close"].iloc[-1]
            margin_ratio = self.risk.get_dynamic_margin_ratio(per_symbol_balance)
            quantity = self.exchange.calculate_quantity(
                balance=per_symbol_balance, price=price, leverage=self.config.leverage, margin_ratio=margin_ratio
            )
            logger.info(f"[{self.symbol}] 포지션 크기: 잔고={per_symbol_balance:.2f}/{balance:.2f} USDT, 증거금비율={margin_ratio:.1%}, 수량={quantity}")
            # df는 이미 calculate_all() 적용된 df_with_indicators이므로
            # Indicators를 재생성하지 않고 ATR을 직접 사용
            atr = df["atr"].iloc[-1]
            if signal == "LONG":
                stop_loss = price - atr * self.strategy.atr_sl_mult
                take_profit = price + atr * self.strategy.atr_tp_mult
            else:
                stop_loss = price + atr * self.strategy.atr_sl_mult
                take_profit = price - atr * self.strategy.atr_tp_mult

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

            await self.risk.register_position(self.symbol, signal)
            self.current_trade_side = signal
            self._entry_price = price
            self._entry_quantity = quantity
            self._entry_time_ms = int(time.time() * 1000)
        self.notifier.notify_open(
            symbol=self.symbol,
            side=signal,
            entry_price=price,
            quantity=quantity,
            leverage=self.config.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_data=signal_snapshot,
        )
        logger.success(
            f"[{self.symbol}] {signal} 진입: 가격={price}, 수량={quantity}, "
            f"SL={stop_loss:.4f}, TP={take_profit:.4f}, "
            f"RSI={signal_snapshot['rsi']:.2f}, "
            f"MACD_H={signal_snapshot['macd_hist']:.6f}, "
            f"ATR={signal_snapshot['atr']:.6f}"
        )

        sl_side = "SELL" if signal == "LONG" else "BUY"
        try:
            await self._place_sl_tp_with_retry(
                sl_side, quantity, stop_loss, take_profit
            )
        except Exception as e:
            logger.error(
                f"[{self.symbol}] SL/TP 배치 최종 실패 — 긴급 청산: {e}"
            )
            await self._emergency_close(side, quantity)

    _SL_TP_MAX_RETRIES = 3

    async def _place_sl_tp_with_retry(
        self, sl_side: str, quantity: float, stop_loss: float, take_profit: float
    ) -> None:
        """SL/TP 주문을 재시도 로직과 함께 배치한다. 최종 실패 시 예외를 raise."""
        sl_placed = False
        tp_placed = False
        last_error = None

        for attempt in range(1, self._SL_TP_MAX_RETRIES + 1):
            try:
                if not sl_placed:
                    await self.exchange.place_order(
                        side=sl_side,
                        quantity=quantity,
                        order_type="STOP_MARKET",
                        stop_price=self.exchange._round_price(stop_loss),
                        reduce_only=True,
                    )
                    sl_placed = True
                if not tp_placed:
                    await self.exchange.place_order(
                        side=sl_side,
                        quantity=quantity,
                        order_type="TAKE_PROFIT_MARKET",
                        stop_price=self.exchange._round_price(take_profit),
                        reduce_only=True,
                    )
                    tp_placed = True
                return  # 둘 다 성공
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{self.symbol}] SL/TP 배치 실패 (시도 {attempt}/{self._SL_TP_MAX_RETRIES}): {e}"
                )
                if attempt < self._SL_TP_MAX_RETRIES:
                    await asyncio.sleep(1)

        raise last_error  # 모든 재시도 실패

    async def _emergency_close(self, entry_side: str, quantity: float) -> None:
        """SL/TP 배치 실패 시 포지션을 긴급 시장가 청산한다."""
        try:
            close_side = "SELL" if entry_side == "BUY" else "BUY"
            await self.exchange.cancel_all_orders()
            await self.exchange.place_order(
                side=close_side, quantity=quantity, reduce_only=True
            )
            await self.risk.close_position(self.symbol, 0.0)
            self.current_trade_side = None
            self._entry_price = None
            self._entry_quantity = None
            self.notifier.notify_info(
                f"🚨 [{self.symbol}] SL/TP 배치 실패 → 긴급 청산 완료"
            )
            logger.warning(f"[{self.symbol}] 긴급 청산 완료")
        except Exception as e:
            logger.critical(
                f"[{self.symbol}] 긴급 청산마저 실패! 수동 개입 필요: {e}"
            )
            self.notifier.notify_info(
                f"🔴 [{self.symbol}] 긴급 청산 실패! 수동 청산 필요: {e}"
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
        async with self._close_lock:
            # 이미 Flat 상태면 중복 처리 방지 (SYNC 또는 process_candle에서 먼저 처리됨)
            if self.current_trade_side is None and not self._is_reentering:
                logger.debug(f"[{self.symbol}] 이미 Flat 상태 — 콜백 건너뜀")
                self._close_event.set()
                return

            estimated_pnl = self._calc_estimated_pnl(exit_price)
            diff = net_pnl - estimated_pnl

            await self.risk.close_position(self.symbol, net_pnl)

            self.notifier.notify_close(
                symbol=self.symbol,
                side=self.current_trade_side or "UNKNOWN",
                close_reason=close_reason,
                exit_price=exit_price,
                estimated_pnl=estimated_pnl,
                net_pnl=net_pnl,
                diff=diff,
            )

            logger.success(
                f"[{self.symbol}] 포지션 청산({close_reason}): 예상={estimated_pnl:+.4f}, "
                f"순수익={net_pnl:+.4f}, 차이={diff:+.4f} USDT"
            )

            # 거래 기록 저장 + 킬스위치 검사 (청산 후 항상 수행)
            self._append_trade(net_pnl, close_reason)
            self._check_kill_switch()

            # _close_and_reenter 대기 해제
            self._close_event.set()

            # _close_and_reenter 중이면 신규 포지션 상태를 덮어쓰지 않는다
            if self._is_reentering:
                return

            # Flat 상태로 초기화
            self.current_trade_side = None
            self._entry_price = None
            self._entry_quantity = None
            self._entry_time_ms = None

    _MONITOR_INTERVAL = 300  # 5분

    async def _position_monitor(self):
        """포지션 보유 중일 때 5분마다 현재가·미실현 PnL을 로깅한다.
        또한 Binance API를 조회하여 WebSocket 이벤트 누락 시 청산을 감지한다."""
        while True:
            await asyncio.sleep(self._MONITOR_INTERVAL)

            # ── 폴백: Binance API로 실제 포지션 상태 확인 ──
            if self.current_trade_side is not None and not self._is_reentering:
                try:
                    actual_pos = await self.exchange.get_position()
                    if actual_pos is None:
                        async with self._close_lock:
                            # Lock 획득 후 재확인 (콜백이 먼저 처리했을 수 있음)
                            if self.current_trade_side is None:
                                continue
                            logger.warning(
                                f"[{self.symbol}] 포지션 불일치 감지: "
                                f"봇={self.current_trade_side}, 바이낸스=포지션 없음 — 상태 동기화"
                            )
                            # Binance income API에서 실제 PnL 조회
                            realized_pnl = 0.0
                            commission = 0.0
                            exit_price = 0.0
                            try:
                                pnl_rows, comm_rows = await self.exchange.get_recent_income(
                                    limit=10, start_time=self._entry_time_ms,
                                )
                                if pnl_rows:
                                    realized_pnl = sum(float(r.get("income", "0")) for r in pnl_rows)
                                if comm_rows:
                                    commission = sum(abs(float(r.get("income", "0"))) for r in comm_rows)
                            except Exception:
                                pass
                            net_pnl = realized_pnl - commission
                            # exit_price 추정: 진입가 + PnL/수량
                            if self._entry_quantity and self._entry_quantity > 0 and self._entry_price:
                                if self.current_trade_side == "LONG":
                                    exit_price = self._entry_price + realized_pnl / self._entry_quantity
                                else:
                                    exit_price = self._entry_price - realized_pnl / self._entry_quantity

                            await self.risk.close_position(self.symbol, net_pnl)
                            self.notifier.notify_close(
                                symbol=self.symbol,
                                side=self.current_trade_side,
                                close_reason="SYNC",
                                exit_price=exit_price,
                                estimated_pnl=realized_pnl,
                                net_pnl=net_pnl,
                                diff=net_pnl - realized_pnl,
                            )
                            logger.info(
                                f"[{self.symbol}] 청산 감지(SYNC): exit={exit_price:.4f}, "
                                f"rp={realized_pnl:+.4f}, commission={commission:.4f}, "
                                f"net_pnl={net_pnl:+.4f}"
                            )
                            self._append_trade(net_pnl, "SYNC")
                            self._check_kill_switch()
                            self.current_trade_side = None
                            self._entry_price = None
                            self._entry_quantity = None
                            self._entry_time_ms = None
                            self._close_event.set()
                            continue
                except Exception as e:
                    logger.debug(f"[{self.symbol}] 포지션 동기화 확인 실패 (무시): {e}")

            if self.current_trade_side is None:
                continue
            price = self.stream.latest_price
            if price is None or self._entry_price is None or self._entry_quantity is None:
                continue
            pnl = self._calc_estimated_pnl(price)
            cost = self._entry_price * self._entry_quantity
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
            logger.info(
                f"[{self.symbol}] 포지션 모니터 | {self.current_trade_side} | "
                f"현재가={price:.4f} | PnL={pnl:+.4f} USDT ({pnl_pct:+.2f}%) | "
                f"진입가={self._entry_price:.4f}"
            )

    async def _close_position(self, position: dict):
        """포지션 청산 주문만 실행한다. PnL 기록/알림은 _on_position_closed 콜백이 담당."""
        amt = abs(float(position["positionAmt"]))
        side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
        await self.exchange.cancel_all_orders()
        await self.exchange.place_order(side=side, quantity=amt, reduce_only=True)
        logger.info(f"[{self.symbol}] 청산 주문 전송 완료 (side={side}, qty={amt})")

    async def _close_and_reenter(
        self,
        position: dict,
        signal: str,
        df,
        btc_df=None,
        eth_df=None,
        oi_change: float = 0.0,
        funding_rate: float = 0.0,
        oi_change_ma5: float = 0.0,
        oi_price_spread: float = 0.0,
    ) -> None:
        """기존 포지션을 청산하고, ML 필터 통과 시 반대 방향으로 즉시 재진입한다."""
        # 재진입 플래그: User Data Stream 콜백이 신규 포지션 상태를 초기화하지 않도록 보호
        self._is_reentering = True
        self._close_event.clear()
        try:
            await self._close_position(position)

            # 콜백이 PnL을 기록할 때까지 대기 (최대 10초)
            try:
                await asyncio.wait_for(self._close_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.symbol}] 청산 콜백 타임아웃 — 수동 동기화")
                await self.risk.close_position(self.symbol, 0.0)

            # 로컬 상태를 Flat으로 전환
            self.current_trade_side = None
            self._entry_price = None
            self._entry_quantity = None
            self._entry_time_ms = None

            if self._killed:
                logger.info(f"[{self.symbol}] 킬스위치 활성 — 재진입 건너뜀 (청산만 수행)")
                return

            if not await self.risk.can_open_new_position(self.symbol, signal):
                logger.info(f"[{self.symbol}] 최대 포지션 수 도달 — 재진입 건너뜀")
                return

            if self.ml_filter.is_model_loaded():
                features = build_features_aligned(
                    df, signal,
                    btc_df=btc_df, eth_df=eth_df,
                    oi_change=oi_change, funding_rate=funding_rate,
                    oi_change_ma5=oi_change_ma5, oi_price_spread=oi_price_spread,
                    oi_history=list(self._oi_history),
                    funding_history=list(self._funding_history),
                )
                if not self.ml_filter.should_enter(features):
                    logger.info(f"[{self.symbol}] ML 필터 차단: {signal} 재진입 무시")
                    return

            await self._open_position(signal, df)
        finally:
            self._is_reentering = False

    async def run(self):
        s = self.strategy
        logger.info(
            f"[{self.symbol}] 봇 시작, 레버리지 {self.config.leverage}x | "
            f"SL={s.atr_sl_mult}x TP={s.atr_tp_mult}x Signal≥{s.signal_threshold} "
            f"ADX≥{s.adx_threshold} Vol≥{s.volume_multiplier}x"
        )
        await self._recover_position()
        await self._init_oi_history()

        user_stream = UserDataStream(
            symbol=self.symbol,
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
            self._position_monitor(),
        )
