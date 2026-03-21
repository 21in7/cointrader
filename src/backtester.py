"""
독립 백테스트 엔진.
봇 코드(src/bot.py)를 수정하지 않고, 기존 모듈을 재활용하여
풀 파이프라인(지표 → 시그널 → ML 필터 → 진입/청산)을 동기 루프로 시뮬레이션한다.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

# 크립토 24/7 시장: 15분봉 × 96봉/일 × 365일 = 35,040
_ANNUALIZE_FACTOR = 35_040


def _calc_trade_stats(trades: list[dict], initial_balance: float) -> dict:
    """거래 리스트에서 통계 요약을 계산한다. Backtester와 WalkForward 공통 사용."""
    if not trades:
        return {
            "total_trades": 0, "total_pnl": 0.0, "return_pct": 0.0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "payoff_ratio": 0.0, "max_consecutive_losses": 0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "total_fees": 0.0, "close_reasons": {},
        }

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    total_fees = sum(t["entry_fee"] + t["exit_fee"] for t in trades)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0

    cumulative = np.cumsum(pnls)
    equity = initial_balance + cumulative
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    mdd = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0.0

    if len(pnls) > 1:
        pnl_arr = np.array(pnls)
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(_ANNUALIZE_FACTOR)) if np.std(pnl_arr) > 0 else 0.0
    else:
        sharpe = 0.0

    avg_w = float(np.mean(wins)) if wins else 0.0
    avg_l = float(np.mean(losses)) if losses else 0.0
    payoff_ratio = round(avg_w / abs(avg_l), 2) if avg_l != 0 else float("inf")

    max_consec_loss = 0
    cur_streak = 0
    for p in pnls:
        if p <= 0:
            cur_streak += 1
            max_consec_loss = max(max_consec_loss, cur_streak)
        else:
            cur_streak = 0

    reasons = {}
    for t in trades:
        r = t["close_reason"]
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 4),
        "return_pct": round(total_pnl / initial_balance * 100, 2),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "avg_win": round(avg_w, 4),
        "avg_loss": round(avg_l, 4),
        "payoff_ratio": payoff_ratio,
        "max_consecutive_losses": max_consec_loss,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_fees": round(total_fees, 4),
        "close_reasons": reasons,
    }

from src.dataset_builder import (
    _calc_indicators, _calc_signals, _calc_features_vectorized,
    generate_dataset_vectorized, stratified_undersample,
)
from src.ml_features import FEATURE_COLS
from src.ml_filter import MLFilter


# ── 설정 ─────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    symbols: list[str] = field(default_factory=lambda: ["XRPUSDT"])
    start: str | None = None
    end: str | None = None
    initial_balance: float = 1000.0
    leverage: int = 10
    fee_pct: float = 0.04        # taker 수수료 (%)
    slippage_pct: float = 0.01   # 슬리피지 (%)
    use_ml: bool = True
    ml_threshold: float = 0.55
    # 리스크
    max_daily_loss_pct: float = 0.05
    max_positions: int = 3
    max_same_direction: int = 2
    # 증거금
    margin_max_ratio: float = 0.50
    margin_min_ratio: float = 0.20
    margin_decay_rate: float = 0.0006
    # SL/TP ATR 배수
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 2.0
    min_notional: float = 5.0
    # 전략 파라미터
    signal_threshold: int = 3
    adx_threshold: float = 25.0
    volume_multiplier: float = 2.5

    WARMUP = 60  # 지표 안정화에 필요한 캔들 수


# ── 포지션 상태 ──────────────────────────────────────────────────────
@dataclass
class Position:
    symbol: str
    side: str           # "LONG" | "SHORT"
    entry_price: float
    quantity: float
    sl: float
    tp: float
    entry_time: pd.Timestamp
    entry_fee: float
    entry_indicators: dict = field(default_factory=dict)
    ml_proba: float | None = None


# ── 동기 RiskManager ─────────────────────────────────────────────────
class BacktestRiskManager:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.daily_pnl: float = 0.0
        self.initial_balance: float = cfg.initial_balance
        self.base_balance: float = cfg.initial_balance
        self.open_positions: dict[str, str] = {}  # {symbol: side}
        self._current_date: str | None = None

    def new_day(self, date_str: str):
        if self._current_date != date_str:
            self._current_date = date_str
            self.daily_pnl = 0.0

    def is_trading_allowed(self) -> bool:
        if self.initial_balance <= 0:
            return True
        if self.daily_pnl < 0 and abs(self.daily_pnl) / self.initial_balance >= self.cfg.max_daily_loss_pct:
            return False
        return True

    def can_open(self, symbol: str, side: str) -> bool:
        if len(self.open_positions) >= self.cfg.max_positions:
            return False
        if symbol in self.open_positions:
            return False
        same_dir = sum(1 for s in self.open_positions.values() if s == side)
        if same_dir >= self.cfg.max_same_direction:
            return False
        return True

    def register(self, symbol: str, side: str):
        self.open_positions[symbol] = side

    def close(self, symbol: str, pnl: float):
        self.open_positions.pop(symbol, None)
        self.daily_pnl += pnl

    def get_dynamic_margin_ratio(self, balance: float) -> float:
        ratio = self.cfg.margin_max_ratio - (
            (balance - self.base_balance) * self.cfg.margin_decay_rate
        )
        return max(self.cfg.margin_min_ratio, min(self.cfg.margin_max_ratio, ratio))


# ── 유틸 ─────────────────────────────────────────────────────────────
def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """시장가 주문의 슬리피지 적용. BUY는 불리하게(+), SELL은 불리하게(-)."""
    factor = slippage_pct / 100.0
    if side == "BUY":
        return price * (1 + factor)
    return price * (1 - factor)


def _calc_fee(price: float, quantity: float, fee_pct: float) -> float:
    return price * quantity * fee_pct / 100.0


def _load_data(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    path = Path(f"data/{symbol.lower()}/combined_15m.parquet")
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    # tz-aware → tz-naive 통일 (UTC 기준)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def _get_ml_proba(ml_filter: MLFilter | None, features: pd.Series) -> float | None:
    """ML 확률을 반환. 모델이 없거나 비활성이면 None."""
    if ml_filter is None or not ml_filter.is_model_loaded():
        return None
    try:
        if ml_filter._onnx_session is not None:
            input_name = ml_filter._onnx_session.get_inputs()[0].name
            X = features[FEATURE_COLS].values.astype(np.float32).reshape(1, -1)
            return float(ml_filter._onnx_session.run(None, {input_name: X})[0][0])
        else:
            available = [c for c in FEATURE_COLS if c in features.index]
            X = pd.DataFrame([features[available].values.astype(np.float64)], columns=available)
            return float(ml_filter._lgbm_model.predict_proba(X)[0][1])
    except Exception as e:
        logger.warning(f"ML PROBA ERROR: {e}")
        return None


# ── 메인 엔진 ────────────────────────────────────────────────────────
class Backtester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.risk = BacktestRiskManager(cfg)
        self.balance = cfg.initial_balance
        self.positions: dict[str, Position] = {}  # {symbol: Position}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self._peak_equity: float = cfg.initial_balance

        # ML 필터 (심볼별)
        self.ml_filters: dict[str, MLFilter | None] = {}
        if cfg.use_ml:
            for sym in cfg.symbols:
                sym_dir = Path(f"models/{sym.lower()}")
                onnx = str(sym_dir / "mlx_filter.weights.onnx")
                lgbm = str(sym_dir / "lgbm_filter.pkl")
                if not sym_dir.exists():
                    onnx = "models/mlx_filter.weights.onnx"
                    lgbm = "models/lgbm_filter.pkl"
                mf = MLFilter(onnx_path=onnx, lgbm_path=lgbm, threshold=cfg.ml_threshold)
                self.ml_filters[sym] = mf if mf.is_model_loaded() else None
        else:
            for sym in cfg.symbols:
                self.ml_filters[sym] = None

    def run(self, ml_models: dict[str, object] | None = None) -> dict:
        """백테스트 실행. 결과 dict(config, summary, trades, validation) 반환.

        ml_models: walk-forward에서 심볼별 사전 학습 모델을 전달할 때 사용.
                   {symbol: lgbm_model} 형태. None이면 기존 파일 기반 MLFilter 사용.
        """
        # 데이터 로드
        all_data: dict[str, pd.DataFrame] = {}
        all_indicators: dict[str, pd.DataFrame] = {}
        all_signals: dict[str, np.ndarray] = {}
        all_features: dict[str, pd.DataFrame] = {}

        for sym in self.cfg.symbols:
            df = _load_data(sym, self.cfg.start, self.cfg.end)
            all_data[sym] = df

            # BTC/ETH 상관 데이터: 임베딩된 컬럼에서 추출 (별도 파일 폴백)
            base_cols = ["open", "high", "low", "close", "volume"]
            btc_df = eth_df = None
            if "close_btc" in df.columns:
                btc_df = df[[c + "_btc" for c in base_cols]].copy()
                btc_df.columns = base_cols
            else:
                btc_df = self._try_load_corr("BTCUSDT")
            if "close_eth" in df.columns:
                eth_df = df[[c + "_eth" for c in base_cols]].copy()
                eth_df.columns = base_cols
            else:
                eth_df = self._try_load_corr("ETHUSDT")

            df_ind = _calc_indicators(df)
            all_indicators[sym] = df_ind
            sig_arr = _calc_signals(
                df_ind,
                signal_threshold=self.cfg.signal_threshold,
                adx_threshold=self.cfg.adx_threshold,
                volume_multiplier=self.cfg.volume_multiplier,
            )
            all_signals[sym] = sig_arr
            # 벡터화 피처 미리 계산 (학습과 동일한 z-score 적용)
            all_features[sym] = _calc_features_vectorized(
                df_ind, sig_arr, btc_df=btc_df, eth_df=eth_df,
            )
            logger.info(f"[{sym}] 데이터 로드: {len(df):,}캔들 ({df.index[0]} ~ {df.index[-1]})")

        # walk-forward 모델 주입 (use_ml=True일 때만)
        if ml_models is not None and self.cfg.use_ml:
            self.ml_filters = {}
            for sym in self.cfg.symbols:
                if sym in ml_models and ml_models[sym] is not None:
                    self.ml_filters[sym] = MLFilter.from_model(
                        ml_models[sym], threshold=self.cfg.ml_threshold
                    )
                else:
                    self.ml_filters[sym] = None

        # 멀티심볼: 타임스탬프 기준 통합 이벤트 생성
        events = self._build_events(all_indicators, all_signals)
        logger.info(f"총 이벤트: {len(events):,}개")

        # 메인 루프
        latest_prices: dict[str, float] = {}
        for ts, sym, candle_idx in events:
            date_str = str(ts.date())
            self.risk.new_day(date_str)

            df_ind = all_indicators[sym]
            signal = all_signals[sym][candle_idx]
            row = df_ind.iloc[candle_idx]
            latest_prices[sym] = float(row["close"])

            # 에퀴티 기록
            self._record_equity(ts, current_prices=latest_prices)

            # 1) 일일 손실 체크
            if not self.risk.is_trading_allowed():
                continue

            # 2) SL/TP 체크 (보유 포지션)
            if sym in self.positions:
                closed = self._check_sl_tp(sym, row, ts)
                if closed:
                    continue

            # 3) 반대 시그널 재진입
            if sym in self.positions and signal != "HOLD":
                pos = self.positions[sym]
                if (pos.side == "LONG" and signal == "SHORT") or \
                   (pos.side == "SHORT" and signal == "LONG"):
                    self._close_position(sym, row["close"], ts, "REVERSE_SIGNAL")
                    # 새 방향으로 재진입 시도
                    if self.risk.can_open(sym, signal):
                        self._try_enter(
                            sym, signal, df_ind, candle_idx,
                            all_features[sym], ts=ts,
                        )
                    continue

            # 4) 신규 진입
            if sym not in self.positions and signal != "HOLD":
                if self.risk.can_open(sym, signal):
                    self._try_enter(
                        sym, signal, df_ind, candle_idx,
                        all_features[sym], ts=ts,
                    )

        # 미청산 포지션 강제 청산
        for sym in list(self.positions.keys()):
            last_df = all_indicators[sym]
            last_price = last_df["close"].iloc[-1]
            last_ts = last_df.index[-1]
            self._close_position(sym, last_price, last_ts, "END_OF_DATA")

        return self._build_result()

    def _try_load_corr(self, symbol: str) -> pd.DataFrame | None:
        path = Path(f"data/{symbol.lower()}/combined_15m.parquet")
        if not path.exists():
            alt = Path(f"data/combined_15m.parquet")
            if not alt.exists():
                return None
            path = alt
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if self.cfg.start:
                df = df[df.index >= pd.Timestamp(self.cfg.start)]
            if self.cfg.end:
                df = df[df.index <= pd.Timestamp(self.cfg.end)]
            return df
        except Exception:
            return None

    def _build_events(
        self,
        all_indicators: dict[str, pd.DataFrame],
        all_signals: dict[str, np.ndarray],
    ) -> list[tuple[pd.Timestamp, str, int]]:
        """모든 심볼의 캔들을 타임스탬프 순서로 정렬한 이벤트 리스트 생성."""
        events = []
        for sym, df_ind in all_indicators.items():
            for i in range(self.cfg.WARMUP, len(df_ind)):
                ts = df_ind.index[i]
                events.append((ts, sym, i))
        events.sort(key=lambda x: (x[0], x[1]))
        return events

    def _check_sl_tp(self, symbol: str, row: pd.Series, ts: pd.Timestamp) -> bool:
        """캔들의 고가/저가로 SL/TP 체크. SL 우선. 청산 시 True 반환."""
        pos = self.positions[symbol]
        high = row["high"]
        low = row["low"]

        if pos.side == "LONG":
            # SL 먼저 (보수적)
            if low <= pos.sl:
                self._close_position(symbol, pos.sl, ts, "STOP_LOSS")
                return True
            if high >= pos.tp:
                self._close_position(symbol, pos.tp, ts, "TAKE_PROFIT")
                return True
        else:  # SHORT
            if high >= pos.sl:
                self._close_position(symbol, pos.sl, ts, "STOP_LOSS")
                return True
            if low <= pos.tp:
                self._close_position(symbol, pos.tp, ts, "TAKE_PROFIT")
                return True
        return False

    def _try_enter(
        self,
        symbol: str,
        signal: str,
        df_ind: pd.DataFrame,
        candle_idx: int,
        feat_df: pd.DataFrame,
        ts: pd.Timestamp,
    ):
        """ML 필터 + 포지션 크기 계산 → 진입."""
        row = df_ind.iloc[candle_idx]

        # 벡터화된 피처에서 해당 행을 lookup (학습과 동일한 z-score 적용)
        available_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
        features = feat_df.iloc[candle_idx][available_cols]

        # ML 필터
        ml_filter = self.ml_filters.get(symbol)
        ml_proba = _get_ml_proba(ml_filter, features)

        if ml_filter is not None and ml_filter.is_model_loaded():
            if ml_proba is not None and ml_proba < self.cfg.ml_threshold:
                return  # ML 차단

        # 포지션 크기 계산
        num_symbols = len(self.cfg.symbols)
        per_symbol_balance = self.balance / num_symbols
        price = float(row["close"])
        margin_ratio = self.risk.get_dynamic_margin_ratio(self.balance)
        notional = per_symbol_balance * margin_ratio * self.cfg.leverage
        if notional < self.cfg.min_notional:
            notional = self.cfg.min_notional
        quantity = round(notional / price, 1)
        if quantity * price < self.cfg.min_notional:
            quantity = round(self.cfg.min_notional / price + 0.05, 1)
        if quantity <= 0 or quantity * price < self.cfg.min_notional:
            return

        # 슬리피지 적용 (시장가 진입)
        buy_side = "BUY" if signal == "LONG" else "SELL"
        entry_price = _apply_slippage(price, buy_side, self.cfg.slippage_pct)

        # 수수료 (청산 시 net_pnl에서 차감하므로 여기서 balance 차감하지 않음)
        entry_fee = _calc_fee(entry_price, quantity, self.cfg.fee_pct)

        # SL/TP 계산
        atr = float(row.get("atr", 0))
        if atr <= 0:
            return
        if signal == "LONG":
            sl = entry_price - atr * self.cfg.atr_sl_mult
            tp = entry_price + atr * self.cfg.atr_tp_mult
        else:
            sl = entry_price + atr * self.cfg.atr_sl_mult
            tp = entry_price - atr * self.cfg.atr_tp_mult

        indicators_snapshot = {
            "rsi": float(row.get("rsi", 0)),
            "macd_hist": float(row.get("macd_hist", 0)),
            "atr": float(atr),
            "adx": float(row.get("adx", 0)),
        }

        pos = Position(
            symbol=symbol,
            side=signal,
            entry_price=entry_price,
            quantity=quantity,
            sl=sl,
            tp=tp,
            entry_time=ts,
            entry_fee=entry_fee,
            entry_indicators=indicators_snapshot,
            ml_proba=ml_proba,
        )
        self.positions[symbol] = pos
        self.risk.register(symbol, signal)

    def _close_position(
        self, symbol: str, exit_price: float, ts: pd.Timestamp, reason: str
    ):
        pos = self.positions.pop(symbol)

        # SL/TP 히트는 지정가이므로 슬리피지 없음. 그 외는 시장가.
        if reason in ("REVERSE_SIGNAL", "END_OF_DATA"):
            close_side = "SELL" if pos.side == "LONG" else "BUY"
            exit_price = _apply_slippage(exit_price, close_side, self.cfg.slippage_pct)

        exit_fee = _calc_fee(exit_price, pos.quantity, self.cfg.fee_pct)

        if pos.side == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - pos.entry_fee - exit_fee
        self.balance += net_pnl
        self.risk.close(symbol, net_pnl)

        trade = {
            "symbol": symbol,
            "side": pos.side,
            "entry_time": str(pos.entry_time),
            "exit_time": str(ts),
            "entry_price": round(pos.entry_price, 6),
            "exit_price": round(exit_price, 6),
            "quantity": pos.quantity,
            "sl": round(pos.sl, 6),
            "tp": round(pos.tp, 6),
            "gross_pnl": round(gross_pnl, 6),
            "entry_fee": round(pos.entry_fee, 6),
            "exit_fee": round(exit_fee, 6),
            "net_pnl": round(net_pnl, 6),
            "close_reason": reason,
            "ml_proba": round(pos.ml_proba, 4) if pos.ml_proba is not None else None,
            "indicators": pos.entry_indicators,
        }
        self.trades.append(trade)

    def _record_equity(self, ts: pd.Timestamp, current_prices: dict[str, float] | None = None):
        unrealized = 0.0
        for sym, pos in self.positions.items():
            price = (current_prices or {}).get(sym)
            if price is not None:
                if pos.side == "LONG":
                    unrealized += (price - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - price) * pos.quantity
        equity = self.balance + unrealized
        self.equity_curve.append({"timestamp": str(ts), "equity": round(equity, 4)})
        if equity > self._peak_equity:
            self._peak_equity = equity

    def _build_result(self) -> dict:
        summary = self._calc_summary()
        from src.backtest_validator import validate
        validation = validate(self.trades, summary, self.cfg)
        return {
            "config": asdict(self.cfg),
            "summary": summary,
            "trades": self.trades,
            "validation": validation,
        }

    def _calc_summary(self) -> dict:
        return _calc_trade_stats(self.trades, self.cfg.initial_balance)


# ── Walk-Forward 백테스트 ─────────────────────────────────────────────
@dataclass
class WalkForwardConfig(BacktestConfig):
    train_months: int = 6       # 학습 윈도우 (개월)
    test_months: int = 1        # 검증 윈도우 (개월)
    time_weight_decay: float = 2.0
    negative_ratio: int = 3


class WalkForwardBacktester:
    """
    Walk-Forward 백테스트: 기간별로 모델을 학습하고 미래 데이터에서만 검증한다.
    look-ahead bias를 완전히 제거한다.
    """

    def __init__(self, cfg: WalkForwardConfig):
        self.cfg = cfg

    def run(self) -> dict:
        # 데이터 로드 (전체 기간)
        all_raw: dict[str, pd.DataFrame] = {}
        for sym in self.cfg.symbols:
            all_raw[sym] = _load_data(sym, self.cfg.start, self.cfg.end)

        # 윈도우 생성
        windows = self._build_windows(all_raw)
        logger.info(f"Walk-Forward: {len(windows)}개 윈도우 "
                     f"(학습 {self.cfg.train_months}개월, 검증 {self.cfg.test_months}개월)")

        all_trades = []
        fold_summaries = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"  폴드 {i+1}/{len(windows)}: "
                         f"학습 {train_start.date()}~{train_end.date()}, "
                         f"검증 {test_start.date()}~{test_end.date()}")

            # 심볼별 모델 학습 (use_ml=True일 때만)
            models = {}
            if self.cfg.use_ml:
                for sym in self.cfg.symbols:
                    model = self._train_model(
                        all_raw[sym], train_start, train_end, sym
                    )
                    models[sym] = model

            # 검증 구간 백테스트
            test_cfg = BacktestConfig(
                symbols=self.cfg.symbols,
                start=str(test_start.date()),
                end=str(test_end.date()),
                initial_balance=self.cfg.initial_balance,
                leverage=self.cfg.leverage,
                fee_pct=self.cfg.fee_pct,
                slippage_pct=self.cfg.slippage_pct,
                use_ml=self.cfg.use_ml,
                ml_threshold=self.cfg.ml_threshold,
                max_daily_loss_pct=self.cfg.max_daily_loss_pct,
                max_positions=self.cfg.max_positions,
                max_same_direction=self.cfg.max_same_direction,
                margin_max_ratio=self.cfg.margin_max_ratio,
                margin_min_ratio=self.cfg.margin_min_ratio,
                margin_decay_rate=self.cfg.margin_decay_rate,
                atr_sl_mult=self.cfg.atr_sl_mult,
                atr_tp_mult=self.cfg.atr_tp_mult,
                min_notional=self.cfg.min_notional,
                signal_threshold=self.cfg.signal_threshold,
                adx_threshold=self.cfg.adx_threshold,
                volume_multiplier=self.cfg.volume_multiplier,
            )
            bt = Backtester(test_cfg)
            result = bt.run(ml_models=models)

            # 폴드별 트레이드에 폴드 번호 추가
            for t in result["trades"]:
                t["fold"] = i + 1
            all_trades.extend(result["trades"])

            fold_summaries.append({
                "fold": i + 1,
                "train_period": f"{train_start.date()} ~ {train_end.date()}",
                "test_period": f"{test_start.date()} ~ {test_end.date()}",
                "summary": result["summary"],
            })

        # 전체 결과 집계
        return self._aggregate_results(all_trades, fold_summaries)

    def _build_windows(
        self, all_raw: dict[str, pd.DataFrame]
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        # 모든 심볼의 공통 기간
        start = max(df.index[0] for df in all_raw.values())
        end = min(df.index[-1] for df in all_raw.values())

        train_delta = pd.DateOffset(months=self.cfg.train_months)
        test_delta = pd.DateOffset(months=self.cfg.test_months)

        windows = []
        cursor = start
        while cursor + train_delta + test_delta <= end:
            train_start = cursor
            train_end = cursor + train_delta
            test_start = train_end
            test_end = test_start + test_delta
            windows.append((train_start, train_end, test_start, test_end))
            cursor = test_start  # 슬라이딩 (겹침 없음)

        return windows

    def _train_model(
        self,
        raw_df: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        symbol: str,
    ) -> object | None:
        """학습 구간 데이터로 LightGBM 모델 학습. 실패 시 None 반환."""
        # tz-naive로 비교
        ts_start = train_start.tz_localize(None) if train_start.tz else train_start
        ts_end = train_end.tz_localize(None) if train_end.tz else train_end
        idx = raw_df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        train_df = raw_df[(idx >= ts_start) & (idx < ts_end)]
        if len(train_df) < 200:
            logger.warning(f"  [{symbol}] 학습 데이터 부족: {len(train_df)}캔들")
            return None

        base_cols = ["open", "high", "low", "close", "volume"]
        df = train_df[base_cols].copy()

        # BTC/ETH 상관 데이터 (있으면)
        btc_df = eth_df = None
        if "close_btc" in train_df.columns:
            btc_df = train_df[[c + "_btc" for c in base_cols]].copy()
            btc_df.columns = base_cols
        if "close_eth" in train_df.columns:
            eth_df = train_df[[c + "_eth" for c in base_cols]].copy()
            eth_df.columns = base_cols

        try:
            dataset = generate_dataset_vectorized(
                df, btc_df=btc_df, eth_df=eth_df,
                time_weight_decay=self.cfg.time_weight_decay,
                negative_ratio=self.cfg.negative_ratio,
                signal_threshold=self.cfg.signal_threshold,
                adx_threshold=self.cfg.adx_threshold,
                volume_multiplier=self.cfg.volume_multiplier,
                atr_sl_mult=self.cfg.atr_sl_mult,
                atr_tp_mult=self.cfg.atr_tp_mult,
            )
        except Exception as e:
            logger.warning(f"  [{symbol}] 데이터셋 생성 실패: {e}")
            return None

        if dataset.empty or "label" not in dataset.columns:
            return None

        actual_cols = [c for c in FEATURE_COLS if c in dataset.columns]
        X = dataset[actual_cols].values
        y = dataset["label"].values
        w = dataset["sample_weight"].values
        source = dataset["source"].values if "source" in dataset.columns else np.full(len(X), "signal")

        # 언더샘플링
        idx = stratified_undersample(y, source, seed=42)

        # LightGBM 파라미터 (active 파일 또는 기본값)
        lgbm_params = self._load_params(symbol)

        model = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[idx], y[idx], sample_weight=w[idx])

        return model

    def _load_params(self, symbol: str) -> dict:
        """심볼별 active 파라미터 로드. 없으면 기본값."""
        params_path = Path(f"models/{symbol.lower()}/active_lgbm_params.json")
        if not params_path.exists():
            params_path = Path("models/active_lgbm_params.json")

        default = {
            "n_estimators": 434,
            "learning_rate": 0.123659,
            "max_depth": 6,
            "num_leaves": 14,
            "min_child_samples": 10,
            "subsample": 0.929062,
            "colsample_bytree": 0.946330,
            "reg_alpha": 0.573971,
            "reg_lambda": 0.000157,
        }

        if params_path.exists():
            import json
            with open(params_path) as f:
                data = json.load(f)
            best = dict(data["best_trial"]["params"])
            best.pop("weight_scale", None)
            default.update(best)

        return default

    def _aggregate_results(
        self, all_trades: list[dict], fold_summaries: list[dict]
    ) -> dict:
        """폴드별 결과를 합산하여 전체 Walk-Forward 결과 생성."""
        from src.backtest_validator import validate

        summary = _calc_trade_stats(all_trades, self.cfg.initial_balance)
        validation = validate(all_trades, summary, self.cfg)

        return {
            "mode": "walk_forward",
            "config": asdict(self.cfg),
            "summary": summary,
            "folds": fold_summaries,
            "trades": all_trades,
            "validation": validation,
        }
