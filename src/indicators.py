import pandas as pd
import pandas_ta as ta
from loguru import logger


class Indicators:
    """
    복합 기술 지표 계산 및 매매 신호 생성.
    공격적 전략: 여러 지표가 동시에 같은 방향을 가리킬 때 진입.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_all(self) -> pd.DataFrame:
        df = self.df

        # RSI (14)
        df["rsi"] = ta.rsi(df["close"], length=14)

        # MACD (12, 26, 9)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"]        = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"]   = macd["MACDh_12_26_9"]

        # 볼린저 밴드 (20, 2)
        bb = ta.bbands(df["close"], length=20, std=2)
        df["bb_upper"] = bb["BBU_20_2.0_2.0"]
        df["bb_mid"]   = bb["BBM_20_2.0_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0_2.0"]

        # EMA (9, 21, 50)
        df["ema9"]  = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)

        # ATR (14) - 변동성 기반 손절 계산용
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Stochastic RSI
        stoch = ta.stochrsi(df["close"], length=14)
        df["stoch_k"] = stoch["STOCHRSIk_14_14_3_3"]
        df["stoch_d"] = stoch["STOCHRSId_14_14_3_3"]

        # ADX (14) — 횡보장 필터
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_df["ADX_14"]

        # 거래량 이동평균
        df["vol_ma20"] = ta.sma(df["volume"], length=20)

        return df

    def get_signal(
        self,
        df: pd.DataFrame,
        signal_threshold: int = 3,
        adx_threshold: float = 25,
        volume_multiplier: float = 2.5,
    ) -> tuple[str, dict]:
        """
        복합 지표 기반 매매 신호 생성.

        signal_threshold: 최소 가중치 합계 (기본 3)
        adx_threshold: ADX 최소값 필터 (0=비활성화, 25=ADX<25이면 HOLD)
        volume_multiplier: 거래량 급증 배수 (기본 1.5)

        Returns:
            (signal, detail) — signal은 "LONG"/"SHORT"/"HOLD",
            detail은 {"long": int, "short": int, "vol_surge": bool, "adx": float|None, "hold_reason": str}
        """
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ADX 필터
        adx = last.get("adx", None)
        adx_val = adx if adx is not None and not pd.isna(adx) else None
        if adx_val is not None:
            logger.debug(f"ADX: {adx_val:.1f}")
            if adx_threshold > 0 and adx_val < adx_threshold:
                detail = {"long": 0, "short": 0, "vol_surge": False, "adx": adx_val, "hold_reason": f"ADX({adx_val:.1f}) < {adx_threshold}"}
                return "HOLD", detail

        long_signals  = 0
        short_signals = 0

        # 1. RSI
        if last["rsi"] < 35:
            long_signals += 1
        elif last["rsi"] > 65:
            short_signals += 1

        # 2. MACD 크로스
        if prev["macd"] < prev["macd_signal"] and last["macd"] > last["macd_signal"]:
            long_signals += 2  # 크로스는 강한 신호
        elif prev["macd"] > prev["macd_signal"] and last["macd"] < last["macd_signal"]:
            short_signals += 2

        # 3. 볼린저 밴드 돌파
        if last["close"] < last["bb_lower"]:
            long_signals += 1
        elif last["close"] > last["bb_upper"]:
            short_signals += 1

        # 4. EMA 정배열/역배열
        if last["ema9"] > last["ema21"] > last["ema50"]:
            long_signals += 1
        elif last["ema9"] < last["ema21"] < last["ema50"]:
            short_signals += 1

        # 5. Stochastic RSI 과매도/과매수
        if last["stoch_k"] < 20 and last["stoch_k"] > last["stoch_d"]:
            long_signals += 1
        elif last["stoch_k"] > 80 and last["stoch_k"] < last["stoch_d"]:
            short_signals += 1

        # 6. 거래량 확인 (신호 강화)
        vol_surge = last["volume"] > last["vol_ma20"] * volume_multiplier

        detail = {"long": long_signals, "short": short_signals, "vol_surge": vol_surge, "adx": adx_val, "hold_reason": ""}

        if long_signals >= signal_threshold and (vol_surge or long_signals >= signal_threshold + 1):
            return "LONG", detail
        elif short_signals >= signal_threshold and (vol_surge or short_signals >= signal_threshold + 1):
            return "SHORT", detail

        # HOLD 사유 구성
        best_side = "LONG" if long_signals >= short_signals else "SHORT"
        best_score = max(long_signals, short_signals)
        reasons = []
        if best_score < signal_threshold:
            reasons.append(f"{best_side} 점수({best_score}) < 임계값({signal_threshold})")
        elif not vol_surge and best_score < signal_threshold + 1:
            reasons.append(f"거래량 미급증 & {best_side} 점수({best_score}) < {signal_threshold + 1}")
        detail["hold_reason"] = ", ".join(reasons) if reasons else "점수 부족"
        return "HOLD", detail

    def get_atr_stop(
        self, df: pd.DataFrame, side: str, entry_price: float,
        atr_sl_mult: float = 2.0, atr_tp_mult: float = 2.0,
    ) -> tuple[float, float]:
        """ATR 기반 손절/익절 가격 반환 (stop_loss, take_profit)"""
        atr = df["atr"].iloc[-1]
        multiplier_sl = atr_sl_mult
        multiplier_tp = atr_tp_mult
        if side == "LONG":
            stop_loss   = entry_price - atr * multiplier_sl
            take_profit = entry_price + atr * multiplier_tp
        else:
            stop_loss   = entry_price + atr * multiplier_sl
            take_profit = entry_price - atr * multiplier_tp
        return stop_loss, take_profit
