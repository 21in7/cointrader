import pandas as pd
import numpy as np

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
]


def build_features(df: pd.DataFrame, signal: str) -> pd.Series:
    """
    기술 지표가 계산된 DataFrame의 마지막 행에서 ML 피처를 추출한다.
    signal: "LONG" | "SHORT"
    """
    last = df.iloc[-1]
    close = last["close"]

    bb_upper = last.get("bb_upper", close)
    bb_lower = last.get("bb_lower", close)
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    ema9  = last.get("ema9",  close)
    ema21 = last.get("ema21", close)
    ema50 = last.get("ema50", close)
    if ema9 > ema21 > ema50:
        ema_align = 1
    elif ema9 < ema21 < ema50:
        ema_align = -1
    else:
        ema_align = 0

    atr = last.get("atr", 0)
    atr_pct = atr / close if close > 0 else 0

    vol_ma20 = last.get("vol_ma20", last.get("volume", 1))
    vol_ratio = last["volume"] / vol_ma20 if vol_ma20 > 0 else 1.0

    closes = df["close"]
    ret_1 = (close - closes.iloc[-2]) / closes.iloc[-2] if len(closes) >= 2 else 0.0
    ret_3 = (close - closes.iloc[-4]) / closes.iloc[-4] if len(closes) >= 4 else 0.0
    ret_5 = (close - closes.iloc[-6]) / closes.iloc[-6] if len(closes) >= 6 else 0.0

    prev = df.iloc[-2] if len(df) >= 2 else last
    strength = 0
    rsi = last.get("rsi", 50)
    macd = last.get("macd", 0)
    macd_sig = last.get("macd_signal", 0)
    prev_macd = prev.get("macd", 0)
    prev_macd_sig = prev.get("macd_signal", 0)
    stoch_k = last.get("stoch_k", 50)
    stoch_d = last.get("stoch_d", 50)

    if signal == "LONG":
        if rsi < 35: strength += 1
        if prev_macd < prev_macd_sig and macd > macd_sig: strength += 2
        if close < last.get("bb_lower", close): strength += 1
        if ema_align == 1: strength += 1
        if stoch_k < 20 and stoch_k > stoch_d: strength += 1
    else:
        if rsi > 65: strength += 1
        if prev_macd > prev_macd_sig and macd < macd_sig: strength += 2
        if close > last.get("bb_upper", close): strength += 1
        if ema_align == -1: strength += 1
        if stoch_k > 80 and stoch_k < stoch_d: strength += 1

    return pd.Series({
        "rsi":            float(rsi),
        "macd_hist":      float(last.get("macd_hist", 0)),
        "bb_pct":         float(bb_pct),
        "ema_align":      float(ema_align),
        "stoch_k":        float(stoch_k),
        "stoch_d":        float(last.get("stoch_d", 50)),
        "atr_pct":        float(atr_pct),
        "vol_ratio":      float(vol_ratio),
        "ret_1":          float(ret_1),
        "ret_3":          float(ret_3),
        "ret_5":          float(ret_5),
        "signal_strength": float(strength),
        "side":           1.0 if signal == "LONG" else 0.0,
    })
