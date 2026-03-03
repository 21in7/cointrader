import pandas as pd
import numpy as np

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
    "btc_ret_1", "btc_ret_3", "btc_ret_5",
    "eth_ret_1", "eth_ret_3", "eth_ret_5",
    "xrp_btc_rs", "xrp_eth_rs",
    # 시장 미시구조: OI 변화율(z-score), 펀딩비(z-score)
    # parquet에 oi_change/funding_rate 컬럼이 없으면 dataset_builder에서 0으로 채움
    "oi_change", "funding_rate",
    "adx",
]


def _calc_ret(closes: pd.Series, n: int) -> float:
    """n캔들 전 대비 수익률. 데이터 부족 시 0.0."""
    if len(closes) < n + 1:
        return 0.0
    prev = closes.iloc[-(n + 1)]
    return (closes.iloc[-1] - prev) / prev if prev != 0 else 0.0


def _calc_rs(xrp_ret: float, other_ret: float) -> float:
    """상대강도 = xrp_ret / other_ret. 분모 0이면 0.0."""
    if other_ret == 0.0:
        return 0.0
    return xrp_ret / other_ret


def build_features(
    df: pd.DataFrame,
    signal: str,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    oi_change: float | None = None,
    funding_rate: float | None = None,
) -> pd.Series:
    """
    기술 지표가 계산된 DataFrame의 마지막 행에서 ML 피처를 추출한다.
    btc_df, eth_df가 제공되면 24개 피처를, 없으면 16개 피처를 반환한다.
    signal: "LONG" | "SHORT"
    oi_change, funding_rate: 실제 값이 제공되면 사용, 없으면 0.0으로 채운다.
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
    ret_1 = _calc_ret(closes, 1)
    ret_3 = _calc_ret(closes, 3)
    ret_5 = _calc_ret(closes, 5)

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

    base = {
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
    }

    if btc_df is not None and eth_df is not None:
        btc_ret_1 = _calc_ret(btc_df["close"], 1)
        btc_ret_3 = _calc_ret(btc_df["close"], 3)
        btc_ret_5 = _calc_ret(btc_df["close"], 5)
        eth_ret_1 = _calc_ret(eth_df["close"], 1)
        eth_ret_3 = _calc_ret(eth_df["close"], 3)
        eth_ret_5 = _calc_ret(eth_df["close"], 5)

        base.update({
            "btc_ret_1":  float(btc_ret_1),
            "btc_ret_3":  float(btc_ret_3),
            "btc_ret_5":  float(btc_ret_5),
            "eth_ret_1":  float(eth_ret_1),
            "eth_ret_3":  float(eth_ret_3),
            "eth_ret_5":  float(eth_ret_5),
            "xrp_btc_rs": float(_calc_rs(ret_1, btc_ret_1)),
            "xrp_eth_rs": float(_calc_rs(ret_1, eth_ret_1)),
        })

    # 실시간에서 실제 값이 제공되면 사용, 없으면 0으로 채운다
    base["oi_change"]    = float(oi_change)    if oi_change    is not None else 0.0
    base["funding_rate"] = float(funding_rate) if funding_rate is not None else 0.0
    base["adx"] = float(last.get("adx", 0))

    return pd.Series(base)
