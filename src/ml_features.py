import pandas as pd
import numpy as np

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_pct", "ema_align",
    "stoch_k", "stoch_d", "atr_pct", "vol_ratio",
    "ret_1", "ret_3", "ret_5", "signal_strength", "side",
    "btc_ret_1", "btc_ret_3", "btc_ret_5",
    "eth_ret_1", "eth_ret_3", "eth_ret_5",
    "primary_btc_rs", "primary_eth_rs",
    # 시장 미시구조: OI 변화율(z-score), 펀딩비(z-score)
    "oi_change", "funding_rate",
    # OI 파생 피처
    "oi_change_ma5", "oi_price_spread",
    "adx",
]

# rolling z-score 윈도우 (학습과 동일)
_ZSCORE_WINDOW = 288      # 일반 피처: 15분봉 × 288 = 3일
_ZSCORE_WINDOW_OI = 96    # OI/펀딩비: 15분봉 × 96 = 1일


def _calc_ret(closes: pd.Series, n: int) -> float:
    """n캔들 전 대비 수익률. 데이터 부족 시 0.0."""
    if len(closes) < n + 1:
        return 0.0
    prev = closes.iloc[-(n + 1)]
    return (closes.iloc[-1] - prev) / prev if prev != 0 else 0.0


def _calc_rs(primary_ret: float, other_ret: float) -> float:
    """상대강도 = primary_ret / other_ret. 분모 0이면 0.0."""
    if other_ret == 0.0:
        return 0.0
    return primary_ret / other_ret


def _rolling_zscore_last(arr: np.ndarray, window: int = _ZSCORE_WINDOW) -> float:
    """배열의 마지막 값에 대한 rolling z-score를 반환한다.
    학습(dataset_builder._rolling_zscore)과 동일한 로직."""
    s = pd.Series(arr, dtype=np.float64)
    r = s.rolling(window=window, min_periods=1)
    mean = r.mean().iloc[-1]
    std = r.std(ddof=0).iloc[-1]
    if std < 1e-8:
        std = 1e-8
    return float((s.iloc[-1] - mean) / std)


def build_features(
    df: pd.DataFrame,
    signal: str,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    oi_change: float | None = None,
    funding_rate: float | None = None,
    oi_change_ma5: float | None = None,
    oi_price_spread: float | None = None,
) -> pd.Series:
    """
    [Deprecated] raw 값 기반 피처. 하위 호환용으로 유지.
    신규 코드는 build_features_aligned()를 사용할 것.
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
            "primary_btc_rs": float(_calc_rs(ret_1, btc_ret_1)),
            "primary_eth_rs": float(_calc_rs(ret_1, eth_ret_1)),
        })

    # 실시간에서 실제 값이 제공되면 사용, 없으면 0으로 채운다
    base["oi_change"]       = float(oi_change)       if oi_change       is not None else 0.0
    base["funding_rate"]    = float(funding_rate)    if funding_rate    is not None else 0.0
    base["oi_change_ma5"]   = float(oi_change_ma5)   if oi_change_ma5   is not None else 0.0
    base["oi_price_spread"] = float(oi_price_spread) if oi_price_spread is not None else 0.0
    base["adx"] = float(last.get("adx", 0))

    return pd.Series(base)


def build_features_aligned(
    df: pd.DataFrame,
    signal: str,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    oi_change: float | None = None,
    funding_rate: float | None = None,
    oi_change_ma5: float | None = None,
    oi_price_spread: float | None = None,
    oi_history: list[float] | None = None,
    funding_history: list[float] | None = None,
) -> pd.Series:
    """
    학습(dataset_builder._calc_features_vectorized)과 동일한 rolling z-score를
    적용한 피처를 반환한다. train-serve skew를 방지한다.

    df: 지표가 이미 계산된 DataFrame (최소 60캔들 이상)
    signal: "LONG" | "SHORT"
    """
    last = df.iloc[-1]
    close_series = df["close"]
    close = float(close_series.iloc[-1])

    # --- raw 값 계산 (z-score 전) ---
    bb_upper = df["bb_upper"] if "bb_upper" in df.columns else pd.Series(close, index=df.index)
    bb_lower = df["bb_lower"] if "bb_lower" in df.columns else pd.Series(close, index=df.index)
    bb_range = bb_upper - bb_lower
    bb_pct_series = (close_series - bb_lower) / (bb_range + 1e-8)

    ema9  = df.get("ema9",  close_series)
    ema21 = df.get("ema21", close_series)
    ema50 = df.get("ema50", close_series)

    ema_align_arr = np.where(
        (ema9 > ema21) & (ema21 > ema50), 1,
        np.where((ema9 < ema21) & (ema21 < ema50), -1, 0)
    ).astype(np.float32)

    atr_series = df["atr"] if "atr" in df.columns else pd.Series(0.0, index=df.index)
    atr_pct_arr = (atr_series / (close_series + 1e-8)).values

    volume = df["volume"]
    vol_ma20 = df["vol_ma20"] if "vol_ma20" in df.columns else pd.Series(1.0, index=df.index)
    vol_ratio_arr = (volume / (vol_ma20 + 1e-8)).values

    ret_1_arr = close_series.pct_change(1).fillna(0).values
    ret_3_arr = close_series.pct_change(3).fillna(0).values
    ret_5_arr = close_series.pct_change(5).fillna(0).values

    # z-score 적용 (학습과 동일)
    atr_pct_z   = _rolling_zscore_last(atr_pct_arr)
    vol_ratio_z = _rolling_zscore_last(vol_ratio_arr)
    ret_1_z     = _rolling_zscore_last(ret_1_arr)
    ret_3_z     = _rolling_zscore_last(ret_3_arr)
    ret_5_z     = _rolling_zscore_last(ret_5_arr)

    # signal_strength
    rsi = float(last.get("rsi", 50))
    macd_val = float(last.get("macd", 0))
    macd_sig_val = float(last.get("macd_signal", 0))
    stoch_k = float(last.get("stoch_k", 50))
    stoch_d = float(last.get("stoch_d", 50))
    prev = df.iloc[-2] if len(df) >= 2 else last
    prev_macd = float(prev.get("macd", 0))
    prev_macd_sig = float(prev.get("macd_signal", 0))

    strength = 0
    if signal == "LONG":
        if rsi < 35: strength += 1
        if prev_macd < prev_macd_sig and macd_val > macd_sig_val: strength += 2
        if close < float(last.get("bb_lower", close)): strength += 1
        if ema_align_arr[-1] == 1: strength += 1
        if stoch_k < 20 and stoch_k > stoch_d: strength += 1
    else:
        if rsi > 65: strength += 1
        if prev_macd > prev_macd_sig and macd_val < macd_sig_val: strength += 2
        if close > float(last.get("bb_upper", close)): strength += 1
        if ema_align_arr[-1] == -1: strength += 1
        if stoch_k > 80 and stoch_k < stoch_d: strength += 1

    # ADX z-score
    adx_arr = df["adx"].values.astype(np.float64) if "adx" in df.columns else np.zeros(len(df))
    adx_z = _rolling_zscore_last(adx_arr)

    base = {
        "rsi":            rsi,
        "macd_hist":      float(last.get("macd_hist", 0)),
        "bb_pct":         float(bb_pct_series.iloc[-1]),
        "ema_align":      float(ema_align_arr[-1]),
        "stoch_k":        stoch_k,
        "stoch_d":        stoch_d,
        "atr_pct":        atr_pct_z,
        "vol_ratio":      vol_ratio_z,
        "ret_1":          ret_1_z,
        "ret_3":          ret_3_z,
        "ret_5":          ret_5_z,
        "signal_strength": float(strength),
        "side":           1.0 if signal == "LONG" else 0.0,
    }

    # BTC/ETH 상관 피처 (z-score)
    if btc_df is not None and eth_df is not None:
        btc_r1 = btc_df["close"].pct_change(1).fillna(0).values
        btc_r3 = btc_df["close"].pct_change(3).fillna(0).values
        btc_r5 = btc_df["close"].pct_change(5).fillna(0).values
        eth_r1 = eth_df["close"].pct_change(1).fillna(0).values
        eth_r3 = eth_df["close"].pct_change(3).fillna(0).values
        eth_r5 = eth_df["close"].pct_change(5).fillna(0).values

        # 길이 맞춤 (btc/eth가 더 길 수 있음)
        n = len(df)
        def _align(arr):
            if len(arr) >= n:
                return arr[-n:]
            return np.concatenate([np.zeros(n - len(arr)), arr])

        btc_r1 = _align(btc_r1)
        btc_r3 = _align(btc_r3)
        btc_r5 = _align(btc_r5)
        eth_r1 = _align(eth_r1)
        eth_r3 = _align(eth_r3)
        eth_r5 = _align(eth_r5)

        # 상대강도 (raw → z-score)
        xrp_r1 = ret_1_arr.astype(np.float32)
        btc_r1_f = btc_r1.astype(np.float32)
        eth_r1_f = eth_r1.astype(np.float32)
        rs_btc = np.divide(xrp_r1, btc_r1_f, out=np.zeros_like(xrp_r1), where=(btc_r1_f != 0))
        rs_eth = np.divide(xrp_r1, eth_r1_f, out=np.zeros_like(xrp_r1), where=(eth_r1_f != 0))

        base.update({
            "btc_ret_1":  _rolling_zscore_last(btc_r1),
            "btc_ret_3":  _rolling_zscore_last(btc_r3),
            "btc_ret_5":  _rolling_zscore_last(btc_r5),
            "eth_ret_1":  _rolling_zscore_last(eth_r1),
            "eth_ret_3":  _rolling_zscore_last(eth_r3),
            "eth_ret_5":  _rolling_zscore_last(eth_r5),
            "primary_btc_rs": _rolling_zscore_last(rs_btc),
            "primary_eth_rs": _rolling_zscore_last(rs_eth),
        })

    # OI/펀딩비 z-score (학습과 동일한 rolling z-score 적용)
    if oi_history and len(oi_history) >= 2 and oi_change is not None:
        oi_arr = np.array(oi_history, dtype=np.float64)
        base["oi_change"] = _rolling_zscore_last(oi_arr, window=_ZSCORE_WINDOW_OI)
    else:
        base["oi_change"] = np.nan

    if funding_history and len(funding_history) >= 2 and funding_rate is not None:
        fr_arr = np.array(funding_history, dtype=np.float64)
        base["funding_rate"] = _rolling_zscore_last(fr_arr, window=_ZSCORE_WINDOW_OI)
    else:
        base["funding_rate"] = np.nan

    if oi_history and len(oi_history) >= 5 and oi_change_ma5 is not None:
        # OI MA5 히스토리로 z-score
        oi_arr = np.array(oi_history, dtype=np.float64)
        ma5 = pd.Series(oi_arr).rolling(5, min_periods=1).mean().values
        base["oi_change_ma5"] = _rolling_zscore_last(ma5, window=_ZSCORE_WINDOW_OI)
    else:
        base["oi_change_ma5"] = np.nan

    base["oi_price_spread"] = float(oi_price_spread) if oi_price_spread is not None else np.nan
    base["adx"] = adx_z

    return pd.Series(base)
