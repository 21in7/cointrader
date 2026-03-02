"""
전체 시계열을 1회 계산하는 벡터화 데이터셋 빌더.
pandas_ta를 130,000번 반복 호출하는 기존 방식 대신
전체 배열에 1번만 적용해 10~30배 속도를 낸다.

봇 실시간 경로(indicators.py, ml_features.py)는 변경하지 않는다.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.ml_features import FEATURE_COLS

LOOKAHEAD    = 24   # 15분봉 × 24 = 6시간 뷰
ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 2.0
WARMUP       = 60   # 15분봉 기준 60캔들 = 15시간 (지표 안정화 충분)


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """전체 시계열에 기술 지표를 1회 계산한다."""
    d = df.copy()
    close  = d["close"]
    high   = d["high"]
    low    = d["low"]
    volume = d["volume"]

    d["rsi"]  = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    d["macd"]        = macd["MACD_12_26_9"]
    d["macd_signal"] = macd["MACDs_12_26_9"]
    d["macd_hist"]   = macd["MACDh_12_26_9"]

    bb = ta.bbands(close, length=20, std=2)
    d["bb_upper"] = bb["BBU_20_2.0_2.0"]
    d["bb_lower"] = bb["BBL_20_2.0_2.0"]

    d["ema9"]  = ta.ema(close, length=9)
    d["ema21"] = ta.ema(close, length=21)
    d["ema50"] = ta.ema(close, length=50)

    d["atr"]      = ta.atr(high, low, close, length=14)
    d["vol_ma20"] = ta.sma(volume, length=20)

    stoch = ta.stochrsi(close, length=14)
    d["stoch_k"] = stoch["STOCHRSIk_14_14_3_3"]
    d["stoch_d"] = stoch["STOCHRSId_14_14_3_3"]

    # ADX (14) — 횡보장 필터
    adx_df = ta.adx(high, low, close, length=14)
    d["adx"] = adx_df["ADX_14"]

    return d


def _calc_signals(d: pd.DataFrame) -> np.ndarray:
    """
    indicators.py get_signal() 로직을 numpy 배열 연산으로 재현한다.
    반환: signal_arr — 각 행에 대해 "LONG" | "SHORT" | "HOLD"
    """
    n = len(d)

    rsi      = d["rsi"].values
    macd     = d["macd"].values
    macd_sig = d["macd_signal"].values
    close    = d["close"].values
    bb_upper = d["bb_upper"].values
    bb_lower = d["bb_lower"].values
    ema9     = d["ema9"].values
    ema21    = d["ema21"].values
    ema50    = d["ema50"].values
    stoch_k  = d["stoch_k"].values
    stoch_d  = d["stoch_d"].values
    volume   = d["volume"].values
    vol_ma20 = d["vol_ma20"].values

    # MACD 크로스: 전 캔들과 비교 (shift(1))
    prev_macd     = np.roll(macd, 1);     prev_macd[0]     = np.nan
    prev_macd_sig = np.roll(macd_sig, 1); prev_macd_sig[0] = np.nan

    long_score  = np.zeros(n, dtype=np.float32)
    short_score = np.zeros(n, dtype=np.float32)

    # 1. RSI
    long_score  += (rsi < 35).astype(np.float32)
    short_score += (rsi > 65).astype(np.float32)

    # 2. MACD 크로스 (가중치 2)
    macd_cross_up   = (prev_macd < prev_macd_sig) & (macd > macd_sig)
    macd_cross_down = (prev_macd > prev_macd_sig) & (macd < macd_sig)
    long_score  += macd_cross_up.astype(np.float32)   * 2
    short_score += macd_cross_down.astype(np.float32) * 2

    # 3. 볼린저 밴드
    long_score  += (close < bb_lower).astype(np.float32)
    short_score += (close > bb_upper).astype(np.float32)

    # 4. EMA 정배열/역배열
    long_score  += ((ema9 > ema21) & (ema21 > ema50)).astype(np.float32)
    short_score += ((ema9 < ema21) & (ema21 < ema50)).astype(np.float32)

    # 5. Stochastic RSI
    long_score  += ((stoch_k < 20) & (stoch_k > stoch_d)).astype(np.float32)
    short_score += ((stoch_k > 80) & (stoch_k < stoch_d)).astype(np.float32)

    # 6. 거래량 급증
    vol_surge = volume > vol_ma20 * 1.5

    long_enter  = (long_score  >= 3) & (vol_surge | (long_score  >= 4))
    short_enter = (short_score >= 3) & (vol_surge | (short_score >= 4))

    signal_arr = np.full(n, "HOLD", dtype=object)
    signal_arr[long_enter]  = "LONG"
    signal_arr[short_enter] = "SHORT"
    # 둘 다 해당하면 HOLD (충돌 방지)
    signal_arr[long_enter & short_enter] = "HOLD"

    # ADX 횡보장 필터: ADX < 25이면 추세 부재로 판단하여 진입 차단
    if "adx" in d.columns:
        adx = d["adx"].values
        low_adx = (~np.isnan(adx)) & (adx < 25)
        signal_arr[low_adx] = "HOLD"

    return signal_arr


def _rolling_zscore(arr: np.ndarray, window: int = 288) -> np.ndarray:
    """rolling window z-score 정규화. nan은 전파된다(nan-safe).
    15분봉 기준 3일(288캔들) 윈도우. min_periods=1로 초반 데이터도 활용."""
    s = pd.Series(arr.astype(np.float64))
    r = s.rolling(window=window, min_periods=1)
    mean = r.mean()   # pandas rolling은 nan을 자동으로 건너뜀
    std  = r.std(ddof=0)
    std  = std.where(std >= 1e-8, other=1e-8)
    z = (s - mean) / std
    return z.values.astype(np.float32)


def _calc_features_vectorized(
    d: pd.DataFrame,
    signal_arr: np.ndarray,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    신호 발생 인덱스에서 ml_features.py build_features() 로직을
    pandas 벡터 연산으로 재현한다.
    """
    close    = d["close"]
    bb_upper = d["bb_upper"]
    bb_lower = d["bb_lower"]
    ema9     = d["ema9"]
    ema21    = d["ema21"]
    ema50    = d["ema50"]
    atr      = d["atr"]
    volume   = d["volume"]
    vol_ma20 = d["vol_ma20"]
    rsi      = d["rsi"]
    macd_hist = d["macd_hist"]
    stoch_k  = d["stoch_k"]
    stoch_d  = d["stoch_d"]
    macd     = d["macd"]
    macd_sig = d["macd_signal"]

    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / (bb_range + 1e-8)

    ema_align = np.where(
        (ema9 > ema21) & (ema21 > ema50),  1,
        np.where(
            (ema9 < ema21) & (ema21 < ema50), -1, 0
        )
    ).astype(np.float32)

    atr_pct   = atr / (close + 1e-8)
    vol_ratio = volume / (vol_ma20 + 1e-8)

    ret_1 = close.pct_change(1).fillna(0).values
    ret_3 = close.pct_change(3).fillna(0).values
    ret_5 = close.pct_change(5).fillna(0).values

    # 절대값 피처를 rolling z-score로 정규화 (레짐 변화에 강하게)
    atr_pct_z   = _rolling_zscore(atr_pct)
    vol_ratio_z = _rolling_zscore(vol_ratio)
    ret_1_z     = _rolling_zscore(ret_1)
    ret_3_z     = _rolling_zscore(ret_3)
    ret_5_z     = _rolling_zscore(ret_5)

    prev_macd     = macd.shift(1).fillna(0).values
    prev_macd_sig = macd_sig.shift(1).fillna(0).values

    # signal_strength: 신호 방향별로 각 조건 점수 합산
    is_long  = (signal_arr == "LONG")
    is_short = (signal_arr == "SHORT")

    strength = np.zeros(len(d), dtype=np.float32)

    # LONG 조건
    strength += is_long * (rsi.values < 35).astype(np.float32)
    strength += is_long * ((prev_macd < prev_macd_sig) & (macd.values > macd_sig.values)).astype(np.float32) * 2
    strength += is_long * (close.values < bb_lower.values).astype(np.float32)
    strength += is_long * (ema_align == 1).astype(np.float32)
    strength += is_long * ((stoch_k.values < 20) & (stoch_k.values > stoch_d.values)).astype(np.float32)

    # SHORT 조건
    strength += is_short * (rsi.values > 65).astype(np.float32)
    strength += is_short * ((prev_macd > prev_macd_sig) & (macd.values < macd_sig.values)).astype(np.float32) * 2
    strength += is_short * (close.values > bb_upper.values).astype(np.float32)
    strength += is_short * (ema_align == -1).astype(np.float32)
    strength += is_short * ((stoch_k.values > 80) & (stoch_k.values < stoch_d.values)).astype(np.float32)

    side = np.where(signal_arr == "LONG", 1.0, 0.0).astype(np.float32)

    result = pd.DataFrame({
        "rsi":             rsi.values.astype(np.float32),
        "macd_hist":       macd_hist.values.astype(np.float32),
        "bb_pct":          bb_pct.astype(np.float32),
        "ema_align":       ema_align,
        "stoch_k":         stoch_k.values.astype(np.float32),
        "stoch_d":         stoch_d.values.astype(np.float32),
        "atr_pct":         atr_pct_z,
        "vol_ratio":       vol_ratio_z,
        "ret_1":           ret_1_z,
        "ret_3":           ret_3_z,
        "ret_5":           ret_5_z,
        "signal_strength": strength,
        "side":            side,
        "_signal":         signal_arr,   # 레이블 계산용 임시 컬럼
    }, index=d.index)

    # BTC/ETH 피처 계산 (제공된 경우)
    if btc_df is not None and eth_df is not None:
        btc_ret_1 = btc_df["close"].pct_change(1).fillna(0).values
        btc_ret_3 = btc_df["close"].pct_change(3).fillna(0).values
        btc_ret_5 = btc_df["close"].pct_change(5).fillna(0).values
        eth_ret_1 = eth_df["close"].pct_change(1).fillna(0).values
        eth_ret_3 = eth_df["close"].pct_change(3).fillna(0).values
        eth_ret_5 = eth_df["close"].pct_change(5).fillna(0).values

        def _align(arr: np.ndarray, target_len: int) -> np.ndarray:
            if len(arr) >= target_len:
                return arr[-target_len:]
            return np.concatenate([np.zeros(target_len - len(arr)), arr])

        n = len(d)
        btc_r1 = _align(btc_ret_1, n).astype(np.float32)
        btc_r3 = _align(btc_ret_3, n).astype(np.float32)
        btc_r5 = _align(btc_ret_5, n).astype(np.float32)
        eth_r1 = _align(eth_ret_1, n).astype(np.float32)
        eth_r3 = _align(eth_ret_3, n).astype(np.float32)
        eth_r5 = _align(eth_ret_5, n).astype(np.float32)

        xrp_r1 = ret_1.astype(np.float32)
        xrp_btc_rs_raw = np.divide(
            xrp_r1, btc_r1,
            out=np.zeros_like(xrp_r1),
            where=(btc_r1 != 0),
        ).astype(np.float32)
        xrp_eth_rs_raw = np.divide(
            xrp_r1, eth_r1,
            out=np.zeros_like(xrp_r1),
            where=(eth_r1 != 0),
        ).astype(np.float32)

        extra = pd.DataFrame({
            "btc_ret_1":  _rolling_zscore(btc_r1),
            "btc_ret_3":  _rolling_zscore(btc_r3),
            "btc_ret_5":  _rolling_zscore(btc_r5),
            "eth_ret_1":  _rolling_zscore(eth_r1),
            "eth_ret_3":  _rolling_zscore(eth_r3),
            "eth_ret_5":  _rolling_zscore(eth_r5),
            "xrp_btc_rs": _rolling_zscore(xrp_btc_rs_raw),
            "xrp_eth_rs": _rolling_zscore(xrp_eth_rs_raw),
        }, index=d.index)
        result = pd.concat([result, extra], axis=1)

    # OI 변화율 / 펀딩비 피처
    # 컬럼 없으면 전체 nan, 있으면 0.0 구간(데이터 미제공 구간)을 nan으로 마스킹
    # LightGBM은 nan을 자체 처리; MLX는 fit()에서 nanmean/nanstd + nan_to_num 처리
    if "oi_change" in d.columns:
        oi_raw = np.where(d["oi_change"].values == 0.0, np.nan, d["oi_change"].values)
    else:
        oi_raw = np.full(len(d), np.nan)

    if "funding_rate" in d.columns:
        fr_raw = np.where(d["funding_rate"].values == 0.0, np.nan, d["funding_rate"].values)
    else:
        fr_raw = np.full(len(d), np.nan)

    result["oi_change"]    = _rolling_zscore(oi_raw.astype(np.float64))
    result["funding_rate"] = _rolling_zscore(fr_raw.astype(np.float64))

    return result


def _calc_labels_vectorized(
    d: pd.DataFrame,
    feat: pd.DataFrame,
    sig_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    label_builder.py build_labels() 로직을 numpy 2D 배열로 벡터화한다.

    각 신호 인덱스 i에 대해 future[i+1 : i+1+LOOKAHEAD] 구간의
    high/low 배열을 (N × LOOKAHEAD) 행렬로 만들어 argmax로 처리한다.
    """
    n_total = len(d)
    highs   = d["high"].values
    lows    = d["low"].values
    closes  = d["close"].values
    atrs    = d["atr"].values

    labels = []
    valid_mask = []

    for idx in sig_idx:
        signal = feat.at[d.index[idx], "_signal"]
        entry  = closes[idx]
        atr    = atrs[idx]
        if atr <= 0:
            valid_mask.append(False)
            continue

        if signal == "LONG":
            sl = entry - atr * ATR_SL_MULT
            tp = entry + atr * ATR_TP_MULT
        else:
            sl = entry + atr * ATR_SL_MULT
            tp = entry - atr * ATR_TP_MULT

        end = min(idx + 1 + LOOKAHEAD, n_total)
        fut_high = highs[idx + 1 : end]
        fut_low  = lows[idx + 1 : end]

        label = 0  # 미도달(타임아웃) 시 실패로 간주

        for h, l in zip(fut_high, fut_low):
            if signal == "LONG":
                if l <= sl:
                    label = 0
                    break
                if h >= tp:
                    label = 1
                    break
            else:  # SHORT
                if h >= sl:
                    label = 0
                    break
                if l <= tp:
                    label = 1
                    break

        labels.append(label)
        valid_mask.append(True)

    return np.array(labels, dtype=np.int8), np.array(valid_mask, dtype=bool)


def generate_dataset_vectorized(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    eth_df: pd.DataFrame | None = None,
    time_weight_decay: float = 0.0,
    negative_ratio: int = 0,
) -> pd.DataFrame:
    """
    전체 시계열을 1회 계산해 학습 데이터셋을 생성한다.
    기존 generate_dataset()의 drop-in 대체제.
    btc_df, eth_df가 제공되면 21개 피처로 확장한다.

    time_weight_decay: 지수 감쇠 강도. 0이면 균등 가중치.
        양수일수록 최신 샘플에 더 높은 가중치를 부여한다.
        예) 2.0 → 최신 샘플이 가장 오래된 샘플보다 e^2 ≈ 7.4배 높은 가중치.
        결과 DataFrame에 'sample_weight' 컬럼으로 포함된다.

    negative_ratio: 시그널 샘플 대비 HOLD negative 샘플 비율.
        0이면 기존 동작 (시그널만). 5면 시그널의 5배만큼 HOLD 샘플 추가.
    """
    print("  [1/3] 전체 시계열 지표 계산 (1회)...")
    d = _calc_indicators(df)

    print("  [2/3] 신호 마스킹 및 피처 추출...")
    signal_arr = _calc_signals(d)
    feat_all   = _calc_features_vectorized(d, signal_arr, btc_df=btc_df, eth_df=eth_df)

    # 신호 발생 + NaN 없음 + 미래 데이터 충분한 인덱스만
    OPTIONAL_COLS = {"oi_change", "funding_rate"}
    available_cols_for_nan_check = [
        c for c in FEATURE_COLS
        if c in feat_all.columns and c not in OPTIONAL_COLS
    ]
    base_valid = (
        (~feat_all[available_cols_for_nan_check].isna().any(axis=1).values) &
        (np.arange(len(d)) >= WARMUP) &
        (np.arange(len(d)) < len(d) - LOOKAHEAD)
    )

    # --- 시그널 캔들 (기존 로직) ---
    sig_valid = base_valid & (signal_arr != "HOLD")
    sig_idx = np.where(sig_valid)[0]
    print(f"  신호 발생 인덱스: {len(sig_idx):,}개")

    print("  [3/3] 레이블 계산...")
    labels, valid_mask = _calc_labels_vectorized(d, feat_all, sig_idx)

    final_sig_idx = sig_idx[valid_mask]
    available_feature_cols = [c for c in FEATURE_COLS if c in feat_all.columns]
    feat_signal = feat_all.iloc[final_sig_idx][available_feature_cols].copy()
    feat_signal["label"] = labels
    feat_signal["source"] = "signal"

    # --- HOLD negative 캔들 ---
    if negative_ratio > 0 and len(final_sig_idx) > 0:
        hold_valid = base_valid & (signal_arr == "HOLD")
        hold_candidates = np.where(hold_valid)[0]
        n_neg = min(len(hold_candidates), len(final_sig_idx) * negative_ratio)

        if n_neg > 0:
            rng = np.random.default_rng(42)
            hold_idx = rng.choice(hold_candidates, size=n_neg, replace=False)
            hold_idx = np.sort(hold_idx)

            feat_hold = feat_all.iloc[hold_idx][available_feature_cols].copy()
            feat_hold["label"] = 0
            feat_hold["source"] = "hold_negative"

            # HOLD 캔들은 시그널이 없으므로 side를 랜덤 할당 (50:50)
            sides = rng.integers(0, 2, size=len(feat_hold)).astype(np.float32)
            feat_hold["side"] = sides

            print(f"  HOLD negative 추가: {len(feat_hold):,}개 "
                  f"(비율 1:{negative_ratio})")

            feat_final = pd.concat([feat_signal, feat_hold], ignore_index=True)
            # 시간 순서 복원 (원본 인덱스 기반 정렬)
            original_order = np.concatenate([final_sig_idx, hold_idx])
            sort_order = np.argsort(original_order)
            feat_final = feat_final.iloc[sort_order].reset_index(drop=True)
        else:
            feat_final = feat_signal.reset_index(drop=True)
    else:
        feat_final = feat_signal.reset_index(drop=True)

    # 시간 가중치
    n = len(feat_final)
    if time_weight_decay > 0 and n > 1:
        weights = np.exp(time_weight_decay * np.linspace(0.0, 1.0, n)).astype(np.float32)
        weights /= weights.mean()
        print(f"  시간 가중치 적용 (decay={time_weight_decay}): "
              f"min={weights.min():.3f}, max={weights.max():.3f}")
    else:
        weights = np.ones(n, dtype=np.float32)

    feat_final["sample_weight"] = weights

    total_sig = (feat_final["source"] == "signal").sum() if "source" in feat_final.columns else len(feat_final)
    total_hold = (feat_final["source"] == "hold_negative").sum() if "source" in feat_final.columns else 0
    print(f"  최종 데이터셋: {n:,}개 (시그널={total_sig:,}, HOLD={total_hold:,})")

    return feat_final
