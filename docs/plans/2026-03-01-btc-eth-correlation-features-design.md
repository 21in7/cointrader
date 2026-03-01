# BTC/ETH 상관관계 피처 추가 설계 문서

**날짜:** 2026-03-01

## 목적

XRP 선물 ML 필터에 BTC/ETH 캔들 데이터를 추가 피처로 활용하여 모델 예측 정확도를 향상시킨다. XRP는 BTC/ETH의 움직임에 강하게 연동되는 경향이 있으므로, 이 컨텍스트를 ML 피처로 명시적으로 제공한다.

---

## 아키텍처 개요

### 변경 전

```
KlineStream(XRPUSDT) → bot.process_candle() → Indicators → MLFilter(13개 피처)
```

### 변경 후

```
MultiSymbolStream(XRP+BTC+ETH, Combined WebSocket)
    ↓ XRP 캔들 닫힐 때
bot.process_candle(xrp_df, btc_df, eth_df)
    ↓
Indicators(XRP) → build_features(xrp_df, btc_df, eth_df, signal)
    ↓
MLFilter(13 + 8 = 21개 피처)
```

---

## 추가 피처 8개

| 피처 | 설명 |
|---|---|
| `btc_ret_1` | BTC 1캔들 수익률 |
| `btc_ret_3` | BTC 3캔들 수익률 |
| `btc_ret_5` | BTC 5캔들 수익률 |
| `eth_ret_1` | ETH 1캔들 수익률 |
| `eth_ret_3` | ETH 3캔들 수익률 |
| `eth_ret_5` | ETH 5캔들 수익률 |
| `xrp_btc_rs` | XRP ret_1 / BTC ret_1 (XRP 상대강도 vs BTC) |
| `xrp_eth_rs` | XRP ret_1 / ETH ret_1 (XRP 상대강도 vs ETH) |

기존 13개 피처(`rsi`, `macd_hist`, `bb_pct`, `ema_align`, `stoch_k`, `stoch_d`, `atr_pct`, `vol_ratio`, `ret_1`, `ret_3`, `ret_5`, `signal_strength`, `side`)는 그대로 유지.

---

## 변경 파일 목록

| 파일 | 변경 유형 | 내용 |
|---|---|---|
| `src/data_stream.py` | 수정 | `KlineStream` → `MultiSymbolStream` (Combined WebSocket) |
| `src/ml_features.py` | 수정 | `build_features(xrp_df, btc_df, eth_df, signal)` — 피처 21개로 확장 |
| `scripts/fetch_history.py` | 수정 | BTC/ETH 동시 수집 후 타임스탬프 기준 병합 저장 |
| `scripts/train_model.py` | 수정 | 병합된 데이터셋으로 21개 피처 학습 |
| `src/bot.py` | 수정 | `MultiSymbolStream` 사용, `process_candle`에 btc_df/eth_df 전달 |
| `src/dataset_builder.py` | 수정 | 레이블 생성 시 BTC/ETH 피처 포함 |

---

## 데이터 흐름

### 실시간 (봇 운영)

```
Binance Combined WebSocket
  ├── xrpusdt@kline_1m  →  xrp_buffer (deque 200)
  ├── btcusdt@kline_1m  →  btc_buffer (deque 200)
  └── ethusdt@kline_1m  →  eth_buffer (deque 200)
         ↓ XRP 캔들 닫힐 때만 트리거
  bot.process_candle(xrp_df, btc_df, eth_df)
```

### 학습 데이터 수집

```
fetch_history.py → XRPUSDT + BTCUSDT + ETHUSDT 각 90일 수집
                 → 타임스탬프 기준 inner join 병합
                 → data/combined_1m.parquet 저장
train_model.py   → 21개 피처로 LightGBM 재학습
                 → models/lgbm_filter.pkl 교체
```

---

## 에러 처리

| 상황 | 처리 방법 |
|---|---|
| BTC/ETH 버퍼 50개 미만 (봇 시작 직후) | btc/eth 피처 전부 0.0으로 채움, 거래는 정상 진행 |
| Combined WebSocket 연결 끊김 | 예외 발생 → 상위에서 재연결 |
| BTC/ETH ret 분모가 0 | `xrp_btc_rs`, `xrp_eth_rs` = 0.0으로 처리 |
| 기존 모델(13개 피처) 파일이 남아있는 경우 | 피처 수 불일치 → MLFilter 폴백(신호 통과)으로 자동 처리 |

---

## 재학습 순서

기존 `lgbm_filter.pkl`(13개 피처)은 새 데이터셋(21개 피처) 재학습 후 자동 교체된다.
**봇 재시작 전 반드시 아래 순서로 실행:**

```bash
# 1. 3심볼 과거 데이터 수집
python scripts/fetch_history.py --symbols XRPUSDT BTCUSDT ETHUSDT --days 90

# 2. 21피처 모델 재학습
python scripts/train_model.py

# 3. 봇 재시작
```

---

## 폴백 정책

- BTC/ETH 버퍼가 비어있으면 해당 피처를 0.0으로 채워 기존 XRP 피처만으로 동작
- 모델 파일이 없으면 ML 필터 전체를 건너뜀 (기존 정책 유지)
