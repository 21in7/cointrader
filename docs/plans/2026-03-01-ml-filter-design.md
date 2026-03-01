# ML 필터 설계 문서

**날짜:** 2026-03-01

## 목적

기존 규칙 기반 신호(LONG/SHORT/HOLD)가 발생했을 때, LightGBM 모델이 해당 진입이 수익으로 끝날 확률을 계산하여 낮은 확률의 진입을 차단하는 보조 필터를 구현한다.

---

## 아키텍처 개요

```
캔들 수신 → 기술 지표 계산 → 규칙 기반 신호(LONG/SHORT/HOLD)
                                          ↓
                              신호 != HOLD 일 때만
                                          ↓
                         [ML 필터] LightGBM.predict_proba()
                                          ↓
                         확률 >= 0.60 이면 진입 허용
                         확률 < 0.60 이면 진입 차단
```

---

## 레이블 정의

- **1 (성공):** 진입 후 `take_profit` 가격에 먼저 도달
- **0 (실패):** 진입 후 `stop_loss` 가격에 먼저 도달
- TP/SL 계산은 기존 `Indicators.get_atr_stop()` 재사용 (ATR 기반)

---

## 피처 목록

| 피처 | 설명 |
|---|---|
| `rsi` | RSI(14) |
| `macd_hist` | MACD 히스토그램 |
| `bb_pct` | 볼린저밴드 내 가격 위치 (0~1) |
| `ema_align` | EMA 정배열 여부 (1=정배열, -1=역배열, 0=혼재) |
| `stoch_k` | Stochastic RSI K |
| `stoch_d` | Stochastic RSI D |
| `atr_pct` | ATR / 현재가 (변동성 비율) |
| `vol_ratio` | 거래량 / vol_ma20 |
| `ret_1` | 1캔들 전 대비 수익률 |
| `ret_3` | 3캔들 전 대비 수익률 |
| `ret_5` | 5캔들 전 대비 수익률 |
| `signal_strength` | 규칙 기반 신호 강도 (long/short_signals 수) |
| `side` | 신호 방향 (1=LONG, 0=SHORT) |

---

## 신규 컴포넌트

| 컴포넌트 | 파일 | 역할 |
|---|---|---|
| 피처 엔지니어링 | `src/ml_features.py` | 기술 지표 → ML 피처 변환 |
| ML 필터 | `src/ml_filter.py` | 모델 로드 + 예측 + 폴백 |
| 재학습 스케줄러 | `src/retrainer.py` | 매일 새벽 재학습 트리거 |
| 데이터 수집 스크립트 | `scripts/fetch_history.py` | 바이낸스 과거 캔들 수집 |
| 학습 스크립트 | `scripts/train_model.py` | LightGBM 학습 + 저장 |

---

## 재학습 스케줄

- **초기:** `scripts/fetch_history.py` + `scripts/train_model.py` 수동 실행
- **이후:** 매일 새벽 3시 (KST) `retrainer.py`가 자동 실행
  - 새 모델 AUC > 기존 모델 AUC → 교체
  - 그렇지 않으면 기존 모델 유지 (롤백)
  - Discord 알림으로 결과 전송

---

## 모델 저장 구조

```
models/
├── lgbm_filter.pkl       ← 현재 사용 중인 모델
├── lgbm_filter_prev.pkl  ← 롤백용 이전 모델
└── training_log.json     ← 재학습 이력 (날짜, AUC, 샘플 수)
```

---

## 폴백 정책

`models/lgbm_filter.pkl` 파일이 없으면 ML 필터를 건너뛰고 기존 규칙 기반 신호 그대로 사용. 봇이 모델 없이도 정상 작동.

---

## bot.py 변경 범위

`process_candle()` 메서드에 3줄 추가:

```python
if signal != "HOLD" and self.ml_filter.is_model_loaded():
    features = build_features(df_with_indicators, signal)
    if not self.ml_filter.should_enter(features):
        signal = "HOLD"
```
