# 코드 리뷰 개선 사항

**날짜**: 2026-03-07
**상태**: 부분 완료 (#1/#2/#4/#5/#6/#8 완료, #9 보류, #3/#7/#10~13 다음 스프린트)

## 목표

전체 코드베이스 리뷰에서 발견된 버그, 엣지 케이스, 로직 오류를 우선순위별로 정리하고 수정한다.

---

## Critical (즉시 수정 필요)

### 1. OI 변화율 계산 시 Division by Zero

**파일**: `src/bot.py:120`

`_prev_oi`가 0.0일 때 `(current_oi - self._prev_oi) / self._prev_oi`에서 ZeroDivisionError 발생. `get_open_interest()` 실패 시 0.0을 반환하므로 실제로 발생 가능.

**수정**: `_prev_oi == 0.0`이면 `oi_change = 0.0`으로 처리.

### 2. 누적 트레이드 수 계산 로직 오류

**파일**: `scripts/weekly_report.py:415-423`

```python
# 현재 (잘못됨) — max()로 비교하여 누적이 아닌 최대값만 가져옴
cumulative = live_count
for rpath in sorted(rdir.glob("report_*.json")):
    cumulative = max(cumulative, prev.get("live_trades", {}).get("count", 0))
```

ML 재학습 트리거 조건(`≥ 150건`)이 제대로 작동하지 않음.

**수정**: 이전 리포트의 `live_trades.count`를 합산하도록 변경.

---

## Important (이번 주 수정 권장)

### 3. Training-Serving Skew (OI/펀딩비 피처)

**파일**: `src/dataset_builder.py` vs `src/ml_features.py`

- 학습 시: OI=0 구간을 NaN으로 마스킹 후 z-score
- 서빙 시: OI 값을 그대로 NaN으로 설정

ML 활성화 시 학습/서빙 간 피처 분포 불일치 발생. 현재 ML OFF이므로 당장은 영향 없지만, ML 재활성화 전 반드시 수정 필요.

### 4. `fetch_history.py` — API 실패/Rate Limit 미처리

**파일**: `scripts/fetch_history.py:46-61`

`futures_klines()` 호출에 retry 로직이 없음. Rate limit(429) 발생 시 예외로 크래시. `weekly_report.py`의 subprocess가 무한 대기할 수 있음.

**수정**: `tenacity` 또는 수동 retry 로직 추가 (최대 3회, exponential backoff).

### 5. Parquet Upsert 시 중복 타임스탬프 미제거

**파일**: `scripts/fetch_history.py:314`

`sort_index()`만 하고 `drop_duplicates()`를 하지 않음. API 응답에 중복 타임스탬프가 있으면 지표 계산이 이중 계산됨.

**수정**: `sort_index()` 앞에 `df[~df.index.duplicated(keep='last')]` 추가.

### 6. `record_pnl()`에 asyncio.Lock 미사용

**파일**: `src/risk_manager.py:55`

`record_pnl()`이 `self.daily_pnl`을 수정하지만 `async with self._lock`을 사용하지 않음. 멀티심볼 환경에서 동시 호출 시 일일 손실 한도 체크가 부정확할 수 있음.

**수정**: `record_pnl()`을 async로 변경하고 `async with self._lock:` 추가.

### 7. 백테스터 Equity Curve 미구현

**파일**: `src/backtester.py:509-510`

`_record_equity()`가 `pass`로 비어 있음. MDD 계산이 실현 PnL 기준이지 포트폴리오 가치(미실현 PnL 포함) 기준이 아님. MDD가 과소평가될 수 있음.

**수정**: 미실현 PnL을 포함한 equity 계산 구현.

### 8. User Data Stream — exit_price 기본값 0.0

**파일**: `src/user_data_stream.py:95`

`order.get("ap", "0")`에서 필드 누락 시 exit_price=0.0으로 설정되어 PnL이 완전히 잘못 계산됨.

**수정**: `exit_price == 0.0`이면 청산 처리를 스킵하고 WARNING 로그 출력.

---

## Minor (다음 스프린트)

### 9. 거래량 급증 진입 조건 의도 불일치

**파일**: `src/indicators.py:115-118`

`(vol_surge or long_signals >= signal_threshold + 1)` — 거래량 급증만으로도 진입 허용됨. "강한 신호 + 거래량 급증"이 의도라면 AND 조건이어야 하는데, 현재 OR로 구현됨. 현재 전략 파라미터 스윕 결과(ADX=25, Vol=2.5)에서는 큰 문제 없으나, 의도를 확인하고 정리 필요.

### 10. ML 모델 피처 불일치 시 Silent Failure

**파일**: `src/ml_filter.py:152`

ONNX 모델과 현재 FEATURE_COLS가 다르면 예외를 잡고 `False`를 반환(모든 신호 차단). 사용자에게 원인이 보이지 않아 디버깅이 어려움.

**수정**: 피처 수 불일치는 WARNING이 아닌 ERROR로 로깅하고, 최초 발생 시 Discord 알림 전송.

### 11. `train_model.py` — 빈 데이터셋 미처리

**파일**: `scripts/train_model.py:196`

`generate_dataset_vectorized()`가 빈 DataFrame을 반환하면 Walk-Forward 검증에서 step=0이 되어 무한 루프 가능.

**수정**: 빈 데이터셋 시 `ValueError("No samples generated")` raise.

### 12. `data_stream.py` — AsyncClient 생성 실패 시 전체 크래시

**파일**: `src/data_stream.py:79-82`

네트워크 단절 상태에서 봇 시작 시 `AsyncClient.create()` 실패로 모든 심볼이 함께 크래시.

**수정**: retry with exponential backoff (최대 5회) 추가.

### 13. `fetch_history.py` — Parquet 타임존 처리 불일치

**파일**: `scripts/fetch_history.py:286-289`

`tz_localize("UTC")` 호출 시 기존 데이터가 실제로 UTC인지 검증하지 않음. 타임존이 다른 데이터가 섞이면 OI/펀딩비 병합이 시간축으로 어긋남.

**수정**: `tz_localize(tz='UTC', ambiguous='raise', nonexistent='raise')` 사용.

---

## 수정 우선순위

| 우선순위 | 이슈 | 난이도 | 영향도 |
|---------|------|--------|--------|
| 즉시 | #1 OI division by zero | 5분 | 봇 크래시 |
| 즉시 | #2 누적 트레이드 계산 | 5분 | ML 트리거 오작동 |
| 이번주 | #4 fetch_history retry | 30분 | 데이터 수집 행 |
| 이번주 | #5 Parquet 중복 제거 | 5분 | 지표 이중 계산 |
| 이번주 | #6 record_pnl Lock | 5분 | 리스크 한도 부정확 |
| 이번주 | #8 exit_price=0 방어 | 10분 | PnL 오계산 |
| ML 재활성화 전 | #3 Training-Serving skew | 30분 | 예측 품질 저하 |
| 다음 스프린트 | #7 Equity curve 구현 | 1시간 | MDD 과소평가 |
| 다음 스프린트 | #9-13 기타 | 각 10-30분 | 안정성 개선 |
