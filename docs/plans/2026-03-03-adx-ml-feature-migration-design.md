# ADX ML 피처 마이그레이션 설계

**Goal:** ADX 하드필터(< 25)를 제거하고, ADX를 ML 피처로 추가하여 횡보장 판단을 ML 모델에 위임한다.

**Background:** 운영 로그 분석 결과, ADX < 25 하드필터가 하루 종일 시그널을 차단하여 ML 필터가 평가할 기회 자체가 없었음. ADX 10~24 구간에서도 수익 가능한 패턴이 존재할 수 있으나, 현재 구조에서는 ML이 이를 학습할 수 없음.

**Tech Stack:** LightGBM, pandas-ta (기존 사용 중)

---

## 변경 사항

### 1. ML 피처에 ADX 추가 (23 → 24 피처)

- `src/ml_features.py`: `FEATURE_COLS`에 `"adx"` 추가
- `build_features()`: ADX 값 추출 로직 추가

### 2. 데이터셋 빌더에서 ADX 하드필터 제거

- `src/dataset_builder.py`: `_calc_signals()`에서 ADX < 25 → HOLD 강제 로직 제거
- ADX 낮은 구간의 시그널도 학습 데이터에 포함됨

### 3. indicators.py ADX 하드필터 제거

- `src/indicators.py`: `get_signal()`에서 ADX < 25 early-return 제거
- ADX 값은 항상 로그에 남김 (대시보드 표시용)

### 4. ADX 로깅 개선

- ADX ≥ 25일 때도 로그 출력 → 대시보드에서 ADX 차트 끊김 해소

### 5. 테스트 업데이트

- ADX 하드필터 관련 기존 테스트 수정/제거
- ML 피처에 ADX 포함 확인 테스트 추가

## 데이터 흐름 (변경 후)

```
캔들 → get_signal() → 지표 가중치 기반 LONG/SHORT/HOLD (ADX 필터 없음)
     → ADX 값 항상 로그 출력
     → signal != HOLD → build_features() [24 피처, ADX 포함]
     → ML 필터 (threshold ≥ 0.55) → 진입 판단
```

## 주의 사항

- 기존 학습된 모델(23 피처)은 24 피처 입력과 호환 안 됨 → **재학습 필수**
- 재학습 전까지 봇 운영 불가 → 배포 시 `train_and_deploy.sh` 먼저 실행
