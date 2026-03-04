# OI 파생 피처 설계

## 목표

기존 `oi_change` 피처에 더해, OI 데이터에서 파생 피처 2개를 만들어 LightGBM 학습 데이터에 추가하고, 피처 추가 전후 검증셋 성능을 자동 비교한다.

## 제약사항

- Binance OI 히스토리 API는 최근 30일분만 제공
- 학습 데이터에서 OI 유효 구간 ≈ 2,880개 15분 캔들
- A/B 비교 결과는 방향성 참고용 (통계적 유의성 제한)

## 파생 피처

### 1. `oi_change_ma5`

- **계산**: OI 변화율의 5캔들(75분) 이동평균
- **의미**: OI 단기 추세. 급감/급증 노이즈 제거된 방향성
- **정규화**: rolling z-score (288캔들 윈도우, 기존 패턴 동일)
- **기존 `oi_change`와의 관계**: smoothed 버전. 상관관계 높을 수 있으나 LightGBM이 자연 선택. importance 낮으면 이후 제거

### 2. `oi_price_spread`

- **계산**: `rolling_zscore(oi_change) - rolling_zscore(price_ret_1)`
- **의미**: OI와 가격 움직임 간 괴리도 (연속값)
  - 양수: OI가 가격 대비 강세 (자금 유입)
  - 음수: OI가 가격 대비 약세 (자금 유출)
- **정규화**: 양쪽 입력이 이미 z-score이므로 추가 정규화 불필요
- **바이너리 대신 연속값 채택 이유**: sign() 기반 바이너리는 미미한 차이도 1/0으로 분류 → 노이즈 과잉. 연속값은 LightGBM이 분할점을 학습

## 수정 대상 파일

### dataset_builder.py

- OI 파생 피처 2개 계산 로직 추가
- 기존 `oi_change` z-score 결과를 재사용하여 `oi_change_ma5` 계산
- `oi_price_spread` = `oi_change` z-score - `ret_1` z-score

### ml_features.py

- `FEATURE_COLS`에 `oi_change_ma5`, `oi_price_spread` 추가 (24→26)
- `build_features()`에 실시간 계산 로직 추가
  - `oi_change_ma5`: bot에서 전달받은 최근 5봉 OI MA
  - `oi_price_spread`: 실시간 z-scored OI - z-scored price change

### train_model.py

- `--compare` 플래그 추가
- Baseline (기존 24피처) vs New (26피처) 자동 비교 출력:
  - Precision, Recall, F1, AUC-ROC
  - Feature importance top 10
  - Best threshold
  - 검증셋 크기 (n=XX) 및 "방향성 참고용" 경고

### bot.py

- OI 변화율 히스토리 deque(maxlen=5) 관리
- `_init_oi_history()`: 봇 시작 시 Binance OI hist API에서 최근 5봉 fetch → cold start 해결
- `_fetch_market_microstructure()` 확장: MA5 계산, price_spread 계산 후 build_features()에 전달

### exchange.py

- `get_oi_history(limit=5)`: 봇 초기화용 최근 OI 히스토리 fetch 메서드 추가

### scripts/collect_oi.py (신규)

- OI 장기 수집 스크립트
- 15분마다 cron 실행
- Binance `/fapi/v1/openInterest` 호출 → `data/oi_history.parquet`에 append
- 기존 fetch_history.py의 30일 데이터 보완용
