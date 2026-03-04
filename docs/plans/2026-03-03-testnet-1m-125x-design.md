# Demo 1분봉 125x 트레이딩 설계

**날짜**: 2026-03-03
**상태**: Approved (testnet → demo 변경 반영)

## 목적

바이낸스 선물 데모(`demo-fapi.binance.com`)에서 XRPUSDT 1분봉, 125x 레버리지로 ML 기반 자동매매를 테스트한다.
로컬 맥미니의 워크트리에서 격리하여 메인 코드베이스에 영향 없이 실험한다.

## 환경 설정

| 항목 | 값 |
|------|-----|
| 네트워크 | Binance Futures Demo (`demo-fapi.binance.com`) |
| 심볼 | XRPUSDT |
| 타임프레임 | 1m (1분봉) |
| 레버리지 | 125x |
| ML Lookahead | 60캔들 (1시간) |
| 작업 방식 | git worktree (격리) |
| 실행 환경 | 로컬 맥미니 (서버 배포 없음) |

## 코드 변경 사항

### 1. Config (`src/config.py`)

- `demo: bool` 플래그 추가
- `BINANCE_DEMO=true`이면 `BINANCE_DEMO_API_KEY/SECRET` 사용
- `INTERVAL` 환경변수 추가 (기본값 `15m` → 데모에서 `1m`)

### 2. Exchange (`src/exchange.py`)

- `config.demo=True`이면 Client의 `FUTURES_URL`을 `demo-fapi.binance.com`으로 오버라이드
- `testnet=True` 미사용 (demo 엔드포인트는 라이브러리 미지원)

### 3. DataStream (`src/data_stream.py`)

- `AsyncClient.create()` 후 demo이면 `FUTURES_URL` 오버라이드
- interval을 Config에서 받도록 수정

### 4. UserDataStream (`src/user_data_stream.py`)

- `AsyncClient.create()` 후 demo이면 `FUTURES_URL` 오버라이드

### 5. Bot (`src/bot.py`)

- `demo` 플래그를 각 stream/exchange에 전달

### 6. 학습 파이프라인

- `fetch_history.py`로 1분봉 데이터 수집 (30일+, 프로덕션 API 사용)
- `dataset_builder.py`에서 `LOOKAHEAD=60` (1시간)
- SL/TP: ATR 기반이므로 자동 적응
- LightGBM 학습 → 로컬 models/ 저장 (서버 배포 없음)

### 7. 환경변수 (`.env`)

```
BINANCE_DEMO=true
BINANCE_DEMO_API_KEY=<demo_key>
BINANCE_DEMO_API_SECRET=<demo_secret>
INTERVAL=1m
LEVERAGE=125
```

## 변경하지 않는 것

- 지표 계산 로직 (RSI, MACD, BB, EMA, StochRSI, ATR, ADX) — 타임프레임 독립
- ML 피처 추출 — 캔들 데이터 기반, 그대로 동작
- 리스크 매니저 — 비율 기반, 자동 적응
- Discord 알림 — 그대로 사용
- ONNX 변환 파이프라인 — 동일
