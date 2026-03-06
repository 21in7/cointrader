# CoinTrader

Binance Futures 자동매매 봇. 복합 기술 지표와 ML 필터(LightGBM / MLX 신경망)를 결합하여 다중 심볼(XRP, TRX, DOGE 등) 선물 포지션을 동시에 자동 진입·청산하며, Discord로 실시간 알림을 전송합니다.

> **아키텍처 문서**: 코드 구조, 레이어별 역할, MLOps 파이프라인, 동작 시나리오를 상세히 설명한 [ARCHITECTURE.md](./ARCHITECTURE.md)를 참고하세요.

---

## 주요 기능

- **복합 기술 지표 신호**: RSI, MACD 크로스, 볼린저 밴드, EMA 정/역배열, Stochastic RSI, 거래량 급증 — 가중치 합계 ≥ 3 시 진입
- **ML 필터 (ONNX 우선 / LightGBM 폴백)**: 기술 지표 신호를 한 번 더 검증하여 오진입 차단. 우선순위: ONNX > LightGBM > 폴백(항상 허용)
- **모델 핫리로드**: 캔들마다 모델 파일 mtime을 감지해 변경 시 자동 리로드 (봇 재시작 불필요)
- **멀티심볼 스트림**: XRP/BTC/ETH 3개 심볼을 단일 Combined WebSocket으로 수신, BTC·ETH 상관관계 피처 활용
- **26개 ML 피처**: XRP 기술 지표 13개 + BTC/ETH 수익률·상대강도 8개 + OI 변화율·펀딩비 2개 + OI 파생 피처 2개(oi_change_ma5, oi_price_spread) + ADX 1개 (캔들 마감 시 실시간 조회, 실패 시 0으로 폴백)
- **점진적 OI 데이터 축적 (Upsert)**: 바이낸스 OI 히스토리 API는 최근 30일치만 제공. `fetch_history.py` 실행 시 기존 parquet의 `oi_change/funding_rate=0` 구간을 신규 값으로 채워 학습 데이터 품질을 점진적으로 개선
- **실시간 OI/펀딩비 조회**: 캔들 마감마다 `get_open_interest()` / `get_funding_rate()`를 비동기 병렬 조회하여 ML 피처에 전달. 이전 캔들 대비 OI 변화율로 변환하여 train-serve skew 해소
- **ATR 기반 손절/익절**: 변동성에 따라 동적으로 SL/TP 계산 (기본 2.0× / 2.0× ATR, 환경변수로 설정 가능)
- **전략 파라미터 스윕**: 324개 파라미터 조합(SL/TP/ADX/신호임계값/거래량배수)을 Walk-Forward 백테스트로 체계적 탐색, 수익 구간 자동 발견
- **주간 전략 리포트**: 매주 자동으로 백테스트 성능 측정, 실전 로그 파싱, 추이 추적, ML 재학습 시점 판단, 성능 저하 시 대안 파라미터 스윕, Discord 알림
- **ML 필터 비활성화 모드**: `NO_ML_FILTER=true` 설정 시 ML 모델 로드 없이 기술 지표 신호만으로 운영 (현재 프로덕션 기본값 — 아래 "ML 필터 현황" 참고)
- **Algo Order API 지원**: 계정 설정에 따라 STOP_MARKET/TAKE_PROFIT_MARKET 주문을 `/fapi/v1/algoOrder` 엔드포인트로 자동 전송 (오류 코드 -4120 대응)
- **동적 증거금 비율**: 잔고 증가에 따라 선형 감소 (최대 50% → 최소 20%)
- **반대 시그널 재진입**: 보유 포지션과 반대 신호 발생 시 즉시 청산 후 ML 필터 통과 시 반대 방향 재진입
- **멀티심볼 동시 거래**: 심볼별 독립 봇 인스턴스를 `asyncio.gather()`로 병렬 실행. 공유 RiskManager로 글로벌 리스크 관리
- **리스크 관리**: 트레이드당 리스크 비율, 최대 포지션 수, 동일 방향 포지션 제한(기본 2개), 일일 손실 한도(5%) 제어
- **포지션 복구**: 봇 재시작 시 기존 포지션 자동 감지 및 상태 복원
- **실시간 TP/SL 감지**: Binance User Data Stream으로 TP/SL 작동을 즉시 감지 (캔들 마감 대기 없음)
- **순수익(Net PnL) 기록**: 바이낸스 `realizedProfit - commission`으로 정확한 순수익 계산
- **Discord 상세 청산 알림**: 예상 수익 vs 실제 순수익 + 슬리피지/수수료 차이 표시
- **listenKey 자동 갱신**: 30분 keepalive + 네트워크 단절 시 자동 재연결. `stream.recv()` 기반으로 수신하며, 라이브러리 내부 에러 페이로드(`{"e":"error"}`) 감지 시 즉시 재연결하여 좀비 커넥션 방지
- **Discord 알림**: 진입·청산·오류 이벤트 실시간 웹훅 알림
- **CI/CD**: Jenkins + Gitea Container Registry 기반 Docker 이미지 자동 빌드·배포 (LXC 운영 서버 자동 적용)

---

## 프로젝트 구조

```
cointrader/
├── main.py                    # 진입점 (심볼별 봇 인스턴스 생성 + asyncio.gather)
├── src/
│   ├── bot.py                 # 메인 트레이딩 루프 (심볼별 독립 인스턴스)
│   ├── config.py              # 환경변수 기반 설정 (symbols 리스트 지원)
│   ├── exchange.py            # Binance Futures API 클라이언트 (심볼별 독립)
│   ├── data_stream.py         # WebSocket 15분봉 멀티심볼 스트림
│   ├── indicators.py          # 기술 지표 계산 및 신호 생성
│   ├── ml_filter.py           # ML 필터 (ONNX 우선 / LightGBM 폴백 / 핫리로드)
│   ├── ml_features.py         # ML 피처 빌더 (26개 피처)
│   ├── mlx_filter.py          # MLX 신경망 필터 (Apple Silicon GPU 학습 + ONNX export)
│   ├── label_builder.py       # 학습 레이블 생성
│   ├── dataset_builder.py     # 벡터화 데이터셋 빌더 (학습용)
│   ├── risk_manager.py        # 공유 리스크 관리 (asyncio.Lock, 동일 방향 제한)
│   ├── notifier.py            # Discord 웹훅 알림
│   └── logger_setup.py        # Loguru 로거 설정
├── scripts/
│   ├── fetch_history.py       # 과거 데이터 수집 (--symbol 단일 / --symbols 다중)
│   ├── train_model.py         # LightGBM 모델 학습 (--symbol 지원)
│   ├── train_mlx_model.py     # MLX 신경망 학습 (Apple Silicon GPU)
│   ├── train_and_deploy.sh    # 전체 파이프라인 (--symbol / --all 지원)
│   ├── tune_hyperparams.py    # Optuna 하이퍼파라미터 자동 탐색 (--symbol 지원)
│   ├── strategy_sweep.py      # 전략 파라미터 그리드 스윕 (324개 조합)
│   ├── weekly_report.py       # 주간 전략 리포트 (백테스트+로그+추이+Discord)
│   ├── run_backtest.py        # 단일 백테스트 CLI
│   ├── deploy_model.sh        # 모델 파일 LXC 서버 전송 (--symbol 지원)
│   └── run_tests.sh           # 전체 테스트 실행
├── dashboard/
│   ├── api/                   # FastAPI 백엔드 (로그 파서 + REST API)
│   └── ui/                    # React 프론트엔드 (Vite + Recharts)
├── models/                    # 학습된 모델 저장 (심볼별 하위 디렉토리)
│   ├── xrpusdt/               #   models/xrpusdt/lgbm_filter.pkl
│   ├── trxusdt/               #   models/trxusdt/lgbm_filter.pkl
│   └── dogeusdt/              #   models/dogeusdt/lgbm_filter.pkl
├── data/                      # 과거 데이터 캐시 (심볼별 하위 디렉토리)
│   ├── xrpusdt/               #   data/xrpusdt/combined_15m.parquet
│   ├── trxusdt/               #   data/trxusdt/combined_15m.parquet
│   └── dogeusdt/              #   data/dogeusdt/combined_15m.parquet
├── results/
│   └── weekly/                # 주간 리포트 JSON 저장
├── logs/                      # 로그 파일
├── docs/plans/                # 설계 문서 및 구현 플랜
├── tests/                     # 테스트 코드
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
└── requirements.txt
```

---

## 빠른 시작

### 1. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 값을 채웁니다.

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
SYMBOLS=XRPUSDT,TRXUSDT,DOGEUSDT
CORRELATION_SYMBOLS=BTCUSDT,ETHUSDT
LEVERAGE=10
MAX_SAME_DIRECTION=2
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### 2. 로컬 실행

```bash
pip install -r requirements.txt
python main.py
```

### 3. Docker Compose로 실행

```bash
docker compose up -d
```

로그 확인:

```bash
docker compose logs -f cointrader
```

---

## ML 모델 학습

봇은 모델 파일이 없으면 ML 필터 없이 동작합니다. 최초 실행 전 또는 수동 재학습 시 아래 순서로 진행합니다.

### 전체 파이프라인 (권장)

맥미니에서 데이터 수집 → 학습 → LXC 배포까지 한 번에 실행합니다.

> **자동 분기**: `data/{symbol}/combined_15m.parquet`가 없으면 1년치(365일) 전체 수집, 있으면 35일치 Upsert로 자동 전환합니다. 서버 이전이나 데이터 유실 시에도 사람의 개입 없이 자동 복구됩니다.

```bash
# 전체 심볼 학습 + 배포 (SYMBOLS 환경변수의 모든 심볼)
bash scripts/train_and_deploy.sh

# 단일 심볼만 학습 + 배포
bash scripts/train_and_deploy.sh --symbol TRXUSDT

# MLX GPU 학습 (단일 심볼)
bash scripts/train_and_deploy.sh mlx --symbol XRPUSDT

# LightGBM + Walk-Forward 3폴드
bash scripts/train_and_deploy.sh lgbm 3

# 학습만 (배포 없이)
bash scripts/train_and_deploy.sh lgbm 0
```

### 단계별 수동 실행

```bash
# 1. 과거 데이터 수집 (단일 심볼 — 상관관계 심볼 자동 추가)
python scripts/fetch_history.py --symbol TRXUSDT --interval 15m --days 365
# → data/trxusdt/combined_15m.parquet 에 저장

# 1-alt. 명시적 심볼 지정 (기존 방식도 지원)
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --output data/combined_15m.parquet

# 2-A. LightGBM 모델 학습 (심볼별)
python scripts/train_model.py --symbol TRXUSDT
# → models/trxusdt/lgbm_filter.pkl 에 저장

# 2-B. MLX 신경망 학습 (Apple Silicon GPU)
python scripts/train_mlx_model.py --data data/xrpusdt/combined_15m.parquet

# 3. LXC 서버에 모델 배포
bash scripts/deploy_model.sh --symbol XRPUSDT
bash scripts/deploy_model.sh mlx --symbol XRPUSDT
```

학습된 모델은 `models/{symbol}/lgbm_filter.pkl` (LightGBM) 또는 `models/{symbol}/mlx_filter.weights.onnx` (MLX) 에 저장됩니다. 심볼별 디렉토리가 없으면 `models/` 루트로 폴백합니다.

> **모델 핫리로드**: 봇이 실행 중일 때 모델 파일을 교체하면, 다음 캔들 마감 시 자동으로 감지해 리로드합니다. 봇 재시작이 필요 없습니다.

### 하이퍼파라미터 자동 튜닝 (Optuna)

봇 성능이 저하되거나 데이터가 충분히 축적되었을 때 Optuna로 최적 LightGBM 파라미터를 탐색합니다.
결과를 확인하고 직접 승인한 후 재학습에 반영하는 **수동 트리거** 방식입니다.

```bash
# 심볼별 튜닝 (50 trials, 5폴드 Walk-Forward, ~30분)
python scripts/tune_hyperparams.py --symbol XRPUSDT

# 빠른 테스트 (10 trials, 3폴드, ~5분)
python scripts/tune_hyperparams.py --symbol TRXUSDT --trials 10 --folds 3

# 베이스라인 측정 없이 탐색만
python scripts/tune_hyperparams.py --symbol XRPUSDT --no-baseline
```

결과는 `models/{symbol}/tune_results_YYYYMMDD_HHMMSS.json`에 저장됩니다.
콘솔에 Best Params, 베이스라인 대비 개선폭, 폴드별 AUC를 출력하므로 직접 확인 후 판단하세요.

> **주의**: Optuna가 찾은 파라미터는 과적합 위험이 있습니다. Best Params를 `train_model.py`에 반영하기 전에 반드시 폴드별 AUC 분산과 개선폭을 검토하세요.

### Apple Silicon GPU 가속 학습 (M1/M2/M3/M4)

M 시리즈 맥에서는 MLX를 사용해 통합 GPU(Metal)로 학습할 수 있습니다.

> **설치**: `mlx`는 Apple Silicon 전용이며 `requirements.txt`에 포함되지 않습니다.
> 맥미니에서 별도 설치: `pip install mlx`

MLX로 학습한 모델은 ONNX 포맷으로 export되어 Linux 서버에서 `onnxruntime`으로 추론합니다.

> **참고**: LightGBM은 Apple Silicon GPU를 공식 지원하지 않습니다. MLX는 Apple이 만든 ML 프레임워크로 통합 GPU를 자동으로 활용합니다.

---

## 매매 전략

### 기술 지표 신호 (15분봉)

| 지표 | 롱 조건 | 숏 조건 | 가중치 |
|------|---------|---------|--------|
| RSI (14) | < 35 | > 65 | 1 |
| MACD 크로스 | 골든크로스 | 데드크로스 | 2 |
| 볼린저 밴드 | 하단 이탈 | 상단 돌파 | 1 |
| EMA 정배열 (9/21/50) | 정배열 | 역배열 | 1 |
| Stochastic RSI | < 20 + K>D | > 80 + K<D | 1 |
| 거래량 | 20MA × 1.5 이상 시 신호 강화 | — | 보조 |

**진입 조건**: 가중치 합계 ≥ `SIGNAL_THRESHOLD`(기본 3) + (거래량 ≥ 20MA × `VOL_MULTIPLIER`(기본 2.5) 또는 가중치 합계 ≥ `SIGNAL_THRESHOLD` + 1)
**ADX 필터**: ADX < `ADX_THRESHOLD`(기본 25) 시 횡보장으로 판단, 진입 차단
**손절/익절**: ATR × `ATR_SL_MULT`(기본 2.0) / ATR × `ATR_TP_MULT`(기본 2.0) — 환경변수로 설정 가능
**ML 필터**: 예측 확률 ≥ 0.55 이어야 최종 진입 (현재 `NO_ML_FILTER=true`로 비활성화 — 아래 참고)

### 반대 시그널 재진입

보유 포지션과 반대 방향 신호가 발생하면:
1. 기존 포지션 즉시 청산 (미체결 SL/TP 주문 취소 포함)
2. ML 필터 통과 시 반대 방향으로 즉시 재진입

### ML 필터 현황 — 왜 현재 ML을 사용하지 않는가

현재 프로덕션 봇은 `NO_ML_FILTER=true`로 ML 필터를 **비활성화**한 상태로 운영 중입니다.

**비활성화 사유:**

1. **학습 데이터 부족**: Walk-Forward 검증(학습 3개월, 테스트 1개월) 시 각 폴드의 학습 세트에서 유효 신호가 약 27건에 불과. LightGBM이 의미 있는 패턴을 학습하기엔 표본 수가 절대적으로 부족.
2. **예측 무차별**: 학습된 모델이 모든 입력에 대해 거의 동일한 확률(~0.55)을 출력하여 필터링 효과가 사실상 없음. 모든 신호를 차단하거나 모든 신호를 통과시키는 극단적 동작.
3. **전략 파라미터 스윕 결과**: ADX 필터(≥25)와 거래량 배수(2.5)를 적용한 기본 기술 지표 전략만으로 PF 1.57~2.39를 달성. ML 없이도 수익성 확보 가능.

**ML 재활성화 조건 (주간 리포트에서 자동 체크):**

- 누적 트레이드 ≥ 150건 (충분한 학습 데이터 확보)
- 현재 PF < 1.0 (기술 지표만으로 수익성 저하)
- PF 3주 연속 하락 추세

3개 조건 중 2개 이상 충족 시 `scripts/weekly_report.py`가 Discord로 ML 재학습 권장 알림을 전송합니다.

---

## 전략 파라미터 스윕

기술 지표 전략의 최적 파라미터를 Walk-Forward 백테스트로 탐색합니다.

```bash
# 전체 스윕 (324개 조합, ~30분)
python scripts/strategy_sweep.py --symbols XRPUSDT --train-months 3 --test-months 1

# 결과 확인
cat results/sweep_*.json | python -m json.tool | head -50
```

5개 파라미터 × 3~4개 값 = 324개 조합을 순차 테스트:

| 파라미터 | 값 | 설명 |
|---------|------|------|
| `ATR_SL_MULT` | 1.0, 1.5, 2.0 | 손절 ATR 배수 |
| `ATR_TP_MULT` | 2.0, 3.0, 4.0 | 익절 ATR 배수 |
| `SIGNAL_THRESHOLD` | 3, 4, 5 | 최소 가중치 점수 |
| `ADX_THRESHOLD` | 0, 20, 25, 30 | ADX 필터 (0=비활성) |
| `VOL_MULTIPLIER` | 1.5, 2.0, 2.5 | 거래량 급증 배수 |

> **핵심 발견**: ADX ≥ 25 필터가 가장 영향력 있는 단일 파라미터. 상위 10개 결과 모두 ADX ≥ 25를 사용하며, 횡보장 노이즈 신호를 효과적으로 필터링.

---

## 주간 전략 리포트

매주 자동으로 전략 성능을 측정하고 Discord로 리포트를 전송합니다.

```bash
# 수동 실행 (데이터 수집 스킵)
python scripts/weekly_report.py --skip-fetch

# 전체 실행 (데이터 수집 포함)
python scripts/weekly_report.py

# 특정 날짜 리포트
python scripts/weekly_report.py --date 2026-03-07
```

**리포트 내용:**
- Walk-Forward 백테스트 성능 (심볼별 PF/승률/MDD)
- 실전 트레이드 로그 파싱 (이번 주 거래 수/순수익/승률)
- 성능 추이 (최근 4주 PF/승률/MDD 변화)
- ML 재도전 체크리스트 (3개 조건 자동 판단)
- PF < 1.0 시 파라미터 스윕 대안 제시

**크론탭 설정 (프로덕션 서버):**
```bash
# 매주 일요일 새벽 3시 KST
0 18 * * 6 cd /app && python scripts/weekly_report.py >> logs/cron.log 2>&1
```

리포트 결과는 `results/weekly/report_YYYY-MM-DD.json`에 저장됩니다.

---

## CI/CD

`main` 브랜치에 푸시하면 Jenkins 파이프라인이 자동으로 실행됩니다.

1. **Notify Build Start** — Discord 빌드 시작 알림
2. **Git Clone from Gitea** — 소스 체크아웃
3. **Build Docker Image** — Docker 이미지 빌드 (`:{BUILD_NUMBER}` + `:latest` 태그)
4. **Push to Gitea Registry** — Gitea Container Registry(`10.1.10.28:3000`)에 푸시
5. **Deploy to Prod LXC** — 운영 LXC 서버(`10.1.10.24`)에 자동 배포 (`docker compose pull && up -d`)
6. **Cleanup** — 빌드 서버 로컬 이미지 정리

빌드 성공/실패 결과는 Discord로 자동 알림됩니다.

---

## 대시보드

봇 로그를 실시간으로 파싱하여 거래 내역, 수익 통계, 차트를 웹에서 조회할 수 있는 모니터링 대시보드입니다.

### 기술 스택

- **프론트엔드**: React 18 + Vite + Recharts, Nginx 정적 서빙
- **백엔드**: FastAPI + SQLite, 로그 파서(5초 주기 폴링)
- **배포**: Docker Compose 3컨테이너 (`dashboard-ui`, `dashboard-api`, `cointrader`)

### 주요 화면

| 탭 | 내용 |
|----|------|
| **Overview** | 총 수익, 승률, 거래 수, 최대 수익/손실 KPI + 일별 PnL 차트 + 누적 수익 곡선 |
| **Trades** | 전체 거래 내역 — 진입/청산가, 방향, 레버리지, 기술 지표(RSI, MACD, ATR), SL/TP, 순익 상세 |
| **Chart** | XRP/USDT 15분봉 가격 차트 + RSI 지표 + ADX 추세 강도 |

### API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /api/position` | 현재 포지션 + 봇 상태 |
| `GET /api/trades` | 청산 거래 내역 (페이지네이션) |
| `GET /api/daily` | 일별 PnL 집계 |
| `GET /api/stats` | 전체 통계 (총 거래, 승률, 수수료 등) |
| `GET /api/candles` | 최근 캔들 + 기술 지표 |
| `GET /api/health` | 헬스 체크 |
| `POST /api/reset` | DB 초기화 + 로그 파서 재시작 |

### 실행

```bash
docker compose up -d
```

대시보드는 `http://<서버IP>:8080`에서 접속할 수 있습니다. 봇 로그를 읽기 전용으로 마운트하여 봇 코드를 수정하지 않는 디커플드 설계입니다.

---

## 테스트

```bash
# 전체 테스트
bash scripts/run_tests.sh

# 특정 키워드 필터
bash scripts/run_tests.sh -k bot

# pytest 직접 실행
pytest tests/ -v
```

---

## 환경변수 레퍼런스

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BINANCE_API_KEY` | — | Binance API 키 |
| `BINANCE_API_SECRET` | — | Binance API 시크릿 |
| `SYMBOLS` | `XRPUSDT` | 거래 심볼 목록 (쉼표 구분, 예: `XRPUSDT,TRXUSDT,DOGEUSDT`) |
| `CORRELATION_SYMBOLS` | `BTCUSDT,ETHUSDT` | 상관관계 심볼 (BTC/ETH 수익률·상대강도 피처용) |
| `LEVERAGE` | `10` | 레버리지 배수 |
| `MAX_SAME_DIRECTION` | `2` | 동일 방향 최대 포지션 수 (LONG 2개면 3번째 LONG 차단) |
| `DISCORD_WEBHOOK_URL` | — | Discord 웹훅 URL |
| `MARGIN_MAX_RATIO` | `0.50` | 최대 증거금 비율 (잔고 대비 50%) |
| `MARGIN_MIN_RATIO` | `0.20` | 최소 증거금 비율 (잔고 대비 20%) |
| `MARGIN_DECAY_RATE` | `0.0006` | 잔고 증가 시 증거금 비율 감소 속도 |
| `NO_ML_FILTER` | `true` | `true`/`1`/`yes` 설정 시 ML 필터 완전 비활성화 — 모델 로드 없이 모든 신호 허용 (현재 프로덕션 기본값) |
| `ML_THRESHOLD` | `0.55` | ML 필터 예측 확률 임계값 — 이 값 이상이어야 진입 허용 |
| `ATR_SL_MULT` | `2.0` | 손절 ATR 배수 (진입가 ± ATR × 이 값) |
| `ATR_TP_MULT` | `2.0` | 익절 ATR 배수 (진입가 ± ATR × 이 값) |
| `SIGNAL_THRESHOLD` | `3` | 진입을 위한 최소 가중치 지표 점수 |
| `ADX_THRESHOLD` | `25` | ADX 횡보장 필터 (이 값 미만이면 진입 차단, 0=비활성) |
| `VOL_MULTIPLIER` | `2.5` | 거래량 급증 감지 배수 (20MA × 이 값 이상 시 급증 판정) |

---

## 주의사항

> **이 봇은 실제 자산을 거래합니다.** 운영 전 반드시 Binance Testnet에서 충분히 검증하세요.  
> 과거 수익이 미래 수익을 보장하지 않습니다. 투자 손실에 대한 책임은 사용자 본인에게 있습니다.  
> 성투기원합니다.
