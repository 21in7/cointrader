# CoinTrader

Binance Futures 자동매매 봇. 복합 기술 지표와 ML 필터(LightGBM / MLX 신경망)를 결합하여 다중 심볼(XRP, TRX, DOGE 등) 선물 포지션을 동시에 자동 진입·청산하며, Discord로 실시간 알림을 전송합니다.

> **이 봇은 실제 자산을 거래합니다.** 운영 전 반드시 Binance Testnet에서 충분히 검증하세요.
> 과거 수익이 미래 수익을 보장하지 않습니다. 투자 손실에 대한 책임은 사용자 본인에게 있습니다.

---

## 주요 기능

- **멀티심볼 동시 거래**: 심볼별 독립 봇 인스턴스를 병렬 실행, 공유 RiskManager로 글로벌 리스크 관리
- **복합 기술 지표 신호**: RSI, MACD, 볼린저 밴드, EMA, Stochastic RSI, ADX, 거래량 급증 — 가중치 합산 시스템
- **ML 필터 (선택)**: LightGBM / ONNX 모델로 오진입 차단 (비활성화 가능)
- **ATR 기반 손절/익절**: 변동성에 따라 동적으로 SL/TP 계산, 환경변수로 배수 조절
- **반대 시그널 재진입**: 보유 포지션과 반대 신호 발생 시 즉시 청산 후 재진입
- **리스크 관리**: 동일 방향 포지션 제한, 일일 손실 한도(5%), 동적 증거금 비율
- **실시간 TP/SL 감지**: Binance User Data Stream으로 즉시 감지
- **Discord 알림**: 진입·청산·오류 이벤트 실시간 웹훅 알림
- **모니터링 대시보드**: 거래 내역, 수익 통계, 차트를 웹에서 조회
- **주간 전략 리포트**: 자동 성능 측정, 추이 추적, ML 재학습 시점 판단

---

# 봇 사용 가이드

봇을 설치하고 운영하려는 사용자를 위한 섹션입니다.

## 요구사항

- Python 3.11+ (또는 Docker)
- Binance Futures 계정 + API 키
- (선택) Discord 웹훅 URL

## 빠른 시작

### 1. 환경변수 설정

```bash
git clone <repository-url>
cd cointrader
cp .env.example .env
```

`.env` 파일을 열어 아래 필수 값을 채웁니다.

```env
# 필수
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
SYMBOLS=XRPUSDT                           # 거래할 심볼 (쉼표 구분, 예: XRPUSDT,TRXUSDT,DOGEUSDT)

# 권장
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
LEVERAGE=10
```

> 처음 사용 시 Binance Testnet에서 먼저 테스트하는 것을 권장합니다. `BINANCE_TESTNET_API_KEY`와 `BINANCE_TESTNET_API_SECRET`을 설정하세요.

### 2-A. Docker로 실행 (권장)

```bash
docker compose up -d
```

로그 확인:
```bash
docker compose logs -f cointrader
```

### 2-B. 로컬 실행

```bash
pip install -r requirements.txt
python main.py
```

### 3. 정상 동작 확인

봇이 정상 실행되면 다음과 같은 로그가 출력됩니다:

```
INFO | 봇 시작: XRPUSDT (레버리지 10x)
INFO | 과거 캔들 200개 프리로드 완료
INFO | WebSocket 연결 완료
```

Discord 웹훅을 설정했다면 진입/청산 시 실시간 알림을 받게 됩니다.

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
| 거래량 | 20MA × `VOL_MULTIPLIER` 이상 시 신호 강화 | — | 보조 |

**진입 조건**: 가중치 합계 ≥ `SIGNAL_THRESHOLD` + (거래량 급증 또는 가중치 합계 ≥ `SIGNAL_THRESHOLD` + 1)
**ADX 필터**: ADX < `ADX_THRESHOLD` 시 횡보장으로 판단, 진입 차단
**손절/익절**: ATR × `ATR_SL_MULT` / ATR × `ATR_TP_MULT`

### 전략 파라미터 조절

환경변수로 전략 파라미터를 조절할 수 있습니다. 기본값은 Walk-Forward 백테스트 스윕 결과에서 선정된 값입니다.

**전역 기본값** (심볼별 오버라이드 없을 때 적용):

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `ATR_SL_MULT` | `2.0` | 손절 ATR 배수 |
| `ATR_TP_MULT` | `2.0` | 익절 ATR 배수 |
| `SIGNAL_THRESHOLD` | `3` | 진입을 위한 최소 가중치 점수 |
| `ADX_THRESHOLD` | `25` | ADX 횡보장 필터 (0=비활성) |
| `VOL_MULTIPLIER` | `2.5` | 거래량 급증 감지 배수 |

**심볼별 오버라이드**: `{환경변수}_{심볼}` 형태로 심볼마다 독립 설정 가능. 미설정 시 전역 기본값 사용.

```env
# 예시: 2026-03-17 스윕 최적화 결과
ATR_SL_MULT_XRPUSDT=1.5
ATR_TP_MULT_XRPUSDT=4.0
ADX_THRESHOLD_XRPUSDT=30

ATR_SL_MULT_TRXUSDT=1.0
ATR_TP_MULT_TRXUSDT=4.0
ADX_THRESHOLD_TRXUSDT=30

ATR_SL_MULT_DOGEUSDT=2.0
ATR_TP_MULT_DOGEUSDT=2.0
ADX_THRESHOLD_DOGEUSDT=30
```

### ML 필터

ML 필터는 기술 지표 신호를 한 번 더 검증하여 오진입을 차단합니다. 기본적으로 **비활성화** 상태입니다.

- `NO_ML_FILTER=true` (기본값) — ML 없이 기술 지표만으로 운영
- `NO_ML_FILTER=false` — ML 필터 활성화 (모델 파일 필요)

> 현재 기본값이 비활성화인 이유: 학습 데이터가 충분히 축적되기 전까지 ML 모델의 예측력이 낮습니다. ADX 필터와 거래량 배수 조합만으로 PF 1.5 이상을 달성하고 있어, 충분한 거래 데이터(150건 이상)가 쌓일 때까지 ML 없이 운영합니다.

---

## 리스크 관리

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `LEVERAGE` | `10` | 레버리지 배수 |
| `MAX_SAME_DIRECTION` | `2` | 동일 방향 최대 포지션 수 |
| `MARGIN_MAX_RATIO` | `0.50` | 최대 증거금 비율 (잔고 대비) |
| `MARGIN_MIN_RATIO` | `0.20` | 최소 증거금 비율 (잔고 대비) |
| `MARGIN_DECAY_RATE` | `0.0006` | 잔고 증가 시 증거금 비율 감소 속도 |

- **일일 손실 한도**: 기준 잔고의 5% 초과 시 당일 거래 중단
- **동적 증거금**: 잔고가 늘어날수록 비율을 선형으로 줄여 과노출 방지
- **포지션 복구**: 봇 재시작 시 기존 포지션 자동 감지 및 상태 복원

---

## 대시보드

봇 로그를 실시간으로 파싱하여 거래 내역, 수익 통계, 차트를 웹에서 조회할 수 있습니다.

```bash
docker compose up -d
# 접속: http://<서버IP>:8080
```

| 탭 | 내용 |
|----|------|
| **Overview** | 총 수익, 승률, 거래 수, 최대 수익/손실 KPI + 일별 PnL 차트 + 누적 수익 곡선 |
| **Trades** | 전체 거래 내역 — 진입/청산가, 방향, 레버리지, 기술 지표, SL/TP, 순익 상세 |
| **Chart** | 15분봉 가격 차트 + RSI 지표 + ADX 추세 강도 |

### API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /api/position` | 현재 포지션 + 봇 상태 |
| `GET /api/trades` | 청산 거래 내역 (페이지네이션) |
| `GET /api/daily` | 일별 PnL 집계 |
| `GET /api/stats` | 전체 통계 (총 거래, 승률, 수수료 등) |
| `GET /api/candles` | 최근 캔들 + 기술 지표 |
| `GET /api/health` | 헬스 체크 |

---

## 환경변수 전체 레퍼런스

| 변수 | 기본값 | 필수 | 설명 |
|------|--------|:----:|------|
| `BINANCE_API_KEY` | — | ✅ | Binance API 키 |
| `BINANCE_API_SECRET` | — | ✅ | Binance API 시크릿 |
| `SYMBOLS` | `XRPUSDT` | | 거래 심볼 목록 (쉼표 구분) |
| `CORRELATION_SYMBOLS` | `BTCUSDT,ETHUSDT` | | 상관관계 심볼 (BTC/ETH 피처용) |
| `LEVERAGE` | `10` | | 레버리지 배수 |
| `MAX_SAME_DIRECTION` | `2` | | 동일 방향 최대 포지션 수 |
| `DISCORD_WEBHOOK_URL` | — | | Discord 웹훅 URL |
| `MARGIN_MAX_RATIO` | `0.50` | | 최대 증거금 비율 |
| `MARGIN_MIN_RATIO` | `0.20` | | 최소 증거금 비율 |
| `MARGIN_DECAY_RATE` | `0.0006` | | 잔고 증가 시 감소 속도 |
| `NO_ML_FILTER` | `true` | | ML 필터 비활성화 |
| `ML_THRESHOLD` | `0.55` | | ML 예측 확률 임계값 |
| `ATR_SL_MULT` | `2.0` | | 손절 ATR 배수 (전역 기본값) |
| `ATR_TP_MULT` | `2.0` | | 익절 ATR 배수 (전역 기본값) |
| `SIGNAL_THRESHOLD` | `3` | | 최소 가중치 점수 (전역 기본값) |
| `ADX_THRESHOLD` | `25` | | ADX 횡보장 필터 (전역 기본값, 0=비활성) |
| `VOL_MULTIPLIER` | `2.5` | | 거래량 급증 배수 (전역 기본값) |
| `ATR_SL_MULT_{SYMBOL}` | — | | 심볼별 손절 ATR 배수 오버라이드 |
| `ATR_TP_MULT_{SYMBOL}` | — | | 심볼별 익절 ATR 배수 오버라이드 |
| `SIGNAL_THRESHOLD_{SYMBOL}` | — | | 심볼별 최소 가중치 점수 오버라이드 |
| `ADX_THRESHOLD_{SYMBOL}` | — | | 심볼별 ADX 필터 오버라이드 |
| `VOL_MULTIPLIER_{SYMBOL}` | — | | 심볼별 거래량 배수 오버라이드 |
| `DASHBOARD_API_URL` | `http://10.1.10.24:8000` | | 대시보드 API 주소 (주간 리포트용) |
| `BINANCE_TESTNET_API_KEY` | — | | Testnet API 키 |
| `BINANCE_TESTNET_API_SECRET` | — | | Testnet API 시크릿 |

---

# 개발 가이드

코드를 수정하거나 기능을 추가하려는 개발자를 위한 섹션입니다.

> **아키텍처 문서**: 5-레이어 구조, 데이터 흐름, MLOps 파이프라인, 동작 시나리오를 상세히 설명한 [ARCHITECTURE.md](./ARCHITECTURE.md)를 참고하세요.

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
│   ├── backtester.py          # 백테스트 엔진 (단일 + Walk-Forward)
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
│   ├── weekly_report.py       # 주간 전략 리포트 (백테스트+대시보드API+추이+Discord)
│   ├── run_backtest.py        # 단일 백테스트 CLI
│   ├── deploy_model.sh        # 모델 파일 LXC 서버 전송 (--symbol 지원)
│   └── run_tests.sh           # 전체 테스트 실행
├── dashboard/
│   ├── api/                   # FastAPI 백엔드 (로그 파서 + REST API)
│   └── ui/                    # React 프론트엔드 (Vite + Recharts)
├── models/                    # 학습된 모델 저장 (심볼별 하위 디렉토리)
├── data/                      # 과거 데이터 캐시 (심볼별 하위 디렉토리)
├── results/
│   └── weekly/                # 주간 리포트 JSON 저장
├── logs/                      # 로그 파일
├── docs/plans/                # 설계 문서 및 구현 플랜
├── tests/                     # 테스트 코드 (15파일, 138개 케이스)
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
└── requirements.txt
```

## 개발 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
```

## 테스트

```bash
# 전체 테스트 (138개)
bash scripts/run_tests.sh

# 특정 키워드 필터
bash scripts/run_tests.sh -k bot

# pytest 직접 실행
pytest tests/ -v
```

모든 외부 API(Binance, Discord)는 `unittest.mock.AsyncMock`으로 대체되며, 비동기 테스트는 `@pytest.mark.asyncio`를 사용합니다.

## ML 모델 학습

봇은 모델 파일이 없으면 ML 필터 없이 동작합니다. 모델을 학습하려면:

### 전체 파이프라인 (권장)

```bash
# 전체 심볼 학습 + 배포
bash scripts/train_and_deploy.sh

# 단일 심볼만 학습 + 배포
bash scripts/train_and_deploy.sh --symbol TRXUSDT

# MLX GPU 학습 (Apple Silicon, 단일 심볼)
bash scripts/train_and_deploy.sh mlx --symbol XRPUSDT

# 학습만 (배포 없이)
bash scripts/train_and_deploy.sh lgbm 0
```

> **자동 분기**: `data/{symbol}/combined_15m.parquet`가 없으면 1년치 전체 수집, 있으면 35일치 Upsert로 자동 전환.

### 단계별 수동 실행

```bash
# 1. 과거 데이터 수집
python scripts/fetch_history.py --symbol TRXUSDT --interval 15m --days 365

# 2. LightGBM 모델 학습
python scripts/train_model.py --symbol TRXUSDT

# 3. 서버에 모델 배포
bash scripts/deploy_model.sh --symbol TRXUSDT
```

> **모델 핫리로드**: 봇 실행 중 모델 파일을 교체하면, 다음 캔들 마감 시 자동으로 감지해 리로드합니다.

### 하이퍼파라미터 튜닝 (Optuna)

```bash
# 심볼별 튜닝 (50 trials, 5폴드 Walk-Forward, ~30분)
python scripts/tune_hyperparams.py --symbol XRPUSDT

# 빠른 테스트 (10 trials, 3폴드, ~5분)
python scripts/tune_hyperparams.py --symbol TRXUSDT --trials 10 --folds 3
```

결과는 `models/{symbol}/tune_results_YYYYMMDD_HHMMSS.json`에 저장됩니다. Optuna가 찾은 파라미터는 과적합 위험이 있으므로 폴드별 AUC 분산과 개선폭을 반드시 검토하세요.

### Apple Silicon GPU 가속 (M1/M2/M3/M4)

```bash
pip install mlx  # Apple Silicon 전용, requirements.txt에 미포함
bash scripts/train_and_deploy.sh mlx --symbol XRPUSDT
```

MLX로 학습한 모델은 ONNX 포맷으로 export되어 Linux 서버에서 `onnxruntime`으로 추론합니다.

## 전략 파라미터 스윕

기술 지표 전략의 최적 파라미터를 Walk-Forward 백테스트로 탐색합니다.

```bash
# 전체 스윕 (324개 조합, ~30분)
python scripts/strategy_sweep.py --symbols XRPUSDT --train-months 3 --test-months 1
```

5개 파라미터 × 3~4개 값 = 324개 조합을 순차 테스트:

| 파라미터 | 값 | 설명 |
|---------|------|------|
| `ATR_SL_MULT` | 1.0, 1.5, 2.0 | 손절 ATR 배수 |
| `ATR_TP_MULT` | 2.0, 3.0, 4.0 | 익절 ATR 배수 |
| `SIGNAL_THRESHOLD` | 3, 4, 5 | 최소 가중치 점수 |
| `ADX_THRESHOLD` | 0, 20, 25, 30 | ADX 필터 |
| `VOL_MULTIPLIER` | 1.5, 2.0, 2.5 | 거래량 급증 배수 |

> **핵심 발견**: ADX ≥ 25 필터가 가장 영향력 있는 파라미터. 횡보장 노이즈 신호를 효과적으로 필터링.

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
- 운영 대시보드 API에서 실전 트레이드 통계 조회 (거래 수/순수익/승률)
- 성능 추이 (최근 4주 PF/승률/MDD 변화)
- ML 재도전 체크리스트 (3개 조건 자동 판단)
- PF < 1.0 시 파라미터 스윕 대안 제시

> 실전 데이터는 운영 대시보드 API(`GET /api/trades`, `GET /api/stats`)에서 조회합니다. `DASHBOARD_API_URL` 환경변수로 주소를 설정하세요.

**크론탭 설정:**
```bash
# 매주 일요일 새벽 3시 KST
0 18 * * 6 cd /app && python scripts/weekly_report.py >> logs/cron.log 2>&1
```

## CI/CD

`main` 브랜치에 푸시하면 Jenkins 파이프라인이 자동 실행됩니다.

1. **Notify Build Start** — Discord 빌드 시작 알림
2. **Git Clone from Gitea** — 소스 체크아웃
3. **Build Docker Image** — Docker 이미지 빌드 (`:{BUILD_NUMBER}` + `:latest`)
4. **Push to Gitea Registry** — Container Registry에 푸시
5. **Deploy to Prod** — 운영 서버에 자동 배포 (`docker compose pull && up -d`)
6. **Cleanup** — 로컬 이미지 정리

빌드 성공/실패 결과는 Discord로 자동 알림됩니다.

## 설계 문서

모든 설계 문서와 구현 계획은 `docs/plans/`에 저장됩니다.

- `YYYY-MM-DD-feature-name-design.md` — 설계 결정 문서
- `YYYY-MM-DD-feature-name-plan.md` — 단계별 구현 계획
- [ARCHITECTURE.md](./ARCHITECTURE.md) — 전체 아키텍처 (5-레이어, MLOps 파이프라인, 동작 시나리오, 테스트 커버리지)
