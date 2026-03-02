# CoinTrader

Binance Futures 자동매매 봇. 복합 기술 지표와 ML 필터(LightGBM / MLX 신경망)를 결합하여 XRPUSDT(기본) 선물 포지션을 자동으로 진입·청산하며, Discord로 실시간 알림을 전송합니다.

---

## 주요 기능

- **복합 기술 지표 신호**: RSI, MACD 크로스, 볼린저 밴드, EMA 정/역배열, Stochastic RSI, 거래량 급증 — 가중치 합계 ≥ 3 시 진입
- **ML 필터 (ONNX 우선 / LightGBM 폴백)**: 기술 지표 신호를 한 번 더 검증하여 오진입 차단. 우선순위: ONNX > LightGBM > 폴백(항상 허용)
- **모델 핫리로드**: 캔들마다 모델 파일 mtime을 감지해 변경 시 자동 리로드 (봇 재시작 불필요)
- **멀티심볼 스트림**: XRP/BTC/ETH 3개 심볼을 단일 Combined WebSocket으로 수신, BTC·ETH 상관관계 피처 활용
- **25개 ML 피처**: XRP 기술 지표 13개 + BTC/ETH 수익률·상대강도 8개 + OI 변화율·펀딩비 2개 (캔들 마감 시 실시간 조회, 실패 시 0으로 폴백)
- **실시간 OI/펀딩비 조회**: 캔들 마감마다 `get_open_interest()` / `get_funding_rate()`를 비동기 병렬 조회하여 ML 피처에 전달. 이전 캔들 대비 OI 변화율로 변환하여 train-serve skew 해소
- **ATR 기반 손절/익절**: 변동성에 따라 동적으로 SL/TP 계산 (1.5× / 3.0× ATR)
- **Algo Order API 지원**: 계정 설정에 따라 STOP_MARKET/TAKE_PROFIT_MARKET 주문을 `/fapi/v1/algoOrder` 엔드포인트로 자동 전송 (오류 코드 -4120 대응)
- **동적 증거금 비율**: 잔고 증가에 따라 선형 감소 (최대 50% → 최소 20%)
- **반대 시그널 재진입**: 보유 포지션과 반대 신호 발생 시 즉시 청산 후 ML 필터 통과 시 반대 방향 재진입
- **리스크 관리**: 트레이드당 리스크 비율, 최대 포지션 수, 일일 손실 한도(5%) 제어
- **포지션 복구**: 봇 재시작 시 기존 포지션 자동 감지 및 상태 복원
- **Discord 알림**: 진입·청산·오류 이벤트 실시간 웹훅 알림
- **CI/CD**: Jenkins + Gitea Container Registry 기반 Docker 이미지 자동 빌드·배포 (LXC 운영 서버 자동 적용)

---

## 프로젝트 구조

```
cointrader/
├── main.py                    # 진입점
├── src/
│   ├── bot.py                 # 메인 트레이딩 루프
│   ├── config.py              # 환경변수 기반 설정
│   ├── exchange.py            # Binance Futures API 클라이언트
│   ├── data_stream.py         # WebSocket 15분봉 멀티심볼 스트림 (XRP/BTC/ETH)
│   ├── indicators.py          # 기술 지표 계산 및 신호 생성
│   ├── ml_filter.py           # ML 필터 (ONNX 우선 / LightGBM 폴백 / 핫리로드)
│   ├── ml_features.py         # ML 피처 빌더 (23개 피처)
│   ├── mlx_filter.py          # MLX 신경망 필터 (Apple Silicon GPU 학습 + ONNX export)
│   ├── label_builder.py       # 학습 레이블 생성
│   ├── dataset_builder.py     # 벡터화 데이터셋 빌더 (학습용)
│   ├── risk_manager.py        # 리스크 관리 (일일 손실 한도, 동적 증거금 비율)
│   ├── notifier.py            # Discord 웹훅 알림
│   └── logger_setup.py        # Loguru 로거 설정
├── scripts/
│   ├── fetch_history.py       # 과거 데이터 수집 (XRP/BTC/ETH + OI/펀딩비)
│   ├── train_model.py         # LightGBM 모델 학습 (CPU)
│   ├── train_mlx_model.py     # MLX 신경망 학습 (Apple Silicon GPU)
│   ├── train_and_deploy.sh    # 전체 파이프라인 (수집 → 학습 → LXC 배포)
│   ├── deploy_model.sh        # 모델 파일 LXC 서버 전송
│   └── run_tests.sh           # 전체 테스트 실행
├── models/                    # 학습된 모델 저장 (.pkl / .onnx)
├── data/                      # 과거 데이터 캐시 (.parquet)
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
SYMBOL=XRPUSDT
LEVERAGE=10
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

```bash
# LightGBM + Walk-Forward 5폴드 (기본값)
bash scripts/train_and_deploy.sh

# MLX GPU 학습 + Walk-Forward 5폴드
bash scripts/train_and_deploy.sh mlx

# LightGBM + Walk-Forward 3폴드
bash scripts/train_and_deploy.sh lgbm 3

# 학습만 (배포 없이)
bash scripts/train_and_deploy.sh lgbm 0
```

### 단계별 수동 실행

```bash
# 1. 과거 데이터 수집 (XRP/BTC/ETH 3심볼, 15분봉, 1년치 + OI/펀딩비)
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --output data/combined_15m.parquet

# 2-A. LightGBM 모델 학습 (CPU)
python scripts/train_model.py --data data/combined_15m.parquet

# 2-B. MLX 신경망 학습 (Apple Silicon GPU)
python scripts/train_mlx_model.py --data data/combined_15m.parquet

# 3. LXC 서버에 모델 배포
bash scripts/deploy_model.sh        # LightGBM
bash scripts/deploy_model.sh mlx    # MLX (ONNX)
```

학습된 모델은 `models/lgbm_filter.pkl` (LightGBM) 또는 `models/mlx_filter.weights.onnx` (MLX) 에 저장됩니다.

> **모델 핫리로드**: 봇이 실행 중일 때 모델 파일을 교체하면, 다음 캔들 마감 시 자동으로 감지해 리로드합니다. 봇 재시작이 필요 없습니다.

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

**진입 조건**: 가중치 합계 ≥ 3 + (거래량 급증 또는 가중치 합계 ≥ 4)  
**손절/익절**: ATR × 1.5 / ATR × 3.0 (리스크:리워드 = 1:2)  
**ML 필터**: 예측 확률 ≥ 0.60 이어야 최종 진입

### 반대 시그널 재진입

보유 포지션과 반대 방향 신호가 발생하면:
1. 기존 포지션 즉시 청산 (미체결 SL/TP 주문 취소 포함)
2. ML 필터 통과 시 반대 방향으로 즉시 재진입

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
| `SYMBOL` | `XRPUSDT` | 거래 심볼 |
| `LEVERAGE` | `10` | 레버리지 배수 |
| `DISCORD_WEBHOOK_URL` | — | Discord 웹훅 URL |
| `MARGIN_MAX_RATIO` | `0.50` | 최대 증거금 비율 (잔고 대비 50%) |
| `MARGIN_MIN_RATIO` | `0.20` | 최소 증거금 비율 (잔고 대비 20%) |
| `MARGIN_DECAY_RATE` | `0.0006` | 잔고 증가 시 증거금 비율 감소 속도 |

---

## 주의사항

> **이 봇은 실제 자산을 거래합니다.** 운영 전 반드시 Binance Testnet에서 충분히 검증하세요.  
> 과거 수익이 미래 수익을 보장하지 않습니다. 투자 손실에 대한 책임은 사용자 본인에게 있습니다.  
> 성투기원합니다.
