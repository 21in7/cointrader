# CoinTrader

Binance Futures 자동매매 봇. 복합 기술 지표와 LightGBM ML 필터를 결합하여 XRPUSDT(기본) 선물 포지션을 자동으로 진입·청산하며, Discord로 실시간 알림을 전송합니다.

---

## 주요 기능

- **복합 기술 지표 신호**: RSI, MACD 크로스, 볼린저 밴드, EMA 정/역배열, Stochastic RSI, 거래량 급증 — 3개 이상 일치 시 진입
- **ML 필터 (LightGBM)**: 기술 지표 신호를 한 번 더 검증하여 오진입 차단 (모델 없으면 자동 폴백)
- **ATR 기반 손절/익절**: 변동성에 따라 동적으로 SL/TP 계산 (1.5× / 3.0× ATR)
- **리스크 관리**: 트레이드당 리스크 비율, 최대 포지션 수, 일일 손실 한도 제어
- **포지션 복구**: 봇 재시작 시 기존 포지션 자동 감지 및 상태 복원
- **자동 재학습**: 매일 새벽 3시 ML 모델 재학습 및 핫 리로드
- **Discord 알림**: 진입·청산·오류 이벤트 실시간 웹훅 알림
- **CI/CD**: Jenkins + Gitea Container Registry 기반 Docker 이미지 자동 빌드·배포

---

## 프로젝트 구조

```
cointrader/
├── main.py                  # 진입점
├── src/
│   ├── bot.py               # 메인 트레이딩 루프
│   ├── config.py            # 환경변수 기반 설정
│   ├── exchange.py          # Binance Futures API 클라이언트
│   ├── data_stream.py       # WebSocket 1분봉 스트림
│   ├── indicators.py        # 기술 지표 계산 및 신호 생성
│   ├── ml_filter.py         # LightGBM 진입 필터
│   ├── ml_features.py       # ML 피처 빌더
│   ├── label_builder.py     # 학습 레이블 생성
│   ├── retrainer.py         # 모델 자동 재학습 스케줄러
│   ├── risk_manager.py      # 리스크 관리
│   ├── notifier.py          # Discord 웹훅 알림
│   └── logger_setup.py      # Loguru 로거 설정
├── scripts/
│   ├── fetch_history.py     # 과거 데이터 수집
│   └── train_model.py       # ML 모델 수동 학습
├── models/                  # 학습된 모델 저장 (.pkl)
├── data/                    # 과거 데이터 캐시
├── logs/                    # 로그 파일
├── tests/                   # 테스트 코드
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
RISK_PER_TRADE=0.02
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

봇은 모델 파일(`models/lgbm_filter.pkl`)이 없으면 ML 필터 없이 동작합니다. 최초 실행 전 또는 수동 재학습 시 아래 순서로 진행합니다.

```bash
# 1. 과거 데이터 수집
python scripts/fetch_history.py

# 2. 모델 학습
python scripts/train_model.py
```

학습된 모델은 `models/lgbm_filter.pkl`에 저장되며, 봇이 실행 중이면 매일 새벽 3시에 자동으로 재학습·리로드됩니다.

---

## 매매 전략

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
**ML 필터**: LightGBM 예측 확률 ≥ 0.60 이어야 최종 진입

---

## CI/CD

`main` 브랜치에 푸시하면 Jenkins 파이프라인이 자동으로 실행됩니다.

1. **Checkout** — 소스 체크아웃
2. **Build Image** — Docker 이미지 빌드 (`:{BUILD_NUMBER}` + `:latest` 태그)
3. **Push** — Gitea Container Registry(`10.1.10.28:3000`)에 푸시
4. **Cleanup** — 로컬 이미지 정리

배포 서버에서 최신 이미지를 반영하려면:

```bash
docker compose pull && docker compose up -d
```

---

## 테스트

```bash
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
| `RISK_PER_TRADE` | `0.02` | 트레이드당 리스크 비율 (2%) |
| `DISCORD_WEBHOOK_URL` | — | Discord 웹훅 URL |

---

## 주의사항

> **이 봇은 실제 자산을 거래합니다.** 운영 전 반드시 Binance Testnet에서 충분히 검증하세요.  
> 과거 수익이 미래 수익을 보장하지 않습니다. 투자 손실에 대한 책임은 사용자 본인에게 있습니다.
