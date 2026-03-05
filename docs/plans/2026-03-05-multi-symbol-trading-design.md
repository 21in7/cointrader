# Multi-Symbol Trading Design

## 개요

현재 XRP 단일 심볼 선물 거래 봇을 TRX, DOGE 등 다중 심볼 동시 거래로 확장한다.

## 요구사항

- **거래 심볼**: XRPUSDT, TRXUSDT, DOGEUSDT (3개, 추후 확장 가능)
- **상관관계 심볼**: BTCUSDT, ETHUSDT (기존과 동일)
- **ML 모델**: 심볼별 개별 학습·배포
- **포지션**: 심볼별 동시 포지션 허용 (최대 3개)
- **리스크**: 심볼별 독립 운영 + 글로벌 한도 (일일 손실 5%)
- **동일 방향 제한**: 같은 방향(LONG/SHORT) 최대 2개까지 (BTC 급락 시 3배 손실 방지)

## 접근법: 심볼별 독립 TradingBot 인스턴스 + 공유 RiskManager

기존 TradingBot의 단일 포지션 상태 머신을 유지하면서, 각 심볼마다 독립 인스턴스를 생성하고 `asyncio.gather()`로 병렬 실행한다. RiskManager만 싱글턴으로 공유하여 글로벌 리스크를 관리한다.

### 선택 이유

- 기존 TradingBot 상태 머신 수정 최소화
- 심볼 간 완전 격리 — 한 심볼의 에러가 다른 심볼에 영향 없음
- 점진적 확장 용이 (새 심볼 = 새 인스턴스 추가)
- 각 단계마다 기존 XRP 단일 모드로 테스트 가능

### 기각된 대안: 단일 Bot + 심볼 라우팅

하나의 TradingBot에서 `Dict[str, PositionState]`로 관리하는 방식. WebSocket 효율적이나 상태 머신 대규모 리팩토링 필요, 한 심볼 에러가 전체에 영향, 복잡도 대폭 증가.

## 설계 상세

### 1. Config 변경

```python
# .env
SYMBOLS=XRPUSDT,TRXUSDT,DOGEUSDT
CORRELATION_SYMBOLS=BTCUSDT,ETHUSDT
MAX_SAME_DIRECTION=2

@dataclass
class Config:
    symbols: list[str]              # ["XRPUSDT", "TRXUSDT", "DOGEUSDT"]
    correlation_symbols: list[str]  # ["BTCUSDT", "ETHUSDT"]
    max_same_direction: int         # 같은 방향 최대 수 (기본 2)
    # symbol: str 필드 제거
```

- 기존 `SYMBOL` 환경변수 제거, `SYMBOLS`로 통일
- `config.symbol` 참조하는 코드 모두 → 각 봇 인스턴스의 `self.symbol`로 전환
- 하위호환: `SYMBOLS` 미설정 시 기존 `SYMBOL` 값을 1개짜리 리스트로 변환

### 2. 실행 구조 (main.py)

```python
async def main():
    config = Config()
    risk = RiskManager(config)  # 공유 싱글턴

    bots = []
    for symbol in config.symbols:
        bot = TradingBot(config, symbol=symbol, risk=risk)
        bots.append(bot)

    await asyncio.gather(*[bot.run() for bot in bots])
```

- 각 봇은 독립적인 MultiSymbolStream, Exchange, UserDataStream 보유
- RiskManager만 공유

### 3. TradingBot 생성자 변경

```python
class TradingBot:
    def __init__(self, config: Config, symbol: str, risk: RiskManager):
        self.symbol = symbol
        self.config = config
        self.exchange = BinanceFuturesClient(config, symbol=symbol)
        self.risk = risk  # 외부에서 주입 (공유)
        self.ml_filter = MLFilter(model_dir=f"models/{symbol.lower()}")
        ...
```

- `config.symbol` 의존 완전 제거
- 각 봇이 자기 심볼을 직접 소유

### 4. Exchange 심볼 분리

```python
class BinanceFuturesClient:
    def __init__(self, config: Config, symbol: str):
        self.symbol = symbol  # config.symbol → self.symbol
```

- 모든 API 호출에서 `self.config.symbol` → `self.symbol`

### 5. RiskManager 공유 설계

```python
class RiskManager:
    def __init__(self, config):
        self.daily_pnl = 0.0
        self.open_positions: dict[str, str] = {}  # {symbol: side}
        self.max_positions = config.max_positions
        self.max_same_direction = config.max_same_direction  # 기본 2
        self._lock = asyncio.Lock()

    async def can_open_new_position(self, symbol: str, side: str) -> bool:
        async with self._lock:
            if len(self.open_positions) >= self.max_positions:
                return False
            if symbol in self.open_positions:
                return False
            same_dir = sum(1 for s in self.open_positions.values() if s == side)
            if same_dir >= self.max_same_direction:
                return False
            return True

    async def register_position(self, symbol: str, side: str):
        async with self._lock:
            self.open_positions[symbol] = side

    async def close_position(self, symbol: str, pnl: float):
        async with self._lock:
            self.open_positions.pop(symbol, None)
            self.daily_pnl += pnl

    def is_trading_allowed(self) -> bool:
        # 글로벌 일일 손실 한도 체크 (기존과 동일)
```

- `asyncio.Lock()`으로 동시 접근 보호
- 동일 방향 2개 제한으로 BTC 급락 시 3배 손실 방지
- 마진은 심볼 수(N)로 균등 배분

### 6. 데이터 스트림

각 TradingBot이 자기만의 MultiSymbolStream 인스턴스를 가짐:

```
XRP Bot:  [XRPUSDT, BTCUSDT, ETHUSDT]
TRX Bot:  [TRXUSDT, BTCUSDT, ETHUSDT]
DOGE Bot: [DOGEUSDT, BTCUSDT, ETHUSDT]
```

- BTC/ETH 데이터 중복 수신되지만 격리성 확보
- 각 stream의 primary_symbol이 달라 candle close 콜백 독립적

### 7. 모델 & 데이터 디렉토리 분리

```
models/
├── xrpusdt/
│   ├── lgbm_filter.pkl
│   └── mlx_filter.weights.onnx
├── trxusdt/
│   └── ...
└── dogeusdt/
    └── ...

data/
├── xrpusdt/
│   └── combined_15m.parquet
├── trxusdt/
│   └── combined_15m.parquet
└── dogeusdt/
    └── combined_15m.parquet
```

- 각 parquet: 해당 심볼이 primary + BTC/ETH가 correlation
- feature 구조 동일 (26 features)

### 8. 학습 파이프라인 CLI 통일

모든 스크립트에 `--symbol`과 `--all` 패턴 적용:

```bash
# 단일 심볼
bash scripts/train_and_deploy.sh --symbol TRXUSDT
python scripts/fetch_history.py --symbol DOGEUSDT
python scripts/train_model.py --symbol TRXUSDT
python scripts/tune_hyperparams.py --symbol DOGEUSDT

# 전체 심볼
bash scripts/train_and_deploy.sh --all
bash scripts/train_and_deploy.sh  # 인자 없으면 --all 동일

# MLX + 단일 심볼
bash scripts/train_and_deploy.sh mlx --symbol DOGEUSDT
```

## 구현 순서

각 단계마다 기존 XRP 단일 모드로 테스트 가능하도록 점진적 전환:

1. **Config** — `symbols` 리스트, `max_same_direction` 추가
2. **RiskManager** — 공유 싱글턴, asyncio.Lock, 동일 방향 제한
3. **exchange.py** — `config.symbol` → `self.symbol` 분리
4. **bot.py** — 생성자에 `symbol`, `risk` 파라미터 추가, `config.symbol` 제거
5. **main.py** — 심볼별 봇 인스턴스 생성 + `asyncio.gather()`
6. **학습 스크립트** — `--symbol`/`--all` CLI, 디렉토리 분리

## 변경 불필요한 컴포넌트

- `src/indicators.py` — 이미 심볼에 독립적
- `src/notifier.py` — 이미 symbol 파라미터 수용
- `src/user_data_stream.py` — 이미 심볼별 필터링 지원
- `src/ml_features.py` — 이미 primary + auxiliary 구조
- `src/label_builder.py` — 이미 범용적
