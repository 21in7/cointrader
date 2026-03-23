# 백테스트 시장 컨텍스트 리포트 설계

**일자**: 2026-03-22
**상태**: 설계 완료, 구현 대기

---

## 목적

Walk-Forward 백테스트 결과를 해석할 때, 각 폴드 기간의 시장 상황(BTC/ETH 추세, L/S ratio)을 함께 보여준다. **"왜 이 폴드에서 졌는가"**를 구조적으로 이해하기 위한 참조 데이터이며, 트레이딩 시그널이나 ML 피처로는 사용하지 않는다.

## 접근 방식

Walk-Forward 폴드 테이블 출력 직후에 시장 컨텍스트 테이블 2개(Market Regime + L/S Ratio)를 추가한다. 기존 `scripts/run_backtest.py`만 수정하며, 별도 CLI 명령어는 만들지 않는다.

---

## 데이터 소스

### 1. BTC/ETH 가격 데이터 (Market Regime)

- **소스**: XRP의 `data/xrpusdt/combined_15m.parquet`에 임베딩된 `close_btc`, `high_btc`, `low_btc`, `close_eth`, `high_eth`, `low_eth` 컬럼
- 별도 `data/btcusdt/combined_15m.parquet` 파일은 로컬/프로덕션 모두 **존재하지 않음**
- 백테스터가 이미 이 임베딩 컬럼을 로딩하므로 추가 데이터 fetch 불필요
- 폴드 기간별로 슬라이싱하여 수익률, ADX 계산

### 2. L/S Ratio 데이터

- **소스**: `data/{symbol}/ls_ratio_15m.parquet` (로컬 파일)
- **심볼**: XRPUSDT, BTCUSDT, ETHUSDT
- **주기**: 15m
- **컬럼**: `timestamp` (datetime64[ms, UTC]), `top_acct_ls_ratio` (float64), `global_ls_ratio` (float64)

#### 현재 데이터 상태

- L/S ratio collector는 운영 LXC(`10.1.10.24`)에서 가동 중 (commit `e2b0454`, 2026-03-22~)
- **프로덕션**: XRP/BTC/ETH 각 3건 (2026-03-22 13:15 ~ 13:45 UTC), 계속 축적 중
- **로컬**: XRP 2건, BTC 2건, ETH 2건 (로컬 collector 테스트 시 생성된 데이터)
- 과거 폴드(2025-06, 2025-09, 2025-12)에 대한 L/S ratio 데이터는 **존재하지 않음**
- Binance API는 최근 30일만 historical 제공 → 과거 데이터 복구 불가능

#### 데이터 동기화

구현 전 프로덕션 LXC에서 L/S ratio parquet 파일을 로컬로 복사해야 한다:

```bash
scp root@10.1.10.24:/root/cointrader/data/xrpusdt/ls_ratio_15m.parquet data/xrpusdt/
scp root@10.1.10.24:/root/cointrader/data/btcusdt/ls_ratio_15m.parquet data/btcusdt/
scp root@10.1.10.24:/root/cointrader/data/ethusdt/ls_ratio_15m.parquet data/ethusdt/
```

#### Fallback 전략

1. **로컬 parquet 우선**: `data/{symbol}/ls_ratio_15m.parquet`에서 폴드 기간 데이터 조회
2. **파일 없거나 해당 기간 데이터 없으면 `N/A`**: 폴드의 L/S ratio 셀을 `N/A`로 표시
3. **전체 폴드가 N/A이면 L/S ratio 테이블 자체를 생략**: 불필요한 N/A 테이블을 출력하지 않음
4. **Binance API에서 실시간 fetch하지 않음**: 백테스트는 오프라인 재현 가능해야 함
5. **시간이 지나면 해결됨**: collector가 계속 수집하므로, 데이터 축적 후 백테스트에 자연스럽게 반영

---

## Market Regime 분류 기준

BTC ADX와 수익률 기반으로 **코드에 명확히 정의**하여 주관적 해석을 방지한다:

| 조건 | 라벨 |
|------|------|
| ADX ≥ 25 and return > 0 | 상승 추세 |
| ADX ≥ 25 and return < 0 | 하락 추세 |
| ADX < 25 | 횡보 |

- ADX는 폴드 기간 내 BTC 15m 캔들(`high_btc`, `low_btc`, `close_btc`)로 계산한 **기간 평균 ADX** (`pandas_ta.adx(length=14)` 사용)
- return은 폴드 시작가 대비 종료가의 **단순 수익률** (`close_btc`)
- 라벨 뒤에 `(BTC ADX {값:.0f})` 형태로 실제 수치 병기

---

## 출력 형식

기존 폴드 테이블 바로 아래에 출력:

```
📊 Market Context per Fold
┌──────┬──────────────┬──────────────┬─────────────────────────────────┐
│ Fold │ BTC Return   │ ETH Return   │ Market Regime                   │
├──────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 1    │ +12.3%       │ +8.7%        │ 상승 추세 (BTC ADX 32)          │
│ 2    │ -2.1%        │ -5.4%        │ 횡보 (BTC ADX 18)               │
│ 3    │ +25.6%       │ +18.2%       │ 상승 추세 (BTC ADX 41)          │
└──────┴──────────────┴──────────────┴─────────────────────────────────┘

📊 L/S Ratio Context per Fold (period avg)
┌──────┬──────────────────┬──────────────────┬──────────────────┐
│ Fold │ XRP Top/Global   │ BTC Top/Global   │ ETH Top/Global   │
├──────┼──────────────────┼──────────────────┼──────────────────┤
│ 1    │ N/A              │ N/A              │ N/A              │
│ 2    │ N/A              │ N/A              │ N/A              │
│ 3    │ 1.15 / 0.98      │ 0.95 / 1.02      │ 1.08 / 1.05      │
└──────┴──────────────────┴──────────────────┴──────────────────┘
  → Fold 1~2: L/S ratio 데이터 없음 (collector 가동 전)
  → Fold 3: 데이터 가용
```

**전체 폴드가 N/A인 경우** (현재 상태에서 과거 데이터만으로 백테스트하면):

```
📊 Market Context per Fold
┌──────┬──────────────┬──────────────┬─────────────────────────────────┐
│ Fold │ BTC Return   │ ETH Return   │ Market Regime                   │
├──────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 1    │ +12.3%       │ +8.7%        │ 상승 추세 (BTC ADX 32)          │
│ 2    │ -2.1%        │ -5.4%        │ 횡보 (BTC ADX 18)               │
│ 3    │ +25.6%       │ +18.2%       │ 상승 추세 (BTC ADX 41)          │
└──────┴──────────────┴──────────────┴─────────────────────────────────┘
  ℹ️ L/S ratio 데이터 없음 — collector 데이터 축적 후 표시됩니다
```

### JSON 출력

walk-forward 결과 JSON에도 `market_context` 필드 추가:

```json
{
  "folds": [
    {
      "fold": 1,
      "test_period": "2025-06-07 ~ 2025-07-06",
      "test_start": "2025-06-07T00:00:00",
      "test_end": "2025-07-06T00:00:00",
      "summary": { "..." : "..." },
      "market_context": {
        "btc_return_pct": 12.3,
        "eth_return_pct": 8.7,
        "btc_avg_adx": 32.1,
        "market_regime": "상승 추세",
        "ls_ratio": null
      }
    },
    {
      "fold": 3,
      "test_period": "2026-03-01 ~ 2026-04-01",
      "test_start": "2026-03-01T00:00:00",
      "test_end": "2026-04-01T00:00:00",
      "summary": { "..." : "..." },
      "market_context": {
        "btc_return_pct": 5.2,
        "eth_return_pct": 3.1,
        "btc_avg_adx": 28.5,
        "market_regime": "상승 추세",
        "ls_ratio": {
          "xrp": { "top_acct_avg": 1.15, "global_avg": 0.98 },
          "btc": { "top_acct_avg": 0.95, "global_avg": 1.02 },
          "eth": { "top_acct_avg": 1.08, "global_avg": 1.05 }
        }
      }
    }
  ]
}
```

---

## 수정 대상 파일

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `scripts/run_backtest.py` | Modify | 시장 컨텍스트 계산 + 출력 함수 추가 |
| `src/backtester.py` | Modify (최소) | 폴드 결과에 `test_start`/`test_end`를 timestamp로 노출 (현재는 문자열 `test_period`만 있음) |

### 변경하지 않는 것

- `src/indicators.py` — ADX 계산은 `run_backtest.py` 내에서 `pandas_ta.adx()` 직접 사용
- `scripts/collect_ls_ratio.py` — 기존 collector 로직 변경 없음
- `src/ml_filter.py`, `src/ml_features.py` — ML 피처와 무관
- `scripts/fetch_history.py` — BTC/ETH 별도 fetch 불필요 (XRP parquet에 임베딩됨)

---

## 구현 전 선행 작업

1. ~~BTC/ETH 히스토리 데이터 fetch~~ → **불필요** (XRP parquet에 `close_btc`, `close_eth` 등 임베딩됨)
2. `backtester.py`에서 `test_start`/`test_end`를 timestamp로 노출하도록 수정
3. 프로덕션 LXC에서 L/S ratio parquet 파일 로컬 동기화

---

## 구현 범위 제한

- **참조 전용**: 시장 컨텍스트는 출력/리포트에만 사용. 트레이딩 로직에 영향 없음
- **오프라인 우선**: Binance API 호출 없음. 로컬 데이터만 사용
- **기존 테스트 영향 없음**: 출력 함수 추가이므로 기존 백테스트 로직 불변
- **L/S ratio 테이블 조건부 출력**: 전체 N/A이면 테이블 생략, 한 줄 안내 메시지만 출력
