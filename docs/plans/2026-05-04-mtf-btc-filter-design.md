# MTF + BTC 추세 필터 백테스트 설계

## 가설

BTC 추세 방향과 일치하는 MTF 풀백 시그널만 실행하면 비용 반영 후에도 PF > 1.2를 달성한다.

## 메인 가설 (사전 확정 — commitment device)

> 메인 가설은 sweep 결과를 보기 **전에** BTC 1h + EMA 50/200 + ADX > 20으로 확정한다.
> sweep 12개 결과에서 가장 좋은 조합으로 사후 변경하지 않는다.
> 4h/1d 결과는 robustness 참고용이며, 메인 가설이 OOS 실패 시
> "다른 조합이 됐으니 PASS"로 구제하지 않는다.

- **BTC 타임프레임**: 1h
- **BTC EMA**: fast=50, slow=200
- **BTC ADX 임계값**: 20
- **선택 근거**: XRP MTF bot의 1h 메타필터와 동일 기준 → 시그널 정합성 확보, 사후 정당화 차단

## 필터 로직

```
BTC_trend = (BTC EMA_fast > BTC EMA_slow) AND (BTC ADX > 20)
    ? (EMA_fast > EMA_slow ? "UP" : "DOWN")
    : "NEUTRAL"

if BTC_trend == "UP":    SHORT 차단, LONG만 허용
if BTC_trend == "DOWN":  LONG 차단, SHORT만 허용
if BTC_trend == "NEUTRAL": 양방향 차단 (추세 불명확)
```

## Sweep 파라미터 (robustness check용)

| 파라미터 | 후보 | 설명 |
|---------|------|------|
| BTC 타임프레임 | 1h, 4h, 1d | BTC 추세 판단 주기 |
| BTC EMA fast | 20, 50 | 단기 EMA |
| BTC EMA slow | 100, 200 | 장기 EMA |

총 조합: 3 × 2 × 2 = 12개. ADX > 20은 전 조합 고정.

## 데이터

- XRP: `data/xrpusdt/combined_15m.parquet` (기존)
- BTC: `data/btcusdt/combined_15m.parquet` (fetch 필요)
- 기간: 최소 6개월 (XRP와 동일 기간)
- BTC 15m → 1h/4h/1d resample 후 EMA/ADX 계산
- merge: `merge_asof(direction="backward")` — look-ahead bias 방지
- fetch 후 XRP/BTC 첫/마지막 timestamp + bar 수 일치 검증 필수

## IS/OOS 분할

- 앞 70% IS, 뒤 30% OOS (단순 시간 분할)
- ML 없고 sweep 12개뿐이므로 walk-forward 불필요

## 합격 기준

| 기준 | 값 | 비고 |
|------|-----|------|
| 메인 가설 OOS fees_only PF | >= 1.2 | 실거래 마진 확보 |
| 메인 가설 OOS realistic PF | >= 1.0 | 슬리피지+펀딩 반영 후 흑자 |
| LONG/SHORT 양쪽 fees_only PF | >= 0.8 | 대칭성 |
| OOS 거래 수 | >= 50 | 통계적 유의성 |
| IS/OOS PF 격차 | < 30% | 과적합 방지 |
| 베이스라인 대비 OOS PF | 명확한 개선 | 절대값보다 차이 중요 |
| IS 거래 수 | >= 100 | 미달 시 조합 자동 제외 |

## 판정 흐름

1. 베이스라인(BTC 필터 없는 MTF) IS/OOS 결과 먼저 산출
2. 12개 조합 IS sweep, 모두 결과 저장 (IS 거래 수 < 100 자동 제외)
3. 메인 가설(1h/EMA50-200/ADX20) OOS 검증 → 합격 기준 통과 시 PASS
4. 나머지 11개도 OOS 검증 → robustness 보고서 (참고용)
5. 메인 가설 실패 시 → 전략 폐기 (다른 조합으로 구제 안 함)

## 산출물

- `scripts/mtf_btc_filter_backtest.py` — 백테스트 스크립트
- `docs/plans/2026-05-04-mtf-btc-filter-result.md` — 결과 문서
- 거래 수준 로그 (CSV) — entry/exit/BTC trend/PnL 포함, 사후 분석용
