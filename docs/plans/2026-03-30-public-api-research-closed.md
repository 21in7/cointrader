# Binance Public API Signal Research: 공식 종료

**Date**: 2026-03-30
**Status**: CLOSED — 단독 edge 가진 피처 없음

## 전수 테스트 결과

| # | 피처 | 상관계수 | 백테스트 Best PF | 판정 |
|---|------|---------|-----------------|------|
| 1 | RSI+MACD+BB+EMA+StochRSI | — | 0.89 | 폐기 (수익성 부족) |
| 2 | top_acct_ls_ratio | r=+0.116 (1h) | 0.86 | 폐기 (전 조합 < 1.0) |
| 3 | global_ls_ratio | r=+0.048 (1h) | — | 약함, 미실시 |
| 4 | top_pos_ls_ratio | r=+0.032 (1h) | — | 약함, 미실시 |
| 5 | Funding Rate | r=-0.054 (4h) | — | 약함, 미실시 |
| 6 | FR × OI변화율(1h) | r=-0.173 (4h) | SHORT 1.88 / LONG 0.50 | 폐기 (대칭성 실패) |
| 7 | Taker Buy/Sell Ratio | r=-0.079 (1h) | 0.93 | 폐기 (전 조합 < 1.0) |
| 8 | Liquidation (allForceOrders) | API 폐기됨 | — | 사용 불가 |

## 핵심 교훈

1. **r < 0.15는 거래비용(0.08%) 커버 불가** — 모든 피처에서 확인
2. **대칭성(LONG+SHORT) 없는 시그널은 시장 베타** — FR×OI에서 확인
3. **8일~29일 데이터는 "방향성 힌트"** — 최종 판정엔 더 긴 기간 필요
4. **PF 1.88도 거절할 수 있어야 함** — 설계 기준 사전 수립이 핵심

## 4월 15일 재검증

- crontab 등록 완료 (프로덕션 10.1.10.24)
- `scripts/revalidate_apr15.py` — L/S ratio + FR×OI 동시 재실행
- 추가 데이터(24일/29일)로 동일 테스트
- 여전히 실패 시 확정 폐기

## 다음 방향 (4월 이후)

Binance 공개 API 한정으로는 단독 edge 불가. 탐색 필요:
- 온체인 데이터 (whale wallet tracking, exchange inflow/outflow)
- 크로스 거래소 데이터 (OKX, Bybit OI/FR 차이)
- 소셜 센티먼트 (Fear & Greed, Twitter/X sentiment)
- 멀티피처 복합 시그널 (ML with L/S + FR + OI + Taker 조합)
- 다른 타임프레임 (1h, 4h candle 기반)
