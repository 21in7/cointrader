# 리드-래그 directional 시그널 precheck 결과 (BTC/ETH → 6알트)

## 맥락
"BTC/ETH 15m 수익률이 알트를 k바 선행하고, 그 예측가능 폭이 비용을 넘게 거래
가능한가?" 세 번째 반증-우선 precheck. 유동성 15m 메이저는 이미 차익소거됐을 확률↑
→ 예상 FAIL. 가장 싸고 결정적인 킬(예측 edge < 비용)을 첫 게이트로. PASS면 기존
`src/backtester.py` 방향성 엔진이 바로 받는 신호.

## 방법론 — falsification-first
PASS/FAIL은 데이터 보기 전 확정(상단 상수). 사후 변경 금지.
- 모듈: `src/leadlag/precheck.py`
- 재사용: `src.backtester`(비용/로더), `src.statarb.scan`(BH), `src.carry`(한글폰트)
- 데이터: 각 알트 `combined_15m.parquet`의 close(alt)·close_btc·close_eth 정렬 내장
  → 파일 내에서 leader/alt 수익률 완벽 정렬. 12쌍(BTC/ETH × 6알트), 알트별 전구간.

### 사전등록 게이트 (쌍별, 전부 충족해야 PASS)
| 게이트 | 기준 |
|---|---|
| 1. 경제성 | 베스트 lag 예측 edge > 왕복비용(10bps)×1.5 = **15bps** (uncond/conditional 중 하나라도) |
| 2. 통계 | block bootstrap p<0.05 AND BH 보정 후 생존 (단순 t 금지 — 자기상관 유의도 부풀림) |
| 3. 진위 | 비대칭비 (leader→alt)/(alt→leader) ≥ 2 (동시상관 누수 아님) |
| 4. 안정성 | IS 베스트 lag의 OOS 부호일치 AND OOS 유의 |

예측 edge ≈ |corr_k| × σ(alt 1bar). 비용 = 단일레그 2 fills × (taker 0.04%+slip 0.01%)
= 10bps. block bootstrap: 이동 블록(96bar=1일), N=1000, 자기상관·변동성클러스터 보존.

## 결과 — 0/12 PASS, ❌ 전 쌍 FAIL

### 게이트 1(경제성)에서 헤드라인 전멸
12쌍 베스트 lag 예측 edge가 **0.2 ~ 2.1bps**로 전부 15bps 임계 미달. 베스트 lag
상관 |corr| ≤ 0.027. 가장 싼 게이트가 결정적 킬.

### 전체 리포트 (예측 edge 내림차순, 발췌)
| pair | lag | corr | edge | bootP | BH | asym | OOS부호 | econ/stat/asym/stab | PASS |
|---|---|---|---|---|---|---|---|---|---|
| BTC→LINK | 1 | −0.014 | 2.1b | 0.124 | · | 1.8 | · | ·/·/·/· | FAIL |
| ETH→LINK | 3 | −0.027 | 1.5b | 0.028 | · | 4.0 | · | ·/·/✓/· | FAIL |
| ETH→AVAX | 3 | −0.025 | 1.4b | 0.014 | · | 4.9 | · | ·/·/✓/· | FAIL |
| **ETH→TRX** | 1 | +0.023 | 1.2b | **0.000** | **✓** | **11.5** | **✓** | ·/✓/✓/· | FAIL |
| **BTC→TRX** | 5 | −0.015 | 0.2b | **0.008** | **✓** | **22.4** | **✓** | ·/✓/✓/· | FAIL |
| (나머지 7쌍) | | |0.4~1.1b| >0.07 | · | <5 | · | ·/·/·/· | FAIL |

## 결론 — FAIL, "진짜지만 거래 불가"
**리드-래그는 일부 쌍(BTC/ETH→TRX)에서 통계적으로 실재한다** — bootstrap p<0.01,
BH 보정 생존, 비대칭비 11~22(진짜 lead, 동시상관 누수 아님). 그러나:

1. **예측 edge가 1~2bps로 ~15bps 비용의 1/10**. 시장이 *거래 가능한* 부분은 이미
   차익소거하고, 경제적으로 쓸모없는 잔차만 남았다. 효율적 시장의 교과서적 모습.
2. **OOS 비유의**(stab ✗): IS 유의가 held-out 30%서 유의도 미달 → 실거래 신뢰 불가.
3. 나머지 10쌍은 통계적으로도 무의(bootP>0.07) — 잡음.

economics-first 순서가 헤드라인 한 줄(0.2~2.1bps)로 전 쌍을 죽였고, 무거운 검정은
"TRX엔 진짜 lead가 있으나 너무 작다"는 정밀 묘사를 더했다. **15m 유동성 메이저→알트
방향성 리드-래그는 비용 스케일에서 비-엣지.** 방향성 backtester로 넘길 신호 없음.

## Caveat
15m close 동기성 가정. tick 데이터 없이 stale-price/비동기거래 lead 누수를 완전
배제할 수 없음(15m 메이저선 작지만). TRX의 진짜 lead 일부도 이 아티팩트일 여지 있음 —
어차피 edge가 비용 미달이라 결론 불변.

## 보존 자산
- `src/leadlag/precheck.py` — CCF·조건부 edge·block bootstrap·BH·비대칭·IS/OOS,
  한글 플롯. 임의 (leader, follower) 쌍에 재사용.
- 산출물: `results/leadlag/leadlag_precheck_2026-05-29.json` + CCF히트맵/최고쌍CCF/rolling 플롯 3.
