# Stat-arb 공적분 precheck 결과 (XRP/BTC + 크로스-알트 스캔)

## 맥락
방향 예측 패러다임이 10전10패로 종결된 뒤, post-mortem(2026-05-04) §5 우선순위 1
**"심볼이 아니라 패러다임을 바꿔라 — stat-arb / pair trading"**을 검증한다.

목표: 과거 실패를 죽인 두 병 — ① 얇은 raw edge가 수수료 못 이김, ② LONG/SHORT
비대칭(=시장 beta 의존) — 을 **방향 중립**으로 구조적으로 회피할 수 있는지.
stat-arb의 전제는 **공적분(cointegration)**: 두 자산이 떨어져도 다시 평형으로
회귀하는 안정적 스프레드가 존재해야 한다. 그 전제부터 검증한다.

## 방법론 — 반증-우선 precheck (백테스트 아님)
**단 하나의 PASS/FAIL 게이트가 산출물.** FAIL이면 백테스트 모듈을 작성하지 않고 중단.
PASS/FAIL 임계값은 **데이터를 보기 전에 사전등록**(코드 상단 상수)하고 결과 후 변경 금지.

- 모듈: `src/statarb/precheck.py`(단일 페어), `src/statarb/scan.py`(전수 스캔)
- 재사용: `src.backtester._load_data / _calc_fee / _apply_slippage`
- 추정기 self-test 통과: OU half-life 복원 오차 5%, RW Hurst R/S 0.537·VR 0.499

### 사전등록 게이트 (전부 충족해야 PASS)
| 게이트 | 기준 |
|---|---|
| I(1) 적분차수 | log-level ADF p>0.10 AND Δlog ADF p<0.05 (양 자산) |
| Engle-Granger 공적분 | p<0.05 (스캔은 Bonferroni α/15=0.0033) |
| Johansen 공적분 | trace r=0 통계량 > 95% 임계값 |
| half-life (OU) | 30분 < HL < 7일 |
| OOS 정상성 | IS-β 스프레드의 OOS-ADF p<0.05 |
| rolling 공적분 | EG p<0.05인 달력시간 비중 ≥ 70% |
| 거래가능성 | 2σ 진폭 > (왕복비용+캐리)×3 |
| Hurst | ★게이트 제외 → 보조 진단 (R/S는 평균회귀서 상향편향) |

비용 모델: 2레그×(진입+청산)=4 fills × (taker 0.04% + slip 0.01%) = 20bps 왕복,
+ 보유 캐리 `CARRY_BPS_PER_DAY=3` × half-life(일). 캐리는 가정값(실측 funding 미반영).

## 데이터
- XRP/BTC: `data/xrpusdt/combined_15m.parquet`, 76,523행, 15분봉(갭 0),
  2024-03-22 ~ 2026-05-28. 페어 = log(close) vs log(close_btc).
- 크로스-알트: AVAX·DOGE·LINK·SOL·TRX·XRP 6심볼, 페어별 자체 공통구간(inner join,
  N=35,040~43,113). 15페어.

## 결과 1 — XRP/BTC 단일 페어: ❌ FAIL (6/7 게이트 탈락)

| # | 게이트 | 결과 | 값 |
|---|---|---|---|
| 1 | I(1) | ✅ | log-level 비정상 / Δlog 정상 |
| 2 | Engle-Granger | ❌ | p=0.393 (BTC~XRP 방향도 0.237) |
| 3 | Johansen | ❌ | trace r=0 13.16 < 15.49 (r=0 기각 실패) |
| 4 | half-life | ❌ | 47.4일 (λ=-0.00015, 사실상 랜덤워크) |
| 5 | OOS 정상성 | ❌ | OOS-ADF p=0.410 |
| 6 | rolling 공적분 | ❌ | 9.9% of 달력시간만 |
| 7 | 거래가능성 | ⚠️"PASS" | 12.1× — **허위** |
| — | Hurst(보조) | — | R/S 0.530 / VR 0.487 ≈ 0.5 (랜덤워크) |

- **EG와 Johansen이 독립적으로 일치** → "공적분 없음". half-life 47일·OOS p=0.41·
  rolling 9.9%·Hurst≈0.5가 전부 같은 결론으로 수렴 = 강건한 FAIL.
- **게이트 7 "PASS"는 함정**: 2σ 진폭(5,889bps)이 큰 이유는 스프레드가 비정상(drift)이라
  σ가 크기 때문. 회귀하지 않는 큰 진폭은 기회가 아니라 비정상성의 부작용 → 무의미.

## 결과 2 — 크로스-알트 전수 스캔: ❌ 0/15 PASS

| 보정 | 통과 |
|---|---|
| raw p<0.05 (보정 전) | 1개 (AVAX/DOGE p=0.0125) — 우연 기대 거짓양성 0.75개와 일치 = 노이즈 |
| Bonferroni (α/15) | **0개** |
| BH-FDR | **0개** |
| SCREEN PASS (Bonf+Johansen+half-life+OOS) | **0개** |

- **half-life 분포: 최소 7.9일 / 중앙값 24일 / 최대 173일. 7일 미만 0개.**
  모든 페어가 랜덤워크 스케일 → 거래가능한 평균회귀 전무.
- numpy matmul 경고(일부 Johansen 고유값 계산)는 결과 비오염 확인(비정상 값 0개).

## 결론 — FAIL, stat-arb 라인 종료
**크립토 알트/메이저는 공통 시장 beta로 함께 움직이지만(co-trending) 공적분되지 않는다**
(다시 회귀하는 평형 스프레드가 없음). 공적분이 없으니 stat-arb pair trading의 전제 자체가
이 자산 유니버스에서 성립하지 않는다. XRP/BTC 단일 페어와 15개 알트 페어가 모두,
다중비교 보정 후, 일관되게 같은 결론.

- **방법론 승리**: 싸고 결정적인 precheck가 비싼 백테스트를 짜기 전에 아이디어를 죽였다.
  반증-우선의 본래 가치. (post-mortem §6 "검증 인프라는 자산"의 실증.)
- **보존 자산**: `src/statarb/precheck.py`·`scan.py`는 임의 페어/유니버스에 재사용 가능.
  외부 자산(주식, 다른 거래소, 온체인 토큰쌍)으로 유니버스를 넓히면 동일 게이트로 재검증 가능.
- 산출물: `results/statarb/xrpbtc_precheck_2026-05-29.json`(+플롯 3),
  `results/statarb/crossalt_scan_2026-05-29.json`(+히트맵).

## 남은 미탐색 (이 결과가 닫지 않은 것)
이번 FAIL은 "Binance 보유 6알트+BTC/ETH"라는 좁은 유니버스에 한정. stat-arb 자체의
사망선고가 아니라 **이 유니버스에 공적분 페어가 없다**는 사실. 진짜 stat-arb는 보통
같은 기초자산의 다른 표현(현물-선물 베이시스, 거래소간 같은 토큰, ETF-NAV 등)에서 나온다.
그쪽은 새 데이터 소스가 필요 → 별도 의사결정.
