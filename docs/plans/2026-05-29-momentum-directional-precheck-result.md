# 저빈도 모멘텀 directional precheck 결과 (TSMOM + XSMOM)

## 맥락
네 번째 precheck. 앞선 셋(stat-arb·carry·lead-lag)과 달리 **FAIL을 예단하지 않음** —
크립토 모멘텀은 문서화된 이상현상이고 저빈도라 비용 문턱이 낮다. 따라서 핵심은
**false positive 방지**(거짓 통과가 자본을 신기루에 태운다): 비용·생존편향·벤치마크·
다중비교를 빡세게.

변형: TSMOM_LS(long/short), TSMOM_LO(long/cash, 가장 현실적), XSMOM(단면 long/short).
헤드라인 검증 단위는 **사전지정**(cherry-pick 방지): TSMOM=BTC·ETH 개별 + 전코인 EW
포트폴리오 / XSMOM=포트폴리오. (per-coin 알트는 진단용.) 그리드 L∈{1,2,4,12}주 ×
skip{0,1일} = 8config × 7단위 = **56테스트**, 전부 사후최적화 없이 BH 보정.

PASS/FAIL은 데이터 보기 전 확정(상단 상수). 재사용: src.carry(한글폰트)·src.statarb.scan(BH).

## 데이터
spot 일봉 전체(2017~), 8심볼. BTC/ETH 8.8yr, 알트 5.7~8.1yr, **갭 0·NaN 0**.
TSMOM은 코인별 전체, XSMOM은 공통구간(2020-09~, 5.7yr, 2021불·2022베어·2024-25불 포함).
일봉 directional은 spot≈perp, 벤치마크 buy&hold도 spot → 내부 일관.

## 사전등록 게이트 (구성별, 전부 충족해야 PASS)
| 게이트 | 기준 |
|---|---|
| 1. economics | 순 Sharpe ≥ 0.5 AND 순수익>0 AND (TSMOM)buy&hold Sharpe 우위 |
| 2. 통계 | block bootstrap p<0.05 AND BH 보정(56) 생존 |
| 3. 리스크 | 전략 MDD ≤ buy&hold MDD |
| 4. 안정성 | IS-Sharpe≥0.5 AND OOS 순수익>0 AND **OOS buy&hold 우위 유지**(불장 beta 배제) |

추가 false-positive 방지(사용자 지시): ①long-only 변형, ②숏 캐리 0/5/10% 민감도 병기,
③헤드라인=majors+포트폴리오, ④OOS는 벤치마크 우위로 판정. 비용=턴오버×10bps(왕복).

## 결과 — 2/56 PASS, 그러나 ⚠️ MARGINAL (강건성 실패)

### 통과한 2개 (둘 다 ETH-TSMOM-LS, L=2주)
| config | 순Sh | bootP | BH | OOS strat vs bench | MDD |
|---|---|---|---|---|---|
| ETH TSMOM_LS L2/sk1 | 1.06 | 0.000 | ✓ | **0.89 vs 0.14** | −67% |
| ETH TSMOM_LS L2/sk0 | 0.98 | 0.002 | ✓ | 0.65 vs 0.14 | −76% |

이 둘은 사전등록 게이트(BH·OOS 포함)를 *진짜로* 충족했다. 정직하게 인정한다.

### 그러나 false-positive 마커 전부 점등 → 이식 불가 단일자산 아티팩트
| 마커 | 증거 |
|---|---|
| **단일 자산** | ETH만. BTC 최고(TSMOM_LO L4) Sh 0.69·bootP 0.056 → BH 탈락 |
| **단일 lookback 스파이크** | ETH-TSMOM-LS: L1 Sh −0.01/0.09 → **L2 0.98/1.06** → L4 0.66/0.54 → L12 0.42. L2만 솟음 |
| **robust 단위 전멸** | EW 포트폴리오 최고 Sh 0.98·OOS 0.66이나 **bootP 0.016 → BH 탈락**. long-only EW Sh 0.94·bootP 0.034 → BH 탈락 |
| **XSMOM 과적합** | IS Sh **1.13** → OOS Sh **0.21**(<0.5) 붕괴 → 안정성 FAIL |
| **MDD 투자불가** | 통과한 ETH조차 −67/−76% |
| **생존편향** | ETH는 생존자 → 낙관 편향(실제 더 낮음) |

벤치마크: buy&hold BTC Sh 0.47 / ETH 0.24 / EW 0.45 (MDD −81~−94%).

## 결론 — 배포 불가. "majors+포트폴리오"가 확증 못 함
사전등록 설계(헤드라인=majors+포트폴리오)는 정확히 이 함정을 잡으려 만들었고, **잡았다**:
진짜 이식 가능한 모멘텀 엣지라면 BTC·EW 포트폴리오·long-only에서 나와야 하는데
**전부 BH 보정 후 탈락**했다. 통과한 건 ETH-2주-LS 하나뿐(skip 변형 둘 = 사실상 1신호),
lookback 민감도가 극단적이고(L1서 소멸) MDD가 −70%대. 56테스트 중 1신호 생존은
이식 가능 이상현상이 아니라 **생존편향+lookback 운의 상단 꼬리**로 읽는 게 정직하다.

→ 자본 투입 대상 아님. ETH-L2를 살리려면 (a) 진짜 walk-forward 홀드아웃으로 *죽이는*
검증이 필요하고(현 OOS는 한 구간), (b) XSMOM의 IS 1.13→OOS 0.21 붕괴는 8코인이 너무
얇다는 증거 → 넓은 유니버스(상위 30~50)로만 단면 재검 의미.

## 진짜 건진 것 (작지만 사실)
모멘텀 변형들은 **2022 베어장 드로다운을 회피**했다 — buy&hold EW MDD −81% vs 변형
−45~−67%. "추세이탈 시 베어 회피"라는 모멘텀의 본질적 가치는 데이터에 실재한다.
다만 그것이 *비용·다중비교·이식성*을 넘어 거래 가능한 알파가 되진 못했다(crash 회피 ≠
수확 가능 엣지). 이는 5번째 라인이 아니라 리스크관리 오버레이로서의 단서다.

## 보존 자산
- `src/momentum/{data,precheck}.py` — TSMOM(LS/LO)·XSMOM, 비용·숏캐리·bootstrap·BH·
  IS/OOS·한글플롯. 넓은 유니버스/다른 자산에 재사용.
- 산출물: `results/momentum/momentum_precheck_2026-05-29.json` + 누적/rollingSharpe/드로다운 플롯 3.

## 4연속 precheck 종합
| 라인 | 결과 | 왜 |
|---|---|---|
| stat-arb 공적분 | FAIL | 알트 공적분 없음(전제 불성립) |
| 펀딩 캐리 | FAIL | 캐리≈자금조달비(이자율 재포장) |
| 리드-래그 | FAIL | 실재하나 ≪비용(거래 불가) |
| 저빈도 모멘텀 | **MARGINAL** | crash회피 실재하나 이식가능 알파는 아님(robust 단위 BH 탈락) |
공통: "엣지의 존재"가 아니라 **"비용·다중비교·이식성 스케일에서의 경제성"**에서 막힘.
