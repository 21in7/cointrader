# 온체인 netflow 디리스크 오버레이 (비방향성) — 설계

**날짜**: 2026-06-08
**상태**: 설계 완료, 구현 진행
**성격**: falsification-first precheck. directional precheck FAIL 후 (c) 비방향성 피벗.

## 배경
같은 날 directional netflow precheck FAIL(0/12). 레그 비대칭이 진단을 줬다 — 롱 레그는
세속 강세장 베타일 뿐, **숏 레그(유입→숏)가 강하게 음**. 즉 netflow는 *방향*을 못
맞추지만 **유입 급증 = 분산(distribution) = 하방 리스크**라는 비대칭 정보는 있을 수 있다.

momentum/crisis-alpha/ETH-L2 연구가 반복 시사: "이 신호들은 방향성 알파가 아니라 레짐/
리스크 오버레이(크래시 회피)." netflow를 **진입 방향**이 아니라 **노출 조절**에 쓴다.

## 핵심 위험 — 리스크 오버레이는 자기기만이 쉽다
"Sharpe 좋아졌다"는 가짜로 만들기 쉽다. 고변동 자산에서 *아무* 디리스크나 크래시 구간
노출을 줄여 Sharpe를 올린다. 그래서 설계 핵심 = **anti-fooling 벤치마크**.

## 가설
거래소 인플로우 급증(netflow z↑)은 하방 리스크를 선행한다. 이때 롱 BTC/ETH 노출을
줄이면 위험조정수익(Sharpe↑, MDD↓)이 buy&hold 대비 개선되며, 이 개선은 (a) 단순 노출
감소나 (b) 몇 개 크래시 에피소드로 설명되지 않는다.

## 전략 (사전등록, 잠금)
- 기본: 롱 BTC/ETH (exposure=1). **숏 없음**(directional서 실패한 곳).
- `netflow_z_{t−1−LAG} > THRESHOLD`(인플로우 급증) → 그날 exposure=0(플랫)
- z-window=90d, LAG=1 (directional과 동일), THRESHOLD=1.0 1차
- 정렬 `shift(1+LAG)` lookahead·flash개정 차단
- 비용: 노출 토글당 1 fill = (taker 0.04%+slip 0.01%)=5bps. 디리스크 1에피소드=왕복 10bps

## Anti-fooling 벤치마크
| 벤치 | 정의 | 차단 대상 |
|---|---|---|
| B0 | buy&hold | 이겨야 할 대상 |
| B1 | **변동성 관리**: 실현변동성(30d) z>임계 시 플랫, **디리스크 빈도를 netflow와 매칭** | netflow가 단순 vol-timing보다 나은가 |
| B2 | **랜덤 디리스크**: netflow와 동일 디리스크 일수 무작위, 부트스트랩 N=1000 | "아무 디리스크나 도움" 착시 |

## 사전등록 게이트
| # | 게이트 | 기준 |
|---|---|---|
| 1 | 개선 | overlay Sharpe > B0 Sharpe **AND** overlay MDD < B0 MDD (비용 후) |
| 2 | 정보(핵심) | 랜덤 null(B2)을 Sharpe-gain p<0.05로 이김 **AND** overlay Sharpe ≥ B1(vol) Sharpe |
| 3 | 레짐 | 사전지정 크래시구간(COVID 2020-03, LUNA 2022-05, FTX 2022-11, 2018-bear) 제거 후에도 Sharpe 개선 부호 생존 |
| 4 | 안정성 | 1차 THRESHOLD가 OOS(마지막 30%)서도 overlay Sharpe>B0 Sharpe |
| P | 포트폴리오 | BTC·ETH 둘 다(또는 EW) 개선 |

**PASS = 1·2·3·4 충족 AND (BTC·ETH 둘 다 or EW)**

### 사전등록 상수
```
Z_WINDOW=90, LAG_DAYS=1, THRESHOLD=1.0 (1차)
GRID_THRESHOLD = [0.5, 1.0, 1.5, 2.0]   # robustness
FEE+SLIP per fill = 5bps; toggle당 부과
VOL_WINDOW=30 (B1 실현변동성)
N_RANDOM=1000 (B2 부트스트랩), SEED=42
IS_FRACTION=0.70
CRASH_WINDOWS = [2018-01~2018-02, 2020-03~2020-04, 2022-05, 2022-11]
ASSETS=[btc,eth] + EW
ANN=365
```

## 산출물
- `src/onchain/overlay.py`
- `results/onchain/onchain_overlay_{date}.json` + 플롯(equity vs B0/B1, MDD 비교, 디리스크일 분포, 크래시구간 분해)
- `docs/plans/2026-06-08-onchain-netflow-overlay-result.md`

## 예상 결과
falsification-first: 개선이 (a) vol-관리(B1)보다 못하거나 (b) 크래시 에피소드 집중(게이트3
탈락=crisis-alpha 함정 재현) 가능성 높음. 단 인플로우가 vol이 놓치는 분산을 *선행*할
여지 있어 잔존 가능성 0은 아님. 게이트 2·3이 정직하게 가린다.
```
