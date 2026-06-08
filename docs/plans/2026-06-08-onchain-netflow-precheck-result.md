# 온체인 거래소 netflow directional precheck 결과 (BTC/ETH 일봉)

## 맥락
"유료 API를 쓰면 수익 가능성이 있나?"에서 출발. 핵심 재구성 — 비용이 아니라
**데이터에 담긴 엣지가 비용·실행제약을 넘느냐**가 변수. 알파 추구 10전10패는 전부
거래소 *내부* 신호(가격·OI·펀딩·L:S·리드래그·모멘텀·공적분·캐리)였고, 유료 데이터 중
**무료 공개 API와 정보 차원이 다른** 유일한 미탐색 후보 = **온체인**.

결제 전 무료 티어로 precheck. 거래소 netflow는 온체인 방향성 신호의 정석.

## 방법론 — falsification-first
PASS/FAIL은 데이터 보기 전 확정(`src/onchain/precheck.py` 상단 상수). 사후 변경 금지.
- 데이터: **Coin Metrics Community API**(키 불필요, CC). BTC 15.1yr(5433행)·ETH 10.8yr(3866행),
  갭 0. `FlowInExNtv`/`FlowOutExNtv`/`PriceUSD`. 결제 0원.
- 신호: `netflow = FlowIn−FlowOut` → rolling z(90d) → `pos = −sign(z)`(유입=숏/유출=롱),
  발행지연 `LAG=1`. always-in ±1. 정렬 `pos.shift(1+LAG)` (lookahead·flash개정 차단).
- 자산: BTC·ETH 개별 + **EW 포트폴리오**(단일자산 아티팩트 차단). robustness 그리드
  z∈{30,90}×LAG∈{1,2} = 12변형, BH 보정.
- 재사용: `src.backtester`(비용), `src.statarb.scan`(BH), `src.carry`(폰트).

### 사전등록 게이트 (전부 충족해야 PASS)
| 게이트 | 기준 |
|---|---|
| 1. 경제성 | 보유기간당 gross edge > 플립비용(10bps)×1.5 = **15bps** |
| 2. 통계 | block bootstrap(블록 20일, N=1000) 평균 net PnL>0 p<0.05 AND BH 생존 |
| 3. 진위/레짐 | 상위 1% \|PnL\|일 제거 후 부호 생존 AND 연도별 ≥60% 양(+) |
| 4. 안정성 | IS(70%) 부호 → OOS(30%) 부호일치 AND OOS bootstrap p<0.05 |
| S. 대칭성 | 롱·숏 레그 각각 net edge>0 (한 레그 의존 차단) |
| P. 포트폴리오 | EW(BTC+ETH) 통과 필수 |

**PASS = EW + (BTC or ETH) 전 게이트 통과**

## 결과 — 0/12 PASS, ❌ 전 변형 FAIL

### 게이트 1(경제성)에서 헤드라인 전멸
12변형 보유기간당 gross edge가 **−15.2 ~ +3.6bps**로 전부 15bps 임계 미달.
최고치(BTC z90L1) +1.8bps = 임계의 1/8. 가장 싼 게이트가 결정적 킬.

### 전체 리포트 (1차 사양 z90L1 발췌)
| label | Sharpe | net% | bh% | bootP | BH | yr+ | L/S leg(bps) | PASS |
|---|---|---|---|---|---|---|---|---|
| BTC z90L1 | −0.17 | −87.7% | +461525% | 0.747 | · | 0.44 | +12.0 / **−18.9** | FAIL |
| ETH z90L1 | −0.28 | −94.5% | +175177% | 0.841 | · | 0.42 | +11.3 / **−28.2** | FAIL |
| EW z90L1 | −0.51 | −96.6% | +54380% | 0.977 | · | 0.33 | −7.2 / **−22.5** | FAIL |
| (robustness 9변형) | −0.65~−0.12 | −99~−67% | | >0.60 | · | 0.25~0.58 | +/− | FAIL |

전 12변형 net Sharpe 음(−0.65~−0.12), bootstrap p 전부 >0.60(평균 net PnL ≤0이
유력), BH 0/12 생존. 게이트 2·3·4·S 전부 미충족.

## 결론 — FAIL, "정보 차원은 다르나 방향성 알파는 없음"

거래소 netflow는 가격과 **정보 차원이 다른** 데이터지만, 일봉 방향성 예측으로는 비용
넘는 엣지가 없다. 결정적 진단은 **레그 비대칭**:

- **롱 레그 +7~+14bps**: "유출→롱"은 약한 +. 단 이건 netflow 알파가 아니라 15년
  세속 강세장의 **베타/드리프트**를 우연히 포착한 것(롱이면 평균적으로 벎).
- **숏 레그 −19~−32bps**: "유입→숏"은 강하게 −. 상승 추세에 숏 = 손실.
- 합치면 방향성 정보 0 → 전략 net −67~−99% vs buy&hold +수만 %. 신호가 시장을
  전혀 못 이긴다. (과거 SHORT 대칭성 킬러의 재현 — 숏 측에 진짜 정보가 없음.)

robustness(z{30,90}×LAG{1,2}) 전부 동일 FAIL → 파라미터 아티팩트 아님. economics-first
순서가 헤드라인 한 줄(0/12)로 결정적 킬, 무거운 검정이 "방향성 정보 부재"를 확증.

**일봉 거래소 netflow → BTC/ETH 방향성은 비용 스케일에서 비-엣지.** 방향성 backtester로
넘길 신호 없음.

## 원질문("유료 API 수익 가능?")에 대한 답
유료 데이터 중 가장 유망했던 카테고리(온체인)의 정석 방향성 신호를, 그 **무료 티어**로
검증 → edge 부재. **이 결과만으로 일봉 netflow 유료 결제는 비추천.** precheck-우선
원칙대로, 무료에서 신호 부재 확인 = 결제 불필요.

## Caveat — 이 결과가 배제하지 *못하는* 것
1. **유료 티어의 고해상도/세분화**: 인트라데이(시간별) 플로우, 거래소별 세분, 엔티티
   레벨(고래/스마트머니) 플로우, 스테이블코인 플로우. 일봉 *집계* netflow는 가장 무딘
   형태 — 이들까진 직접 반증 못 함. 단 일봉서 방향성 정보가 0이라 입증 부담은 매우 높음.
2. **netflow 단일만 검증**(접근1). MVRV/활성주소 바스켓(접근2)은 미검증 — 단 netflow가
   방향성으론 가장 이론적 동기 강함.
3. **비방향성 용도는 별개**: 온체인이 방향성 진입 알파는 아니어도 레짐/리스크 오버레이·
   포지션 사이징엔 쓸 수 있음(momentum/crisis-alpha 교훈과 일치). 이번 precheck는
   방향성 진입만 반증.
4. **CM 거래소 라벨**: 휴리스틱 라벨 주소 기반(일부 거래소·노이즈). flash 개정은 LAG로
   차단(완전배제 불가).

## 보존 자산
- `src/onchain/data.py` — Coin Metrics community 페이지네이션 fetcher(키 불필요, sanity).
  임의 CM 메트릭·자산 재사용.
- `src/onchain/precheck.py` — netflow z-score → directional, 경제성/block bootstrap/BH/
  레짐(top% 제거+연도)/IS-OOS/대칭성/EW포트폴리오, 한글 플롯 4. 바스켓 확장 가능 구조.
- 산출물: `results/onchain/onchain_precheck_2026-06-08.json` + 신호/equity/연도별/rolling Sharpe 플롯 4.
