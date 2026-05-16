# Funding Carry — 백테스트 결과

> 작성일: 2026-05-17 · 상태: **FAIL — 폐기 → from-scratch 재설계 에스컬레이션**
> 설계: `2026-05-17-funding-carry-design.md`

## 가설

델타중립(perp+반대 spot)으로 방향 노출 없이 펀딩 현금흐름을 수확하면
수수료·슬리피지·spot 차입비 차감 후 +캐리가 남는다 (구조적·시장중립 엣지).

## 데이터

6심볼 XRP/SOL/DOGE/TRX/LINK/AVAX, `combined_15m.parquet` 의 `funding_rate`
8h 정산점 추출. XRP 2.1년·나머지 1.0~1.2년. 관측치 수천(N 문제 해소).
이상화 가정: 가격 leg 완전 헤지(basis PnL=0) → **상한** 추정.

## 결과 (연환산 %, 동일가중 포트폴리오)

| 시나리오 | 변형 A net% | A Sharpe | 변형 B net% | B Sharpe |
|----------|-------------|----------|-------------|----------|
| MAKER + borrow 0% (최상한) | **+4.67** | 15.0 | -6.06 | -26 |
| MAKER + borrow 5% | -0.33 | -1.1 | -11.06 | -48 |
| **TAKER + borrow 5% (폐기기준 베이스라인)** | **-0.37** | -1.2 | -23.42 | -101 |
| TAKER + borrow 10% | -5.37 | -17 | -28.42 | -122 |

심볼별 gross signed(비용 전, 변형 A): XRP +5.32, LINK +5.14, DOGE +3.64
**(양)** / SOL -0.62, TRX -2.83, AVAX -0.63 **(음)** → 비용 전에도 3/6만 양.

레짐 분해(변형 A, gross): 양(+)은 **"up" 레짐에 집중** (XRP +8.2 / DOGE
+7.6 / LINK +7.0 / AVAX +6.3), down·chop에선 붕괴/음전환 (SOL -2.2 down,
TRX -6.5 chop, AVAX -4.4 down).

변형 B 부호전환 빈도: **212~286회/년** (사실상 매 8h 정산마다 부호 반전).

## 결론: **FAIL — 사전 폐기기준 4개 전부 위반**

1. **베이스라인(TAKER+5% borrow) net ≤ 0**: 포트 -0.37%/yr → KILL.
   borrow가 비용을 지배(저턴오버라 fee종류 무관) — 이상화 gross ~+4.6%가
   spot leg 자금조달(~5%/yr)을 못 버팀.
2. **집중 실패**: borrow 0%(비현실적 최상)에서도 6심볼 중 3개만 양
   (XRP/DOGE/LINK), <4/6.
3. **레짐 취약**: gross 양(+)이 "up" 레짐에만. 이는 시장중립 알파가 아니라
   **숨은 short-bull-premium 노출**(강세장에 perp 프리미엄 → 숏 수취,
   약세/횡보장엔 역전). 표본이 강세 편향이라 gross만 +로 보였을 뿐.
4. **변형 B 전멸**: 펀딩 부호가 ~250회/년 반전 → 양 leg 반전 fee가 |funding|
   수확을 완전 상쇄. 전 시나리오 Sharpe ≪ 0 (< 0.5 기준 위반).

핵심: "구조적 엣지"는 허상이었다. 유일한 양(+) 성분(변형 A signed)은
(a) 시장중립이 아닌 레짐의존 강세프리미엄 베팅이고, (b) 그조차 현실적
spot 자금조달비를 못 넘는다. 변형 B는 펀딩 부호 churn에 파괴된다.
리테일·이 알트군에서 펀딩 캐리는 비용 차감 후 비-엣지.

## 에스컬레이션 (사용자 2026-05-17 사전 결정)

설계 §5 + 사용자 명시("이것도 아니면 처음부터 재설계")에 따라:
**9전 9패. 방향 예측(8) + 구조적 캐리(1) 양 패러다임 모두 falsified.
→ incremental 신호(#10) 금지. from-scratch 재설계로 에스컬레이션.**

from-scratch는 "전략 #10"이 아니라 **프로젝트 전제 자체를 재검토**하는
단계다(아래 차기 입력 대기). 산출물 처리: `scripts/funding_carry_backtest.py`
+ 설계/결과 문서 유지(재현 가능), 봇/src 무변경.

## 폐기 전략 목록 (갱신)

LS Ratio / FR+OI / Binance 공개 API / ML Filter / MTF / Sentiment Fusion /
**Funding Carry (이상화 gross +4.6%/yr이나 spot 자금조달 못 넘고, 실은
레짐의존 강세프리미엄 — 시장중립 아님; 변형 B는 부호 churn에 전멸)**.
