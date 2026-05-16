# Funding Carry — 구조적 엣지 리서치 설계

> 작성일: 2026-05-17 · 상태: **설계, 백테스트 대기** · 유형: design (strategy-research)
> 맥락: 8연패(방향 예측 패러다임 소진) 후 사용자 결정으로 *구조적·시장중립*
> 엣지로 전환. 이것도 FAIL 시 incremental 신호가 아니라 **from-scratch
> 재설계로 에스컬레이션** (사용자 2026-05-17 명시).

## 1. 가설

무기한 선물은 8h마다(00/08/16 UTC) 롱↔숏 간 펀딩을 지급한다. **델타중립
포지션(perp + 반대 spot)으로 방향 노출 없이 펀딩 현금흐름을 수확**하면,
수수료·슬리피지·spot 차입비 차감 후에도 +캐리가 남는다.

방향을 예측하지 않으므로 8연패 패러다임과 카테고리가 다르다. 펀딩은 8h마다
연속 발생 → 6심볼 2년이면 관측치 수천 개 (이전 N=19 표본부족 문제 해소).

### 두 변형
- **A. 정적 델타중립(always-short-perp + long spot)**: 턴오버 최소. 수확 =
  Σ(signed funding) — 펀딩 +면 받고 −면 낸다. (XRP 이상화 gross ≈ +5.3%/yr)
- **B. 부호추종(sign-following)**: 매 정산 펀딩 받는 쪽으로 양 leg 정렬.
  수확 = Σ|funding| (이상화 gross ≈ +9.0%/yr) **− 부호전환 턴오버 비용**.
  부호 전환 빈도가 핵심 비용 동인.

## 2. 데이터

- 심볼: **XRP, SOL, DOGE, TRX, LINK, AVAX** (6) — `data/{sym}/combined_15m.parquet`
  의 `funding_rate`. (BTC/ETH는 단독 parquet 부재 → 제외)
- 기간: 심볼별 상이 (XRP 2024-03~2026-05 최장, 나머지 2025-03~). 심볼별
  가용 전구간 사용.
- 정산 추출: `funding_rate` 는 8h값의 15m forward-fill. 정산 PnL은 8h
  지점에서 1회만 적용(15m 합산 시 32배 과대계상 — 금지). 검증: XRP
  hour∈{0,8,16}&min==0 추출 2,317관측 ≈ 예상 2,316. 정산 타임스탬프에
  ±15m 지터 있으나 8h 주기 대비 무시 가능(집계 엣지 불변 확인).
- 부호 규약: Binance 표준 — funding>0 = 롱이 숏에 지급 → **short perp가 수취**.

## 3. 비용 모델 (이상화 = 가격 leg 완전 헤지, basis PnL=0 상한 가정)

순캐리 = Σ funding − 선물수수료(턴오버) − 슬리피지 − spot 차입/조달비

- 선물 수수료: taker 0.04% / maker 0.018% per leg flip (둘 다 리포트)
- 슬리피지: 0.01% per fill
- **spot 차입비**: 데이터 부재 → 민감도 스트레스 {0, 5, 10}%/yr 로 차감
- 가격 leg는 perp+spot 완전 헤지로 가정(basis 수렴 PnL≈0). 이는 **상한
  추정** — 이상화조차 음수면 실제(basis risk·불완전헤지 포함)는 확실히 죽음
  → 싸게 폐기 가능 (edge-first 필터의 핵심).

## 4. 메트릭 (캐리용 — 방향 PF 아님)

| 메트릭 | 의미 | 기준 |
|--------|------|------|
| Net 연환산 캐리 | funding − 전비용 | >0 최소, >8%/yr 목표 |
| Net Sharpe | 캐리 PnL 시계열 / 변동성 | ≥0.5 최소, ≥1.0 목표 |
| Max DD (캐리 equity) | 헤지/basis 실패 지표 | <10% |
| 심볼 견고성 | +인 심볼 수 / 6 | ≥4/6 |
| 레짐 견고성 | +인 레짐 수 (상승/하락/횡보) | ≥2/3 |
| 부호전환 턴오버 | 변형 B 비용 동인 | 리포트 |

레짐: 심볼별 가격의 90일 추세(상승/하락/횡보)로 분류, 레짐별 캐리 분해.

## 5. 사전 폐기 기준 (실행 전 확정 — 변경 금지)

**하나라도 해당 시 즉시 FAIL → 폐기 → from-scratch 재설계 에스컬레이션:**

1. 베이스라인(taker fee + 5%/yr 차입 가정) **net 연환산 캐리 ≤ 0** (전 기간)
2. **집중 실패**: 최대 기여 심볼 1개 제거 시 포트폴리오 net ≤ 0
3. **레짐 취약**: 단일 레짐에서만 + (≥2/3 레짐 + 실패)
4. **net Sharpe < 0.5** (운영·basis 리스크 대비 너무 노이지)

**PASS (다음 단계 진입 = 체결가능성/2-leg 실행 설계, *봇 코드 아님*):**
위 4개 모두 통과 + 변형 A 또는 B 중 최소 하나가 maker-fee 기준 net >5%/yr.

PASS여도 즉시 봇 연결 안 함 — 라이브는 spot+perp 2-leg 구조 필요(현 단일-leg
선물 봇의 큰 변경)이므로 별도 실행 설계·승인 후.

## 6. 백테스트 설계 (커스텀 스크립트)

기존 `run_backtest.py`(방향 전략용)는 부적합 → 신규 연구 스크립트
`scripts/funding_carry_backtest.py` (봇/src 무변경, 데이터 전용):

1. 심볼별 `combined_15m.parquet` 로드 → 8h 정산점 추출
2. 변형 A: 고정 notional, PnL_t = +funding_t (always-short-perp)
   변형 B: PnL_t = +|funding_t|, 부호변경 시 flip turnover 비용 가산
3. 비용 차감(fee/slip/borrow 시나리오 매트릭스: taker|maker × borrow{0,5,10})
4. 심볼별/포트폴리오(동일가중) equity, 연환산 캐리, Sharpe, MaxDD
5. 레짐 분해(심볼 90일 추세), 부호전환 빈도
6. 결과 → `docs/plans/2026-05-17-funding-carry-result.md`,
   CLAUDE.md plan history + 메모리 갱신

## 7. 폐기 전략 참조 (반복 방지)

LS Ratio / FR+OI / Binance 공개 API / ML Filter / MTF / TradingAgents
Sentiment — 전부 *방향 예측*. 본 건은 *시장중립 캐리*로 패러다임이 다름.
단 FR+OI(2026-03-30)에서 funding을 *방향 신호*로 쓴 건 실패했음 — 본 건은
funding을 *현금흐름*으로 수확하는 것이라 구분됨.
