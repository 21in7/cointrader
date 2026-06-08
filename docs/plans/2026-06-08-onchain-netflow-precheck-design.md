# 온체인 netflow directional precheck — 설계

**날짜**: 2026-06-08
**상태**: 설계 완료, 구현 진행
**성격**: falsification-first precheck (백테스트 아님). 사전등록 게이트.

## 배경 / 동기

알파 추구 라인은 10전10패로 종결됐고, 그 실패는 전부 **거래소 내부 신호**(가격·OI·펀딩·L:S비율·리드래그·모멘텀·공적분·캐리)에 대한 것이었다. post-mortem 결론: "공개 시그널 방향 예측 패러다임 한계".

"유료 API를 쓰면 달라지나?"라는 질문에서 출발 — 핵심 재구성: **비용이 결정 변수가 아니라, 데이터에 담긴 엣지가 비용(~15bps)·실행제약(타임스케일)을 넘느냐**가 변수다. 과거 실패 대부분은 "엣지 < 비용"(lead-lag) 또는 "엣지 부재"(cointegration)였지 데이터 품질이 아니었다.

유료 데이터 중 **무료 공개 API와 정보 차원 자체가 다른** 유일한 후보 = **온체인 데이터**. 지금까지 테스트한 건 전부 거래소 *내부* 신호였다. 온체인 거래소 in/outflow는 아직 *덜* 차익소거됐을 가능성이 있는 미탐색 영역.

**결제 전 무료로 precheck** — 신호 부재가 확인되면 유료 결제 불필요. flicker가 보일 때만 유료 정당화. 지금까지 지켜온 반증-우선 원칙 그대로.

## 데이터 feasibility (검증 완료)

**Coin Metrics Community API** (키 불필요, rate 10req/6s, CC 라이선스):
- `FlowInExNtv` / `FlowOutExNtv` — 거래소 인/아웃플로우(코인 단위). BTC는 **2012-12-30부터** (13년+)
- `CapMVRVCur`(MVRV, 진단용), `PriceUSD`(벤치마크 종가), `AdrActCnt`/`TxCnt`(확장용)
- BTC·ETH 모두 무료 확인 (실제 API 호출로 검증)
- 엔드포인트: `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics`

## 가설

거래소 일일 netflow가 다음날 BTC/ETH 방향성 수익률을 비용 넘게 예측한다.
- 거래소로 **유입**(netflow>0) = 매도 의향 → 숏
- 거래소에서 **유출**(netflow<0) = 축적/hodl → 롱

## 타깃 / 타임스케일

**일봉 BTC/ETH 신규 독립전략.** 온체인은 본질적으로 일봉·BTC/ETH 중심이라 현 15m 알트봇과는 별개 전략. (사용자 결정: 옵션 A)

## 신호 구성 (사전등록 — 데이터 보기 전 잠금)

- `netflow_t = FlowInExNtv_t − FlowOutExNtv_t` (코인 단위)
- `z_t = (netflow_t − rolling_mean_90d) / rolling_std_90d` — 13년 스케일 변화 정규화
- **포지션 `pos_t = −sign(z_{t−LAG_DAYS})`** — always-in ±1
- **LAG_DAYS = 1** (발행지연/`flash` 개정 누수 차단) — 1차 사양
- 수익률 = `PriceUSD` close-to-close 다음날 로그수익률
- 신호 PnL = `pos_t × ret_{t+1}`

## 게이트 (사전등록 PASS/FAIL — 잠금, 사후 변경 금지)

| # | 게이트 | 기준 | 근거 |
|---|---|---|---|
| 1 | **경제성** (가장 싸고 결정적) | 평균 보유기간당 gross edge > 왕복비용 × COST_MARGIN | 회전율↑→비용이 죽임. 먼저 kill |
| 2 | **통계** | block bootstrap(블록 20일, N=1000) 평균PnL>0 p<0.05 + BH 보정 | leadlag 동일 |
| 3 | **진위/레짐** | 상위 1% \|PnL\|일 제거 후 edge 부호 생존 **AND** 연도별 ≥60% 양(+) | crisis-alpha/ETH-L2 교훈 — 크래시 몇 에피소드 집중 차단 |
| 4 | **안정성** | IS(70%) 부호 → OOS(30%) 부호일치 + OOS bootstrap p<0.05 | leadlag 동일 |
| S | **대칭성** | LONG·SHORT 레그 각각 net edge 부호 일치(한 레그 의존 차단) | 과거 반복 킬러(SHORT 대칭성) |
| P | **포트폴리오 확증** | EW(BTC+ETH) 통과 필수 | momentum 교훈 — 단일자산 아티팩트 차단 |

**PASS = EW 포트폴리오 + (BTC 또는 ETH)가 전 게이트 통과**

### 사전등록 상수
```
Z_WINDOW = 90            # rolling z-score lookback (days)
LAG_DAYS = 1             # publication lag
FEE_PCT_PER_SIDE = 0.04  # taker %
SLIPPAGE_PCT_PER_SIDE = 0.01
N_FILLS_PER_FLIP = 2     # 플립 = close + open
COST_MARGIN = 1.5
BOOTSTRAP_P_MAX = 0.05
BH_ALPHA = 0.05
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20          # days
IS_FRACTION = 0.70
REGIME_TOPK_PCT = 0.01   # 상위 1% |PnL|일 제거
REGIME_YEAR_MIN_FRAC = 0.60  # 연도별 양(+) 비율 최소
SEED = 42
ASSETS = ["btc", "eth"]  # + EW 포트폴리오
```

### robustness 그리드 (BH에 포함)
- z-window ∈ {30, 90}, LAG ∈ {1, 2} → 4 변형 × {btc, eth, EW}

## 벤치마크 & 산출물

- buy&hold 대비 (표준)
- 플롯 4종: (a) 신호 z vs 가격, (b) equity curve, (c) rolling 상관, (d) 연도별 PnL 바
- `results/onchain/onchain_precheck_{date}.json`
- `docs/plans/2026-06-08-onchain-netflow-precheck-result.md`

## 모듈 구조

- `src/onchain/data.py` — Coin Metrics community fetch + sanity 체크포인트 (momentum/data.py 패턴)
- `src/onchain/precheck.py` — 사전등록 게이트 (leadlag/precheck.py 패턴)
- 재사용: `src.backtester._calc_fee/_apply_slippage`, `src.statarb.scan._benjamini_hochberg`, `src.carry.setup_korean_font`

## 명시 caveat

1. **데이터 개정** — 플로우에 `flash` 상태, 나중 개정. LAG로 차단하나 완전 배제 불가
2. **일봉 해상도** — 현 15m 알트봇 직접 투입 불가(별도 전략)
3. **거래소 라벨 노이즈** — CM 플로우는 휴리스틱 라벨 주소 기반 = 일부 거래소·노이즈 포함

## 예상 결과

falsification-first 관점: **FAIL이 기본 시나리오**(거래소 플로우도 차익소거 진행됐을 것). 단 가격과 정보 차원이 다르므로 이전 10전과 달리 잔존 edge 가능성 0은 아님. 게이트가 정직하게 가린다.
