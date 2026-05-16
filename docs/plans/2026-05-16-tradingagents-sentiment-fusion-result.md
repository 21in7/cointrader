# TradingAgents Sentiment Fusion — 백테스트 결과 (게이트 B)

> 작성일: 2026-05-17 · 상태: **FAIL — 폐기** · 설계: `2026-05-16-tradingagents-sentiment-fusion-design.md`

## 가설

군중/뉴스 센티먼트의 극단값이 일반 봇 진입 신호의 수익성을 높인다
(순응이 아닌 contrarian/veto 게이트로 적용 시). 베이스라인(센티먼트 OFF)
대비 OOS PF 개선 + LONG/SHORT 대칭 + 거래 수 >50.

## 데이터

- 심볼: XRPUSDT 단독, `data/xrpusdt/combined_15m.parquet` (74,123캔들)
- 기간: 2024-03-22 ~ 2026-05-03 (~25.5개월)
- 센티먼트: Alpha Vantage NEWS_SENTIMENT `CRYPTO:XRP` 2,453기사 →
  로컬 MLX gemma-4-e4b(temp=0, 결정론) 12h cadence 1,237판정 →
  15m forward-fill `data/xrpusdt/sentiment_15m.parquet` (74,055행, 커버 100%)
- **센티먼트 라벨 분포: Neutral 75.7% / Bullish 17.0% / Bearish 7.2%,
  극단(VeryBullish/VeryBearish ±1.0) 0건.** 백테스트는 뉴스 only
  (StockTwits/Reddit는 라이브 전용)라 모델이 보수적 → contrarian은
  extreme-band 0.5로 평가.

## 결과 (signal-threshold=2, full period, ML OFF)

| 모드 | 거래수 | PF | 승률 | 수익률 | MDD | LONG n/PF | SHORT n/PF |
|------|--------|-----|------|--------|-----|-----------|------------|
| off (baseline) | 19 | 0.77 | 42.1% | -8.7% | 19% | 9 / 0.62 | 10 / 0.92 |
| veto | 19 | 0.77 | 42.1% | -8.7% | 19% | 9 / 0.62 | 10 / 0.92 |
| contrarian | 19 | 0.77 | 42.1% | -8.7% | 19% | 9 / 0.62 | 10 / 0.92 |
| confirm | 15 | 0.39 | 20.0% | -21.3% | 25% | 3 / 0.45 | 12 / 0.37 |

운영 파라미터(adx25/vol2.5/tp4.0)에서는 베이스라인이 25개월에 **15거래**뿐
이라, 통계 N 확보 위해 느슨한 기본 파라미터+signal-threshold 2로 통일
측정(전 모드 동일, 상대 델타 비교). signal-threshold 1까지 낮춰도 최대
22거래 — 트레이드 수는 구조적으로 ≤22.

## 결론: **FAIL — 폐기**

설계 §2 폐기 기준 다중 위반:

1. **거래 수 < 30** (N=19) → 즉시 폐기 기준 해당. signal-threshold/파라미터를
   어떻게 조정해도 XRP 단독 전략은 킬스위치(연속손실 차단) + 낮은 신호빈도로
   25.5개월 전체에서 ≤22거래 → **단일심볼 백테스트로는 진입 게이트 효과를
   통계적으로 평가할 표본 자체가 없다.**
2. **OOS PF < 1.0**: 베이스라인 PF 0.77, 모든 모드 < 1.0. 베이스라인 전략
   자체가 이 기간 edge 없음.
3. **veto/contrarian = 베이스라인과 완전 동일 (0건 필터)**: ~19개 진입
   타임스탬프가 충돌 방향의 non-Neutral 센티먼트 봉과 거의 겹치지 않음
   (센티먼트 76% Neutral × 트레이드 19개 → 충돌 교집합 ≈ 0). 게이트
   파이프라인은 정상(confirm이 19→15로 결과를 바꿔 wiring·데이터 동작 입증)
   이므로 무효과는 버그가 아니라 **신호 밀도 부족이 실재**함을 의미.
4. **confirm(추세순응)만 효과 → 악화** (PF 0.77→0.39): post-mortem의
   "공개 시그널 순응 패러다임 실패" 명제와 정합. 페이드가 아닌 순응은 해롭다.

## 근본 원인

- **Gate B 설계 전제의 결함**: 단일심볼 XRP 백테스트는 진입 게이트 ablation에
  필요한 트레이드 표본(>30~50)을 어떤 파라미터로도 생성하지 못한다.
- **Modality gap이 치명적(설계 R1 실증)**: 백테스트 신호원(뉴스 only,
  76% Neutral, 극단 0)이 라이브 의도 신호원(뉴스+StockTwits+Reddit 군중)의
  매우 약한 대리. gemma-4-e4b는 뉴스만으론 actionable 밀도를 못 만든다.

## 폐기 처리 & 다음 방향

본 전략을 **폐기**한다. 단, 두 결과를 구분:
- *순응(confirm)*은 명확히 해롭다 → 재시도 금지 (post-mortem 8번째 확증).
- *contrarian/veto edge 자체*는 "없음"이 아니라 **"이 백테스트 패러다임으로는
  검증 불가"** (표본·modality 한계). 진짜로 보려면 둘 중 하나가 필요:
  1. **멀티심볼 트레이드 풀링**으로 N 확보 (XRP 단독 한계 우회), 또는
  2. **라이브 섀도우**(설계 게이트 C) — 뉴스+StockTwits+Reddit 풍부한 신호원으로
     무거래 로깅 후 평가. 단 7+1전 전패 맥락상 신규 투자 가치는 낮음.

**권장: 폐기 목록 추가.** 추가 투자(멀티심볼 풀링/라이브 섀도우)는 비용
대비 기대가 낮으므로 사용자 명시 요청 시에만 진행.

## 산출물 처리

- 게이트 A 코드(`src/sentiment_provider.py`, `scripts/build_sentiment_dataset.py`)
  + 게이트 B 코드(`src/backtester.py` 센티먼트 게이트, `run_backtest.py` 인자)는
  **유지** (재현·재평가 가능, `--sentiment-mode off` 기본값이라 운영 무영향).
- `src/bot.py` 연결은 **진행하지 않음** (게이트 B FAIL → 설계 §7 게이트 B에
  따라 종료).
- 테스트(`tests/test_sentiment_provider.py`, `tests/test_sentiment_gate.py`)
  유지 — 게이트 로직 회귀 가드.

## 폐기 전략 목록 (갱신)

LS Ratio / FR+OI / Binance 공개 API / ML Filter / MTF Pullback /
**TradingAgents Sentiment Fusion (contrarian·veto 무효과, confirm 악화 —
단일심볼 백테스트 표본 부족 + 뉴스-only modality gap)**.
