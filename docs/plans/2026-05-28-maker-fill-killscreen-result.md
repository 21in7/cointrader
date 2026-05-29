# Maker 체결 전환 kill-screen 결과

## 가설
MTF Pullback 전략을 taker(MARKET) 대신 maker(limit) 체결로 운영하면
수수료가 절반(왕복 8bps→4bps)으로 줄어 PF가 1.0을 넘는다.
(post-mortem의 미해결 "maker caveat" 검증)

## 검증 설계 — 반증-우선 2단계
- **Q1 (비용, 싸고 결정적)**: maker 수수료를 가정하고 fees_only PF 재계산.
  100% 체결·동일가격을 가정한 **최선의 천장값**. 코드 로직 변경 없이
  `COST_MODEL`의 bps 상수만 재조합.
- **Q2 (체결 가능성, 데이터 한계)**: 지정가가 실제 체결될지(미체결·역선택)
  모델링. Q1이 통과해야만 의미 있음.

## 데이터
- 세트: `data/trade_history/mtf_xrpusdtusdt.jsonl` — OOS dry-run 30건 (LONG 18 / SHORT 12)
- 기준 비용: taker 4.0bps, maker 2.0bps per side (`src/config.py` COST_MODEL, VIP 0)
- SL은 STOP_MARKET이라 본질상 taker로 고정(지정가화 불가)

## Q1 결과 (fees_only: slippage=0, funding=0)

| fee 구성 | TOTAL PF | LONG PF | SHORT PF | TOTAL CumPnL |
|---|---|---|---|---|
| baseline (전부 taker) | 0.84 | 1.04 | 0.56 | −178.4 bps |
| maker-real (진입·TP=maker, SL=taker) | 0.92 | 1.12 | 0.62 | −92.4 bps |
| all-maker (전부 maker, *비현실적 상한*) | 0.95 | 1.15 | 0.64 | −58.4 bps |

baseline은 기존 최종결과(`2026-05-04-mtf-oos-final-result.md`)와 정확히 일치 → 재현 검증 통과.

## 결론 — FAIL, Q2 불필요
- 물리적으로 불가능한 전부-maker 상한(0.95)조차 TOTAL PF 1.0 미달.
- 이 수치는 "100% 체결 + 동일가격" 천장값. 실제 maker는 미체결·역선택으로
  이보다 나쁠 수밖에 없으므로, 살릴 headroom이 없어 Q2(체결 모델링)는 무의미.
- **SHORT 대칭성 실패가 fee와 무관하게 지속**(0.56→0.64). maker는 비용만 깎고
  방향 edge를 만들지 못함.
- LONG-only는 1.0 초과(1.12)이나 baseline에서 이미 알던 사실 + N=18 유의성 미달 +
  SHORT 대칭성 실패(폐기 기준 위반) + best-case 천장값. 단방향 베팅이지 edge 아님.

수수료(8bps)는 증상이지 병이 아니었음을 재확인. 병은 edge 부재 —
fees_only가 raw PF 1.06으로 이미 증명. post-mortem의 maker caveat 종결.
