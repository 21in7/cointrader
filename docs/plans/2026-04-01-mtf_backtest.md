【MTF 백테스트: 기존 vs 개선】

데이터:
├─ XRPUSDT kline (15m, 2026-02-01 ~ 현재)
├─ 타겟: 다음 4시간 수익률
└─ 기간: 2개월

Step 1: 기존 신호 백테스트
├─ 신호: 15분봉 RSI+MACD (메타필터 없음)
├─ SL: ATR 1.5x, TP: ATR 2.3x
├─ 결과 기대값: PF ≈ 0.89 (이미 알고 있는 값)
└─ 거래수: X건

Step 2: MTF 필터링 적용 백테스트
├─ Meta Filter 1: 1h EMA 50 vs EMA 200
├─ Meta Filter 2: 1h ADX > 20
├─ Trigger: 15분봉 RSI+MACD (필터 조건과 일치할 때만)
├─ SL: ATR 1.5x, TP: ATR 2.3x (동일)
├─ 결과 기대값: PF > 1.0 (목표)
└─ 거래수: Y건 (X의 40~60%)

Step 3: 비교 분석
├─ 기존: Trades X, PF 0.89, PnL A
├─ MTF: Trades Y, PF ?, PnL B
├─ 개선도: (PF_new - 0.89) / 0.89 = ?%

출력:
결과를 테이블 형식으로 정렬
│ 구분 │ Trades │ Win Rate │ PF │ PnL │ 개선율 │
└─────────────────────────────────────────
