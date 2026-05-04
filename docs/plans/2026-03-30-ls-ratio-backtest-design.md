# L/S Ratio 단독 백테스트 설계

**Date**: 2026-03-30
**Status**: Approved
**Goal**: L/S ratio의 독립적 edge 유무를 빠르게 판정

## 배경

- 기존 시그널(RSI+MACD+BB+EMA+StochRSI)은 PF 0.89, 수익성 부족 확정 (2026-03-29)
- L/S ratio 수집 중 (3/22~, 프로덕션 716행)
- 이전 분석에서 XRP top_acct_ls_ratio → 1h return 상관계수 +0.1158 (Momentum)

## Phase 1: Pure Edge Test

### 데이터
- L/S ratio: 프로덕션 수집 (716행, 3/22~3/30)
- Kline: Binance XRPUSDT 15m (같은 기간)
- 비교용: BTC/ETH L/S ratio

### 전략 로직
- 진입: `top_acct_ls_ratio` 백분위수 기반 임계값
- 보유: 고정 4캔들 (1시간)
- 청산: 1시간 후 종가
- 수수료: 0.04% × 2 = 0.08%
- 포지션: 1건씩만

### 테스트 조합 (6개)

| 임계값 | 방향 | 신호 |
|--------|------|------|
| 75th %ile | LONG | 모멘텀 강함 |
| 75th %ile | SHORT | 모멘텀 약함 |
| 50th %ile | LONG | 모멘텀 중간 |
| 50th %ile | SHORT | 모멘텀 중간 |
| 25th %ile | LONG | 모멘텀 약함 |
| 25th %ile | SHORT | 모멘텀 강함 |

### PF 계산
- PF = Σ(Gross Profit - 비용) / Σ(|Gross Loss| + 비용)
- 비용: 0.08% per trade

### 판정 기준 (3단계 필터)

**필터 1: 명확한 신호**
- PF > 1.5 → edge 있음
- PF < 0.5 → 실패
- 0.5~1.5 → 판단 보류

**필터 2: 거래수 신뢰도**
- < 20건 → 🔴 폐기
- 20~50건 → 🟡 참고만
- 50~100건 → 🟢 검토
- 100건+ → 🟢 우선

**필터 3: 대칭성**
- Case 1: LONG > 1.5 AND SHORT > 1.5 → 진정한 edge → Phase 2 진행
- Case 2: 한쪽만 성공 → 시장 베타/우연 → 폐기
- Case 3: 한쪽 강함 + 한쪽 애매 → 부분적 edge → 낮은 신뢰도로 Phase 2

## Phase 2: Bot Simulation (조건부)

Phase 1 필터 통과 시에만:
- L/S ratio로 RSI/MACD 완전 대체
- ATR SL 1.5x, TP 2.3x (기존 동일)
- 역신호 시 반대 포지션 재진입

## Phase 3: 4월 15일 재검증

- 4/1~4/15 데이터로 동일 로직 재실행
- 기준: PF > 1.15
- GO/STOP/HOLD 최종 판정

## 한계
- 8일 = ~700 시그널 포인트, 조합당 ~115개
- 극심한 과적합 위험 → 8일 결과는 "방향성 힌트"
- 최종 판정은 4월 15일 재검증 기반
