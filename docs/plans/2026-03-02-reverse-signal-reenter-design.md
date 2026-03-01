# 반대 시그널 시 청산 후 즉시 재진입 설계

- **날짜**: 2026-03-02
- **파일**: `src/bot.py`
- **상태**: 설계 완료, 구현 대기

---

## 배경

현재 `TradingBot.process_candle`은 반대 방향 시그널이 오면 기존 포지션을 청산만 하고 종료한다.
새 포지션은 다음 캔들에서 시그널이 다시 나와야 잡힌다.

```
현재: 반대 시그널 → 청산 → 다음 캔들 대기
목표: 반대 시그널 → 청산 → (ML 필터 통과 시) 즉시 반대 방향 재진입
```

같은 방향 시그널이 오거나 HOLD이면 기존 포지션을 그대로 유지한다.

---

## 요구사항

| 항목 | 결정 |
|------|------|
| 포지션 크기 | 재진입 시점 잔고 + 동적 증거금 비율로 새로 계산 |
| SL/TP | 청산 시 기존 주문 전부 취소, 재진입 시 새로 설정 |
| ML 필터 | 재진입에도 동일하게 적용 (차단 시 청산만 하고 대기) |
| 같은 방향 시그널 | 포지션 유지 (변경 없음) |
| HOLD 시그널 | 포지션 유지 (변경 없음) |

---

## 설계

### 변경 범위

`src/bot.py` 한 파일만 수정한다.

1. `_close_and_reenter` 메서드 신규 추가
2. `process_candle` 내 반대 시그널 분기에서 `_close_position` 대신 `_close_and_reenter` 호출

### 데이터 흐름

```
process_candle()
  └─ 반대 시그널 감지
       └─ _close_and_reenter(position, signal, df, btc_df, eth_df)
            ├─ _close_position(position)          # 청산 + cancel_all_orders
            ├─ risk.can_open_new_position() 체크
            │    └─ 불가 → 로그 + 종료
            ├─ ML 필터 체크 (ml_filter.is_model_loaded())
            │    ├─ 차단 → 로그 + 종료 (포지션 없는 상태로 대기)
            │    └─ 통과 → 계속
            └─ _open_position(signal, df)          # 재진입 + 새 SL/TP 설정
```

### `process_candle` 수정

```python
# 변경 전
elif position is not None:
    pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
    if (pos_side == "LONG" and signal == "SHORT") or \
       (pos_side == "SHORT" and signal == "LONG"):
        await self._close_position(position)

# 변경 후
elif position is not None:
    pos_side = "LONG" if float(position["positionAmt"]) > 0 else "SHORT"
    if (pos_side == "LONG" and signal == "SHORT") or \
       (pos_side == "SHORT" and signal == "LONG"):
        await self._close_and_reenter(position, signal, df_with_indicators, btc_df, eth_df)
```

### 신규 메서드 `_close_and_reenter`

```python
async def _close_and_reenter(
    self,
    position: dict,
    signal: str,
    df,
    btc_df=None,
    eth_df=None,
) -> None:
    """기존 포지션을 청산하고, ML 필터 통과 시 반대 방향으로 즉시 재진입한다."""
    await self._close_position(position)

    if not self.risk.can_open_new_position():
        logger.info("최대 포지션 수 도달 — 재진입 건너뜀")
        return

    if self.ml_filter.is_model_loaded():
        features = build_features(df, signal, btc_df=btc_df, eth_df=eth_df)
        if not self.ml_filter.should_enter(features):
            logger.info(f"ML 필터 차단: {signal} 재진입 무시")
            return

    await self._open_position(signal, df)
```

---

## 엣지 케이스

| 상황 | 처리 |
|------|------|
| 청산 후 ML 필터 차단 | 청산만 하고 포지션 없는 상태로 대기 |
| 청산 후 잔고 부족 (명목금액 미달) | `_open_position` 내부 경고 후 건너뜀 (기존 로직) |
| 청산 후 최대 포지션 수 초과 | 재진입 건너뜀 |
| 같은 방향 시그널 | 포지션 유지 (변경 없음) |
| HOLD 시그널 | 포지션 유지 (변경 없음) |
| 봇 재시작 후 포지션 복구 | `_recover_position` 로직 변경 없음 |

---

## 영향 없는 코드

- `_close_position` — 변경 없음
- `_open_position` — 변경 없음
- `_recover_position` — 변경 없음
- `RiskManager` — 변경 없음
- `MLFilter` — 변경 없음
