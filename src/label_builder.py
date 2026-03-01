from typing import Optional


def build_labels(
    future_closes: list[float],
    future_highs: list[float],
    future_lows: list[float],
    take_profit: float,
    stop_loss: float,
    side: str,
) -> Optional[int]:
    for high, low in zip(future_highs, future_lows):
        if side == "LONG":
            # 보수적 접근: 손절(SL)을 먼저 체크
            if low <= stop_loss:
                return 0
            if high >= take_profit:
                return 1
        else:  # SHORT
            # 보수적 접근: 손절(SL)을 먼저 체크
            if high >= stop_loss:
                return 0
            if low <= take_profit:
                return 1
    return None
