from typing import Optional


def build_labels(
    future_closes: list[float],
    future_highs: list[float],
    future_lows: list[float],
    take_profit: float,
    stop_loss: float,
    side: str,
) -> Optional[int]:
    """
    진입 이후 미래 캔들을 순서대로 확인해 TP/SL 도달 여부를 판단한다.
    LONG: high >= TP → 1, low <= SL → 0
    SHORT: low <= TP → 1, high >= SL → 0
    둘 다 미도달 → None (학습 데이터에서 제외)
    """
    for high, low in zip(future_highs, future_lows):
        if side == "LONG":
            if high >= take_profit:
                return 1
            if low <= stop_loss:
                return 0
        else:  # SHORT
            if low <= take_profit:
                return 1
            if high >= stop_loss:
                return 0
    return None
