import pandas as pd
import numpy as np
import pytest
from src.label_builder import build_labels


def make_signal_df():
    """
    신호 발생 시점 이후 가격이 TP에 도달하는 시나리오
    entry=100, TP=103, SL=98.5
    """
    future_closes = [100.5, 101.0, 101.8, 102.5, 103.1, 103.5]
    future_highs  = [c + 0.3 for c in future_closes]
    future_lows   = [c - 0.3 for c in future_closes]
    return future_closes, future_highs, future_lows


def test_label_tp_reached():
    closes, highs, lows = make_signal_df()
    label = build_labels(
        future_closes=closes,
        future_highs=highs,
        future_lows=lows,
        take_profit=103.0,
        stop_loss=98.5,
        side="LONG",
    )
    assert label == 1, "TP 먼저 도달해야 레이블 1"


def test_label_sl_reached():
    future_closes = [99.5, 99.0, 98.8, 98.4, 98.0]
    future_highs  = [c + 0.3 for c in future_closes]
    future_lows   = [c - 0.3 for c in future_closes]
    label = build_labels(
        future_closes=future_closes,
        future_highs=future_highs,
        future_lows=future_lows,
        take_profit=103.0,
        stop_loss=98.5,
        side="LONG",
    )
    assert label == 0, "SL 먼저 도달해야 레이블 0"


def test_label_neither_reached_returns_none():
    future_closes = [100.1, 100.2, 100.3]
    future_highs  = [c + 0.1 for c in future_closes]
    future_lows   = [c - 0.1 for c in future_closes]
    label = build_labels(
        future_closes=future_closes,
        future_highs=future_highs,
        future_lows=future_lows,
        take_profit=103.0,
        stop_loss=98.5,
        side="LONG",
    )
    assert label is None, "미결 시 None 반환"


def test_label_short_tp():
    future_closes = [99.5, 99.0, 98.0, 97.0]
    future_highs  = [c + 0.3 for c in future_closes]
    future_lows   = [c - 0.3 for c in future_closes]
    label = build_labels(
        future_closes=future_closes,
        future_highs=future_highs,
        future_lows=future_lows,
        take_profit=97.0,
        stop_loss=101.5,
        side="SHORT",
    )
    assert label == 1
