"""Spot-perp 베이시스 / 펀딩 캐리 리서치 모듈.

이전 funding-carry 연구(2026-05-17, 6알트, FAIL)는 BTC/ETH를 다루지 않았고
가격 leg을 이상화(basis PnL=0)했다. 이 모듈은 BTC/ETH를 대상으로 실제 spot-perp
베이시스(공적분·half-life·최대 역이격→안전 레버리지)와 캐리를 특성화한다.

data.py:     spot+perp klines + funding 취득/정렬 (체크포인트)
precheck.py: 코인별 PASS/FAIL 게이트 + 캐리 리스크 프로파일
"""
from __future__ import annotations


def setup_korean_font() -> None:
    """matplotlib 한글 폰트 설정 (지난 stat-arb 플롯 □ 깨짐 방지)."""
    import matplotlib
    import matplotlib.pyplot as plt

    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
                  "Malgun Gothic", "DejaVu Sans"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
