"""저빈도(일~주봉) 모멘텀 directional 리서치 모듈.

(1) TSMOM 시계열 모멘텀: 각 코인 과거수익률 부호로 롱/숏(추세추종).
(2) XSMOM 단면 모멘텀: 코인 랭킹 상위 롱/하위 숏(market-neutral).

data.py:     일봉 전체 히스토리 취득/sanity (체크포인트)
precheck.py: 변형·L별 PASS/FAIL (false positive 방지 — 비용·생존편향·벤치마크 빡세게)
"""
