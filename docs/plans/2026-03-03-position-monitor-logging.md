# 포지션 모니터 로깅 (실시간 가격 추적)

**Goal:** 포지션 보유 중 5분마다 현재가 기준 미실현 손익을 로그로 출력하여, 봇 운영 중 포지션 상태를 실시간 모니터링할 수 있게 한다.

**Status:** Completed

---

## 변경 사항

### 1. MultiSymbolStream에 latest_price 속성 추가

- `src/data_stream.py`: `self.latest_price: float | None = None` 초기화
- `handle_message()`에서 **모든 kline 메시지** (미확정 캔들 포함)에 대해 primary symbol(XRPUSDT)의 close 가격으로 업데이트
- 기존에는 확정 캔들만 처리했으나, 실시간 가격 추적을 위해 미확정 캔들도 반영
- BTC/ETH 등 비주 심볼은 latest_price 갱신 안 함

### 2. _position_monitor() 코루틴 추가

- `src/bot.py`: `_MONITOR_INTERVAL = 300` (5분) 클래스 상수 정의
- `async def _position_monitor()` 무한 루프: 5분마다 실행
- 포지션 없으면(`current_trade_side is None`) skip
- 포지션 있으면: `_calc_estimated_pnl(price)`로 미실현 PnL 계산, 퍼센트 산출 후 INFO 로그 출력
- `asyncio.gather()`에 추가하여 기존 user_data_stream, candle processing과 병렬 실행

### 3. 테스트

- `tests/test_bot.py`: 포지션 보유 시 PnL 로깅 확인, 포지션 없을 때 정상 skip 확인 (2 cases)
- `tests/test_data_stream.py`: 미확정 캔들로 latest_price 갱신, 비주 심볼은 무시 확인 (1 case)

## 설계 결정

- WebSocket 스트림 재사용 (추가 API 연결 불필요)
- `_MONITOR_INTERVAL`은 클래스 상수로 정의 (테스트에서 0으로 오버라이드 가능)
- 가격/진입가/수량 중 하나라도 None이면 graceful skip
