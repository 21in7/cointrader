import httpx
from loguru import logger


class DiscordNotifier:
    """Discord 웹훅으로 거래 알림을 전송하는 노티파이어."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self._enabled = bool(webhook_url)

    def _send(self, content: str) -> None:
        if not self._enabled:
            logger.debug("Discord 웹훅 URL 미설정 - 알림 건너뜀")
            return
        try:
            resp = httpx.post(
                self.webhook_url,
                json={"content": content},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Discord 알림 전송 실패: {e}")

    def notify_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
        signal_data: dict = None,
    ) -> None:
        rsi = (signal_data or {}).get("rsi", 0)
        macd = (signal_data or {}).get("macd_hist", 0)
        atr = (signal_data or {}).get("atr", 0)
        msg = (
            f"**[{symbol}] {side} 진입**\n"
            f"진입가: `{entry_price:.4f}` | 수량: `{quantity}` | 레버리지: `{leverage}x`\n"
            f"SL: `{stop_loss:.4f}` | TP: `{take_profit:.4f}`\n"
            f"RSI: `{rsi:.2f}` | MACD Hist: `{macd:.6f}` | ATR: `{atr:.6f}`"
        )
        self._send(msg)

    def notify_close(
        self,
        symbol: str,
        side: str,
        exit_price: float,
        pnl: float,
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} **[{symbol}] {side} 청산**\n"
            f"청산가: `{exit_price:.4f}` | PnL: `{pnl:+.4f} USDT`"
        )
        self._send(msg)

    def notify_info(self, message: str) -> None:
        self._send(f"ℹ️ {message}")
