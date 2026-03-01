import json
from datetime import datetime, timezone
from notion_client import Client
from loguru import logger


class TradeRepository:
    """Notion 데이터베이스에 거래 이력을 저장하는 레포지토리."""

    def __init__(self, token: str, database_id: str):
        self.client = Client(auth=token)
        self.database_id = database_id

    def save_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        signal_data: dict = None,
    ) -> dict:
        properties = {
            "Symbol": {"title": [{"text": {"content": symbol}}]},
            "Side": {"select": {"name": side}},
            "Entry Price": {"number": entry_price},
            "Quantity": {"number": quantity},
            "Leverage": {"number": leverage},
            "Status": {"select": {"name": "OPEN"}},
            "Signal Data": {
                "rich_text": [
                    {"text": {"content": json.dumps(signal_data or {}, ensure_ascii=False)}}
                ]
            },
            "Opened At": {
                "date": {"start": datetime.now(timezone.utc).isoformat()}
            },
        }
        result = self.client.pages.create(
            parent={"database_id": self.database_id},
            properties=properties,
        )
        logger.info(f"거래 저장: {result['id']}")
        return result

    def close_trade(self, trade_id: str, exit_price: float, pnl: float) -> dict:
        properties = {
            "Exit Price": {"number": exit_price},
            "PnL": {"number": pnl},
            "Status": {"select": {"name": "CLOSED"}},
            "Closed At": {
                "date": {"start": datetime.now(timezone.utc).isoformat()}
            },
        }
        result = self.client.pages.update(
            page_id=trade_id,
            properties=properties,
        )
        logger.info(f"거래 종료: {trade_id}, PnL: {pnl:.4f}")
        return result

    def get_open_trades(self, symbol: str) -> list[dict]:
        response = self.client.databases.query(
            database_id=self.database_id,
            filter={
                "and": [
                    {"property": "Symbol", "title": {"equals": symbol}},
                    {"property": "Status", "select": {"equals": "OPEN"}},
                ]
            },
        )
        return response.get("results", [])
