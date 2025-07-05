import threading
from time import sleep
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Literal

from binance import Client

@dataclass
class Position:
    id: int
    direction: Literal["LONG", "SHORT"]
    quantity: float
    entry_price: float
    sl_order_id: int
    tp_order_id: int
    expiry: datetime
    status: Literal["OPEN", "CLOSED", "EXPIRED"] = field(default="OPEN")

class OMS:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        leverage: int = 10,
        testnet: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 0.2,
    ):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.symbol = symbol.upper()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.positions: dict[int, Position] = {}

        self._retry(self.client.futures_change_position_mode, dualSidePosition=True)
        self._retry(self.client.futures_change_margin_type,
                    symbol=self.symbol, marginType="ISOLATED")
        self._retry(self.client.futures_change_leverage,
                    symbol=self.symbol, leverage=leverage)

        t = threading.Thread(target=self._background_loop, daemon=True)
        t.start()

    def _retry(self, func, note: str = "", **kwargs):
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return func(**kwargs)
            except Exception:
                if attempt == self.retry_attempts:
                    raise
                sleep(self.retry_delay)

    def place_order(
        self,
        direction: Literal["LONG", "SHORT"],
        quantity: float,
        stop_loss: float,
        take_profit: float,
        expiry_ts: datetime,
    ) -> int:
        side = "BUY" if direction == "LONG" else "SELL"
        try:
            entry = self._retry(
                self.client.futures_create_order,
                symbol=self.symbol,
                side=side,
                positionSide=direction,
                type="MARKET",
                quantity=quantity,
                priceProtect=False,
            )
        except Exception:
            raise RuntimeError("Entry order failed; aborting without SL/TP.")

        filled_qty = float(entry.get("executedQty", 0))
        entry_price = float(entry.get("avgPrice") or entry["fills"][0]["price"])
        pos_id = entry["orderId"]

        try:
            tp = self._retry(
                self.client.futures_create_order,
                symbol=self.symbol,
                side=("SELL" if direction == "LONG" else "BUY"),
                positionSide=direction,
                type="TAKE_PROFIT_MARKET",
                stopPrice=take_profit,
                quantity=filled_qty,
                timeInForce="GTC",
            )
        except Exception:
            self._flat_and_mark_closed(pos_id, filled_qty, direction)
            raise RuntimeError("TP placement failed; position flat-tened.")

        try:
            sl = self._retry(
                self.client.futures_create_order,
                symbol=self.symbol,
                side=("SELL" if direction == "LONG" else "BUY"),
                positionSide=direction,
                type="STOP_MARKET",
                stopPrice=stop_loss,
                quantity=filled_qty,
                timeInForce="GTC",
            )
        except Exception:
            self.client.futures_cancel_order(
                symbol=self.symbol, orderId=tp["orderId"]
            )
            self._flat_and_mark_closed(pos_id, filled_qty, direction)
            raise RuntimeError("SL placement failed; TP canceled and position flat-tened.")

        self.positions[pos_id] = Position(
            id=pos_id,
            direction=direction,
            quantity=filled_qty,
            entry_price=entry_price,
            sl_order_id=sl["orderId"],
            tp_order_id=tp["orderId"],
            expiry=expiry_ts,
        )
        return pos_id

    def _flat_and_mark_closed(self, pos_id: int, qty: float, direction: str):
        """Force-flat via market and mark status CLOSED."""
        side = "SELL" if direction == "LONG" else "BUY"
        self._retry(
            self.client.futures_create_order,
            symbol=self.symbol,
            side=side,
            positionSide=direction,
            type="MARKET",
            quantity=qty,
            reduceOnly=True,
        )
        if pos_id in self.positions:
            pos = self.positions[pos_id]
            for oid in (pos.sl_order_id, pos.tp_order_id):
                try:
                    self.client.futures_cancel_order(
                        symbol=self.symbol, orderId=oid
                    )
                except:
                    pass
            pos.status = "CLOSED"

    def process_fills(self):
        """Check for SL/TP fills and cancel the sibling on execution."""
        for pos in list(self.positions.values()):
            if pos.status != "OPEN":
                continue

            sl_status = self._retry(
                self.client.futures_get_order,
                symbol=self.symbol,
                orderId=pos.sl_order_id,
            )
            tp_status = self._retry(
                self.client.futures_get_order,
                symbol=self.symbol,
                orderId=pos.tp_order_id,
            )

            if sl_status["status"] == "FILLED":
                self.client.futures_cancel_order(
                    symbol=self.symbol, orderId=pos.tp_order_id
                )
                pos.status = "CLOSED"

            elif tp_status["status"] == "FILLED":
                self.client.futures_cancel_order(
                    symbol=self.symbol, orderId=pos.sl_order_id
                )
                pos.status = "CLOSED"

    def enforce_expiry(self):
        """Force-flat any OPEN position whose expiry has passed."""
        now = datetime.now(timezone.utc)
        for pos in list(self.positions.values()):
            if pos.status != "OPEN":
                continue
            if now >= pos.expiry:
                self._flat_and_mark_closed(pos.id, pos.quantity, pos.direction)
                pos.status = "EXPIRED"

    def _background_loop(self):
        """Daemon thread: wake up every 5 s to process fills + expiry."""
        while True:
            try:
                self.process_fills()
                self.enforce_expiry()
            except Exception:
                pass
            sleep(5)
