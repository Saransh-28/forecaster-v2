# tests/test_oms.py

import unittest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta, timezone

from oms import OMS, Position

class TestOMS(unittest.TestCase):
    @patch('oms.Client')
    def setUp(self, mock_client_cls):
        self.mock_client = MagicMock()
        mock_client_cls.return_value = self.mock_client

        self.mock_client.futures_create_order.return_value = {
            "orderId": 1,
            "executedQty": "0.5",
            "avgPrice": "100.0",
            "fills": [{"price": "100.0"}]
        }
        self.mock_client.futures_get_order.return_value = {"status": "NEW"}
        self.mock_client.futures_cancel_order.return_value = {}

        self.oms = OMS(
            api_key="dummy",
            api_secret="dummy",
            symbol="BTCUSDT",
            leverage=5,
            testnet=True,
            retry_attempts=2,
            retry_delay=0
        )

    def test_init_calls_margin_and_hedge(self):
        self.mock_client.futures_change_position_mode.assert_called_once_with(dualSidePosition=True)
        self.mock_client.futures_change_margin_type.assert_called_once_with(symbol="BTCUSDT", marginType="ISOLATED")
        self.mock_client.futures_change_leverage.assert_called_once_with(symbol="BTCUSDT", leverage=5)

    def test_place_order_success(self):
        expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
        pos_id = self.oms.place_order(
            direction="LONG",
            quantity=0.5,
            stop_loss=95.0,
            take_profit=110.0,
            expiry_ts=expiry
        )

        self.assertEqual(self.mock_client.futures_create_order.call_count, 3)
        self.assertIn(pos_id, self.oms.positions)
        pos = self.oms.positions[pos_id]
        self.assertEqual(pos.direction, "LONG")
        self.assertEqual(pos.quantity, 0.5)
        self.assertEqual(pos.entry_price, 100.0)
        self.assertEqual(pos.expiry, expiry)
        self.assertEqual(pos.status, "OPEN")

    def test_place_order_entry_failure(self):
        self.mock_client.futures_create_order.side_effect = Exception("fail")

        with self.assertRaises(RuntimeError) as cm:
            self.oms.place_order(
                "LONG", 0.5, 95.0, 110.0,
                datetime.now(timezone.utc)
            )
        self.assertIn("Entry order failed", str(cm.exception))
        self.assertEqual(self.mock_client.futures_create_order.call_count, self.oms.retry_attempts)

    def test_place_order_tp_failure_flattens(self):
        def side_effect(**kwargs):
            if kwargs.get("type") == "MARKET":
                return {"orderId": 2, "executedQty": "0.5", "avgPrice": "100.0", "fills": [{"price": "100.0"}]}
            raise Exception("fail")
        self.mock_client.futures_create_order.side_effect = side_effect

        with self.assertRaises(RuntimeError):
            self.oms.place_order(
                "SHORT", 0.5, 105.0, 90.0,
                datetime.now(timezone.utc)
            )

        self.assertIn(
            call(
                symbol="BTCUSDT",
                side="BUY",
                positionSide="SHORT",
                type="MARKET",
                quantity=0.5,
                reduceOnly=True,
            ),
            self.mock_client.futures_create_order.mock_calls
        )

    def test_place_order_sl_failure_cancels_tp_and_flattens(self):
        seq = []
        def create_side_effect(**kwargs):
            seq.append(kwargs["type"])
            if kwargs["type"] == "STOP_MARKET":
                raise Exception("fail")
            return {"orderId": len(seq), "executedQty": "0.5", "avgPrice": "100.0", "fills": [{"price": "100.0"}]}

        self.mock_client.futures_create_order.side_effect = create_side_effect

        with self.assertRaises(RuntimeError):
            self.oms.place_order(
                "LONG", 0.5, 95.0, 110.0,
                datetime.now(timezone.utc)
            )

        self.mock_client.futures_cancel_order.assert_any_call(
            symbol="BTCUSDT", orderId=2
        )

    def test_process_fills_sl_filled(self):
        pos = Position(
            id=10,
            direction="LONG",
            quantity=0.5,
            entry_price=100.0,
            sl_order_id=20,
            tp_order_id=21,
            expiry=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        self.oms.positions[10] = pos

        def get_order_side_effect(symbol, orderId):
            return {"status": "FILLED"} if orderId == 20 else {"status": "NEW"}
        self.mock_client.futures_get_order.side_effect = get_order_side_effect

        self.oms.process_fills()
        self.mock_client.futures_cancel_order.assert_called_once_with(
            symbol="BTCUSDT", orderId=21
        )
        self.assertEqual(self.oms.positions[10].status, "CLOSED")

    def test_enforce_expiry(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        pos = Position(
            id=30,
            direction="SHORT",
            quantity=1.0,
            entry_price=200.0,
            sl_order_id=40,
            tp_order_id=41,
            expiry=past,
        )
        self.oms.positions[30] = pos

        self.oms.enforce_expiry()
        self.assertEqual(self.oms.positions[30].status, "EXPIRED")

        reduce_calls = [
            call_args for call_args in self.mock_client.futures_create_order.call_args_list
            if call_args.kwargs.get("type") == "MARKET" and call_args.kwargs.get("reduceOnly")
        ]
        self.assertTrue(reduce_calls)

if __name__ == '__main__':
    unittest.main()
