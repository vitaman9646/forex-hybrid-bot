"""
Execution Engine
Отправка ордеров в MT5, управление позициями, трейлинг стоп
"""

import MetaTrader5 as mt5
from typing import Optional, Dict, List
from datetime import datetime
from strategies.base_strategy import TradeSignal, SignalType
from core.risk_manager import RiskCheck
from config import trading_config, risk_config


class ExecutionEngine:
    """Исполнение торговых ордеров"""

    def __init__(self):
        self.last_order_result = None

    def execute_trade(
        self,
        signal: TradeSignal,
        risk_check: RiskCheck
    ) -> Optional[Dict]:
        """
        Открытие позиции
        Возвращает информацию о сделке или None
        """
        if not risk_check.approved:
            print(f"[EXEC] Сделка отклонена: {risk_check.reason}")
            return None

        # Тип ордера
        if signal.signal_type == SignalType.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(signal.symbol).ask
        elif signal.signal_type == SignalType.SELL:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(signal.symbol).bid
        else:
            return None

        # Формируем запрос
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": risk_check.lot_size,
            "type": order_type,
            "price": price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "deviation": 20,              # допустимое проскальзывание
            "magic": trading_config.magic_number,
            "comment": f"HybridBot|{signal.strategy_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Отправляем ордер
        result = mt5.order_send(request)

        if result is None:
            print(f"[EXEC] Ошибка отправки: {mt5.last_error()}")
            return None

        self.last_order_result = result

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                f"[EXEC] Ордер отклонен: "
                f"code={result.retcode}, "
                f"comment={result.comment}"
            )
            return None

        trade_info = {
            "ticket": result.order,
            "symbol": signal.symbol,
            "type": signal.signal_type.value,
            "volume": risk_check.lot_size,
            "price": result.price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "strategy": signal.strategy_name,
            "reason": signal.reason,
            "confidence": signal.confidence,
            "time": datetime.now().isoformat()
        }

        print(
            f"[EXEC] ✅ Ордер исполнен: "
            f"{signal.signal_type.value} {signal.symbol} "
            f"@ {result.price} | "
            f"lot={risk_check.lot_size} | "
            f"SL={signal.stop_loss} TP={signal.take_profit}"
        )

        return trade_info

    def close_position(
        self,
        ticket: int,
        symbol: str = None
    ) -> bool:
        """Закрытие позиции по тикету"""
        # Находим позицию
        positions = mt5.positions_get(ticket=ticket)

        if positions is None or len(positions) == 0:
            print(f"[EXEC] Позиция {ticket} не найдена")
            return False

        position = positions[0]
        symbol = symbol or position.symbol

        # Обратный ордер
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": trading_config.magic_number,
            "comment": "HybridBot|close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[EXEC] ✅ Позиция {ticket} закрыта")
            return True

        print(f"[EXEC] ❌ Ошибка закрытия {ticket}: {result}")
        return False

    def close_all_positions(self, symbol: str = None) -> int:
        """Закрыть все позиции (или по символу)"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic == trading_config.magic_number:
                if self.close_position(pos.ticket, pos.symbol):
                    closed += 1

        print(f"[EXEC] Закрыто позиций: {closed}")
        return closed

    def update_trailing_stop(
        self,
        atr_values: Dict[str, float]
    ):
        """
        Трейлинг стоп на основе ATR
        Сдвигает SL по мере роста прибыли
        """
        if not risk_config.trailing_stop:
            return

        positions = mt5.positions_get()
        if positions is None:
            return

        for pos in positions:
            if pos.magic != trading_config.magic_number:
                continue

            atr = atr_values.get(pos.symbol, 0)
            if atr == 0:
                continue

            trailing_distance = atr * risk_config.trailing_atr_mult
            current_price_info = mt5.symbol_info_tick(pos.symbol)

            if current_price_info is None:
                continue

            new_sl = None

            # BUY позиция
            if pos.type == mt5.POSITION_TYPE_BUY:
                potential_sl = current_price_info.bid - trailing_distance
                if potential_sl > pos.sl and potential_sl > pos.price_open:
                    new_sl = round(potential_sl, 5)

            # SELL позиция
            elif pos.type == mt5.POSITION_TYPE_SELL:
                potential_sl = current_price_info.ask + trailing_distance
                if potential_sl < pos.sl and potential_sl < pos.price_open:
                    new_sl = round(potential_sl, 5)

            if new_sl is not None:
                self._modify_sl(pos.ticket, pos.symbol, new_sl, pos.tp)

    def _modify_sl(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        tp: float
    ):
        """Изменение SL позиции"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": tp,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(
                f"[EXEC] Trailing SL → {new_sl} "
                f"для позиции {ticket}"
            )

    def get_open_positions(self) -> List[Dict]:
        """Список открытых позиций бота"""
        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            if pos.magic == trading_config.magic_number:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "comment": pos.comment
                })

        return result
