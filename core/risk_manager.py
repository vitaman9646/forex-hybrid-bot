"""
Risk Manager
Контроль рисков, расчёт размера позиции, лимиты
"""

import MetaTrader5 as mt5
from datetime import datetime, date
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from strategies.base_strategy import TradeSignal, SignalType
from config import risk_config, trading_config


@dataclass
class RiskCheck:
    """Результат проверки рисков"""
    approved: bool
    lot_size: float
    reason: str
    details: Dict = None


@dataclass
class DailyStats:
    """Статистика за день"""
    date: date = None
    total_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0


class RiskManager:
    """Управление рисками"""

    def __init__(self):
        self.daily_stats = DailyStats(date=date.today())
        self.open_trades: List[Dict] = []

    def check_trade(self, signal: TradeSignal) -> RiskCheck:
        """
        Полная проверка перед открытием сделки
        Возвращает RiskCheck с решением
        """
        # Обновляем дату
        self._update_daily_stats()

        # 1. Проверка типа сигнала
        if signal.signal_type == SignalType.NONE:
            return RiskCheck(
                approved=False,
                lot_size=0,
                reason="Нет сигнала"
            )

        # 2. Проверка дневного лимита убытков
        if not self._check_daily_loss():
            return RiskCheck(
                approved=False,
                lot_size=0,
                reason=f"Дневной лимит убытков достигнут: "
                       f"{self.daily_stats.total_pnl:.2f}"
            )

        # 3. Проверка количества открытых позиций
        if not self._check_max_positions():
            return RiskCheck(
                approved=False,
                lot_size=0,
                reason=f"Макс позиций достигнуто: "
                       f"{risk_config.max_open_trades}"
            )

        # 4. Проверка спреда
        if not self._check_spread(signal.symbol):
            return RiskCheck(
                approved=False,
                lot_size=0,
                reason="Спред слишком высокий"
            )

        # 5. Расчёт размера позиции
        lot_size = self._calculate_position_size(signal)

        if lot_size <= 0:
            return RiskCheck(
                approved=False,
                lot_size=0,
                reason="Невозможно рассчитать размер позиции"
            )

        # 6. Проверка минимального/максимального лота
        symbol_info = mt5.symbol_info(signal.symbol)
        if symbol_info:
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step

            if lot_size < min_lot:
                return RiskCheck(
                    approved=False,
                    lot_size=0,
                    reason=f"Лот {lot_size} меньше минимума {min_lot}"
                )

            # Округляем до lot_step
            lot_size = round(
                round(lot_size / lot_step) * lot_step, 2
            )
            lot_size = min(lot_size, max_lot)

        return RiskCheck(
            approved=True,
            lot_size=lot_size,
            reason="Одобрено",
            details={
                "risk_amount": self._get_account_balance()
                              * risk_config.risk_per_trade,
                "daily_pnl": self.daily_stats.total_pnl,
                "open_positions": len(self.open_trades)
            }
        )

    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """
        Расчёт размера позиции на основе риска
        lot_size = (balance × risk%) / (SL в пунктах × pip_value)
        """
        balance = self._get_account_balance()

        if balance <= 0:
            return 0

        risk_amount = balance * risk_config.risk_per_trade

        # Расчёт SL в пунктах
        sl_distance = abs(signal.entry_price - signal.stop_loss)

        if sl_distance == 0:
            return 0

        # Информация о символе
        symbol_info = mt5.symbol_info(signal.symbol)

        if symbol_info is None:
            return 0

        # Стоимость пункта
        point = symbol_info.point
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size

        if tick_size == 0:
            return 0

        # SL в тиках
        sl_ticks = sl_distance / tick_size

        # Размер лота
        if sl_ticks * tick_value > 0:
            lot_size = risk_amount / (sl_ticks * tick_value)
        else:
            lot_size = symbol_info.volume_min

        return round(lot_size, 2)

    def _check_daily_loss(self) -> bool:
        """Проверка дневного лимита убытков"""
        balance = self._get_account_balance()
        max_loss = balance * risk_config.max_daily_loss

        return self.daily_stats.total_pnl > -max_loss

    def _check_max_positions(self) -> bool:
        """Проверка количества открытых позиций"""
        positions = mt5.positions_total()
        if positions is None:
            positions = 0
        return positions < risk_config.max_open_trades

    def _check_spread(self, symbol: str) -> bool:
        """Проверка спреда"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        return info.spread <= risk_config.max_spread_points

    @staticmethod
    def _get_account_balance() -> float:
        """Баланс аккаунта"""
        info = mt5.account_info()
        if info is None:
            return 0
        return info.balance

    @staticmethod
    def _get_account_equity() -> float:
        """Эквити аккаунта"""
        info = mt5.account_info()
        if info is None:
            return 0
        return info.equity

    def _update_daily_stats(self):
        """Обновление дневной статистики"""
        today = date.today()
        if self.daily_stats.date != today:
            # Новый день — сбрасываем
            self.daily_stats = DailyStats(date=today)

    def update_pnl(self, pnl: float, is_win: bool):
        """Обновить P&L после закрытия сделки"""
        self._update_daily_stats()
        self.daily_stats.total_pnl += pnl
        self.daily_stats.trades_count += 1
        if is_win:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1

    def get_stats(self) -> Dict:
        """Текущая статистика"""
        balance = self._get_account_balance()
        equity = self._get_account_equity()

        return {
            "balance": balance,
            "equity": equity,
            "daily_pnl": self.daily_stats.total_pnl,
            "daily_trades": self.daily_stats.trades_count,
            "daily_wins": self.daily_stats.wins,
            "daily_losses": self.daily_stats.losses,
            "open_positions": mt5.positions_total() or 0,
            "daily_loss_limit": balance * risk_config.max_daily_loss,
            "risk_per_trade": balance * risk_config.risk_per_trade
        }
