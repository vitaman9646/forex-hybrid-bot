"""
Base Strategy
Абстрактный базовый класс для всех стратегий
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class SignalType(Enum):
    """Тип торгового сигнала"""
    BUY = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass
class TradeSignal:
    """Торговый сигнал от стратегии"""
    signal_type: SignalType
    strategy_name: str
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float           # 0.0 — 1.0
    reason: str                 # причина входа
    metadata: dict = None       # доп данные


class BaseStrategy(ABC):
    """Базовый класс стратегии"""

    def __init__(self, name: str):
        self.name = name
        self.is_active = True

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:
        """Генерация торгового сигнала — реализуется в каждой стратегии"""
        pass

    @abstractmethod
    def get_required_indicators(self) -> list:
        """Список необходимых индикаторов"""
        pass

    def _no_signal(self, symbol: str) -> TradeSignal:
        """Возвращает пустой сигнал"""
        return TradeSignal(
            signal_type=SignalType.NONE,
            strategy_name=self.name,
            symbol=symbol,
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            confidence=0,
            reason="Нет сигнала"
        )
