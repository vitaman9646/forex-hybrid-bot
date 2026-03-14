"""
strategies/session_strategy.py

Session Models Strategy (по методологии Trade With BP)

Ключевые концепции:
1. Азия создаёт рэндж (ликвидность с двух сторон)
2. Франкфурт делает манипуляцию (снимает одну сторону Азии)
3. Лондон задаёт истинное направление дня
4. На НЙ — либо продолжение, либо разворот слабого Лондона

Работает на M15 данных, анализируя поведение каждой сессии.

Время (UTC):
  Asia:      00:00 - 07:00
  Frankfurt: 07:00 - 08:00
  London:    08:00 - 13:00
  New York:  13:00 - 17:00

(В PDF время дано в EST, здесь конвертировано в UTC)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from enum import Enum

from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from config import risk_config


# ═══════════════════════════════════════════════
#  Типы и структуры данных
# ═══════════════════════════════════════════════

class SessionType(Enum):
    ASIA = "asia"
    FRANKFURT = "frankfurt"
    LONDON = "london"
    PRE_NY = "pre_ny"
    NEW_YORK = "new_york"
    CLOSED = "closed"


class AsiaProfile(Enum):
    """Профиль Азиатской сессии"""
    RANGING = "ranging"               # Консолидация с ликвидностью
    TRENDING_CLEAN = "trending_clean"  # Тренд без ликвидности позади
    TRENDING_WITH_LIQ = "trending_liq" # Тренд, но оставляет пулы
    STRONG_IMBALANCE = "strong_imb"   # Сильный 1H имбаланс


class FrankfurtAction(Enum):
    """Действие Франкфурта"""
    MANIPULATION_HIGH = "manip_high"   # Снял верх Азии
    MANIPULATION_LOW = "manip_low"     # Снял низ Азии
    CONTINUATION = "continuation"       # Продолжил движение Азии
    CORRECTION_INTO_IMB = "correction"  # Коррекция в имбаланс
    NEUTRAL = "neutral"                 # Слабое движение


class LondonProfile(Enum):
    """Профиль Лондонской сессии"""
    STRONG_WITH_IMB = "strong_imb"      # Резкий с имбалансами
    WEAK_WITH_LIQ = "weak_liq"          # Слабый, оставляет ликвидность
    ONE_SIDED = "one_sided"             # Одностороннее движение
    CONSOLIDATION = "consolidation"     # Консолидация
    REVERSAL_AT_ASIA = "reversal_asia"  # Разворот у границы Азии


@dataclass
class SessionRange:
    """Рэндж торговой сессии"""
    high: float
    low: float
    open_price: float
    close_price: float
    direction: int           # 1=up, -1=down, 0=flat
    range_size: float        # high - low
    has_imbalance: bool      # есть ли сильный имбаланс
    imbalance_zones: List[Tuple[float, float]] = None
    liquidity_above: bool = False   # есть ли пулы выше
    liquidity_below: bool = False   # есть ли пулы ниже
    swing_count: int = 0            # количество свингов


@dataclass
class SessionContext:
    """Полный контекст текущего дня"""
    asia: Optional[SessionRange] = None
    frankfurt: Optional[SessionRange] = None
    london: Optional[SessionRange] = None
    asia_profile: AsiaProfile = AsiaProfile.RANGING
    frankfurt_action: FrankfurtAction = FrankfurtAction.NEUTRAL
    london_profile: LondonProfile = LondonProfile.CONSOLIDATION
    asia_high_swept: bool = False
    asia_low_swept: bool = False
    htf_bias: int = 0            # 1=bullish, -1=bearish, 0=neutral


# ═══════════════════════════════════════════════
#  Session Analyzer — анализ поведения сессий
# ═══════════════════════════════════════════════

class SessionAnalyzer:
    """
    Анализирует каждую сессию и определяет её профиль
    """

    # Границы сессий (UTC часы)
    SESSION_TIMES = {
        SessionType.ASIA:      (0, 7),
        SessionType.FRANKFURT:  (7, 8),
        SessionType.LONDON:     (8, 13),
        SessionType.PRE_NY:     (12, 13),
        SessionType.NEW_YORK:   (13, 17),
    }

    def __init__(self, utc_offset: int = 0):
        """
        utc_offset: смещение времени сервера MT5 от UTC.
        Например если MT5 в UTC+2, передайте 2.
        """
        self.utc_offset = utc_offset

    def get_current_session(self, utc_hour: int) -> SessionType:
        """Определить текущую сессию"""
        for session, (start, end) in self.SESSION_TIMES.items():
            if session in (SessionType.PRE_NY,):
                continue
            if start <= utc_hour < end:
                return session
        return SessionType.CLOSED

    def extract_session_data(
        self,
        df: pd.DataFrame,
        session: SessionType,
        target_date: pd.Timestamp = None
    ) -> Optional[pd.DataFrame]:
        """Извлечь данные конкретной сессии"""
        if df is None or len(df) == 0:
            return None

        start_h, end_h = self.SESSION_TIMES[session]

        # Корректируем на UTC offset сервера
        adj_start = (start_h + self.utc_offset) % 24
        adj_end = (end_h + self.utc_offset) % 24

        if hasattr(df.index, 'hour'):
            hours = df.index.hour
        else:
            return None

        if adj_start < adj_end:
            mask = (hours >= adj_start) & (hours < adj_end)
        else:
            # Переход через полночь
            mask = (hours >= adj_start) | (hours < adj_end)

        # Фильтр по дате если указана
        if target_date is not None:
            date_mask = df.index.date == target_date.date()
            mask = mask & date_mask

        session_data = df[mask]
        return session_data if len(session_data) > 0 else None

    def calculate_session_range(
        self,
        session_df: pd.DataFrame
    ) -> SessionRange:
        """Рассчитать рэндж сессии"""
        high = session_df["high"].max()
        low = session_df["low"].min()
        open_p = session_df["open"].iloc[0]
        close_p = session_df["close"].iloc[-1]
        range_size = high - low

        # Направление
        if close_p > open_p + range_size * 0.2:
            direction = 1
        elif close_p < open_p - range_size * 0.2:
            direction = -1
        else:
            direction = 0

        # Имбалансы (Fair Value Gaps на 1H+)
        imbalances = self._find_imbalances(session_df)
        has_imb = len(imbalances) > 0

        # Ликвидность (равные хаи/лоу = пулы)
        liq_above, liq_below = self._detect_liquidity_pools(
            session_df
        )

        # Количество свингов
        swings = self._count_swings(session_df)

        
