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

        return SessionRange(
            high=high,
            low=low,
            open_price=open_p,
            close_price=close_p,
            direction=direction,
            range_size=range_size,
            has_imbalance=has_imb,
            imbalance_zones=imbalances,
            liquidity_above=liq_above,
            liquidity_below=liq_below,
            swing_count=swings
        )

    def classify_asia(
        self,
        asia_range: SessionRange
    ) -> AsiaProfile:
        """Классифицировать профиль Азии"""
        has_liq = asia_range.liquidity_above or asia_range.liquidity_below

        # Консолидация: нет чёткого направления, есть пулы с двух сторон
        if (
            asia_range.direction == 0
            and asia_range.liquidity_above
            and asia_range.liquidity_below
        ):
            return AsiaProfile.RANGING

        # Сильный имбаланс
        if asia_range.has_imbalance and abs(asia_range.direction) == 1:
            if not has_liq:
                return AsiaProfile.STRONG_IMBALANCE
            return AsiaProfile.TRENDING_WITH_LIQ

        # Тренд без ликвидности
        if abs(asia_range.direction) == 1 and not has_liq:
            return AsiaProfile.TRENDING_CLEAN

        # Тренд с ликвидностью
        if abs(asia_range.direction) == 1 and has_liq:
            return AsiaProfile.TRENDING_WITH_LIQ

        return AsiaProfile.RANGING

    def classify_frankfurt(
        self,
        frankfurt_range: SessionRange,
        asia_range: SessionRange
    ) -> FrankfurtAction:
        """Классифицировать действие Франкфурта"""
        fk_high = frankfurt_range.high
        fk_low = frankfurt_range.low

        # Снял верх Азии?
        swept_high = fk_high > asia_range.high
        # Снял низ Азии?
        swept_low = fk_low < asia_range.low

        if swept_high and not swept_low:
            return FrankfurtAction.MANIPULATION_HIGH

        if swept_low and not swept_high:
            return FrankfurtAction.MANIPULATION_LOW

        # Коррекция в зону имбаланса
        if asia_range.has_imbalance and asia_range.imbalance_zones:
            for imb_high, imb_low in asia_range.imbalance_zones:
                if fk_low <= imb_high and fk_high >= imb_low:
                    return FrankfurtAction.CORRECTION_INTO_IMB

        # Продолжение
        if (
            asia_range.direction == 1
            and frankfurt_range.direction == 1
        ):
            return FrankfurtAction.CONTINUATION
        if (
            asia_range.direction == -1
            and frankfurt_range.direction == -1
        ):
            return FrankfurtAction.CONTINUATION

        return FrankfurtAction.NEUTRAL

    def classify_london(
        self,
        london_df: pd.DataFrame,
        london_range: SessionRange,
        asia_range: SessionRange,
        frankfurt_action: FrankfurtAction
    ) -> LondonProfile:
        """Классифицировать профиль Лондона"""

        # Проверяем снял ли Лондон противоположную границу Азии
        london_swept_asia_high = london_range.high > asia_range.high
        london_swept_asia_low = london_range.low < asia_range.low

        # Формация 2: резкий Лондон с имбалансами
        if london_range.has_imbalance and london_range.range_size > 0:
            return LondonProfile.STRONG_WITH_IMB

        # Формация 1: слабый Лондон без имбалансов, с пулами
        if (
            not london_range.has_imbalance
            and (london_range.liquidity_above or london_range.liquidity_below)
        ):
            return LondonProfile.WEAK_WITH_LIQ

        # Формация 3: односторонний Лондон
        if london_range.swing_count <= 2 and abs(london_range.direction) == 1:
            return LondonProfile.ONE_SIDED

        # Формация 4: разворот у границы Азии
        if (
            london_swept_asia_low or london_swept_asia_high
        ):
            # Проверяем что после снятия был разворот
            if london_range.direction != 0:
                return LondonProfile.REVERSAL_AT_ASIA

        # Формация 7: консолидация
        if london_range.swing_count >= 4 and london_range.direction == 0:
            return LondonProfile.CONSOLIDATION

        return LondonProfile.WEAK_WITH_LIQ

    def get_htf_bias(self, df: pd.DataFrame) -> int:
        """
        Определить bias старшего таймфрейма
        Используем EMA50/200 на текущих данных
        """
        if df is None or len(df) < 200:
            return 0

        if "ema_50" in df.columns and "ema_200" in df.columns:
            ema50 = df["ema_50"].iloc[-1]
            ema200 = df["ema_200"].iloc[-1]

            if ema50 > ema200:
                return 1
            elif ema50 < ema200:
                return -1

        return 0

    # ─── Вспомогательные методы ──────────────────

    @staticmethod
    def _find_imbalances(
        df: pd.DataFrame,
    ) -> List[Tuple[float, float]]:
        """
        Найти имбалансы (FVG — Fair Value Gaps)
        FVG: gap между high[i-1] и low[i+1] (бычий)
             или low[i-1] и high[i+1] (медвежий)
        """
        imbalances = []

        if len(df) < 3:
            return imbalances

        for i in range(1, len(df) - 1):
            prev_high = df["high"].iloc[i - 1]
            curr_close = df["close"].iloc[i]
            curr_open = df["open"].iloc[i]
            next_low = df["low"].iloc[i + 1]
            next_high = df["high"].iloc[i + 1]
            prev_low = df["low"].iloc[i - 1]

            # Бычий FVG: gap между high[i-1] и low[i+1]
            if next_low > prev_high:
                gap_size = next_low - prev_high
                avg_range = df["high"].iloc[i] - df["low"].iloc[i]
                if avg_range > 0 and gap_size > avg_range * 0.3:
                    imbalances.append((next_low, prev_high))

            # Медвежий FVG: gap между low[i-1] и high[i+1]
            if next_high < prev_low:
                gap_size = prev_low - next_high
                avg_range = df["high"].iloc[i] - df["low"].iloc[i]
                if avg_range > 0 and gap_size > avg_range * 0.3:
                    imbalances.append((prev_low, next_high))

        return imbalances

    @staticmethod
    def _detect_liquidity_pools(
        df: pd.DataFrame,
        tolerance_pct: float = 0.001
    ) -> Tuple[bool, bool]:
        """
        Обнаружить пулы ликвидности (equal highs/lows)
        Если 2+ свинговых хая/лоу находятся рядом — это пул
        """
        if len(df) < 5:
            return False, False

        # Свинговые хаи
        swing_highs = []
        swing_lows = []

        for i in range(2, len(df) - 2):
            h = df["high"].iloc[i]
            if (
                h > df["high"].iloc[i-1]
                and h > df["high"].iloc[i-2]
                and h > df["high"].iloc[i+1]
                and h > df["high"].iloc[i+2]
            ):
                swing_highs.append(h)

            lo = df["low"].iloc[i]
            if (
                lo < df["low"].iloc[i-1]
                and lo < df["low"].iloc[i-2]
                and lo < df["low"].iloc[i+1]
                and lo < df["low"].iloc[i+2]
            ):
                swing_lows.append(lo)

        # Проверяем equal highs
        liq_above = False
        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                avg = (swing_highs[i] + swing_highs[j]) / 2
                if avg > 0:
                    diff = abs(swing_highs[i] - swing_highs[j]) / avg
                    if diff < tolerance_pct:
                        liq_above = True
                        break

        # Проверяем equal lows
        liq_below = False
        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                avg = (swing_lows[i] + swing_lows[j]) / 2
                if avg > 0:
                    diff = abs(swing_lows[i] - swing_lows[j]) / avg
                    if diff < tolerance_pct:
                        liq_below = True
                        break

        return liq_above, liq_below

    @staticmethod
    def _count_swings(
        df: pd.DataFrame,
        lookback: int = 2
    ) -> int:
        """Подсчёт свингов (смена направления)"""
        if len(df) < lookback * 2 + 1:
            return 0

        swings = 0
        for i in range(lookback, len(df) - lookback):
            # Свинг-хай
            is_high = all(
                df["high"].iloc[i] > df["high"].iloc[i + k]
                for k in range(-lookback, lookback + 1)
                if k != 0
            )
            # Свинг-лоу
            is_low = all(
                df["low"].iloc[i] < df["low"].iloc[i + k]
                for k in range(-lookback, lookback + 1)
                if k != 0
            )
            if is_high or is_low:
                swings += 1

        return swings


# ═══════════════════════════════════════════════
#  Session Strategy — основная стратегия
# ═══════════════════════════════════════════════

class SessionStrategy(BaseStrategy):
    """
    Session Models Strategy

    Логика торговли по 3 основных модели:

    MODEL A — Asia Continuation on London:
      Условия: Азия с сильным имбалансом, без ликвидности позади.
      Франкфурт корректируется в имбаланс.
      Вход: На Лондоне в направлении Азии, от имбаланса.
      Цель: Расширение движения Азии.

    MODEL B — Frankfurt Manipulation Reversal:
      Условия: Азия в рэндже (ликвидность с двух сторон).
      Франкфурт снимает одну сторону Азии.
      Лондон НЕ оставляет сильных имбалансов.
      Вход: Против Франкфурта, к противоположной границе Азии.
      Цель: Противоположная граница Азии.

    MODEL C — London Continuation on NY:
      Условия: Лондон дал резкое движение с имбалансами.
      Вход: На НЙ от зоны имбаланса/OF Лондона, по направлению.
      Цель: Продолжение движения Лондона.
    """

    def __init__(self, utc_offset: int = 0):
        super().__init__("session_strategy")
        self.analyzer = SessionAnalyzer(utc_offset=utc_offset)
        self.context: Optional[SessionContext] = None
        self._last_analysis_date = None

    def get_required_indicators(self) -> list:
        return ["atr", "ema_50", "ema_200", "rsi"]

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:
        """Генерация сигнала на основе сессионных моделей"""

        if df is None or len(df) < 50:
            return self._no_signal(symbol)

        # Определяем текущую сессию
        now = df.index[-1]
        utc_hour = (now.hour - self.analyzer.utc_offset) % 24
        current_session = self.analyzer.get_current_session(utc_hour)

        # Анализируем контекст дня (один раз в день или при новых данных)
        self._update_context(df, now)

        if self.context is None:
            return self._no_signal(symbol)

        # Получаем ATR для SL/TP
        atr = df["atr"].iloc[-1] if "atr" in df.columns else 0
        if atr == 0:
            return self._no_signal(symbol)

        close = df["close"].iloc[-1]

        # ═══ MODEL A: Asia Continuation ═══
        if current_session in (SessionType.LONDON,):
            signal = self._model_a_asia_continuation(
                df, symbol, close, atr
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        # ═══ MODEL B: Frankfurt Manipulation ═══
        if current_session in (SessionType.LONDON,):
            signal = self._model_b_frankfurt_reversal(
                df, symbol, close, atr
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        # ═══ MODEL C: London Continuation on NY ═══
        if current_session in (SessionType.NEW_YORK, SessionType.PRE_NY):
            signal = self._model_c_london_continuation(
                df, symbol, close, atr
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        return self._no_signal(symbol)

    # ─── MODEL A ─────────────────────────────────

    def _model_a_asia_continuation(
        self,
        df: pd.DataFrame,
        symbol: str,
        close: float,
        atr: float
    ) -> TradeSignal:
        """
        Продолжение движения Азии на Лондоне

        Условия:
        - Азия с сильным имбалансом или чистым трендом
        - Нет ликвидности позади движения
        - Франкфурт корректируется в имбаланс (или продолжает)
        - HTF bias совпадает с направлением Азии
        """
        ctx = self.context

        if ctx.asia is None:
            return self._no_signal(symbol)

        # Профиль Азии: только тренд без ликвидности или с имбалансом
        valid_profiles = (
            AsiaProfile.STRONG_IMBALANCE,
            AsiaProfile.TRENDING_CLEAN
        )
        if ctx.asia_profile not in valid_profiles:
            return self._no_signal(symbol)

        # Франкфурт: коррекция в имбаланс или продолжение
        valid_fk = (
            FrankfurtAction.CORRECTION_INTO_IMB,
            FrankfurtAction.CONTINUATION,
            FrankfurtAction.NEUTRAL
        )
        if ctx.frankfurt_action not in valid_fk:
            return self._no_signal(symbol)

        # HTF bias должен совпадать
        if ctx.htf_bias != 0 and ctx.htf_bias != ctx.asia.direction:
            return self._no_signal(symbol)

        asia_dir = ctx.asia.direction

        if asia_dir == 0:
            return self._no_signal(symbol)

        # Проверяем что цена откатила в зону имбаланса
        entry_valid = False
        if ctx.asia.imbalance_zones:
            for imb_high, imb_low in ctx.asia.imbalance_zones:
                if imb_low <= close <= imb_high:
                    entry_valid = True
                    break

        if not entry_valid:
            # Проверяем что цена около уровня коррекции
            if asia_dir == 1:
                fib_618 = ctx.asia.high - (ctx.asia.range_size * 0.618)
                if close <= fib_618 + atr * 0.3:
                    entry_valid = True
            else:
                fib_618 = ctx.asia.low + (ctx.asia.range_size * 0.618)
                if close >= fib_618 - atr * 0.3:
                    entry_valid = True

        if not entry_valid:
            return self._no_signal(symbol)

        # Генерируем сигнал
        if asia_dir == 1:
            sl = close - atr * 1.5
            tp = ctx.asia.high + atr * 2.0
            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=0.7,
                reason=(
                    f"MODEL A: Asia continuation BUY | "
                    f"Asia={ctx.asia_profile.value} | "
                    f"FK={ctx.frankfurt_action.value}"
                ),
                metadata={
                    "model": "A",
                    "asia_profile": ctx.asia_profile.value,
                    "asia_high": ctx.asia.high,
                    "asia_low": ctx.asia.low,
                }
            )
        else:
            sl = close + atr * 1.5
            tp = ctx.asia.low - atr * 2.0
            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=0.7,
                reason=(
                    f"MODEL A: Asia continuation SELL | "
                    f"Asia={ctx.asia_profile.value} | "
                    f"FK={ctx.frankfurt_action.value}"
                ),
                metadata={
                    "model": "A",
                    "asia_profile": ctx.asia_profile.value,
                    "asia_high": ctx.asia.high,
                    "asia_low": ctx.asia.low,
                }
            )

    # ─── MODEL B ─────────────────────────────────

    def _model_b_frankfurt_reversal(
        self,
        df: pd.DataFrame,
        symbol: str,
        close: float,
        atr: float
    ) -> TradeSignal:
        """
        Разворот манипуляции Франкфурта на Лондоне

        Условия:
        - Азия в рэндже (ликвидность с двух сторон)
        - Франкфурт снимает одну сторону Азии (манипуляция)
        - Лондон НЕ даёт сильного продолжения с имбалансами
        - Цена разворачивается к противоположной границе Азии
        """
        ctx = self.context

        if ctx.asia is None or ctx.frankfurt is None:
            return self._no_signal(symbol)

        # Азия должна быть в рэндже или с ликвидностью
        if ctx.asia_profile not in (
            AsiaProfile.RANGING,
            AsiaProfile.TRENDING_WITH_LIQ
        ):
            return self._no_signal(symbol)

        # Франкфурт должен сделать манипуляцию
        if ctx.frankfurt_action not in (
            FrankfurtAction.MANIPULATION_HIGH,
            FrankfurtAction.MANIPULATION_LOW
        ):
            return self._no_signal(symbol)

        # Лондон НЕ должен быть с сильными имбалансами
        if (
            ctx.london_profile == LondonProfile.STRONG_WITH_IMB
        ):
            return self._no_signal(symbol)

        # Направление: ПРОТИВ манипуляции Франкфурта
        if ctx.frankfurt_action == FrankfurtAction.MANIPULATION_HIGH:
            # Франкфурт снял верх → ожидаем SELL
            target = ctx.asia.low
            sl = ctx.frankfurt.high + atr * 0.5
            tp = target - atr * 0.5
            confidence = 0.65

            # Проверяем что цена уже развернулась
            if close > ctx.asia.high:
                return self._no_signal(symbol)

            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=confidence,
                reason=(
                    f"MODEL B: FK manip high → SELL to Asia low | "
                    f"FK swept {ctx.frankfurt.high:.5f}"
                ),
                metadata={
                    "model": "B",
                    "fk_action": ctx.frankfurt_action.value,
                    "target": target,
                    "asia_high": ctx.asia.high,
                    "asia_low": ctx.asia.low,
                }
            )

        elif ctx.frankfurt_action == FrankfurtAction.MANIPULATION_LOW:
            # Франкфурт снял низ → ожидаем BUY
            target = ctx.asia.high
            sl = ctx.frankfurt.low - atr * 0.5
            tp = target + atr * 0.5
            confidence = 0.65

            if close < ctx.asia.low:
                return self._no_signal(symbol)

            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=confidence,
                reason=(
                    f"MODEL B: FK manip low → BUY to Asia high | "
                    f"FK swept {ctx.frankfurt.low:.5f}"
                ),
                metadata={
                    "model": "B",
                    "fk_action": ctx.frankfurt_action.value,
                    "target": target,
                    "asia_high": ctx.asia.high,
                    "asia_low": ctx.asia.low,
                }
            )

        return self._no_signal(symbol)

    # ─── MODEL C ─────────────────────────────────

    def _model_c_london_continuation(
        self,
        df: pd.DataFrame,
        symbol: str,
        close: float,
        atr: float
    ) -> TradeSignal:
        """
        Продолжение сильного Лондона на Нью-Йорке

        Условия:
        - Лондон дал резкое движение с 1H имбалансами
        - НЙ откатывает в зону имбаланса/Order Flow Лондона
        - Вход от этой зоны по направлению Лондона
        """
        ctx = self.context

        if ctx.london is None:
            return self._no_signal(symbol)

        # Лондон должен быть с имбалансами
        if ctx.london_profile != LondonProfile.STRONG_WITH_IMB:
            return self._no_signal(symbol)

        london_dir = ctx.london.direction
        if london_dir == 0:
            return self._no_signal(symbol)

        # Проверяем откат к зоне имбаланса Лондона
        entry_valid = False

        if ctx.london.imbalance_zones:
            for imb_high, imb_low in ctx.london.imbalance_zones:
                if imb_low <= close <= imb_high:
                    entry_valid = True
                    break

        if not entry_valid:
            # Откат к 50% рэнджа Лондона
            mid = (ctx.london.high + ctx.london.low) / 2
            if abs(close - mid) < atr * 0.5:
                entry_valid = True

        if not entry_valid:
            return self._no_signal(symbol)

        # Генерируем сигнал по направлению Лондона
        if london_dir == 1:
            sl = close - atr * 1.5
            tp = ctx.london.high + atr * 1.5
            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=0.6,
                reason=(
                    f"MODEL C: London continuation BUY on NY | "
                    f"London={ctx.london_profile.value}"
                ),
                metadata={
                    "model": "C",
                    "london_profile": ctx.london_profile.value,
                    "london_high": ctx.london.high,
                    "london_low": ctx.london.low,
                }
            )
        else:
            sl = close + atr * 1.5
            tp = ctx.london.low - atr * 1.5
            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=0.6,
                reason=(
                    f"MODEL C: London continuation SELL on NY | "
                    f"London={ctx.london_profile.value}"
                ),
                metadata={
                    "model": "C",
                    "london_profile": ctx.london_profile.value,
                    "london_high": ctx.london.high,
                    "london_low": ctx.london.low,
                }
            )

    # ─── Context Update ──────────────────────────

    def _update_context(
        self,
        df: pd.DataFrame,
        now: pd.Timestamp
    ):
        """Обновить контекст текущего дня"""
        current_date = now.date()

        if self._last_analysis_date == current_date and self.context:
            # Обновляем только London/NY если появились новые данные
            self._update_live_sessions(df, now)
            return

        self._last_analysis_date = current_date
        ctx = SessionContext()

        # HTF bias
        ctx.htf_bias = self.analyzer.get_htf_bias(df)

        # Азия
        asia_df = self.analyzer.extract_session_data(
            df, SessionType.ASIA, now
        )
        if asia_df is not None and len(asia_df) >= 5:
            ctx.asia = self.analyzer.calculate_session_range(asia_df)
            ctx.asia_profile = self.analyzer.classify_asia(ctx.asia)

        # Франкфурт
        fk_df = self.analyzer.extract_session_data(
            df, SessionType.FRANKFURT, now
        )
        if fk_df is not None and len(fk_df) >= 2 and ctx.asia:
            ctx.frankfurt = self.analyzer.calculate_session_range(fk_df)
            ctx.frankfurt_action = self.analyzer.classify_frankfurt(
                ctx.frankfurt, ctx.asia
            )
            ctx.asia_high_swept = ctx.frankfurt.high > ctx.asia.high
            ctx.asia_low_swept = ctx.frankfurt.low < ctx.asia.low

        # Лондон
        ldn_df = self.analyzer.extract_session_data(
            df, SessionType.LONDON, now
        )
        if ldn_df is not None and len(ldn_df) >= 3 and ctx.asia:
            ctx.london = self.analyzer.calculate_session_range(ldn_df)
            ctx.london_profile = self.analyzer.classify_london(
                ldn_df, ctx.london, ctx.asia, ctx.frankfurt_action
            )
            # Обновляем sweep статусы
            if ctx.london.high > ctx.asia.high:
                ctx.asia_high_swept = True
            if ctx.london.low < ctx.asia.low:
                ctx.asia_low_swept = True

        self.context = ctx

    def _update_live_sessions(
        self,
        df: pd.DataFrame,
        now: pd.Timestamp
    ):
        """Обновить текущие live-сессии без пересчёта всего"""
        ctx = self.context

        utc_hour = (now.hour - self.analyzer.utc_offset) % 24
        current = self.analyzer.get_current_session(utc_hour)

        if current == SessionType.LONDON and ctx.asia:
            ldn_df = self.analyzer.extract_session_data(
                df, SessionType.LONDON, now
            )
            if ldn_df is not None and len(ldn_df) >= 2:
                ctx.london = self.analyzer.calculate_session_range(ldn_df)
                ctx.london_profile = self.analyzer.classify_london(
                    ldn_df, ctx.london, ctx.asia, ctx.frankfurt_action
                )

        elif current == SessionType.NEW_YORK and ctx.london:
            pass  # Контекст уже собран
