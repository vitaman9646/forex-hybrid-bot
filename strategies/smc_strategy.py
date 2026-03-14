"""
strategies/smc_strategy.py

Smart Money Concepts Strategy
Автоматизируемые элементы:
- Swing structure (HH/HL/LH/LL)
- Break of Structure (BOS)
- Order Blocks
- Fair Value Gaps (FVG)
- Liquidity sweeps
- Killzone timing
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum

from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from config import risk_config


# ═══════════════════════════════════════════════
#  SMC Data Types
# ═══════════════════════════════════════════════

class MarketStructure(Enum):
    BULLISH = "bullish"     # HH-HL
    BEARISH = "bearish"     # LL-LH
    RANGING = "ranging"


class SwingType(Enum):
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_LOW = "LL"
    LOWER_HIGH = "LH"


@dataclass
class SwingPoint:
    """Свинговая точка"""
    index: int
    price: float
    time: datetime
    swing_type: str        # "high" или "low"
    label: SwingType = None
    is_broken: bool = False


@dataclass
class OrderBlock:
    """Order Block"""
    high: float
    low: float
    time: datetime
    direction: str          # "bullish" или "bearish"
    is_valid: bool = True
    touches: int = 0
    source_index: int = 0


@dataclass
class FairValueGap:
    """Fair Value Gap (имбаланс)"""
    high: float             # верхняя граница
    low: float              # нижняя граница
    direction: str          # "bullish" или "bearish"
    time: datetime
    is_filled: bool = False
    fill_percentage: float = 0.0
    source_index: int = 0


@dataclass
class LiquidityPool:
    """Пул ликвидности"""
    price: float
    side: str               # "above" или "below"
    strength: int           # сколько равных уровней
    is_swept: bool = False
    time: datetime = None


@dataclass
class SMCContext:
    """Полный SMC контекст"""
    # Структура
    htf_structure: MarketStructure = MarketStructure.RANGING
    ltf_structure: MarketStructure = MarketStructure.RANGING
    swing_points: List[SwingPoint] = field(default_factory=list)
    last_bos_direction: int = 0     # 1=bullish, -1=bearish

    # POI
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FairValueGap] = field(default_factory=list)

    # Ликвидность
    liquidity_above: List[LiquidityPool] = field(default_factory=list)
    liquidity_below: List[LiquidityPool] = field(default_factory=list)

    # Premium / Discount
    equilibrium: float = 0.0
    is_discount: bool = False
    is_premium: bool = False


# ═══════════════════════════════════════════════
#  Swing Structure Analyzer
# ═══════════════════════════════════════════════

class SwingAnalyzer:
    """
    Анализ свинговой структуры рынка
    Использует 5-свечные фракталы (как в методике)
    """

    def __init__(self, fractal_bars: int = 5):
        self.fractal_bars = fractal_bars

    def find_swing_points(
        self,
        df: pd.DataFrame
    ) -> List[SwingPoint]:
        """
        Найти все свинговые точки (фракталы)

        Свинг-хай: high[i] > high всех соседних N баров
        Свинг-лоу: low[i] < low всех соседних N баров
        """
        swings = []
        n = self.fractal_bars

        for i in range(n, len(df) - n):
            # Свинг хай
            is_swing_high = True
            for j in range(1, n + 1):
                if (
                    df["high"].iloc[i] <= df["high"].iloc[i - j]
                    or df["high"].iloc[i] <= df["high"].iloc[i + j]
                ):
                    is_swing_high = False
                    break

            if is_swing_high:
                swings.append(SwingPoint(
                    index=i,
                    price=df["high"].iloc[i],
                    time=df.index[i],
                    swing_type="high"
                ))

            # Свинг лоу
            is_swing_low = True
            for j in range(1, n + 1):
                if (
                    df["low"].iloc[i] >= df["low"].iloc[i - j]
                    or df["low"].iloc[i] >= df["low"].iloc[i + j]
                ):
                    is_swing_low = False
                    break

            if is_swing_low:
                swings.append(SwingPoint(
                    index=i,
                    price=df["low"].iloc[i],
                    time=df.index[i],
                    swing_type="low"
                ))

        return swings

    def classify_structure(
        self,
        swings: List[SwingPoint]
    ) -> Tuple[MarketStructure, List[SwingPoint]]:
        """
        Классифицировать структуру: HH-HL или LL-LH

        Логика:
        1. Находим последовательность свинг-хаев и свинг-лоу
        2. Если каждый хай выше предыдущего И каждый лоу выше
           предыдущего → bullish
        3. Наоборот → bearish
        """
        if len(swings) < 4:
            return MarketStructure.RANGING, swings

        # Разделяем хаи и лоу
        highs = [s for s in swings if s.swing_type == "high"]
        lows = [s for s in swings if s.swing_type == "low"]

        if len(highs) < 2 or len(lows) < 2:
            return MarketStructure.RANGING, swings

        # Последние 3 свинг-хая и лоу
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        # Проверяем HH-HL (бычий)
        hh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i-1].price:
                hh_count += 1
                recent_highs[i].label = SwingType.HIGHER_HIGH

        hl_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i-1].price:
                hl_count += 1
                recent_lows[i].label = SwingType.HIGHER_LOW

        # Проверяем LL-LH (медвежий)
        ll_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price < recent_lows[i-1].price:
                ll_count += 1
                recent_lows[i].label = SwingType.LOWER_LOW

        lh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price < recent_highs[i-1].price:
                lh_count += 1
                recent_highs[i].label = SwingType.LOWER_HIGH

        if hh_count >= 1 and hl_count >= 1:
            return MarketStructure.BULLISH, swings

        if ll_count >= 1 and lh_count >= 1:
            return MarketStructure.BEARISH, swings

        return MarketStructure.RANGING, swings

    def detect_bos(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint]
    ) -> Dict:
        """
        Обнаружить Break of Structure (BOS)

        Bullish BOS: цена пробивает последний swing high
        Bearish BOS: цена пробивает последний swing low
        """
        if len(swings) < 2:
            return {"direction": 0, "level": 0, "broken": False}

        highs = [s for s in swings if s.swing_type == "high"]
        lows = [s for s in swings if s.swing_type == "low"]

        current_close = df["close"].iloc[-1]

        # Bullish BOS
        if highs:
            last_high = highs[-1]
            if (
                current_close > last_high.price
                and not last_high.is_broken
            ):
                last_high.is_broken = True
                return {
                    "direction": 1,
                    "level": last_high.price,
                    "broken": True,
                    "type": "bullish_bos"
                }

        # Bearish BOS
        if lows:
            last_low = lows[-1]
            if (
                current_close < last_low.price
                and not last_low.is_broken
            ):
                last_low.is_broken = True
                return {
                    "direction": -1,
                    "level": last_low.price,
                    "broken": True,
                    "type": "bearish_bos"
                }

        return {"direction": 0, "level": 0, "broken": False}


# ═══════════════════════════════════════════════
#  POI Detector (Order Blocks, FVG, Liquidity)
# ═══════════════════════════════════════════════

class POIDetector:
    """
    Обнаружение зон интереса Smart Money:
    - Order Blocks
    - Fair Value Gaps
    - Liquidity Pools
    """

    @staticmethod
    def find_order_blocks(
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[OrderBlock]:
        """
        Найти Order Blocks

        Bullish OB: последняя медвежья свеча перед
        импульсным движением вверх

        Bearish OB: последняя бычья свеча перед
        импульсным движением вниз

        Импульс = движение более 1.5 ATR за 1-3 свечи
        """
        obs = []

        if len(df) < lookback or "atr" not in df.columns:
            return obs

        recent = df.iloc[-lookback:]
        atr_avg = recent["atr"].mean()

        for i in range(2, len(recent) - 2):
            curr = recent.iloc[i]
            prev = recent.iloc[i - 1]
            next1 = recent.iloc[i + 1]
            next2 = recent.iloc[i + 2] if i + 2 < len(recent) else next1

            body_size = abs(curr["close"] - curr["open"])

            # Импульс вверх после медвежьей свечи
            impulse_up = (
                next1["close"] - curr["close"]
            )
            if (
                curr["close"] < curr["open"]           # медвежья свеча
                and impulse_up > atr_avg * 1.5         # сильный импульс
                and body_size > atr_avg * 0.3          # нормальное тело
            ):
                obs.append(OrderBlock(
                    high=curr["high"],
                    low=curr["low"],
                    time=recent.index[i],
                    direction="bullish",
                    source_index=i
                ))

            # Импульс вниз после бычьей свечи
            impulse_down = (
                curr["close"] - next1["close"]
            )
            if (
                curr["close"] > curr["open"]           # бычья свеча
                and impulse_down > atr_avg * 1.5       # сильный импульс
                and body_size > atr_avg * 0.3
            ):
                obs.append(OrderBlock(
                    high=curr["high"],
                    low=curr["low"],
                    time=recent.index[i],
                    direction="bearish",
                    source_index=i
                ))

        # Проверяем валидность (не пробит ли OB)
        current_price = df["close"].iloc[-1]
        for ob in obs:
            if ob.direction == "bullish" and current_price < ob.low:
                ob.is_valid = False
            if ob.direction == "bearish" and current_price > ob.high:
                ob.is_valid = False

        return [ob for ob in obs if ob.is_valid]

    @staticmethod
    def find_fvg(
        df: pd.DataFrame,
        lookback: int = 50,
        min_gap_atr: float = 0.3
    ) -> List[FairValueGap]:
        """
        Найти Fair Value Gaps (FVG)

        Bullish FVG: low[i+1] > high[i-1]
        (gap между 1-й и 3-й свечой)

        Bearish FVG: high[i+1] < low[i-1]
        """
        fvgs = []

        if len(df) < lookback or "atr" not in df.columns:
            return fvgs

        recent = df.iloc[-lookback:]
        atr_avg = recent["atr"].mean()
        min_gap = atr_avg * min_gap_atr

        for i in range(1, len(recent) - 1):
            prev = recent.iloc[i - 1]
            curr = recent.iloc[i]
            nxt = recent.iloc[i + 1]

            # Bullish FVG
            gap = nxt["low"] - prev["high"]
            if gap > min_gap:
                fvgs.append(FairValueGap(
                    high=nxt["low"],
                    low=prev["high"],
                    direction="bullish",
                    time=recent.index[i],
                    source_index=i
                ))

            # Bearish FVG
            gap = prev["low"] - nxt["high"]
            if gap > min_gap:
                fvgs.append(FairValueGap(
                    high=prev["low"],
                    low=nxt["high"],
                    direction="bearish",
                    time=recent.index[i],
                    source_index=i
                ))

        # Проверяем заполненность
        current_price = df["close"].iloc[-1]
        valid_fvgs = []
        for fvg in fvgs:
            if fvg.direction == "bullish":
                if current_price <= fvg.high:
                    # Частично или полностью не заполнен
                    if current_price > fvg.low:
                        fvg.fill_percentage = (
                            (current_price - fvg.low)
                            / (fvg.high - fvg.low)
                        )
                    valid_fvgs.append(fvg)

            elif fvg.direction == "bearish":
                if current_price >= fvg.low:
                    if current_price < fvg.high:
                        fvg.fill_percentage = (
                            (fvg.high - current_price)
                            / (fvg.high - fvg.low)
                        )
                    valid_fvgs.append(fvg)

        return valid_fvgs

    @staticmethod
    def find_liquidity_pools(
        df: pd.DataFrame,
        swings: List[SwingPoint],
        tolerance_pct: float = 0.001
    ) -> Tuple[List[LiquidityPool], List[LiquidityPool]]:
        """
        Найти пулы ликвидности

        Equal highs = sell-side liquidity (стопы лонгов)
        Equal lows = buy-side liquidity (стопы шортов)

        PDH/PDL = ликвидность предыдущего дня
        """
        above = []
        below = []

        swing_highs = [
            s for s in swings if s.swing_type == "high"
        ]
        swing_lows = [
            s for s in swings if s.swing_type == "low"
        ]

        current = df["close"].iloc[-1]

        # Equal highs (ликвидность сверху)
        for i in range(len(swing_highs)):
            equal_count = 1
            for j in range(i + 1, len(swing_highs)):
                avg = (swing_highs[i].price + swing_highs[j].price) / 2
                if avg > 0:
                    diff = (
                        abs(swing_highs[i].price - swing_highs[j].price)
                        / avg
                    )
                    if diff < tolerance_pct:
                        equal_count += 1

            if equal_count >= 2 and swing_highs[i].price > current:
                above.append(LiquidityPool(
                    price=swing_highs[i].price,
                    side="above",
                    strength=equal_count,
                    time=swing_highs[i].time
                ))

        # Equal lows (ликвидность снизу)
        for i in range(len(swing_lows)):
            equal_count = 1
            for j in range(i + 1, len(swing_lows)):
                avg = (swing_lows[i].price + swing_lows[j].price) / 2
                if avg > 0:
                    diff = (
                        abs(swing_lows[i].price - swing_lows[j].price)
                        / avg
                    )
                    if diff < tolerance_pct:
                        equal_count += 1

            if equal_count >= 2 and swing_lows[i].price < current:
                below.append(LiquidityPool(
                    price=swing_lows[i].price,
                    side="below",
                    strength=equal_count,
                    time=swing_lows[i].time
                ))

        # Сортируем по близости к текущей цене
        above.sort(key=lambda x: x.price)
        below.sort(key=lambda x: x.price, reverse=True)

        return above, below


# ═══════════════════════════════════════════════
#  Killzone Manager
# ═══════════════════════════════════════════════

class KillzoneManager:
    """
    Определение торговых окон (Killzone)

    Все значения в UTC
    Если сервер MT5 в UTC+2, нужен offset
    """

    # Killzones (UTC часы)
    KILLZONES = {
        "AO":  (0, 5),     # Asian Open
        "LO":  (7, 10),    # London Open
        "NYO": (12, 15),   # New York Open
        "LC":  (15, 17),   # London Close
    }

    def __init__(self, utc_offset: int = 0):
        self.utc_offset = utc_offset

    def get_current_killzone(
        self,
        bar_time: datetime
    ) -> Optional[str]:
        """Определить текущую Killzone"""
        utc_hour = (bar_time.hour - self.utc_offset) % 24

        for kz_name, (start, end) in self.KILLZONES.items():
            if start <= utc_hour < end:
                return kz_name

        return None

    def is_in_killzone(self, bar_time: datetime) -> bool:
        """Находимся ли мы в Killzone"""
        return self.get_current_killzone(bar_time) is not None

    def get_kz_for_trade(
        self,
        direction: int,
        structure: MarketStructure
    ) -> List[str]:
        """
        Рекомендуемые KZ для торговли

        Bullish bias + LO = искать лонги от PDL
        Bearish bias + NYO = искать шорты от манипуляции
        """
        if structure == MarketStructure.BULLISH:
            if direction == 1:
                return ["LO", "NYO"]
            else:
                return ["LC"]  # Коррекционные шорты

        elif structure == MarketStructure.BEARISH:
            if direction == -1:
                return ["LO", "NYO"]
            else:
                return ["LC"]

        return ["LO", "NYO"]


# ═══════════════════════════════════════════════
#  Premium / Discount Calculator
# ═══════════════════════════════════════════════

class PremiumDiscount:
    """
    Определение Premium / Discount зон

    Range = (Swing High - Swing Low)
    Equilibrium = 50% range
    Premium = выше 50% (зона продаж)
    Discount = ниже 50% (зона покупок)
    """

    @staticmethod
    def calculate(
        swing_high: float,
        swing_low: float,
        current_price: float
    ) -> Dict:
        """Рассчитать Premium/Discount"""
        if swing_high <= swing_low:
            return {
                "zone": "neutral",
                "equilibrium": current_price,
                "fib_level": 0.5
            }

        rng = swing_high - swing_low
        eq = swing_low + rng * 0.5

        # Fib level (0 = low, 1 = high)
        fib = (current_price - swing_low) / rng

        if current_price > eq:
            zone = "premium"
        elif current_price < eq:
            zone = "discount"
        else:
            zone = "equilibrium"

        return {
            "zone": zone,
            "equilibrium": eq,
            "fib_level": round(fib, 3),
            "ote_zone": 0.62 <= fib <= 0.79,  # Optimal Trade Entry
        }


# ═══════════════════════════════════════════════
#  SMC Strategy — Main
# ═══════════════════════════════════════════════

class SMCStrategy(BaseStrategy):
    """
    Smart Money Concepts Strategy

    Полная логика входа:
    1. HTF bias (D1/H4): определить структуру (bullish/bearish)
    2. Killzone: торговать только в LO/NYO/LC
    3. POI: найти Order Block или FVG в Premium/Discount
    4. Confirmation: BOS на LTF + цена в зоне POI
    5. Entry: ретест POI после BOS
    6. SL: за POI
    7. TP: следующий пул ликвидности (RR >= 3:1)
    """

    def __init__(
        self,
        utc_offset: int = 0,
        min_rr: float = 3.0
    ):
        super().__init__("smc_strategy")

        self.swing_analyzer = SwingAnalyzer(fractal_bars=5)
        self.poi_detector = POIDetector()
        self.kz_manager = KillzoneManager(utc_offset=utc_offset)
        self.pd_calc = PremiumDiscount()
        self.min_rr = min_rr

        self._context: Optional[SMCContext] = None
        self._last_update = None

    def get_required_indicators(self) -> list:
        return ["atr", "ema_50", "ema_200"]

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:
        """Генерация сигнала по SMC"""

        if df is None or len(df) < 100:
            return self._no_signal(symbol)

        bar_time = df.index[-1]
        close = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1] if "atr" in df.columns else 0

        if atr == 0:
            return self._no_signal(symbol)

        # ═══ CHECK 1: Killzone ═══
        current_kz = self.kz_manager.get_current_killzone(bar_time)
        if current_kz is None:
            return self._no_signal(symbol)

        # ═══ CHECK 2: HTF Structure ═══
        self._update_context(df)
        ctx = self._context

        if ctx.htf_structure == MarketStructure.RANGING:
            return self._no_signal(symbol)

        # ═══ CHECK 3: Premium / Discount ═══
        if not ctx.swing_points:
            return self._no_signal(symbol)

        highs = [s for s in ctx.swing_points if s.swing_type == "high"]
        lows = [s for s in ctx.swing_points if s.swing_type == "low"]

        if not highs or not lows:
            return self._no_signal(symbol)

        recent_high = max(s.price for s in highs[-5:])
        recent_low = min(s.price for s in lows[-5:])

        pd_info = self.pd_calc.calculate(
            recent_high, recent_low, close
        )

        # ═══ CHECK 4: POI Match ═══
        # ═══ BULLISH SETUP ═══
        if ctx.htf_structure == MarketStructure.BULLISH:
            if pd_info["zone"] != "discount":
                return self._no_signal(symbol)

            if current_kz not in ("LO", "NYO"):
                return self._no_signal(symbol)

            signal = self._bullish_entry(
                df, ctx, close, atr, symbol, current_kz, pd_info
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        # ═══ BEARISH SETUP ═══
        elif ctx.htf_structure == MarketStructure.BEARISH:
            if pd_info["zone"] != "premium":
                return self._no_signal(symbol)

            if current_kz not in ("LO", "NYO"):
                return self._no_signal(symbol)

            signal = self._bearish_entry(
                df, ctx, close, atr, symbol, current_kz, pd_info
            )
            if signal.signal_type != SignalType.NONE:
                return signal

        return self._no_signal(symbol)

    def _bullish_entry(
        self,
        df, ctx, close, atr, symbol, kz, pd_info
    ) -> TradeSignal:
        """Бычий вход от POI в дисконте"""

        # Ищем bullish OB или FVG в зоне дисконта
        poi_found = None
        poi_type = ""

        # Приоритет 1: Order Block
        for ob in ctx.order_blocks:
            if (
                ob.direction == "bullish"
                and ob.low <= close <= ob.high
                and ob.is_valid
            ):
                poi_found = ob
                poi_type = "OB"
                break

        # Приоритет 2: FVG
        if poi_found is None:
            for fvg in ctx.fvgs:
                if (
                    fvg.direction == "bullish"
                    and fvg.low <= close <= fvg.high
                    and fvg.fill_percentage < 0.7
                ):
                    poi_found = fvg
                    poi_type = "FVG"
                    break

        if poi_found is None:
            return self._no_signal(symbol)

        # BOS confirmation
        if ctx.last_bos_direction != 1:
            return self._no_signal(symbol)

        # SL/TP
        sl = poi_found.low - atr * 0.3

        # TP = ближайшая ликвидность сверху
        tp = close + atr * 4.0  # default
        if ctx.liquidity_above:
            tp = ctx.liquidity_above[0].price

        # Проверяем RR
        risk = close - sl
        reward = tp - close
        if risk <= 0:
            return self._no_signal(symbol)

        rr = reward / risk
        if rr < self.min_rr:
            return self._no_signal(symbol)

        confidence = self._calc_confidence(
            kz, pd_info, poi_type, rr
        )

        return TradeSignal(
            signal_type=SignalType.BUY,
            strategy_name=self.name,
            symbol=symbol,
            entry_price=close,
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            confidence=confidence,
            reason=(
                f"SMC BUY: {poi_type} in discount | "
                f"KZ={kz} | RR={rr:.1f} | "
                f"Fib={pd_info['fib_level']:.2f} | "
                f"Structure=Bullish"
            ),
            metadata={
                "model": "SMC",
                "poi_type": poi_type,
                "killzone": kz,
                "rr": round(rr, 2),
                "fib_level": pd_info["fib_level"],
                "structure": "bullish",
                "ote": pd_info["ote_zone"],
            }
        )

    def _bearish_entry(
        self,
        df, ctx, close, atr, symbol, kz, pd_info
    ) -> TradeSignal:
        """Медвежий вход от POI в премиуме"""

        poi_found = None
        poi_type = ""

        for ob in ctx.order_blocks:
            if (
                ob.direction == "bearish"
                and ob.low <= close <= ob.high
                and ob.is_valid
            ):
                poi_found = ob
                poi_type = "OB"
                break

        if poi_found is None:
            for fvg in ctx.fvgs:
                if (
                    fvg.direction == "bearish"
                    and fvg.low <= close <= fvg.high
                    and fvg.fill_percentage < 0.7
                ):
                    poi_found = fvg
                    poi_type = "FVG"
                    break

        if poi_found is None:
            return self._no_signal(symbol)

        if ctx.last_bos_direction != -1:
            return self._no_signal(symbol)

        sl = poi_found.high + atr * 0.3

        tp = close - atr * 4.0
        if ctx.liquidity_below:
            tp = ctx.liquidity_below[0].price

        risk = sl - close
        reward = close - tp
        if risk <= 0:
            return self._no_signal(symbol)

        rr = reward / risk
        if rr < self.min_rr:
            return self._no_signal(symbol)

        confidence = self._calc_confidence(
            kz, pd_info, poi_type, rr
        )

        return TradeSignal(
            signal_type=SignalType.SELL,
            strategy_name=self.name,
            symbol=symbol,
            entry_price=close,
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            confidence=confidence,
            reason=(
                f"SMC SELL: {poi_type} in premium | "
                f"KZ={kz} | RR={rr:.1f} | "
                f"Fib={pd_info['fib_level']:.2f} | "
                f"Structure=Bearish"
            ),
            metadata={
                "model": "SMC",
                "poi_type": poi_type,
                "killzone": kz,
                "rr": round(rr, 2),
                "fib_level": pd_info["fib_level"],
                "structure": "bearish",
                "ote": pd_info["ote_zone"],
            }
        )

    def _calc_confidence(
        self,
        kz: str,
        pd_info: Dict,
        poi_type: str,
        rr: float
    ) -> float:
        """Расчёт уверенности в сигнале"""
        conf = 0.3

        # Killzone boost
        if kz == "LO":
            conf += 0.15
        elif kz == "NYO":
            conf += 0.1

        # OTE zone (0.618–0.79 fib)
        if pd_info.get("ote_zone"):
            conf += 0.15

        # POI type
        if poi_type == "OB":
            conf += 0.1
        elif poi_type == "FVG":
            conf += 0.05

        # RR
        if rr >= 5:
            conf += 0.15
        elif rr >= 3:
            conf += 0.1

        return min(conf, 0.95)

    def _update_context(self, df: pd.DataFrame):
        """Обновить SMC контекст"""
        ctx = SMCContext()

        # Свинги
        swings = self.swing_analyzer.find_swing_points(df)
        ctx.swing_points = swings

        # Структура
        ctx.htf_structure, _ = self.swing_analyzer.classify_structure(
            swings
        )

        # BOS
        bos = self.swing_analyzer.detect_bos(df, swings)
        ctx.last_bos_direction = bos["direction"]

        # Order Blocks
        ctx.order_blocks = self.poi_detector.find_order_blocks(df)

        # FVG
        ctx.fvgs = self.poi_detector.find_fvg(df)

        # Ликвидность
        liq_above, liq_below = self.poi_detector.find_liquidity_pools(
            df, swings
        )
        ctx.liquidity_above = liq_above
        ctx.liquidity_below = liq_below

        self._context = ctx
