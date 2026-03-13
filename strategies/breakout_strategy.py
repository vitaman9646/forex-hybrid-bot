"""
Breakout Strategy
Пробой уровней high/low с подтверждением объёмом
"""

import pandas as pd
from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from config import risk_config


class BreakoutStrategy(BaseStrategy):
    """
    Логика:
    BUY:  close > 20-bar high AND volume > avg AND BB expanding
    SELL: close < 20-bar low AND volume > avg AND BB expanding
    """

    def __init__(self, lookback: int = 20):
        super().__init__("breakout_strategy")
        self.lookback = lookback

    def get_required_indicators(self) -> list:
        return [
            "high_20", "low_20", "atr", "volume_ratio",
            "bb_width", "adx"
        ]

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:

        if df is None or len(df) < self.lookback + 5:
            return self._no_signal(symbol)

        row = df.iloc[-1]
        prev = df.iloc[-2]

        close = row["close"]
        atr = row["atr"]
        volume_ratio = row["volume_ratio"]
        bb_width = row["bb_width"]
        bb_width_prev = prev["bb_width"]

        # Определяем уровни пробоя (исключая текущую свечу)
        recent = df.iloc[-(self.lookback + 1):-1]
        high_level = recent["high"].max()
        low_level = recent["low"].min()

        # BB должен расширяться
        bb_expanding = bb_width > bb_width_prev

        # Объём выше среднего
        volume_confirm = volume_ratio > 1.2

        # ─── BUY BREAKOUT ────────────────────
        if (
            close > high_level and
            prev["close"] <= high_level and    # именно пробой
            volume_confirm and
            bb_expanding
        ):
            sl = close - atr * risk_config.default_sl_atr_mult
            tp = close + atr * risk_config.default_tp_atr_mult
            confidence = self._calc_breakout_confidence(
                volume_ratio, bb_width, atr
            )

            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=round(confidence, 2),
                reason=f"Пробой вверх {high_level:.5f}, "
                       f"vol×{volume_ratio:.1f}",
                metadata={
                    "breakout_level": high_level,
                    "volume_ratio": volume_ratio,
                    "bb_width": bb_width
                }
            )

        # ─── SELL BREAKOUT ───────────────────
        if (
            close < low_level and
            prev["close"] >= low_level and
            volume_confirm and
            bb_expanding
        ):
            sl = close + atr * risk_config.default_sl_atr_mult
            tp = close - atr * risk_config.default_tp_atr_mult
            confidence = self._calc_breakout_confidence(
                volume_ratio, bb_width, atr
            )

            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=round(confidence, 2),
                reason=f"Пробой вниз {low_level:.5f}, "
                       f"vol×{volume_ratio:.1f}",
                metadata={
                    "breakout_level": low_level,
                    "volume_ratio": volume_ratio,
                    "bb_width": bb_width
                }
            )

        return self._no_signal(symbol)

    @staticmethod
    def _calc_breakout_confidence(
        volume_ratio: float,
        bb_width: float,
        atr: float
    ) -> float:
        """Уверенность пробоя"""
        vol_score = min(volume_ratio / 2.0, 0.4)
        bb_score = min(bb_width * 10, 0.3)
        return min(vol_score + bb_score + 0.3, 1.0)
