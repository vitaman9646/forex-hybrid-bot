"""
Range / Mean Reversion Strategy
Торговля от уровней RSI + Bollinger Bands + Stochastic
"""

import pandas as pd
from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from config import risk_config


class RangeStrategy(BaseStrategy):
    """
    Логика:
    BUY:  RSI < 30 AND close < BB lower AND Stoch K < 20
    SELL: RSI > 70 AND close > BB upper AND Stoch K > 80
    
    SL/TP через ATR
    """

    def __init__(self):
        super().__init__("range_strategy")

    def get_required_indicators(self) -> list:
        return [
            "rsi", "bb_upper", "bb_lower", "bb_middle",
            "stoch_k", "stoch_d", "atr", "adx"
        ]

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:

        if df is None or len(df) < 5:
            return self._no_signal(symbol)

        row = df.iloc[-1]

        rsi = row["rsi"]
        close = row["close"]
        bb_upper = row["bb_upper"]
        bb_lower = row["bb_lower"]
        bb_middle = row["bb_middle"]
        stoch_k = row["stoch_k"]
        atr = row["atr"]
        adx = row["adx"]

        # Не торгуем если сильный тренд
        if adx > 30:
            return self._no_signal(symbol)

        # ─── BUY (перепроданность) ──────────
        if (
            rsi < 30 and
            close < bb_lower and
            stoch_k < 20
        ):
            sl = close - atr * risk_config.default_sl_atr_mult
            tp = bb_middle                     # цель = середина BB
            confidence = self._calc_confidence(rsi, stoch_k, "buy")

            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=round(confidence, 2),
                reason=f"Перепроданность RSI={rsi:.1f}, "
                       f"Stoch={stoch_k:.1f}",
                metadata={
                    "rsi": rsi,
                    "stoch_k": stoch_k,
                    "bb_lower": bb_lower
                }
            )

        # ─── SELL (перекупленность) ─────────
        if (
            rsi > 70 and
            close > bb_upper and
            stoch_k > 80
        ):
            sl = close + atr * risk_config.default_sl_atr_mult
            tp = bb_middle
            confidence = self._calc_confidence(rsi, stoch_k, "sell")

            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=round(confidence, 2),
                reason=f"Перекупленность RSI={rsi:.1f}, "
                       f"Stoch={stoch_k:.1f}",
                metadata={
                    "rsi": rsi,
                    "stoch_k": stoch_k,
                    "bb_upper": bb_upper
                }
            )

        return self._no_signal(symbol)

    @staticmethod
    def _calc_confidence(
        rsi: float,
        stoch_k: float,
        direction: str
    ) -> float:
        """Уверенность на основе экстремальности показателей"""
        if direction == "buy":
            rsi_score = max(0, (30 - rsi) / 30)
            stoch_score = max(0, (20 - stoch_k) / 20)
        else:
            rsi_score = max(0, (rsi - 70) / 30)
            stoch_score = max(0, (stoch_k - 80) / 20)

        return min((rsi_score + stoch_score) / 2 + 0.3, 1.0)
