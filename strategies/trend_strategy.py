"""
Trend Following Strategy
Торговля по тренду с EMA + MACD + ADX подтверждением
"""

import pandas as pd
from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from config import risk_config


class TrendStrategy(BaseStrategy):
    """
    Логика:
    BUY:  EMA50 > EMA200 AND MACD > Signal AND ADX > 25
    SELL: EMA50 < EMA200 AND MACD < Signal AND ADX > 25
    
    SL: 1.5 × ATR
    TP: 3.0 × ATR (RR 1:2)
    """

    def __init__(self):
        super().__init__("trend_strategy")

    def get_required_indicators(self) -> list:
        return ["ema_50", "ema_200", "macd", "macd_signal", "adx", "atr"]

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:

        if df is None or len(df) < 5:
            return self._no_signal(symbol)

        # Последние значения
        row = df.iloc[-1]
        prev = df.iloc[-2]

        ema_50 = row["ema_50"]
        ema_200 = row["ema_200"]
        macd = row["macd"]
        macd_signal = row["macd_signal"]
        adx = row["adx"]
        atr = row["atr"]
        close = row["close"]

        # Нет тренда — нет сигнала
        if adx < 25:
            return self._no_signal(symbol)

        # ─── BUY ──────────────────────────────
        if (
            ema_50 > ema_200 and              # тренд вверх
            macd > macd_signal and             # MACD подтверждает
            prev["macd"] <= prev["macd_signal"]  # свежий кроссовер
        ):
            sl = close - atr * risk_config.default_sl_atr_mult
            tp = close + atr * risk_config.default_tp_atr_mult
            confidence = min(adx / 50, 1.0)    # чем сильнее тренд

            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=round(confidence, 2),
                reason=f"EMA50>200, MACD cross up, ADX={adx:.1f}",
                metadata={"adx": adx, "atr": atr}
            )

        # ─── SELL ─────────────────────────────
        if (
            ema_50 < ema_200 and
            macd < macd_signal and
            prev["macd"] >= prev["macd_signal"]
        ):
            sl = close + atr * risk_config.default_sl_atr_mult
            tp = close - atr * risk_config.default_tp_atr_mult
            confidence = min(adx / 50, 1.0)

            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=round(confidence, 2),
                reason=f"EMA50<200, MACD cross down, ADX={adx:.1f}",
                metadata={"adx": adx, "atr": atr}
            )

        return self._no_signal(symbol)
