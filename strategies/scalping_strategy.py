"""
Scalping Strategy
Быстрые сделки на малых таймфреймах
RSI + EMA + Momentum
"""

import pandas as pd
from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from config import risk_config


class ScalpingStrategy(BaseStrategy):
    """
    Логика скальпинга:
    BUY:  EMA20 > EMA50 AND RSI 40-60 (не экстремум) AND momentum > 0
    SELL: EMA20 < EMA50 AND RSI 40-60 AND momentum < 0
    
    Маленький SL/TP, быстрый вход-выход
    """

    def __init__(self):
        super().__init__("scalping_strategy")
        self.sl_atr_mult = 0.8       # тесный SL
        self.tp_atr_mult = 1.2       # быстрый TP

    def get_required_indicators(self) -> list:
        return ["ema_20", "ema_50", "rsi", "momentum", "atr", "adx"]

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradeSignal:

        if df is None or len(df) < 5:
            return self._no_signal(symbol)

        row = df.iloc[-1]
        prev = df.iloc[-2]

        close = row["close"]
        ema_20 = row["ema_20"]
        ema_50 = row["ema_50"]
        rsi = row["rsi"]
        momentum = row["momentum"]
        atr = row["atr"]

        # RSI не должен быть в экстремуме (иначе скальп опасен)
        rsi_ok = 35 < rsi < 65

        # Спред проверяем в execution, тут — логика

        # ─── SCALP BUY ──────────────────────
        if (
            ema_20 > ema_50 and
            prev["ema_20"] <= prev["ema_50"] and    # свежий кросс
            momentum > 0 and
            rsi_ok
        ):
            sl = close - atr * self.sl_atr_mult
            tp = close + atr * self.tp_atr_mult

            return TradeSignal(
                signal_type=SignalType.BUY,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=0.5,
                reason=f"Scalp BUY: EMA cross, mom={momentum:.4f}",
                metadata={"momentum": momentum, "rsi": rsi}
            )

        # ─── SCALP SELL ─────────────────────
        if (
            ema_20 < ema_50 and
            prev["ema_20"] >= prev["ema_50"] and
            momentum < 0 and
            rsi_ok
        ):
            sl = close + atr * self.sl_atr_mult
            tp = close - atr * self.tp_atr_mult

            return TradeSignal(
                signal_type=SignalType.SELL,
                strategy_name=self.name,
                symbol=symbol,
                entry_price=close,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                confidence=0.5,
                reason=f"Scalp SELL: EMA cross, mom={momentum:.4f}",
                metadata={"momentum": momentum, "rsi": rsi}
            )

        return self._no_signal(symbol)
