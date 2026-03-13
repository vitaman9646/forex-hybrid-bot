"""
Market Regime Detector
Определяет текущий тип рынка: TREND, RANGE, VOLATILE, NEWS
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from config import regime_config


class MarketRegime(Enum):
    """Режимы рынка"""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOLATILE = "volatile"
    SQUEEZE = "squeeze"      # перед пробоем
    UNKNOWN = "unknown"


@dataclass
class RegimeInfo:
    """Информация о текущем режиме"""
    regime: MarketRegime
    confidence: float           # 0.0 — 1.0
    adx_value: float
    atr_value: float
    atr_percentile: float
    bb_width: float
    trend_direction: int        # 1 = up, -1 = down, 0 = flat
    description: str


class MarketDetector:
    """Определяет режим рынка по нескольким метрикам"""

    def __init__(self):
        self.history: list = []  # история режимов

    def detect(self, df: pd.DataFrame) -> RegimeInfo:
        """Основной метод определения режима"""
        if df is None or len(df) < 50:
            return RegimeInfo(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                adx_value=0,
                atr_value=0,
                atr_percentile=0,
                bb_width=0,
                trend_direction=0,
                description="Недостаточно данных"
            )

        # Считываем текущие значения индикаторов
        adx = df["adx"].iloc[-1]
        atr = df["atr"].iloc[-1]
        bb_width = df["bb_width"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        ema_200 = df["ema_200"].iloc[-1]
        rsi = df["rsi"].iloc[-1]
        close = df["close"].iloc[-1]

        # ATR percentile (насколько текущий ATR высок исторически)
        atr_percentile = (
            (df["atr"] < atr).sum() / len(df["atr"])
        ) * 100

        # Направление тренда
        if ema_50 > ema_200:
            trend_direction = 1
        elif ema_50 < ema_200:
            trend_direction = -1
        else:
            trend_direction = 0

        # Определяем режим
        regime, confidence, description = self._classify(
            adx=adx,
            atr_percentile=atr_percentile,
            bb_width=bb_width,
            trend_direction=trend_direction,
            rsi=rsi
        )

        info = RegimeInfo(
            regime=regime,
            confidence=confidence,
            adx_value=adx,
            atr_value=atr,
            atr_percentile=atr_percentile,
            bb_width=bb_width,
            trend_direction=trend_direction,
            description=description
        )

        self.history.append(info)
        return info

    def _classify(
        self,
        adx: float,
        atr_percentile: float,
        bb_width: float,
        trend_direction: int,
        rsi: float
    ) -> tuple:
        """Классификация режима по правилам"""

        scores = {
            MarketRegime.TREND_UP: 0.0,
            MarketRegime.TREND_DOWN: 0.0,
            MarketRegime.RANGE: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.SQUEEZE: 0.0,
        }

        # ─── Тренд ──────────────────────────────
        if adx > regime_config.adx_trend_threshold:
            if trend_direction == 1:
                scores[MarketRegime.TREND_UP] += 0.4
            elif trend_direction == -1:
                scores[MarketRegime.TREND_DOWN] += 0.4

        # Подтверждение RSI
        if trend_direction == 1 and rsi > 50:
            scores[MarketRegime.TREND_UP] += 0.15
        elif trend_direction == -1 and rsi < 50:
            scores[MarketRegime.TREND_DOWN] += 0.15

        # Подтверждение ADX силы
        if adx > 35:
            if trend_direction == 1:
                scores[MarketRegime.TREND_UP] += 0.15
            else:
                scores[MarketRegime.TREND_DOWN] += 0.15

        # ─── Флэт ──────────────────────────────
        if adx < regime_config.adx_range_threshold:
            scores[MarketRegime.RANGE] += 0.35

        if 40 < rsi < 60:
            scores[MarketRegime.RANGE] += 0.15

        # ─── Волатильность ──────────────────────
        if atr_percentile > regime_config.atr_volatility_percentile:
            scores[MarketRegime.VOLATILE] += 0.35

        if bb_width > 0.05:
            scores[MarketRegime.VOLATILE] += 0.15

        # ─── Сжатие (перед пробоем) ──────────────
        if bb_width < regime_config.bb_squeeze_threshold:
            scores[MarketRegime.SQUEEZE] += 0.4

        if adx < 15:
            scores[MarketRegime.SQUEEZE] += 0.15

        # Определяем победителя
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # Нормализуем уверенность
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0

        # Описание
        descriptions = {
            MarketRegime.TREND_UP: f"Восходящий тренд (ADX={adx:.1f})",
            MarketRegime.TREND_DOWN: f"Нисходящий тренд (ADX={adx:.1f})",
            MarketRegime.RANGE: f"Боковой рынок (ADX={adx:.1f})",
            MarketRegime.VOLATILE: f"Высокая волатильность (ATR perc={atr_percentile:.0f}%)",
            MarketRegime.SQUEEZE: f"Сжатие — жди пробой (BB width={bb_width:.4f})",
        }

        return best_regime, confidence, descriptions.get(best_regime, "")

    def get_recommended_strategy(self, regime: MarketRegime) -> str:
        """Какую стратегию использовать для данного режима"""
        mapping = {
            MarketRegime.TREND_UP: "trend",
            MarketRegime.TREND_DOWN: "trend",
            MarketRegime.RANGE: "range",
            MarketRegime.VOLATILE: "breakout",
            MarketRegime.SQUEEZE: "breakout",
            MarketRegime.UNKNOWN: "none",
        }
        return mapping.get(regime, "none")
