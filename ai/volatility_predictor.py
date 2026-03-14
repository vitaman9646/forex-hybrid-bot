"""
ai/volatility_predictor.py

Предсказание волатильности и импульсов
Использует GARCH + ML ensemble для прогноза ATR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


@dataclass
class VolatilityForecast:
    """Прогноз волатильности"""
    current_atr: float
    predicted_atr: float             # прогноз ATR
    predicted_direction: str         # expanding / contracting / stable
    expansion_probability: float     # вероятность импульса
    regime: str                      # low / normal / high / extreme
    squeeze_detected: bool           # сжатие (перед пробоем)
    impulse_expected: bool           # ожидается ли импульс
    confidence: float
    features: Dict                   # все использованные метрики
    recommendation: str              # trade / wait / reduce_size


class GARCHSimple:
    """
    Упрощённая GARCH(1,1) модель
    sigma²(t) = omega + alpha * r²(t-1) + beta * sigma²(t-1)
    """

    def __init__(
        self,
        omega: float = 0.00001,
        alpha: float = 0.1,
        beta: float = 0.85
    ):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def fit(self, returns: np.ndarray):
        """Подгонка параметров методом максимального правдоподобия"""
        if len(returns) < 30:
            return

        var = np.var(returns)
        best_ll = -np.inf
        best_params = (self.omega, self.alpha, self.beta)

        # Grid search по параметрам
        for alpha in np.arange(0.01, 0.3, 0.02):
            for beta in np.arange(0.6, 0.95, 0.02):
                if alpha + beta >= 1.0:
                    continue
                omega = var * (1 - alpha - beta)
                if omega <= 0:
                    continue

                ll = self._log_likelihood(
                    returns, omega, alpha, beta
                )
                if ll > best_ll:
                    best_ll = ll
                    best_params = (omega, alpha, beta)

        self.omega, self.alpha, self.beta = best_params

    def _log_likelihood(
        self,
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float
    ) -> float:
        """Log-likelihood для GARCH"""
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = (
                omega
                + alpha * returns[t - 1] ** 2
                + beta * sigma2[t - 1]
            )
            if sigma2[t] <= 0:
                return -np.inf

        ll = -0.5 * np.sum(
            np.log(2 * np.pi * sigma2[1:])
            + returns[1:] ** 2 / sigma2[1:]
        )
        return ll

    def forecast(
        self,
        returns: np.ndarray,
        horizon: int = 5
    ) -> np.ndarray:
        """Прогноз волатильности на horizon шагов вперёд"""
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = (
                self.omega
                + self.alpha * returns[t - 1] ** 2
                + self.beta * sigma2[t - 1]
            )

        # Прогноз
        forecasts = np.zeros(horizon)
        last_sigma2 = sigma2[-1]
        last_r2 = returns[-1] ** 2

        for h in range(horizon):
            if h == 0:
                forecasts[h] = (
                    self.omega
                    + self.alpha * last_r2
                    + self.beta * last_sigma2
                )
            else:
                forecasts[h] = (
                    self.omega
                    + (self.alpha + self.beta) * forecasts[h - 1]
                )

        return np.sqrt(forecasts)  # возвращаем стандартные отклонения


class VolatilityPredictor:
    """
    Предсказание волатильности

    Комбинирует:
    1. GARCH модель (статистический подход)
    2. ATR-based фичи (технический анализ)
    3. Volume analysis (объёмный анализ)
    4. Session analysis (время суток)
    5. Squeeze detection (сжатие BB)
    """

    def __init__(self):
        self.garch = GARCHSimple()
        self.atr_history: List[float] = []
        self.vol_regimes = {
            "low": (0, 25),
            "normal": (25, 60),
            "high": (60, 85),
            "extreme": (85, 100)
        }

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "EURUSD"
    ) -> VolatilityForecast:
        """
        Полный прогноз волатильности

        Args:
            df: DataFrame с индикаторами
            symbol: торговый символ
        """
        if df is None or len(df) < 50:
            return self._default_forecast()

        # Текущие значения
        current_atr = df["atr"].iloc[-1]
        close = df["close"].iloc[-1]

        # ═══ 1. GARCH прогноз ═══
        returns = df["close"].pct_change().dropna().values
        garch_forecast = self._garch_predict(returns)

        # ═══ 2. ATR анализ ═══
        atr_features = self._analyze_atr(df)

        # ═══ 3. Bollinger Squeeze ═══
        squeeze = self._detect_squeeze(df)

        # ═══ 4. Volume анализ ═══
        volume_signal = self._analyze_volume(df)

        # ═══ 5. Session анализ ═══
        session_factor = self._get_session_factor()

        # ═══ 6. Комбинируем прогнозы ═══
        predicted_atr = self._combine_forecasts(
            current_atr=current_atr,
            garch_vol=garch_forecast,
            atr_trend=atr_features["trend"],
            squeeze_factor=squeeze["factor"],
            volume_factor=volume_signal["factor"],
            session_factor=session_factor
        )

        # ═══ 7. Определяем режим ═══
        atr_percentile = atr_features["percentile"]
        regime = self._classify_regime(atr_percentile)

        # ═══ 8. Вероятность импульса ═══
        expansion_prob = self._calc_expansion_probability(
            squeeze=squeeze,
            volume=volume_signal,
            atr_features=atr_features,
            garch_forecast=garch_forecast,
            current_atr=current_atr
        )

        # ═══ 9. Направление ═══
        if predicted_atr > current_atr * 1.1:
            direction = "expanding"
        elif predicted_atr < current_atr * 0.9:
            direction = "contracting"
        else:
            direction = "stable"

        # ═══ 10. Рекомендация ═══
        recommendation = self._get_recommendation(
            regime=regime,
            expansion_prob=expansion_prob,
            squeeze=squeeze["detected"],
            direction=direction
        )

        return VolatilityForecast(
            current_atr=round(current_atr, 6),
            predicted_atr=round(predicted_atr, 6),
            predicted_direction=direction,
            expansion_probability=round(expansion_prob, 3),
            regime=regime,
            squeeze_detected=squeeze["detected"],
            impulse_expected=expansion_prob > 0.65,
            confidence=round(
                self._calc_confidence(df, atr_features), 3
            ),
            features={
                "atr_percentile": round(atr_percentile, 1),
                "atr_trend": atr_features["trend"],
                "bb_width": round(squeeze["bb_width"], 6),
                "volume_ratio": round(
                    volume_signal["ratio"], 2
                ),
                "garch_forecast": round(garch_forecast, 6),
                "session": self._get_current_session(),
                "session_factor": round(session_factor, 2)
            },
            recommendation=recommendation
        )

    def _garch_predict(self, returns: np.ndarray) -> float:
        """GARCH прогноз"""
        if len(returns) < 30:
            return np.std(returns) if len(returns) > 0 else 0

        try:
            self.garch.fit(returns[-200:])
            forecast = self.garch.forecast(returns[-200:], horizon=5)
            return float(forecast.mean())
        except Exception:
            return float(np.std(returns[-20:]))

    def _analyze_atr(self, df: pd.DataFrame) -> Dict:
        """Анализ ATR"""
        atr = df["atr"]
        current = atr.iloc[-1]

        # Перцентиль
        percentile = (
            (atr < current).sum() / len(atr) * 100
        )

        # Тренд ATR (растёт или падает)
        atr_sma_5 = atr.rolling(5).mean().iloc[-1]
        atr_sma_20 = atr.rolling(20).mean().iloc[-1]

        if atr_sma_5 > atr_sma_20 * 1.05:
            trend = "rising"
        elif atr_sma_5 < atr_sma_20 * 0.95:
            trend = "falling"
        else:
            trend = "flat"

        # Скорость изменения ATR
        atr_roc = (
            (current - atr.iloc[-5]) / atr.iloc[-5]
            if atr.iloc[-5] != 0 else 0
        )

        return {
            "current": current,
            "percentile": percentile,
            "trend": trend,
            "sma_5": atr_sma_5,
            "sma_20": atr_sma_20,
            "roc": atr_roc,
            "min_20": atr.rolling(20).min().iloc[-1],
            "max_20": atr.rolling(20).max().iloc[-1]
        }

    def _detect_squeeze(self, df: pd.DataFrame) -> Dict:
        """Обнаружение сжатия Bollinger Bands"""
        bb_width = df["bb_width"]
        current_width = bb_width.iloc[-1]

        # Минимальная ширина за 50 баров
        min_width_50 = bb_width.rolling(50).min().iloc[-1]

        # Сжатие если текущая ширина близка к минимуму
        is_squeeze = current_width <= min_width_50 * 1.1

        # Как долго длится сжатие
        squeeze_bars = 0
        threshold = bb_width.rolling(20).mean().iloc[-1] * 0.7

        for i in range(1, min(50, len(bb_width))):
            if bb_width.iloc[-i] < threshold:
                squeeze_bars += 1
            else:
                break

        # Фактор: чем дольше сжатие, тем сильнее пробой
        if is_squeeze and squeeze_bars > 5:
            factor = min(1.5 + squeeze_bars * 0.05, 2.5)
        elif is_squeeze:
            factor = 1.3
        else:
            factor = 1.0

        return {
            "detected": is_squeeze,
            "bb_width": current_width,
            "min_width": min_width_50,
            "squeeze_bars": squeeze_bars,
            "factor": factor
        }

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Анализ объёма"""
        if "volume" not in df.columns:
            return {"ratio": 1.0, "trend": "flat", "factor": 1.0}

        vol = df["volume"]
        current = vol.iloc[-1]
        avg_20 = vol.rolling(20).mean().iloc[-1]

        ratio = current / avg_20 if avg_20 > 0 else 1.0

        # Объёмный тренд
        vol_sma_5 = vol.rolling(5).mean().iloc[-1]
        if vol_sma_5 > avg_20 * 1.2:
            trend = "rising"
            factor = 1.2
        elif vol_sma_5 < avg_20 * 0.8:
            trend = "falling"
            factor = 0.9
        else:
            trend = "flat"
            factor = 1.0

        return {
            "current": current,
            "avg_20": avg_20,
            "ratio": ratio,
            "trend": trend,
            "factor": factor
        }

    @staticmethod
    def _get_session_factor() -> float:
        """
        Фактор волатильности по торговой сессии

        London + New York overlap = самая высокая волатильность
        Asian session = самая низкая
        """
        hour = datetime.utcnow().hour

        # Азия: 00-07 UTC (низкая)
        if 0 <= hour < 7:
            return 0.7

        # Лондон: 07-12 UTC (высокая)
        if 7 <= hour < 12:
            return 1.3

        # London + NY overlap: 12-16 UTC (максимальная)
        if 12 <= hour < 16:
            return 1.5

        # NY afternoon: 16-21 UTC (средняя)
        if 16 <= hour < 21:
            return 1.0

        # Закрытие: 21-00 UTC (низкая)
        return 0.8

    @staticmethod
    def _get_current_session() -> str:
        """Текущая торговая сессия"""
        hour = datetime.utcnow().hour

        if 0 <= hour < 7:
            return "Asian"
        if 7 <= hour < 12:
            return "London"
        if 12 <= hour < 16:
            return "London-NY Overlap"
        if 16 <= hour < 21:
            return "New York"
        return "Close"

    def _combine_forecasts(
        self,
        current_atr: float,
        garch_vol: float,
        atr_trend: str,
        squeeze_factor: float,
        volume_factor: float,
        session_factor: float
    ) -> float:
        """Комбинация всех прогнозов в один"""
        # Базовый прогноз = GARCH
        # Масштабируем GARCH vol к уровню ATR
        if garch_vol > 0 and current_atr > 0:
            garch_scaled = current_atr * (
                1 + (garch_vol - np.mean([garch_vol, current_atr * 0.01]))
                / max(current_atr * 0.01, 0.0001)
                * 0.1
            )
        else:
            garch_scaled = current_atr

        # Тренд ATR
        if atr_trend == "rising":
            trend_factor = 1.1
        elif atr_trend == "falling":
            trend_factor = 0.9
        else:
            trend_factor = 1.0

        # Комбинация
        predicted = (
            garch_scaled * 0.3
            + current_atr * trend_factor * 0.3
            + current_atr * squeeze_factor * 0.15
            + current_atr * volume_factor * 0.1
            + current_atr * session_factor * 0.15
        )

        return max(predicted, current_atr * 0.5)

    def _calc_expansion_probability(
        self,
        squeeze: Dict,
        volume: Dict,
        atr_features: Dict,
        garch_forecast: float,
        current_atr: float
    ) -> float:
        """Вероятность расширения волатильности (импульса)"""
        prob = 0.3  # базовая

        # Сжатие BB → сильный сигнал
        if squeeze["detected"]:
            prob += 0.25
            if squeeze["squeeze_bars"] > 10:
                prob += 0.1

        # Рост объёма
        if volume["ratio"] > 1.5:
            prob += 0.1
        if volume["ratio"] > 2.0:
            prob += 0.1

        # ATR на минимуме
        if atr_features["percentile"] < 20:
            prob += 0.15

        # ATR начинает расти
        if atr_features["trend"] == "rising":
            prob += 0.1

        # GARCH предсказывает рост
        if garch_forecast > current_atr * 0.01 * 1.2:
            prob += 0.1

        return min(prob, 0.95)

    def _classify_regime(self, percentile: float) -> str:
        """Классификация режима волатильности"""
        for regime, (low, high) in self.vol_regimes.items():
            if low <= percentile < high:
                return regime
        return "normal"

    @staticmethod
    def _get_recommendation(
        regime: str,
        expansion_prob: float,
        squeeze: bool,
        direction: str
    ) -> str:
        """Рекомендация по торговле"""
        if squeeze and expansion_prob > 0.7:
            return "WAIT_FOR_BREAKOUT"

        if regime == "extreme":
            return "REDUCE_SIZE"

        if regime == "low" and direction == "contracting":
            return "WAIT"

        if expansion_prob > 0.6:
            return "PREPARE_BREAKOUT"

        return "TRADE_NORMAL"

    @staticmethod
    def _calc_confidence(
        df: pd.DataFrame,
        atr_features: Dict
    ) -> float:
        """Уверенность в прогнозе"""
        data_points = len(df)
        data_score = min(data_points / 200, 0.4)

        # Стабильность ATR
        atr_std = df["atr"].std() / df["atr"].mean()
        stability_score = max(0, 0.3 - atr_std)

        return min(data_score + stability_score + 0.3, 1.0)

    @staticmethod
    def _default_forecast() -> VolatilityForecast:
        """Дефолтный прогноз при недостатке данных"""
        return VolatilityForecast(
            current_atr=0,
            predicted_atr=0,
            predicted_direction="unknown",
            expansion_probability=0.5,
            regime="unknown",
            squeeze_detected=False,
            impulse_expected=False,
            confidence=0,
            features={},
            recommendation="WAIT"
          )
