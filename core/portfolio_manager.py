"""
Portfolio Manager
Управление портфелем стратегий — распределение капитала
"""

from typing import Dict, List, Optional
from strategies.base_strategy import TradeSignal, SignalType
from core.trade_logger import TradeLogger
from config import strategy_weights


class PortfolioManager:
    """
    Управляет несколькими стратегиями одновременно
    Распределяет капитал по весам
    Адаптирует веса на основе результатов
    """

    def __init__(self, logger: TradeLogger):
        self.weights = dict(strategy_weights.weights)
        self.logger = logger
        self.performance: Dict[str, Dict] = {}

    def get_weight(self, strategy_name: str) -> float:
        """Вес стратегии в портфеле"""
        # Маппинг полных имён
        name_map = {
            "trend_strategy": "trend",
            "range_strategy": "range",
            "breakout_strategy": "breakout",
            "scalping_strategy": "scalping"
        }

        key = name_map.get(strategy_name, strategy_name)
        return self.weights.get(key, 0.25)

    def adjust_risk_by_weight(
        self,
        signal: TradeSignal,
        base_risk: float
    ) -> float:
        """
        Корректирует риск на основе веса стратегии
        Стратегия с весом 0.30 получит 30% от base_risk
        """
        weight = self.get_weight(signal.strategy_name)
        return base_risk * weight

    def rank_signals(
        self,
        signals: List[TradeSignal]
    ) -> List[TradeSignal]:
        """
        Ранжирует несколько сигналов по приоритету
        Учитывает: confidence × weight × historical_performance
        """
        scored_signals = []

        for signal in signals:
            if signal.signal_type == SignalType.NONE:
                continue

            weight = self.get_weight(signal.strategy_name)
            perf_score = self._get_performance_score(
                signal.strategy_name
            )
            total_score = signal.confidence * weight * perf_score

            scored_signals.append((total_score, signal))

        # Сортируем по убыванию скора
        scored_signals.sort(key=lambda x: x[0], reverse=True)

        return [s[1] for s in scored_signals]

    def select_best_signal(
        self,
        signals: List[TradeSignal]
    ) -> Optional[TradeSignal]:
        """Выбрать лучший сигнал из нескольких"""
        ranked = self.rank_signals(signals)

        if not ranked:
            return None

        best = ranked[0]
        print(
            f"[PORTFOLIO] Лучший сигнал: "
            f"{best.strategy_name} | "
            f"{best.signal_type.value} | "
            f"confidence={best.confidence}"
        )
        return best

    def _get_performance_score(self, strategy_name: str) -> float:
        """
        Историческая эффективность стратегии
        Базируется на winrate и profit factor
        """
        stats = self.logger.get_strategy_stats(strategy_name)

        if stats["total"] < 5:
            return 1.0  # мало данных — нейтральный скор

        winrate = stats["winrate"] / 100
        avg_profit = stats["avg_profit"]

        # Скор от 0.5 до 1.5
        score = 0.5 + winrate
        if avg_profit > 0:
            score += 0.2

        return min(max(score, 0.3), 1.5)

    def update_weights_from_performance(self):
        """
        Адаптация весов на основе реальных результатов
        Стратегии которые работают лучше — получают больший вес
        """
        strategies = ["trend", "range", "breakout", "scalping"]
        scores = {}

        strategy_names = {
            "trend": "trend_strategy",
            "range": "range_strategy",
            "breakout": "breakout_strategy",
            "scalping": "scalping_strategy"
        }

        for key in strategies:
            full_name = strategy_names[key]
            score = self._get_performance_score(full_name)
            scores[key] = score

        # Нормализуем
        total = sum(scores.values())
        if total > 0:
            for key in strategies:
                self.weights[key] = round(scores[key] / total, 3)

        print(f"[PORTFOLIO] Обновлены веса: {self.weights}")

    def get_portfolio_stats(self) -> Dict:
        """Полная статистика портфеля"""
        strategy_names = [
            "trend_strategy",
            "range_strategy",
            "breakout_strategy",
            "scalping_strategy"
        ]

        stats = {}
        for name in strategy_names:
            stats[name] = self.logger.get_strategy_stats(name)
            stats[name]["weight"] = self.get_weight(name)

        overall = self.logger.get_overall_stats()
        stats["overall"] = overall
        stats["weights"] = dict(self.weights)

        return stats
