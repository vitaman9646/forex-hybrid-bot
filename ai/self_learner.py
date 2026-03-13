"""
Self-Learning Module
Автоматическое переобучение моделей
"""

from datetime import datetime
from typing import Dict
from ai.strategy_selector import AIStrategySelector
from ai.signal_filter import AISignalFilter
from core.portfolio_manager import PortfolioManager
from core.trade_logger import TradeLogger
from config import ai_config


class SelfLearner:
    """
    Самообучающийся модуль:
    1. Собирает данные из trade_log
    2. Переобучает AI модели
    3. Адаптирует веса портфеля
    """

    def __init__(
        self,
        strategy_selector: AIStrategySelector,
        signal_filter: AISignalFilter,
        portfolio_manager: PortfolioManager,
        trade_logger: TradeLogger
    ):
        self.strategy_selector = strategy_selector
        self.signal_filter = signal_filter
        self.portfolio_manager = portfolio_manager
        self.trade_logger = trade_logger
        self.last_learn_time = None

    def should_learn(self) -> bool:
        """Пора ли переобучаться"""
        if self.last_learn_time is None:
            return True

        hours = (
            datetime.now() - self.last_learn_time
        ).total_seconds() / 3600

        return hours >= ai_config.retrain_interval_hours

    def learn(self) -> Dict:
        """
        Полный цикл обучения
        """
        print("\n[SELF-LEARN] 🧠 Запуск цикла обучения...")
        results = {}

        # 1. Переобучение Strategy Selector
        print("[SELF-LEARN] Обучение Strategy Selector...")
        try:
            self.strategy_selector.train()
            results["strategy_selector"] = "OK"
        except Exception as e:
            results["strategy_selector"] = f"Error: {e}"

        # 2. Переобучение Signal Filter
        print("[SELF-LEARN] Обучение Signal Filter...")
        try:
            self.signal_filter.train()
            results["signal_filter"] = "OK"
        except Exception as e:
            results["signal_filter"] = f"Error: {e}"

        # 3. Адаптация весов портфеля
        print("[SELF-LEARN] Обновление весов портфеля...")
        try:
            self.portfolio_manager.update_weights_from_performance()
            results["portfolio_weights"] = dict(
                self.portfolio_manager.weights
            )
        except Exception as e:
            results["portfolio_weights"] = f"Error: {e}"

        # 4. Статистика
        overall = self.trade_logger.get_overall_stats()
        results["overall_stats"] = overall

        self.last_learn_time = datetime.now()

        print("[SELF-LEARN] ✅ Цикл обучения завершён")
        print(f"[SELF-LEARN] Результаты: {results}")

        return results
