"""
Forex Hybrid Bot — Main Entry Point
Основной цикл работы бота
"""

import time
import traceback
from datetime import datetime, time as dt_time

# Core
from core.data_processor import DataProcessor
from core.market_detector import MarketDetector, MarketRegime
from core.risk_manager import RiskManager
from core.execution_engine import ExecutionEngine
from core.trade_logger import TradeLogger
from core.portfolio_manager import PortfolioManager

# Strategies
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.base_strategy import SignalType

# AI
from ai.strategy_selector import AIStrategySelector
from ai.signal_filter import AISignalFilter
from ai.self_learner import SelfLearner

# Notifications
from notifications.telegram_bot import TelegramNotifier

# Config
from config import (
    trading_config, risk_config,
    ai_config, regime_config
)


class ForexHybridBot:
    """
    Гибридный Forex бот
    
    Pipeline:
    Market Data → Indicators → Regime Detection →
    Strategy Selection → Signal Generation →
    AI Filter → Risk Check → Execution → Logging
    """

    def __init__(self):
        print("=" * 60)
        print("  FOREX HYBRID BOT")
        print("  Запуск системы...")
        print("=" * 60)

        # Core модули
        self.data_processor = DataProcessor()
        self.market_detector = MarketDetector()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine()
        self.trade_logger = TradeLogger()

        # Стратегии
        self.strategies = {
            "trend": TrendStrategy(),
            "range": RangeStrategy(),
            "breakout": BreakoutStrategy(),
            "scalping": ScalpingStrategy(),
        }

        # Portfolio Manager
        self.portfolio_manager = PortfolioManager(self.trade_logger)

        # AI модули
        self.strategy_selector = AIStrategySelector()
        self.signal_filter = AISignalFilter()

        # Self-Learning
        self.self_learner = SelfLearner(
            strategy_selector=self.strategy_selector,
            signal_filter=self.signal_filter,
            portfolio_manager=self.portfolio_manager,
            trade_logger=self.trade_logger
        )

        # Telegram
        self.telegram = TelegramNotifier()

        # Состояние
        self.is_running = False
        self.current_regime = None
        self.iteration_count = 0

    def start(self):
        """Запуск бота"""
        # Подключение к MT5
        if not self.data_processor.connect():
            print("[BOT] ❌ Не удалось подключиться к MT5")
            return

        self.is_running = True
        self.telegram.send_message("🤖 Forex Hybrid Bot запущен!")

        print("\n[BOT] ✅ Бот запущен. Начинаю торговый цикл...\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[BOT] Остановка по Ctrl+C")
        except Exception as e:
            error_msg = f"Критическая ошибка: {e}\n{traceback.format_exc()}"
            print(f"[BOT] ❌ {error_msg}")
            self.telegram.notify_error(error_msg)
        finally:
            self.stop()

    def stop(self):
        """Остановка бота"""
        self.is_running = False
        self.data_processor.disconnect()

        # Финальная статистика
        stats = self.trade_logger.get_overall_stats()
        print(f"\n[BOT] Финальная статистика: {stats}")

        self.telegram.send_message(
            f"🔴 Бот остановлен\n"
            f"Статистика: {stats}"
        )

        print("[BOT] Бот остановлен")

    def _main_loop(self):
        """Основной торговый цикл"""
        while self.is_running:
            self.iteration_count += 1

            try:
                # Проверяем торговое время
                if not self._is_trading_time():
                    print("[BOT] Вне торгового времени, ожидание...")
                    time.sleep(60)
                    continue

                # ═══ PIPELINE ═══════════════════════════

                # 1. Получение данных
                self._process_symbols()

                # 2. Self-learning (периодически)
                if self.self_learner.should_learn():
                    self.self_learner.learn()

                # 3. Трейлинг стоп
                self._update_trailing_stops()

                # 4. Периодический отчёт
                if self.iteration_count % 60 == 0:
                    self._send_periodic_report()

            except Exception as e:
                print(f"[BOT] Ошибка в цикле: {e}")
                traceback.print_exc()

            # Пауза между итерациями
            self._wait_next_candle()

    def _process_symbols(self):
        """Обработка всех торгуемых символов"""
        for symbol in trading_config.symbols:
            try:
                self._process_single_symbol(symbol)
            except Exception as e:
                print(f"[BOT] Ошибка обработки {symbol}: {e}")

    def _process_single_symbol(self, symbol: str):
        """Полный пайплайн для одного символа"""

        # ═══ STEP 1: Данные ═══
        df = self.data_processor.get_processed_data(
            symbol=symbol,
            timeframe=trading_config.timeframe,
            count=500
        )

        if df is None or len(df) < 50:
            return

        # ═══ STEP 2: Режим рынка ═══
        regime_info = self.market_detector.detect(df)

        # Логируем смену режима
        old_regime = self.current_regime
        if old_regime != regime_info.regime:
            print(
                f"[BOT] 🔄 {symbol}: Режим рынка → "
                f"{regime_info.regime.value} "
                f"({regime_info.confidence:.1%})"
            )
            self.telegram.notify_regime_change(
                old_regime.value if old_regime else "none",
                regime_info.regime.value
            )
            self.current_regime = regime_info.regime

        # ═══ STEP 3: Выбор стратегии ═══
        selected_strategy = self._select_strategy(
            df, regime_info
        )

        if selected_strategy is None:
            return

        # ═══ STEP 4: Генерация сигнала ═══
        signal = selected_strategy.generate_signal(df, symbol)

        if signal.signal_type == SignalType.NONE:
            return

        print(
            f"[BOT] 📡 Сигнал: {signal.signal_type.value} "
            f"{symbol} | "
            f"стратегия={signal.strategy_name} | "
            f"confidence={signal.confidence}"
        )

        # ═══ STEP 5: AI фильтр ═══
        if ai_config.enabled:
            ai_check = self.signal_filter.should_trade(
                signal, df, regime_info.regime.value
            )
            if not ai_check["approved"]:
                print(f"[BOT] 🤖 AI отклонил: {ai_check['reason']}")
                return

        # ═══ STEP 6: Риск-менеджмент ═══
        risk_check = self.risk_manager.check_trade(signal)

        if not risk_check.approved:
            print(
                f"[BOT] 🛑 Риск-менеджер отклонил: "
                f"{risk_check.reason}"
            )
            return

        # ═══ STEP 7: Исполнение ═══
        trade_info = self.execution_engine.execute_trade(
            signal, risk_check
        )

        if trade_info is None:
            return

        # ═══ STEP 8: Логирование ═══
        market_info = {
            "regime": regime_info.regime.value,
            "adx": regime_info.adx_value,
            "atr": regime_info.atr_value,
            "rsi": df["rsi"].iloc[-1],
            "volatility": df["volatility"].iloc[-1],
            "spread": self.data_processor.get_spread_points(symbol)
        }

        self.trade_logger.log_trade(
            trade_info=trade_info,
            market_info=market_info
        )

        # ═══ STEP 9: Уведомление ═══
        self.telegram.notify_trade_open(trade_info)

    def _select_strategy(self, df, regime_info):
        """Выбор стратегии (правила или AI)"""

        # Попробуем AI
        if ai_config.enabled:
            ai_prediction = self.strategy_selector.predict_best_strategy(df)

            if (
                ai_prediction
                and ai_prediction["confidence"]
                > ai_config.confidence_threshold
            ):
                strategy_name = ai_prediction["strategy"]

                # Маппинг имён
                name_map = {
                    "trend_strategy": "trend",
                    "range_strategy": "range",
                    "breakout_strategy": "breakout",
                    "scalping_strategy": "scalping"
                }

                key = name_map.get(strategy_name, strategy_name)
                strategy = self.strategies.get(key)

                if strategy:
                    print(
                        f"[BOT] 🤖 AI выбрал: {strategy_name} "
                        f"({ai_prediction['confidence']:.1%})"
                    )
                    return strategy

        # Fallback: правила на основе режима
        strategy_key = self.market_detector.get_recommended_strategy(
            regime_info.regime
        )

        if strategy_key == "none":
            return None

        strategy = self.strategies.get(strategy_key)

        if strategy:
            print(
                f"[BOT] 📋 Режим {regime_info.regime.value} → "
                f"стратегия: {strategy_key}"
            )

        return strategy

    def _update_trailing_stops(self):
        """Обновление трейлинг стопов для открытых позиций"""
        atr_values = {}

        for symbol in trading_config.symbols:
            df = self.data_processor.get_processed_data(
                symbol=symbol, count=50
            )
            if df is not None and "atr" in df.columns:
                atr_values[symbol] = df["atr"].iloc[-1]

        self.execution_engine.update_trailing_stop(atr_values)

    def _is_trading_time(self) -> bool:
        """Проверка торгового времени (не торгуем выходные)"""
        now = datetime.now()

        # Не торгуем в субботу и воскресенье
        if now.weekday() >= 5:
            return False

        # Не торгуем в ночное время (опционально)
        current_time = now.time()
        trading_start = dt_time(1, 0)    # 01:00
        trading_end = dt_time(23, 0)      # 23:00

        return trading_start <= current_time <= trading_end

    def _wait_next_candle(self):
        """
        Ожидание следующей свечи
        Для H1 — ждём ~60 минут
        Для M15 — ~15 минут
        """
        timeframe_seconds = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        }

        wait = timeframe_seconds.get(
            trading_config.timeframe, 3600
        )

        # Уменьшаем для проверки трейлинга
        actual_wait = min(wait, 300)  # макс 5 минут между проверками

        print(
            f"[BOT] ⏳ Ожидание {actual_wait} сек "
            f"(итерация #{self.iteration_count})..."
        )
        time.sleep(actual_wait)

    def _send_periodic_report(self):
        """Периодический отчёт"""
        stats = self.risk_manager.get_stats()
        self.telegram.notify_daily_stats(stats)

        portfolio_stats = self.portfolio_manager.get_portfolio_stats()
        print(f"[BOT] 📊 Портфель: {portfolio_stats}")


def run_backtest():
    """Запуск бэктеста (отдельная функция)"""
    from backtesting.backtester import Backtester

    print("\n" + "=" * 60)
    print("  BACKTEST MODE")
    print("=" * 60)

    processor = DataProcessor()

    if not processor.connect():
        print("Не удалось подключиться к MT5")
        return

    # Загружаем данные
    df = processor.get_processed_data(
        symbol="EURUSD",
        timeframe="H1",
        count=5000
    )

    processor.disconnect()

    if df is None:
        print("Нет данных для бэктеста")
        return

    backtester = Backtester()

    # Тестируем каждую стратегию
    strategies = [
        TrendStrategy(),
        RangeStrategy(),
        BreakoutStrategy(),
        ScalpingStrategy(),
    ]

    results = {}
    for strategy in strategies:
        result = backtester.run(strategy, df)
        results[strategy.name] = result

    # Сравнение
    print("\n" + "=" * 60)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 60)
    print(
        f"{'Strategy':<25} {'Trades':<8} "
        f"{'WR%':<8} {'Profit':<12} "
        f"{'PF':<8} {'MaxDD%':<8}"
    )
    print("-" * 60)

    for name, r in results.items():
        print(
            f"{name:<25} {r.total_trades:<8} "
            f"{r.winrate:<8.1f} {r.total_profit:<12.2f} "
            f"{r.profit_factor:<8.2f} {r.max_drawdown:<8.2f}"
        )


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtest()
    else:
        bot = ForexHybridBot()
        bot.start()
