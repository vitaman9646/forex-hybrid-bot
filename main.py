"""
Forex Hybrid Bot — Main Entry Point
С интеграцией Session Strategy (Trade With BP)

Режимы:
  python main.py            — Live торговля
  python main.py demo       — Demo торговля
  python main.py backtest   — Обычный бэктест
  python main.py supertest  — Ультра-быстрый бэктест
  python main.py sessions   — Только Session Strategy
"""

import sys
import time
import traceback
from datetime import datetime, time as dt_time, date

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
from strategies.session_strategy import (
    SessionStrategy, SessionType
)
from strategies.base_strategy import SignalType

# AI
from ai.strategy_selector import AIStrategySelector
from ai.signal_filter import AISignalFilter
from ai.self_learner import SelfLearner

# Advanced modules
from meta_ai.capital_allocator import MetaAICapitalAllocator
from ai.volatility_predictor import VolatilityPredictor

# Notifications
from notifications.telegram_bot import TelegramNotifier

# Config
from config import (
    trading_config, risk_config,
    ai_config, regime_config
)


class ForexHybridBot:
    """
    Гибридный Forex бот с Session Strategy

    Pipeline:
    1. Market Data (M15 + H1)
    2. Session Analysis (Asia → Frankfurt → London → NY)
    3. Market Regime Detection
    4. Strategy Selection (AI + rules + session context)
    5. Signal Generation
    6. AI Signal Filter
    7. Risk Management
    8. Execution
    9. Logging + Telegram
    10. Self-Learning
    """

    def __init__(self, mode: str = "live"):
        print("=" * 60)
        print("  FOREX HYBRID BOT v2.0")
        print("  Session Models + Quant Strategies")
        print(f"  Mode: {mode.upper()}")
        print("=" * 60)

        self.mode = mode

        # ═══ Core Modules ═══
        self.data_processor = DataProcessor()
        self.market_detector = MarketDetector()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine()
        self.trade_logger = TradeLogger()

        # ═══ Strategies (включая Session) ═══
        self.strategies = {
            "trend": TrendStrategy(),
            "range": RangeStrategy(),
            "breakout": BreakoutStrategy(),
            "scalping": ScalpingStrategy(),
            "session": SessionStrategy(
                utc_offset=self._detect_mt5_utc_offset()
            ),
        }

        # ═══ Portfolio Manager ═══
        self.portfolio_manager = PortfolioManager(self.trade_logger)

        # ═══ Meta-AI Capital Allocator ═══
        self.capital_allocator = MetaAICapitalAllocator()
        self.capital_allocator.load_state()

        # ═══ Volatility Predictor ═══
        self.vol_predictor = VolatilityPredictor()

        # ═══ AI Modules ═══
        self.strategy_selector = AIStrategySelector()
        self.signal_filter = AISignalFilter()

        # ═══ Self-Learning ═══
        self.self_learner = SelfLearner(
            strategy_selector=self.strategy_selector,
            signal_filter=self.signal_filter,
            portfolio_manager=self.portfolio_manager,
            trade_logger=self.trade_logger
        )

        # ═══ Telegram ═══
        self.telegram = TelegramNotifier()

        # ═══ State ═══
        self.is_running = False
        self.current_regime = None
        self.iteration_count = 0
        self.today_session_signals = {}    # Какие сессионные модели сегодня
        self._daily_session_log = {}       # Лог сессий за сегодня

    @staticmethod
    def _detect_mt5_utc_offset() -> int:
        """
        Определить UTC offset сервера MT5.
        Большинство брокеров используют UTC+2 (зима) / UTC+3 (лето).
        """
        # TODO: автоопределение через mt5.symbol_info_tick().time
        return 2  # По умолчанию UTC+2

    # ═══════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════

    def start(self):
        """Запуск бота"""
        if not self.data_processor.connect():
            print("[BOT] ❌ Не удалось подключиться к MT5")
            return

        self.is_running = True
        self.telegram.send_message(
            "🤖 Forex Hybrid Bot v2.0 запущен!\n"
            "📊 Режим: Session Models + Quant\n"
            f"📋 Стратегий: {len(self.strategies)}"
        )

        print("\n[BOT] ✅ Бот запущен\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[BOT] Остановка по Ctrl+C")
        except Exception as e:
            error = (
                f"Критическая ошибка: {e}\n"
                f"{traceback.format_exc()}"
            )
            print(f"[BOT] ❌ {error}")
            self.telegram.notify_error(error)
        finally:
            self.stop()

    def stop(self):
        """Остановка бота"""
        self.is_running = False

        # Сохраняем состояние Meta-AI
        self.capital_allocator.save_state()

        self.data_processor.disconnect()

        stats = self.trade_logger.get_overall_stats()
        print(f"\n[BOT] Статистика: {stats}")
        self.telegram.send_message(
            f"🔴 Бот остановлен\n{stats}"
        )

    # ═══════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════

    def _main_loop(self):
        """Основной торговый цикл"""
        while self.is_running:
            self.iteration_count += 1

            try:
                if not self._is_trading_time():
                    self._handle_off_hours()
                    continue

                # ═══ PIPELINE ═══
                self._process_all_symbols()

                # Self-learning (периодически)
                if self.self_learner.should_learn():
                    results = self.self_learner.learn()
                    # Обновляем Meta-AI веса
                    self.capital_allocator.get_optimal_weights()

                # Трейлинг стоп
                self._update_trailing_stops()

                # Периодический отчёт (каждый час)
                if self.iteration_count % 12 == 0:
                    self._send_periodic_report()

            except Exception as e:
                print(f"[BOT] Ошибка в цикле: {e}")
                traceback.print_exc()

            self._wait_next_candle()

    # ═══════════════════════════════════════════
    #  SYMBOL PROCESSING
    # ═══════════════════════════════════════════

    def _process_all_symbols(self):
        """Обработка всех символов"""
        for symbol in trading_config.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                print(f"[BOT] Ошибка {symbol}: {e}")

    def _process_symbol(self, symbol: str):
        """Полный пайплайн для одного символа"""

        # ═══ STEP 1: Данные (M15 для sessions, H1 для остальных) ═══
        df_m15 = self.data_processor.get_processed_data(
            symbol=symbol, timeframe="M15", count=500
        )
        df_h1 = self.data_processor.get_processed_data(
            symbol=symbol, timeframe=trading_config.timeframe, count=500
        )

        if df_h1 is None or len(df_h1) < 50:
            return

        # ═══ STEP 2: Volatility Forecast ═══
        vol_forecast = self.vol_predictor.predict(df_h1, symbol)

        if vol_forecast.recommendation == "WAIT":
            return

        # ═══ STEP 3: Режим рынка ═══
        regime_info = self.market_detector.detect(df_h1)
        old_regime = self.current_regime

        if old_regime != regime_info.regime:
            print(
                f"[BOT] 🔄 {symbol}: "
                f"{regime_info.regime.value} "
                f"({regime_info.confidence:.0%})"
            )
            if old_regime:
                self.telegram.notify_regime_change(
                    old_regime.value, regime_info.regime.value
                )
            self.current_regime = regime_info.regime

        # ═══ STEP 4: Генерация сигналов от ВСЕХ стратегий ═══
        all_signals = []

        # 4a. Session Strategy (на M15 данных)
        if df_m15 is not None and len(df_m15) >= 50:
            session_signal = self.strategies["session"].generate_signal(
                df_m15, symbol
            )
            if session_signal.signal_type != SignalType.NONE:
                # Session strategy получает приоритетный буст
                session_signal.confidence = min(
                    session_signal.confidence + 0.1, 1.0
                )
                all_signals.append(session_signal)
                print(
                    f"[BOT] 🕐 SESSION: {session_signal.reason}"
                )

        # 4b. Quant Strategies (на H1 данных)
        selected_quant = self._select_quant_strategy(
            df_h1, regime_info
        )
        if selected_quant:
            quant_signal = selected_quant.generate_signal(df_h1, symbol)
            if quant_signal.signal_type != SignalType.NONE:
                all_signals.append(quant_signal)

        if not all_signals:
            return

        # ═══ STEP 5: Выбор лучшего сигнала ═══
        best_signal = self.portfolio_manager.select_best_signal(
            all_signals
        )

        if best_signal is None:
            return

        print(
            f"[BOT] 📡 {best_signal.signal_type.value} "
            f"{symbol} | {best_signal.strategy_name} | "
            f"conf={best_signal.confidence}"
        )

        # ═══ STEP 6: AI фильтр ═══
        if ai_config.enabled:
            ai_check = self.signal_filter.should_trade(
                best_signal, df_h1, regime_info.regime.value
            )
            if not ai_check["approved"]:
                print(f"[BOT] 🤖 AI отклонил: {ai_check['reason']}")
                return

        # ═══ STEP 7: Volatility adjustment ═══
        if vol_forecast.recommendation == "REDUCE_SIZE":
            # Уменьшаем размер позиции при экстремальной волатильности
            best_signal.confidence *= 0.7
            print("[BOT] ⚠️ Vol extreme: уменьшен размер")

        # ═══ STEP 8: Meta-AI Capital Allocation ═══
        weights = self.capital_allocator.get_optimal_weights()
        strategy_key = best_signal.strategy_name
        weight = weights.get(strategy_key, 0.2)

        # ═══ STEP 9: Risk Management ═══
        risk_check = self.risk_manager.check_trade(best_signal)

        if not risk_check.approved:
            print(f"[BOT] 🛑 Risk: {risk_check.reason}")
            return

        # Корректируем лот на вес стратегии
        risk_check.lot_size = round(
            risk_check.lot_size * weight * 4, 2
        )  # ×4 потому что вес ~0.25

        # ═══ STEP 10: Execution ═══
        trade_info = self.execution_engine.execute_trade(
            best_signal, risk_check
        )

        if trade_info is None:
            return

        # ═══ STEP 11: Logging ═══
        market_info = {
            "regime": regime_info.regime.value,
            "adx": regime_info.adx_value,
            "atr": regime_info.atr_value,
            "rsi": df_h1["rsi"].iloc[-1],
            "volatility": df_h1["volatility"].iloc[-1],
            "spread": self.data_processor.get_spread_points(symbol),
            "vol_regime": vol_forecast.regime,
            "session": vol_forecast.features.get("session", ""),
        }

        self.trade_logger.log_trade(
            trade_info=trade_info,
            market_info=market_info
        )

        # Обновляем Meta-AI performance
        # (P&L будет обновлён при закрытии, тут регистрируем открытие)

        # ═══ STEP 12: Telegram ═══
        self.telegram.notify_trade_open(trade_info)

    # ═══════════════════════════════════════════
    #  STRATEGY SELECTION
    # ═══════════════════════════════════════════

    def _select_quant_strategy(self, df, regime_info):
        """Выбор количественной стратегии (без session)"""

        # AI prediction
        if ai_config.enabled:
            pred = self.strategy_selector.predict_best_strategy(df)
            if (
                pred
                and pred["confidence"] > ai_config.confidence_threshold
            ):
                name_map = {
                    "trend_strategy": "trend",
                    "range_strategy": "range",
                    "breakout_strategy": "breakout",
                    "scalping_strategy": "scalping"
                }
                key = name_map.get(pred["strategy"], pred["strategy"])
                if key != "session":
                    strategy = self.strategies.get(key)
                    if strategy:
                        return strategy

        # Rules-based fallback
        strategy_key = self.market_detector.get_recommended_strategy(
            regime_info.regime
        )
        if strategy_key == "none":
            return None

        return self.strategies.get(strategy_key)

    # ═══════════════════════════════════════════
    #  TRAILING / TIME / REPORTING
    # ═══════════════════════════════════════════

    def _update_trailing_stops(self):
        """Обновление трейлинг стопов"""
        atr_values = {}
        for symbol in trading_config.symbols:
            df = self.data_processor.get_processed_data(
                symbol=symbol, count=50
            )
            if df is not None and "atr" in df.columns:
                atr_values[symbol] = df["atr"].iloc[-1]

        self.execution_engine.update_trailing_stop(atr_values)

    def _is_trading_time(self) -> bool:
        """Проверка торгового времени"""
        now = datetime.utcnow()

        # Не торгуем в выходные
        if now.weekday() >= 5:
            return False

        return True

    def _handle_off_hours(self):
        """Действия вне торгового времени"""
        print("[BOT] Выходные, ожидание...")
        time.sleep(300)

    def _wait_next_candle(self):
        """Ожидание следующей итерации"""
        # Для session strategy нужен M15
        wait = 300  # 5 минут
        print(
            f"[BOT] ⏳ Ждём {wait}с "
            f"(итерация #{self.iteration_count})"
        )
        time.sleep(wait)

    def _send_periodic_report(self):
        """Периодический отчёт"""
        stats = self.risk_manager.get_stats()
        self.telegram.notify_daily_stats(stats)

        # Session strategy контекст
        session_strat = self.strategies["session"]
        if session_strat.context:
            ctx = session_strat.context
            session_msg = (
                "📊 Session Context:\n"
                f"  Asia: {ctx.asia_profile.value if ctx.asia else 'N/A'}\n"
                f"  Frankfurt: {ctx.frankfurt_action.value}\n"
                f"  London: {ctx.london_profile.value if ctx.london else 'N/A'}\n"
                f"  HTF Bias: {'🟢 Bull' if ctx.htf_bias > 0 else '🔴 Bear' if ctx.htf_bias < 0 else '⚪ Neutral'}"
            )
            self.telegram.send_message(session_msg)

        # Meta-AI allocation
        report = self.capital_allocator.get_allocation_report()
        print(f"[BOT] Meta-AI weights: {report['current_weights']}")


# ═══════════════════════════════════════════════
#  BACKTEST FUNCTIONS
# ═══════════════════════════════════════════════

def run_backtest():
    """Стандартный бэктест"""
    from backtesting.backtester import Backtester

    print("\n" + "=" * 60)
    print("  BACKTEST MODE")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data(
        symbol="EURUSD", timeframe="H1", count=5000
    )
    processor.disconnect()

    if df is None:
        print("Нет данных")
        return

    bt = Backtester()
    strategies = [
        TrendStrategy(),
        RangeStrategy(),
        BreakoutStrategy(),
        ScalpingStrategy(),
    ]

    results = {}
    for s in strategies:
        results[s.name] = bt.run(s, df)

    print("\n" + "=" * 60)
    print("  СРАВНЕНИЕ")
    print("=" * 60)
    print(
        f"{'Strategy':<25} {'Trades':<8} {'WR%':<8} "
        f"{'Profit':<12} {'PF':<8}"
    )
    print("-" * 60)
    for name, r in results.items():
        print(
            f"{name:<25} {r.total_trades:<8} "
            f"{r.winrate:<8.1f} {r.total_profit:<12.2f} "
            f"{r.profit_factor:<8.2f}"
        )


def run_session_backtest():
    """Бэктест Session Strategy на M15"""
    from backtesting.backtester import Backtester

    print("\n" + "=" * 60)
    print("  SESSION STRATEGY BACKTEST")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    # Загружаем M15 данные (больше баров = больше дней)
    df = processor.get_processed_data(
        symbol="EURUSD", timeframe="M15", count=20000
    )
    processor.disconnect()

    if df is None:
        print("Нет M15 данных")
        return

    print(f"Загружено {len(df)} M15 баров")
    print(f"Период: {df.index[0]} — {df.index[-1]}")

    session_strat = SessionStrategy(utc_offset=2)
    bt = Backtester()
    result = bt.run(session_strat, df)

    print("\n[SESSION BT] Готово!")


def run_super_backtest():
    """Ультра-быстрый бэктест"""
    from backtest.super_backtester import SuperBacktester

    print("\n" + "=" * 60)
    print("  SUPER BACKTEST")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data(
        symbol="EURUSD", timeframe="H1", count=50000
    )
    processor.disconnect()

    if df is None:
        return

    bt = SuperBacktester()
    configs = [
        {"name": "Trend", "signal_func": bt.trend_signals},
        {"name": "Range", "signal_func": bt.range_signals},
        {"name": "Breakout", "signal_func": bt.breakout_signals},
        {"name": "Scalping", "signal_func": bt.scalping_signals,
         "params": {"sl_atr_mult": 0.8, "tp_atr_mult": 1.2}},
    ]

    results = bt.run_parallel(configs, df)
    bt.compare_strategies(results)

    # Walk-Forward лучшей
    best = max(results, key=lambda k: results[k].sharpe_ratio)
    best_func = next(
        c["signal_func"] for c in configs if c["name"] == best
    )
    wf = bt.walk_forward_analysis(best_func, df, strategy_name=best)

    # Monte Carlo
    mc = bt.monte_carlo_analysis(wf, n_simulations=5000)
    print(f"  Prob Profit: {mc['probability_of_profit']:.1%}")
    print(f"  Prob Ruin:   {mc['probability_of_ruin']:.1%}")

    bt.print_results(wf)
    bt.save_results()


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"

    if mode == "backtest":
        run_backtest()

    elif mode == "supertest":
        run_super_backtest()

    elif mode == "sessions":
        run_session_backtest()

    elif mode == "demo":
        bot = ForexHybridBot(mode="demo")
        bot.start()

    else:
        bot = ForexHybridBot(mode="live")
        bot.start()
