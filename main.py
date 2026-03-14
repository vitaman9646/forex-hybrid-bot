"""
main.py — Forex Hybrid Bot v3.0
Полная версия со всеми модулями

Режимы запуска:
  python main.py              — Live торговля
  python main.py demo         — Demo торговля
  python main.py backtest     — Бэктест
  python main.py supertest    — Квантовый бэктест
  python main.py ultratest    — Ультра-быстрый бэктест
  python main.py sessions     — Session strategy бэктест
  python main.py train_ai     — Обучить AI (anti-overfit)
  python main.py train_impulse — Обучить Impulse AI
  python main.py features     — Анализ Feature Factory
  python main.py collect      — Загрузить данные в Data Lake
  python main.py lake_info    — Информация о Data Lake
  python main.py validate     — Проверить данные
"""

import sys
import os
import time
import traceback
import numpy as np
from datetime import datetime, time as dt_time, date


# ═══════════════════════════════════════════════
#  IMPORTS — Core
# ═══════════════════════════════════════════════

from core.data_processor import DataProcessor
from core.market_detector import MarketDetector, MarketRegime
from core.risk_manager import RiskManager
from core.execution_engine import ExecutionEngine
from core.trade_logger import TradeLogger

# ═══════════════════════════════════════════════
#  IMPORTS — Strategies
# ═══════════════════════════════════════════════

from strategies.base_strategy import SignalType
from strategies.trend_strategy import TrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.breakout_strategy import BreakoutStrategy

# Optional strategies (могут быть не созданы ещё)
try:
    from strategies.scalping_strategy import ScalpingStrategy
    HAS_SCALPING = True
except ImportError:
    HAS_SCALPING = False

try:
    from strategies.session_strategy import SessionStrategy
    HAS_SESSION = True
except ImportError:
    HAS_SESSION = False

try:
    from strategies.smc_strategy import SMCStrategy
    HAS_SMC = True
except ImportError:
    HAS_SMC = False

# ═══════════════════════════════════════════════
#  IMPORTS — AI (optional)
# ═══════════════════════════════════════════════

try:
    from ai.impulse_predictor import ImpulsePredictor
    HAS_IMPULSE = True
except ImportError:
    HAS_IMPULSE = False

try:
    from ai.anti_overfit_trainer import AntiOverfitTrainer
    HAS_TRAINER = True
except ImportError:
    HAS_TRAINER = False

# ═══════════════════════════════════════════════
#  IMPORTS — Advanced (optional)
# ═══════════════════════════════════════════════

try:
    from meta_ai.hedge_fund_allocator import (
        HedgeFundAllocator, MarketRegimeType
    )
    HAS_ALLOCATOR = True
except ImportError:
    HAS_ALLOCATOR = False

try:
    from ai.volatility_predictor import VolatilityPredictor
    HAS_VOL_PRED = True
except ImportError:
    HAS_VOL_PRED = False

try:
    from features.feature_factory import FeatureFactory
    HAS_FACTORY = True
except ImportError:
    HAS_FACTORY = False

try:
    from infrastructure.service_bus import ServiceBus
    HAS_BUS = True
except ImportError:
    HAS_BUS = False

try:
    from infrastructure.monitoring import MetricsCollector
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

# ═══════════════════════════════════════════════
#  IMPORTS — Notifications
# ═══════════════════════════════════════════════

from notifications.telegram_bot import TelegramNotifier

# ═══════════════════════════════════════════════
#  IMPORTS — Config
# ═══════════════════════════════════════════════

from config import trading_config, risk_config


# ═══════════════════════════════════════════════
#  MAIN BOT CLASS
# ═══════════════════════════════════════════════

class ForexHybridBot:
    """
    Forex Hybrid Bot v3.0

    Full Pipeline:
    1. Market Data (M15 + H1)
    2. Volatility Forecast
    3. Impulse AI Prediction
    4. Market Regime Detection
    5. Strategy Selection (regime-based + AI)
    6. Signal Generation (all strategies)
    7. Signal Ranking (portfolio manager)
    8. AI Signal Filter
    9. Meta-AI Capital Allocation
    10. Risk Management
    11. Execution
    12. Logging + Telegram
    13. Monitoring metrics
    14. Self-Learning (periodic)
    """

    def __init__(self, mode: str = "live"):
        print("=" * 60)
        print("  FOREX HYBRID BOT v3.0")
        print(f"  Mode: {mode.upper()}")
        print("=" * 60)

        self.mode = mode

        # ═══ Core (обязательные) ═══
        self.data_processor = DataProcessor()
        self.market_detector = MarketDetector()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine()
        self.trade_logger = TradeLogger()
        self.telegram = TelegramNotifier()

        # ═══ Strategies ═══
        self.strategies = {
            "trend": TrendStrategy(),
            "range": RangeStrategy(),
            "breakout": BreakoutStrategy(),
        }

        if HAS_SCALPING:
            self.strategies["scalping"] = ScalpingStrategy()
            print("  + Scalping Strategy")

        if HAS_SESSION:
            self.strategies["session"] = SessionStrategy(
                utc_offset=self._detect_utc_offset()
            )
            print("  + Session Strategy")

        if HAS_SMC:
            self.strategies["smc"] = SMCStrategy(
                utc_offset=self._detect_utc_offset()
            )
            print("  + SMC Strategy")

        print(f"  Strategies loaded: {len(self.strategies)}")

        # ═══ AI Modules (optional) ═══
        self.impulse_predictor = None
        if HAS_IMPULSE:
            self.impulse_predictor = ImpulsePredictor()
            print("  + Impulse Predictor")

        self.vol_predictor = None
        if HAS_VOL_PRED:
            self.vol_predictor = VolatilityPredictor()
            print("  + Volatility Predictor")

        # ═══ Meta-AI Allocator (optional) ═══
        self.allocator = None
        if HAS_ALLOCATOR:
            self.allocator = HedgeFundAllocator()
            self.allocator.load()
            print("  + Hedge Fund Allocator")

        # ═══ Feature Factory (optional) ═══
        self.feature_factory = None
        if HAS_FACTORY:
            self.feature_factory = FeatureFactory()
            self.feature_factory.load()
            print("  + Feature Factory")

        # ═══ Monitoring (optional) ═══
        self.metrics = None
        if HAS_METRICS:
            try:
                self.metrics = MetricsCollector(
                    service_name="forex_bot",
                    port=8000
                )
                print("  + Prometheus Metrics :8000")
            except Exception:
                pass

        # ═══ Service Bus (optional) ═══
        self.bus = None
        if HAS_BUS:
            try:
                self.bus = ServiceBus(service_name="trading")
                print("  + Service Bus (Redis)")
            except Exception:
                pass

        # ═══ State ═══
        self.is_running = False
        self.current_regime = None
        self.iteration = 0

        print(f"\n  Ready to {'trade' if mode == 'live' else mode}!")
        print("=" * 60)

    @staticmethod
    def _detect_utc_offset() -> int:
        """UTC offset сервера MT5 (большинство: UTC+2)"""
        return 2

    # ═══════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════

    def start(self):
        """Запуск бота"""
        if not self.data_processor.connect():
            print("Failed to connect to MT5!")
            return

        self.is_running = True
        self.telegram.send(
            f"🤖 Forex Bot v3.0 started!\n"
            f"Mode: {self.mode}\n"
            f"Strategies: {len(self.strategies)}\n"
            f"Symbols: {', '.join(trading_config.symbols)}"
        )

        if self.metrics:
            self.metrics.set_mt5_status(True)

        print("\nBot is running...\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nStopping by Ctrl+C...")
        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            print(f"CRITICAL ERROR: {error_msg}")
            self.telegram.notify_error(error_msg)
            if self.metrics:
                self.metrics.record_error("critical")
        finally:
            self.stop()

    def stop(self):
        """Остановка"""
        self.is_running = False

        # Сохраняем состояние
        if self.allocator:
            self.allocator.save()
        if self.feature_factory:
            self.feature_factory.save()

        stats = self.trade_logger.get_stats()
        print(f"\nFinal stats: {stats}")

        self.telegram.send(f"🔴 Bot stopped\nStats: {stats}")
        self.data_processor.disconnect()

        if self.metrics:
            self.metrics.set_mt5_status(False)

    # ═══════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════

    def _main_loop(self):
        """Основной торговый цикл"""
        while self.is_running:
            self.iteration += 1
            iter_start = time.time()

            try:
                # Проверяем торговое время
                if not self._is_trading_time():
                    print(f"[{self._now()}] Outside trading hours")
                    time.sleep(300)
                    continue

                # Обрабатываем все символы
                for symbol in trading_config.symbols:
                    try:
                        self._process_symbol(symbol)
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
                        if self.metrics:
                            self.metrics.record_error("symbol_processing")

                # Обновляем trailing stops
                self._update_trailing_stops()

                # Периодический отчёт (каждый час)
                if self.iteration % 12 == 0:
                    self._send_report()

            except Exception as e:
                print(f"Loop error: {e}")
                traceback.print_exc()
                if self.metrics:
                    self.metrics.record_error("loop")

            # Записываем время итерации
            iter_time = time.time() - iter_start
            if self.metrics:
                self.metrics.record_latency(iter_time)

            self._wait()

    # ═══════════════════════════════════════════
    #  PROCESS SYMBOL — полный пайплайн
    # ═══════════════════════════════════════════

    def _process_symbol(self, symbol: str):
        """Полный пайплайн для одного символа"""

        # ═══ STEP 1: Данные ═══
        df = self.data_processor.get_processed_data(
            symbol=symbol,
            timeframe=trading_config.timeframe,
            count=500
        )

        if df is None or len(df) < 50:
            return

        close = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]

        # ═══ STEP 2: Volatility Forecast (optional) ═══
        vol_recommendation = "TRADE_NORMAL"
        if self.vol_predictor:
            try:
                vol_forecast = self.vol_predictor.predict(df, symbol)
                vol_recommendation = vol_forecast.recommendation

                if vol_recommendation == "WAIT":
                    return

                if self.metrics:
                    self.metrics.publish_metric(
                        "volatility_regime",
                        hash(vol_forecast.regime) % 100
                    )
            except Exception:
                pass

        # ═══ STEP 3: Impulse Prediction (optional) ═══
        impulse_prob = 0.5
        impulse_dir = 0
        if self.impulse_predictor and self.impulse_predictor.is_trained:
            try:
                impulse = self.impulse_predictor.predict(df)
                impulse_prob = impulse.probability
                impulse_dir = impulse.predicted_direction

                if impulse_prob > 0.7:
                    print(
                        f"  ⚡ {symbol}: Impulse "
                        f"prob={impulse_prob:.0%} "
                        f"dir={'UP' if impulse_dir > 0 else 'DOWN'}"
                    )

                if self.metrics:
                    self.metrics.update_impulse_prob(
                        symbol, impulse_prob
                    )
            except Exception:
                pass

        # ═══ STEP 4: Market Regime ═══
        regime_info = self.market_detector.detect(df)

        if self.current_regime != regime_info.regime:
            print(
                f"  🔄 {symbol}: {regime_info.regime.value} "
                f"({regime_info.description})"
            )
            self.current_regime = regime_info.regime

        # ═══ STEP 5: Generate signals from ALL strategies ═══
        all_signals = []

        for strat_name, strategy in self.strategies.items():
            try:
                # Session & SMC используют M15 если доступен
                if strat_name in ("session", "smc"):
                    m15 = self.data_processor.get_processed_data(
                        symbol=symbol, timeframe="M15", count=500
                    )
                    if m15 is not None and len(m15) >= 50:
                        signal = strategy.generate_signal(m15, symbol)
                    else:
                        signal = strategy.generate_signal(df, symbol)
                else:
                    signal = strategy.generate_signal(df, symbol)

                if signal.signal_type != SignalType.NONE:
                    # Impulse AI boost
                    if impulse_prob > 0.7 and impulse_dir != 0:
                        if (
                            (impulse_dir == 1
                             and signal.signal_type == SignalType.BUY)
                            or
                            (impulse_dir == -1
                             and signal.signal_type == SignalType.SELL)
                        ):
                            signal.confidence = min(
                                signal.confidence + 0.15, 0.95
                            )

                    all_signals.append(signal)

                    if self.metrics:
                        self.metrics.record_signal(
                            strat_name,
                            signal.signal_type.value
                        )

            except Exception as e:
                print(f"  Strategy {strat_name} error: {e}")

        if not all_signals:
            return

        # ═══ STEP 6: Select best signal ═══
        # Сортируем по confidence
        all_signals.sort(
            key=lambda s: s.confidence, reverse=True
        )
        best_signal = all_signals[0]

        print(
            f"  📡 {best_signal.signal_type.value} {symbol} | "
            f"{best_signal.strategy_name} | "
            f"conf={best_signal.confidence:.2f} | "
            f"{best_signal.reason}"
        )

        # ═══ STEP 7: Volatility adjustment ═══
        if vol_recommendation == "REDUCE_SIZE":
            best_signal.confidence *= 0.7

        # ═══ STEP 8: Meta-AI weight (optional) ═══
        capital_weight = 1.0
        if self.allocator:
            try:
                regime_map = {
                    "trend_up": MarketRegimeType.TRENDING,
                    "trend_down": MarketRegimeType.TRENDING,
                    "range": MarketRegimeType.RANGING,
                    "volatile": MarketRegimeType.VOLATILE,
                }
                self.allocator.set_regime(
                    regime_map.get(
                        regime_info.regime.value,
                        MarketRegimeType.CALM
                    )
                )
                weights = self.allocator.get_weights()
                capital_weight = weights.get(
                    best_signal.strategy_name, 0.2
                )

                if self.metrics:
                    self.metrics.update_strategy_weights(weights)

            except Exception:
                capital_weight = 1.0

        # ═══ STEP 9: Risk Management ═══
        risk_check = self.risk_manager.check_trade(best_signal)

        if not risk_check.approved:
            print(f"  🛑 Risk rejected: {risk_check.reason}")
            return

        # Корректируем лот на вес стратегии
        if self.allocator and capital_weight < 1.0:
            n_strats = len(self.strategies)
            adjusted_lot = round(
                risk_check.lot_size * capital_weight * n_strats, 2
            )
            risk_check.lot_size = max(adjusted_lot, 0.01)

        # ═══ STEP 10: Execution ═══
        trade_info = self.execution_engine.execute_trade(
            best_signal, risk_check
        )

        if trade_info is None:
            return

        # ═══ STEP 11: Logging ═══
        self.trade_logger.log_trade(trade_info)

        # ═══ STEP 12: Telegram ═══
        self.telegram.notify_trade(trade_info)

        # ═══ STEP 13: Metrics ═══
        if self.metrics:
            self.metrics.record_trade(
                strategy=best_signal.strategy_name,
                direction=best_signal.signal_type.value,
                pnl=0,
                is_win=True
            )

        # ═══ STEP 14: Service Bus ═══
        if self.bus:
            self.bus.publish_trade("open", trade_info)

    # ═══════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════

    def _update_trailing_stops(self):
        """Обновить trailing stops"""
        if not risk_config.trailing_stop:
            return

        positions = self.execution_engine.get_open_positions()
        if not positions:
            return

        for pos in positions:
            try:
                df = self.data_processor.get_processed_data(
                    symbol=pos.symbol, count=50
                )
                if df is None or "atr" not in df.columns:
                    continue
                # Trailing logic handled by execution engine
            except Exception:
                pass

    def _is_trading_time(self) -> bool:
        """Проверка торгового времени"""
        now = datetime.utcnow()
        # Не торгуем в выходные
        if now.weekday() >= 5:
            return False
        return True

    def _wait(self):
        """Ожидание до следующей итерации"""
        wait_map = {
            "M1": 60, "M5": 300, "M15": 900,
            "M30": 1800, "H1": 3600, "H4": 14400,
        }
        wait = wait_map.get(trading_config.timeframe, 3600)
        wait = min(wait, 300)  # Макс 5 минут

        print(
            f"[{self._now()}] Waiting {wait}s "
            f"(iteration #{self.iteration})"
        )
        time.sleep(wait)

    def _send_report(self):
        """Периодический отчёт"""
        stats = self.trade_logger.get_stats()

        msg = (
            f"📊 <b>Periodic Report</b>\n\n"
            f"Iteration: #{self.iteration}\n"
            f"Trades: {stats.get('total', 0)}\n"
            f"Winrate: {stats.get('winrate', 0)}%\n"
        )

        if self.allocator:
            weights = self.allocator.get_weights()
            msg += f"\nWeights: {weights}"

        self.telegram.send(msg)

        if self.metrics:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                self.metrics.update_equity(
                    equity=info.equity,
                    daily_pnl=0,
                    total_pnl=info.profit,
                    drawdown_pct=0,
                    n_positions=mt5.positions_total() or 0
                )

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M:%S")


# ═══════════════════════════════════════════════
#  BACKTEST FUNCTIONS
# ═══════════════════════════════════════════════

def run_backtest():
    """Стандартный бэктест"""
    try:
        from backtest.backtester import Backtester
    except ImportError:
        print("backtest/backtester.py not found")
        return

    print("\n" + "=" * 60)
    print("  BACKTEST MODE")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data("EURUSD", "H1", 5000)
    processor.disconnect()

    if df is None:
        print("No data")
        return

    bt = Backtester()
    strategies = [
        TrendStrategy(),
        RangeStrategy(),
        BreakoutStrategy(),
    ]

    if HAS_SCALPING:
        strategies.append(ScalpingStrategy())

    print(f"\nTesting {len(strategies)} strategies "
          f"on {len(df)} bars\n")

    for s in strategies:
        result = bt.run(s, df)
        print(f"  {s.name}: {result.total_trades} trades, "
              f"WR={result.winrate:.1f}%, "
              f"PF={result.profit_factor:.2f}")


def run_super_backtest():
    """Квантовый бэктест с Walk-Forward"""
    try:
        from backtest.quant_backtester import (
            QuantBacktester, CostModel, FillModel
        )
    except ImportError:
        print("backtest/quant_backtester.py not found")
        return

    print("\n" + "=" * 60)
    print("  QUANT BACKTEST")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data("EURUSD", "H1", 20000)
    processor.disconnect()

    if df is None:
        print("No data")
        return

    cost = CostModel(
        commission_per_lot=7.0,
        slippage_pips=0.5,
        fill_model=FillModel.NEXT_BAR
    )

    bt = QuantBacktester(cost_model=cost)

    # Пример signal function для trend strategy
    def trend_signal_func(window_df):
        if len(window_df) < 5:
            return {"action": "none"}

        row = window_df.iloc[-1]
        prev = window_df.iloc[-2]
        atr = row.get("atr", 0)

        if atr <= 0 or row.get("adx", 0) < 25:
            return {"action": "none"}

        if (
            row.get("ema_50", 0) > row.get("ema_200", 0)
            and row.get("macd", 0) > row.get("macd_signal", 0)
            and prev.get("macd", 0) <= prev.get("macd_signal", 0)
        ):
            return {
                "action": "buy",
                "sl_distance": atr * 1.5,
                "tp_distance": atr * 4.5
            }

        if (
            row.get("ema_50", 0) < row.get("ema_200", 0)
            and row.get("macd", 0) < row.get("macd_signal", 0)
            and prev.get("macd", 0) >= prev.get("macd_signal", 0)
        ):
            return {
                "action": "sell",
                "sl_distance": atr * 1.5,
                "tp_distance": atr * 4.5
            }

        return {"action": "none"}

    # Walk-Forward
    metrics, windows = bt.walk_forward(
        trend_signal_func, df,
        n_splits=5, strategy_name="trend"
    )

    # Monte Carlo
    mc = bt.monte_carlo(n_simulations=2000)

    bt.print_report(metrics, "Trend Strategy")

    print(f"\nMonte Carlo:")
    print(f"  P(Profit): {mc.get('prob_profit', 0):.1%}")
    print(f"  P(Ruin):   {mc.get('prob_ruin', 0):.1%}")
    print(f"  95% MaxDD: {mc.get('dd_95', 0):.1f}%")


def run_ultra_backtest():
    """Ультра-быстрый бэктест"""
    try:
        from backtest.ultra_fast_backtester import UltraFastBacktester
    except ImportError:
        print("backtest/ultra_fast_backtester.py not found")
        return

    print("\n" + "=" * 60)
    print("  ULTRA-FAST BACKTEST")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    data = {}

    for sym in symbols:
        df = processor.get_candles(sym, "H1", 50000)
        if df is not None:
            data[sym] = df
            print(f"  {sym}: {len(df)} bars")

    processor.disconnect()

    if not data:
        print("No data")
        return

    bt = UltraFastBacktester()
    results = bt.run_multi_pair(data)

    for sym, result in results.items():
        bt.print_results(result)


def run_session_backtest():
    """Session strategy бэктест"""
    if not HAS_SESSION:
        print("Session strategy not available")
        return

    try:
        from backtest.backtester import Backtester
    except ImportError:
        print("backtest/backtester.py not found")
        return

    print("\n" + "=" * 60)
    print("  SESSION STRATEGY BACKTEST")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data("EURUSD", "M15", 20000)
    processor.disconnect()

    if df is None:
        print("No M15 data")
        return

    print(f"Data: {len(df)} M15 bars")

    strategy = SessionStrategy(utc_offset=2)
    bt = Backtester()
    result = bt.run(strategy, df)

    print(f"\nResult: {result.total_trades} trades, "
          f"WR={result.winrate:.1f}%")


def run_train_ai():
    """Обучить AI с защитой от переобучения"""
    if not HAS_TRAINER:
        print("AntiOverfitTrainer not available")
        print("Create ai/anti_overfit_trainer.py first")
        return

    print("\n" + "=" * 60)
    print("  AI TRAINING (Anti-Overfit)")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data("EURUSD", "H1", 30000)
    processor.disconnect()

    if df is None:
        print("No data")
        return

    def build_target(data):
        c = data["close"].values
        atr_vals = data["atr"].values
        n = len(c)
        target = np.zeros(n)
        for i in range(n - 3):
            future = c[i + 1:i + 4]
            move = max(
                np.max(future) - c[i],
                c[i] - np.min(future)
            )
            if atr_vals[i] > 0 and move > atr_vals[i]:
                target[i] = 1
        return target

    trainer = AntiOverfitTrainer(
        n_folds=5,
        purge_bars=20,
        holdout_ratio=0.15,
        max_features=30,
    )

    report = trainer.train(
        df=df,
        target_builder=build_target,
        model_name="impulse_v1"
    )

    if not report.is_overfitted:
        trainer.save()
        print("\n✅ Model saved — safe to use")
    else:
        print("\n❌ Model overfitted — NOT saved")


def run_train_impulse():
    """Обучить Impulse Predictor"""
    if not HAS_IMPULSE:
        print("ImpulsePredictor not available")
        return

    print("\n" + "=" * 60)
    print("  IMPULSE PREDICTOR TRAINING")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data("EURUSD", "H1", 30000)
    processor.disconnect()

    if df is None:
        print("No data")
        return

    predictor = ImpulsePredictor()
    results = predictor.train(df, horizon=3, threshold_atr=1.0)
    print(f"\nResults: {results}")


def run_features():
    """Анализ Feature Factory"""
    if not HAS_FACTORY:
        print("FeatureFactory not available")
        return

    print("\n" + "=" * 60)
    print("  FEATURE FACTORY")
    print("=" * 60)

    processor = DataProcessor()
    if not processor.connect():
        return

    df = processor.get_processed_data("EURUSD", "H1", 20000)
    processor.disconnect()

    if df is None:
        print("No data")
        return

    factory = FeatureFactory(max_features=35)
    features = factory.build(df)

    # Target
    c = df["close"].values
    atr_vals = df["atr"].values
    target = np.zeros(len(c))
    for i in range(len(c) - 3):
        future = c[i + 1:i + 4]
        move = max(np.max(future) - c[i], c[i] - np.min(future))
        if atr_vals[i] > 0 and move > atr_vals[i]:
            target[i] = 1

    min_len = min(len(features), len(target))
    features = features.iloc[:min_len]
    target = target[:min_len]

    selected, report = factory.select(features, target)

    print(f"\n  Generated:  {report.total_generated}")
    print(f"  Selected:   {report.final_selected}")
    print(f"  Categories: {report.categories}")

    print(f"\n  Top features:")
    for fi in report.top_features[:15]:
        print(f"    {fi.name:<30} imp={fi.importance:.4f}")

    factory.save()


def run_collect():
    """Загрузить данные в Data Lake"""
    try:
        from data_lake.lake import DataLake
        from data_lake.collector import DataCollector
    except ImportError:
        print("data_lake modules not found")
        return

    lake = DataLake()
    collector = DataCollector(lake)

    if not collector.connect():
        return

    collector.collect_historical(years_back=10)
    collector.disconnect()
    lake.print_info()


def run_lake_info():
    """Info о Data Lake"""
    try:
        from data_lake.lake import DataLake
    except ImportError:
        print("data_lake/lake.py not found")
        return

    lake = DataLake()
    lake.print_info()


def run_validate():
    """Проверка данных"""
    try:
        from data_lake.lake import DataLake
    except ImportError:
        print("data_lake/lake.py not found")
        return

    lake = DataLake()
    for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
        for tf in ["H1", "D1"]:
            result = lake.validate(sym, tf)
            ok = "✅" if result.get("valid") else "❌"
            print(f"  {ok} {sym}/{tf}: {result}")


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"

    commands = {
        "live":           lambda: ForexHybridBot("live").start(),
        "demo":           lambda: ForexHybridBot("demo").start(),
        "backtest":       run_backtest,
        "supertest":      run_super_backtest,
        "ultratest":      run_ultra_backtest,
        "sessions":       run_session_backtest,
        "train_ai":       run_train_ai,
        "train_impulse":  run_train_impulse,
        "features":       run_features,
        "collect":        run_collect,
        "lake_info":      run_lake_info,
        "validate":       run_validate,
    }

    if mode in commands:
        commands[mode]()
    else:
        print(f"Unknown mode: {mode}")
        print(f"Available: {', '.join(commands.keys())}")
