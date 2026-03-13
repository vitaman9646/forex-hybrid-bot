"""
Backtester
Тестирование стратегий на исторических данных
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from strategies.base_strategy import (
    BaseStrategy, TradeSignal, SignalType
)
from core.data_processor import DataProcessor
from core.market_detector import MarketDetector, MarketRegime
from config import backtest_config, risk_config


@dataclass
class BacktestTrade:
    """Сделка в бэктесте"""
    entry_time: datetime
    exit_time: datetime = None
    signal_type: str = ""
    entry_price: float = 0
    exit_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    profit: float = 0
    strategy: str = ""


@dataclass
class BacktestResult:
    """Результат бэктеста"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0
    total_profit: float = 0
    max_drawdown: float = 0
    profit_factor: float = 0
    sharpe_ratio: float = 0
    avg_trade: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """Бэктестер стратегий"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.market_detector = MarketDetector()

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        initial_balance: float = None
    ) -> BacktestResult:
        """
        Запуск бэктеста стратегии на данных
        
        Args:
            strategy: экземпляр стратегии
            df: DataFrame с OHLCV данными + индикаторы
            initial_balance: начальный баланс
        """
        balance = initial_balance or backtest_config.initial_balance
        initial = balance
        trades = []
        equity_curve = [balance]
        current_trade: Optional[BacktestTrade] = None

        print(
            f"\n[BACKTEST] Запуск: {strategy.name} | "
            f"Баров: {len(df)} | "
            f"Баланс: {balance}"
        )

        # Проходим по каждому бару
        for i in range(50, len(df)):
            # Данные до текущего бара (без заглядывания вперёд)
            window = df.iloc[:i + 1].copy()
            row = df.iloc[i]

            # Если есть открытая позиция — проверяем SL/TP
            if current_trade is not None:
                closed = self._check_exit(
                    current_trade, row
                )

                if closed:
                    balance += current_trade.profit
                    trades.append(current_trade)
                    current_trade = None

                equity_curve.append(balance)
                continue

            # Генерируем сигнал
            signal = strategy.generate_signal(window, "BACKTEST")

            if signal.signal_type != SignalType.NONE:
                current_trade = BacktestTrade(
                    entry_time=row.name if hasattr(
                        row, 'name'
                    ) else datetime.now(),
                    signal_type=signal.signal_type.value,
                    entry_price=row["close"],
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy=strategy.name
                )

            equity_curve.append(balance)

        # Закрываем если осталась открытая
        if current_trade is not None:
            current_trade.exit_price = df.iloc[-1]["close"]
            current_trade.exit_time = df.index[-1]

            if current_trade.signal_type == "buy":
                current_trade.profit = (
                    current_trade.exit_price
                    - current_trade.entry_price
                ) * 100000 * 0.01  # примерный расчёт
            else:
                current_trade.profit = (
                    current_trade.entry_price
                    - current_trade.exit_price
                ) * 100000 * 0.01

            balance += current_trade.profit
            trades.append(current_trade)

        # Рассчитываем метрики
        result = self._calculate_metrics(
            trades, equity_curve, initial
        )

        self._print_results(result, strategy.name)
        return result

    def _check_exit(
        self,
        trade: BacktestTrade,
        row: pd.Series
    ) -> bool:
        """Проверка выхода по SL или TP"""
        high = row["high"]
        low = row["low"]

        if trade.signal_type == "buy":
            # Стоп-лосс
            if low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.profit = (
                    trade.stop_loss - trade.entry_price
                ) * 100000 * 0.01
                trade.exit_time = row.name
                return True

            # Тейк-профит
            if high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.profit = (
                    trade.take_profit - trade.entry_price
                ) * 100000 * 0.01
                trade.exit_time = row.name
                return True

        elif trade.signal_type == "sell":
            # Стоп-лосс
            if high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.profit = (
                    trade.entry_price - trade.stop_loss
                ) * 100000 * 0.01
                trade.exit_time = row.name
                return True

            # Тейк-профит
            if low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.profit = (
                    trade.entry_price - trade.take_profit
                ) * 100000 * 0.01
                trade.exit_time = row.name
                return True

        return False

    @staticmethod
    def _calculate_metrics(
        trades: List[BacktestTrade],
        equity_curve: List[float],
        initial_balance: float
    ) -> BacktestResult:
        """Расчёт метрик"""
        result = BacktestResult()
        result.trades = trades
        result.equity_curve = equity_curve
        result.total_trades = len(trades)

        if not trades:
            return result

        profits = [t.profit for t in trades]

        result.wins = sum(1 for p in profits if p > 0)
        result.losses = sum(1 for p in profits if p < 0)
        result.total_profit = sum(profits)

        result.winrate = (
            result.wins / result.total_trades * 100
            if result.total_trades > 0 else 0
        )

        result.avg_trade = np.mean(profits) if profits else 0
        result.best_trade = max(profits) if profits else 0
        result.worst_trade = min(profits) if profits else 0

        # Profit Factor
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )

        # Max Drawdown
        peak = initial_balance
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd

        # Sharpe Ratio (упрощённый)
        if len(profits) > 1:
            returns = np.array(profits) / initial_balance
            if np.std(returns) > 0:
                result.sharpe_ratio = (
                    np.mean(returns) / np.std(returns)
                    * np.sqrt(252)
                )

        return result

    @staticmethod
    def _print_results(result: BacktestResult, name: str):
        """Вывод результатов"""
        print(f"\n{'='*50}")
        print(f"  BACKTEST RESULT: {name}")
        print(f"{'='*50}")
        print(f"  Сделок:        {result.total_trades}")
        print(f"  Wins:          {result.wins}")
        print(f"  Losses:        {result.losses}")
        print(f"  Winrate:       {result.winrate:.1f}%")
        print(f"  Total Profit:  {result.total_profit:.2f}")
        print(f"  Avg Trade:     {result.avg_trade:.2f}")
        print(f"  Best Trade:    {result.best_trade:.2f}")
        print(f"  Worst Trade:   {result.worst_trade:.2f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown:  {result.max_drawdown:.2f}%")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
        print(f"{'='*50}\n")
