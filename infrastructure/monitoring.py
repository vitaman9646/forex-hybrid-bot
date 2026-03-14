"""
infrastructure/monitoring.py

Мониторинг и метрики для Prometheus + Grafana
+ Health checks + Alerting
"""

import time
from typing import Dict, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
import threading
import json
import os

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        start_http_server, CollectorRegistry
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


@dataclass
class HealthStatus:
    """Статус здоровья сервиса"""
    service: str
    is_healthy: bool
    uptime_seconds: float
    last_trade_time: Optional[str] = None
    open_positions: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    errors_today: int = 0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    mt5_connected: bool = False
    redis_connected: bool = False


class MetricsCollector:
    """
    Сборщик метрик для Prometheus

    Метрики:
    - Торговые (P&L, сделки, winrate)
    - Системные (CPU, RAM, latency)
    - AI (accuracy, prediction count)
    - Инфраструктурные (uptime, errors)
    """

    def __init__(
        self,
        service_name: str = "forex_bot",
        port: int = 8000
    ):
        self.service_name = service_name
        self.start_time = time.time()
        self.errors_today = 0
        self._today = date.today()

        if not HAS_PROMETHEUS:
            print("[METRICS] prometheus_client not installed")
            return

        # ═══ Trading Metrics ═══
        self.trades_total = Counter(
            "forex_trades_total",
            "Total number of trades",
            ["strategy", "direction", "result"]
        )

        self.pnl_total = Gauge(
            "forex_pnl_total",
            "Total P&L in USD"
        )

        self.pnl_daily = Gauge(
            "forex_pnl_daily",
            "Daily P&L in USD"
        )

        self.equity = Gauge(
            "forex_equity",
            "Current equity"
        )

        self.drawdown = Gauge(
            "forex_drawdown_pct",
            "Current drawdown percentage"
        )

        self.open_positions = Gauge(
            "forex_open_positions",
            "Number of open positions"
        )

        self.winrate = Gauge(
            "forex_winrate",
            "Current win rate",
            ["strategy"]
        )

        # ═══ Strategy Metrics ═══
        self.strategy_weight = Gauge(
            "forex_strategy_weight",
            "Strategy weight from Meta-AI",
            ["strategy"]
        )

        self.signal_count = Counter(
            "forex_signals_total",
            "Total signals generated",
            ["strategy", "type"]
        )

        # ═══ AI Metrics ═══
        self.ai_prediction_count = Counter(
            "forex_ai_predictions_total",
            "Total AI predictions",
            ["model"]
        )

        self.ai_accuracy = Gauge(
            "forex_ai_accuracy",
            "AI model accuracy",
            ["model"]
        )

        self.impulse_probability = Gauge(
            "forex_impulse_probability",
            "Current impulse probability",
            ["symbol"]
        )

        # ═══ System Metrics ═══
        self.errors = Counter(
            "forex_errors_total",
            "Total errors",
            ["type"]
        )

        self.latency = Histogram(
            "forex_processing_seconds",
            "Processing time per iteration",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
        )

        self.mt5_connected = Gauge(
            "forex_mt5_connected",
            "MT5 connection status"
        )

        # Start metrics server
        try:
            start_http_server(port)
            print(f"[METRICS] Prometheus metrics on :{port}")
        except Exception as e:
            print(f"[METRICS] Failed to start: {e}")

    # ─── Trade Recording ─────────────────────

    def record_trade(
        self,
        strategy: str,
        direction: str,
        pnl: float,
        is_win: bool
    ):
        """Записать сделку"""
        if not HAS_PROMETHEUS:
            return

        result = "win" if is_win else "loss"
        self.trades_total.labels(
            strategy=strategy,
            direction=direction,
            result=result
        ).inc()

    def update_equity(
        self,
        equity: float,
        daily_pnl: float,
        total_pnl: float,
        drawdown_pct: float,
        n_positions: int
    ):
        """Обновить equity метрики"""
        if not HAS_PROMETHEUS:
            return

        self.equity.set(equity)
        self.pnl_daily.set(daily_pnl)
        self.pnl_total.set(total_pnl)
        self.drawdown.set(drawdown_pct)
        self.open_positions.set(n_positions)

    def update_strategy_weights(self, weights: Dict[str, float]):
        """Обновить веса стратегий"""
        if not HAS_PROMETHEUS:
            return
        for strategy, weight in weights.items():
            self.strategy_weight.labels(strategy=strategy).set(weight)

    def update_winrate(self, strategy: str, rate: float):
        """Обновить winrate"""
        if not HAS_PROMETHEUS:
            return
        self.winrate.labels(strategy=strategy).set(rate)

    # ─── AI Recording ────────────────────────

    def record_prediction(self, model: str):
        """Записать предсказание AI"""
        if not HAS_PROMETHEUS:
            return
        self.ai_prediction_count.labels(model=model).inc()

    def update_ai_accuracy(self, model: str, accuracy: float):
        """Обновить accuracy AI"""
        if not HAS_PROMETHEUS:
            return
        self.ai_accuracy.labels(model=model).set(accuracy)

    def update_impulse_prob(self, symbol: str, prob: float):
        """Обновить вероятность импульса"""
        if not HAS_PROMETHEUS:
            return
        self.impulse_probability.labels(symbol=symbol).set(prob)

    # ─── Signal Recording ────────────────────

    def record_signal(self, strategy: str, signal_type: str):
        """Записать сигнал"""
        if not HAS_PROMETHEUS:
            return
        self.signal_count.labels(
            strategy=strategy, type=signal_type
        ).inc()

    # ─── System ──────────────────────────────

    def record_error(self, error_type: str):
        """Записать ошибку"""
        self.errors_today += 1
        if HAS_PROMETHEUS:
            self.errors.labels(type=error_type).inc()

    def record_latency(self, seconds: float):
        """Записать latency итерации"""
        if not HAS_PROMETHEUS:
            return
        self.latency.observe(seconds)

    def set_mt5_status(self, connected: bool):
        """Статус MT5"""
        if not HAS_PROMETHEUS:
            return
        self.mt5_connected.set(1 if connected else 0)

    # ─── Health Check ────────────────────────

    def get_health(self) -> HealthStatus:
        """Получить статус здоровья"""
        uptime = time.time() - self.start_time

        # Проверяем дату для сброса ежедневных счётчиков
        today = date.today()
        if today != self._today:
            self._today = today
            self.errors_today = 0

        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_pct = process.cpu_percent()
        except ImportError:
            memory_mb = 0
            cpu_pct = 0

        return HealthStatus(
            service=self.service_name,
            is_healthy=self.errors_today < 100,
            uptime_seconds=uptime,
            errors_today=self.errors_today,
            memory_mb=round(memory_mb, 1),
            cpu_percent=round(cpu_pct, 1)
      )
