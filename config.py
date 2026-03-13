"""
Конфигурация Forex Hybrid Bot
Все настройки в одном месте
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MT5Config:
    """Настройки подключения к MetaTrader 5"""
    login: int = 12345678
    password: str = "your_password"
    server: str = "YourBroker-Demo"
    path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"


@dataclass
class TradingConfig:
    """Настройки торговли"""
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    magic_number: int = 234000

    # Мультивалютность
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"
    ])


@dataclass
class RiskConfig:
    """Настройки риск-менеджмента"""
    risk_per_trade: float = 0.01          # 1% от депозита на сделку
    max_daily_loss: float = 0.03          # 3% макс дневной убыток
    max_open_trades: int = 3              # Макс открытых позиций
    max_correlation_trades: int = 2       # Макс коррелированных пар
    default_sl_atr_mult: float = 1.5      # SL = 1.5 × ATR
    default_tp_atr_mult: float = 3.0      # TP = 3.0 × ATR (RR 1:2)
    max_spread_points: int = 30           # Макс допустимый спред
    trailing_stop: bool = True            # Трейлинг стоп
    trailing_atr_mult: float = 1.0        # Трейлинг = 1 × ATR


@dataclass
class StrategyWeights:
    """Веса стратегий в портфеле"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 0.30,
        "range": 0.25,
        "breakout": 0.20,
        "scalping": 0.25
    })


@dataclass
class AIConfig:
    """Настройки AI модуля"""
    enabled: bool = True
    model_path: str = "data/strategy_model.pkl"
    retrain_interval_hours: int = 24
    min_training_samples: int = 100
    confidence_threshold: float = 0.6     # Мин уверенность для входа


@dataclass
class MarketRegimeConfig:
    """Пороги для определения режима рынка"""
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_volatility_percentile: float = 75.0
    bb_squeeze_threshold: float = 0.02


@dataclass
class TelegramConfig:
    """Настройки Telegram бота"""
    enabled: bool = True
    bot_token: str = "YOUR_BOT_TOKEN"
    chat_id: str = "YOUR_CHAT_ID"


@dataclass
class BacktestConfig:
    """Настройки бэктестирования"""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0


# Глобальные экземпляры конфигов
mt5_config = MT5Config()
trading_config = TradingConfig()
risk_config = RiskConfig()
strategy_weights = StrategyWeights()
ai_config = AIConfig()
regime_config = MarketRegimeConfig()
telegram_config = TelegramConfig()
backtest_config = BacktestConfig()
