"""
config.py — Forex Hybrid Bot v3.0
Все настройки в одном месте

ВАЖНО: Заполни свои данные в MT5Config перед запуском!
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ═══════════════════════════════════════════════
#  MT5 — Подключение к MetaTrader 5
# ═══════════════════════════════════════════════

@dataclass
class MT5Config:
    """
    ЗАПОЛНИ СВОИМИ ДАННЫМИ!

    Где взять:
    1. Открой MT5
    2. Файл → Открыть счёт (или используй существующий)
    3. Login = номер счёта
    4. Password = пароль
    5. Server = имя сервера (видно при логине)
    6. Path = путь к terminal64.exe на твоём компьютере
    """
    login: int = 12345678
    password: str = "your_password_here"
    server: str = "YourBroker-Demo"
    path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"


# ═══════════════════════════════════════════════
#  TRADING — Торговые настройки
# ═══════════════════════════════════════════════

@dataclass
class TradingConfig:
    """Основные торговые параметры"""

    # Основной символ
    symbol: str = "EURUSD"

    # Таймфрейм (H1 рекомендуется для начала)
    timeframe: str = "H1"

    # Magic number — уникальный ID бота
    # Чтобы отличать сделки бота от ручных
    magic_number: int = 234000

    # Список пар для торговли
    # Для $100 счёта — только 3 пары с низким спредом
    # Для $1000+ можно добавить больше
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD",
        "USDJPY",
        "GBPUSD",
    ])

    # Расширенный список (для больших счетов)
    # symbols_extended: List[str] = field(default_factory=lambda: [
    #     "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    #     "AUDUSD", "USDCAD", "NZDUSD",
    #     "EURGBP", "EURJPY", "GBPJPY",
    #     "EURAUD", "AUDNZD", "AUDJPY",
    # ])


# ═══════════════════════════════════════════════
#  RISK — Управление рисками
# ═══════════════════════════════════════════════

@dataclass
class RiskConfig:
    """
    Настройки риск-менеджмента

    САМЫЙ ВАЖНЫЙ РАЗДЕЛ!
    Неправильные настройки = потеря депозита
    """

    # ─── Размер риска ────────────────────────

    # Риск на одну сделку (% от депозита)
    # $100 счёт: 0.02 (2%) = $2 риска
    # $1000 счёт: 0.01 (1%) = $10 риска
    # $10000 счёт: 0.005 (0.5%) = $50 риска
    risk_per_trade: float = 0.02

    # Максимальный дневной убыток (% от депозита)
    # При достижении — бот останавливает торговлю до следующего дня
    max_daily_loss: float = 0.05

    # ─── Позиции ─────────────────────────────

    # Максимум одновременно открытых позиций
    # $100: 2 позиции
    # $1000+: 3-5 позиций
    max_open_trades: int = 2

    # Максимум коррелированных позиций
    # (например EURUSD и GBPUSD — коррелируют)
    max_correlated_trades: int = 1

    # ─── SL / TP ─────────────────────────────

    # Стоп-лосс = ATR × множитель
    default_sl_atr_mult: float = 1.5

    # Тейк-профит = ATR × множитель
    # TP/SL = 4.5/1.5 = RR 3:1
    default_tp_atr_mult: float = 4.5

    # Минимальный Risk:Reward для входа
    min_risk_reward: float = 2.5

    # ─── Спред ───────────────────────────────

    # Максимально допустимый спред (в пунктах)
    # Если спред выше — сделка не открывается
    # EURUSD обычно 10-15 пунктов
    max_spread_points: int = 20

    # ─── Trailing Stop ───────────────────────

    # Включить трейлинг стоп
    trailing_stop: bool = True

    # Трейлинг = ATR × множитель
    trailing_atr_mult: float = 1.0


# ═══════════════════════════════════════════════
#  STRATEGIES — Веса стратегий
# ═══════════════════════════════════════════════

@dataclass
class StrategyWeights:
    """
    Распределение капитала между стратегиями

    Сумма должна быть ~1.0
    Meta-AI может менять эти веса автоматически
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "trend_strategy": 0.25,
        "range_strategy": 0.15,
        "breakout_strategy": 0.15,
        "scalping_strategy": 0.10,
        "session_strategy": 0.20,
        "smc_strategy": 0.15,
    })


# ═══════════════════════════════════════════════
#  AI — Настройки искусственного интеллекта
# ═══════════════════════════════════════════════

@dataclass
class AIConfig:
    """Настройки AI модулей"""

    # Включить AI (выключи пока нет обученной модели)
    enabled: bool = False

    # Путь к сохранённой модели
    model_path: str = "data/strategy_model.pkl"

    # Как часто переобучать (в часах)
    retrain_interval_hours: int = 24

    # Минимум сделок для обучения
    min_training_samples: int = 100

    # Минимальная уверенность AI для входа
    # 0.6 = модель должна быть уверена на 60%+
    confidence_threshold: float = 0.6


# ═══════════════════════════════════════════════
#  MARKET REGIME — Определение рынка
# ═══════════════════════════════════════════════

@dataclass
class MarketRegimeConfig:
    """Пороги для определения режима рынка"""

    # ADX > этого = тренд
    adx_trend_threshold: float = 25.0

    # ADX < этого = рэндж
    adx_range_threshold: float = 20.0

    # ATR percentile > этого = высокая волатильность
    atr_volatility_percentile: float = 75.0

    # BB width < этого = сжатие (скоро пробой)
    bb_squeeze_threshold: float = 0.02


# ═══════════════════════════════════════════════
#  TELEGRAM — Уведомления
# ═══════════════════════════════════════════════

@dataclass
class TelegramConfig:
    """
    Telegram бот для уведомлений

    Как настроить:
    1. Найди @BotFather в Telegram
    2. Напиши /newbot
    3. Получи token
    4. Найди @userinfobot
    5. Получи свой chat_id
    """
    enabled: bool = False
    bot_token: str = "YOUR_BOT_TOKEN_HERE"
    chat_id: str = "YOUR_CHAT_ID_HERE"


# ═══════════════════════════════════════════════
#  BACKTEST — Бэктестирование
# ═══════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """Настройки бэктеста"""

    # Начальный баланс для тестов
    initial_balance: float = 10000.0

    # Комиссия за лот ($)
    commission_per_lot: float = 7.0

    # Проскальзывание (пипсы)
    slippage_pips: float = 0.5

    # Спред для симуляции (пипсы)
    simulated_spread: float = 1.5


# ═══════════════════════════════════════════════
#  DATA LAKE — Хранилище данных
# ═══════════════════════════════════════════════

@dataclass
class DataLakeConfig:
    """Настройки Data Lake"""

    # Путь к хранилищу
    base_path: str = "data/lake"

    # Символы для сбора
    collect_symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY",
    ])

    # Таймфреймы для сбора
    collect_timeframes: List[str] = field(default_factory=lambda: [
        "M15", "H1", "H4", "D1"
    ])

    # Интервал сбора live данных (секунды)
    collect_interval: int = 60


# ═══════════════════════════════════════════════
#  MONITORING — Мониторинг
# ═══════════════════════════════════════════════

@dataclass
class MonitoringConfig:
    """Настройки мониторинга"""

    # Prometheus metrics
    prometheus_enabled: bool = False
    prometheus_port: int = 8000

    # Redis для Service Bus
    redis_host: str = "localhost"
    redis_port: int = 6379


# ═══════════════════════════════════════════════
#  SESSION STRATEGY — Настройки сессий
# ═══════════════════════════════════════════════

@dataclass
class SessionConfig:
    """
    Настройки для Session Strategy

    Время в UTC!
    Если MT5 сервер в UTC+2, utc_offset = 2
    """

    # UTC offset сервера MT5
    # Большинство брокеров: UTC+2 (зима) / UTC+3 (лето)
    utc_offset: int = 2

    # Killzone окна (UTC часы)
    # Asia Open
    asia_start: int = 0
    asia_end: int = 7

    # London Open
    london_start: int = 7
    london_end: int = 10

    # New York Open
    ny_start: int = 12
    ny_end: int = 16

    # London Close
    lc_start: int = 15
    lc_end: int = 17


# ═══════════════════════════════════════════════
#  PRESETS — Готовые пресеты
# ═══════════════════════════════════════════════

class Presets:
    """
    Готовые конфигурации для разных размеров счёта

    Использование:
        Presets.apply_small_account()   # $100-500
        Presets.apply_medium_account()  # $1000-5000
        Presets.apply_large_account()   # $10000+
    """

    @staticmethod
    def apply_small_account():
        """$100-500: Консервативные настройки"""
        risk_config.risk_per_trade = 0.02
        risk_config.max_daily_loss = 0.05
        risk_config.max_open_trades = 2
        risk_config.default_tp_atr_mult = 4.5
        risk_config.min_risk_reward = 3.0
        trading_config.symbols = ["EURUSD", "USDJPY", "GBPUSD"]
        trading_config.timeframe = "H1"
        print("Applied: Small Account preset ($100-500)")

    @staticmethod
    def apply_medium_account():
        """$1000-5000: Сбалансированные настройки"""
        risk_config.risk_per_trade = 0.01
        risk_config.max_daily_loss = 0.04
        risk_config.max_open_trades = 3
        risk_config.default_tp_atr_mult = 3.0
        risk_config.min_risk_reward = 2.0
        trading_config.symbols = [
            "EURUSD", "GBPUSD", "USDJPY",
            "AUDUSD", "USDCAD"
        ]
        trading_config.timeframe = "H1"
        print("Applied: Medium Account preset ($1000-5000)")

    @staticmethod
    def apply_large_account():
        """$10000+: Полный портфель"""
        risk_config.risk_per_trade = 0.005
        risk_config.max_daily_loss = 0.03
        risk_config.max_open_trades = 5
        risk_config.default_tp_atr_mult = 3.0
        risk_config.min_risk_reward = 2.0
        trading_config.symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "USDCAD", "NZDUSD",
            "EURGBP", "EURJPY", "GBPJPY",
        ]
        trading_config.timeframe = "H1"
        ai_config.enabled = True
        print("Applied: Large Account preset ($10000+)")


# ═══════════════════════════════════════════════
#  GLOBAL INSTANCES — Глобальные экземпляры
# ═══════════════════════════════════════════════

# Создаём экземпляры которые импортируются другими модулями
# from config import mt5_config, trading_config, risk_config ...

mt5_config = MT5Config()
trading_config = TradingConfig()
risk_config = RiskConfig()
strategy_weights = StrategyWeights()
ai_config = AIConfig()
regime_config = MarketRegimeConfig()
telegram_config = TelegramConfig()
backtest_config = BacktestConfig()
datalake_config = DataLakeConfig()
monitoring_config = MonitoringConfig()
session_config = SessionConfig()


# ═══════════════════════════════════════════════
#  AUTO-APPLY PRESET (раскомментируй нужный)
# ═══════════════════════════════════════════════

# Presets.apply_small_account()     # ← Для $100
# Presets.apply_medium_account()    # ← Для $1000
# Presets.apply_large_account()     # ← Для $10000+
