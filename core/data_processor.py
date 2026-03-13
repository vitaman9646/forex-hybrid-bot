"""
Data Processor
Получение данных из MT5, расчёт индикаторов, feature engineering
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from config import mt5_config, trading_config


class DataProcessor:
    """Обработка рыночных данных и расчёт индикаторов"""

    # Маппинг таймфреймов
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        """Подключение к MT5"""
        if not mt5.initialize(path=mt5_config.path):
            print(f"[DATA] Ошибка инициализации MT5: {mt5.last_error()}")
            return False

        authorized = mt5.login(
            login=mt5_config.login,
            password=mt5_config.password,
            server=mt5_config.server
        )

        if not authorized:
            print(f"[DATA] Ошибка авторизации: {mt5.last_error()}")
            return False

        self.connected = True
        print("[DATA] Подключение к MT5 успешно")
        return True

    def disconnect(self):
        """Отключение от MT5"""
        mt5.shutdown()
        self.connected = False
        print("[DATA] Отключено от MT5")

    def get_candles(
        self,
        symbol: str = None,
        timeframe: str = None,
        count: int = 500
    ) -> Optional[pd.DataFrame]:
        """Получение свечей из MT5"""
        symbol = symbol or trading_config.symbol
        timeframe = timeframe or trading_config.timeframe
        mt5_tf = self.TIMEFRAME_MAP.get(timeframe)

        if mt5_tf is None:
            print(f"[DATA] Неизвестный таймфрейм: {timeframe}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)

        if rates is None or len(rates) == 0:
            print(f"[DATA] Нет данных для {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume"
        }, inplace=True)

        return df[["open", "high", "low", "close", "volume"]]

    def get_current_price(self, symbol: str = None) -> Optional[Dict]:
        """Текущая цена (bid/ask/spread)"""
        symbol = symbol or trading_config.symbol
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "time": datetime.fromtimestamp(tick.time)
        }

    def get_spread_points(self, symbol: str = None) -> float:
        """Спред в пунктах"""
        symbol = symbol or trading_config.symbol
        info = mt5.symbol_info(symbol)

        if info is None:
            return 999  # большой спред = не торговать

        return info.spread

    # ─── Индикаторы ───────────────────────────────────────

    @staticmethod
    def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return df["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_sma(df: pd.DataFrame, period: int) -> pd.Series:
        """Simple Moving Average"""
        return df["close"].rolling(window=period).mean()

    @staticmethod
    def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM и -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where(
            (plus_dm > minus_dm) & (plus_dm > 0), 0.0
        )
        minus_dm = minus_dm.where(
            (minus_dm > plus_dm) & (minus_dm > 0), 0.0
        )

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Сглаживание
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX и ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def calc_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        return {
            "upper": sma + std_dev * std,
            "middle": sma,
            "lower": sma - std_dev * std,
            "width": (sma + std_dev * std - (sma - std_dev * std)) / sma
        }

    @staticmethod
    def calc_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD"""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }

    @staticmethod
    def calc_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()

        return {"k": k, "d": d}

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет ВСЕ индикаторы в DataFrame"""
        df = df.copy()

        # EMA
        df["ema_20"] = self.calc_ema(df, 20)
        df["ema_50"] = self.calc_ema(df, 50)
        df["ema_200"] = self.calc_ema(df, 200)

        # RSI
        df["rsi"] = self.calc_rsi(df, 14)

        # ATR
        df["atr"] = self.calc_atr(df, 14)

        # ADX
        df["adx"] = self.calc_adx(df, 14)

        # Bollinger Bands
        bb = self.calc_bollinger_bands(df)
        df["bb_upper"] = bb["upper"]
        df["bb_middle"] = bb["middle"]
        df["bb_lower"] = bb["lower"]
        df["bb_width"] = bb["width"]

        # MACD
        macd = self.calc_macd(df)
        df["macd"] = macd["macd"]
        df["macd_signal"] = macd["signal"]
        df["macd_histogram"] = macd["histogram"]

        # Stochastic
        stoch = self.calc_stochastic(df)
        df["stoch_k"] = stoch["k"]
        df["stoch_d"] = stoch["d"]

        # Дополнительные фичи
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        df["momentum"] = df["close"].pct_change(10)
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Уровни для breakout
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()

        # Убираем NaN
        df.dropna(inplace=True)

        return df

    def get_processed_data(
        self,
        symbol: str = None,
        timeframe: str = None,
        count: int = 500
    ) -> Optional[pd.DataFrame]:
        """Получить данные с индикаторами — основной метод"""
        df = self.get_candles(symbol, timeframe, count)

        if df is None:
            return None

        return self.add_all_indicators(df)
