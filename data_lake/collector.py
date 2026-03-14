"""
data_lake/collector.py

Автоматический сборщик данных из MT5 в Data Lake
Запускается как отдельный сервис
"""

import time
import traceback
from datetime import datetime, timedelta
from typing import List
import os

import MetaTrader5 as mt5
import pandas as pd

from data_lake.lake import DataLake


class DataCollector:
    """
    Сборщик данных MT5 → Data Lake

    Режимы:
    1. Historical: загрузка исторических данных
    2. Live: инкрементальное обновление каждую минуту
    """

    SYMBOLS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY",
        "EURAUD", "AUDNZD", "AUDJPY",
        "CADJPY", "CHFJPY", "GBPCAD",
    ]

    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(self, lake: DataLake = None):
        self.lake = lake or DataLake()
        self.connected = False

    def connect(self) -> bool:
        """Подключение к MT5"""
        login = int(os.getenv("MT5_LOGIN", 0))
        password = os.getenv("MT5_PASSWORD", "")
        server = os.getenv("MT5_SERVER", "")

        if not mt5.initialize():
            print(f"[COLLECTOR] MT5 init failed: {mt5.last_error()}")
            return False

        if login:
            if not mt5.login(login, password, server):
                print(f"[COLLECTOR] Login failed: {mt5.last_error()}")
                return False

        self.connected = True
        print("[COLLECTOR] Connected to MT5")
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def collect_historical(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        years_back: int = 10
    ):
        """
        Загрузить исторические данные

        Может занять несколько минут для 20 пар × 10 лет
        """
        symbols = symbols or self.SYMBOLS
        timeframes = timeframes or ["M15", "H1", "H4", "D1"]

        print(f"\n[COLLECTOR] Historical download:")
        print(f"  Symbols: {len(symbols)}")
        print(f"  Timeframes: {timeframes}")
        print(f"  Years: {years_back}")

        start_time = time.time()
        total_bars = 0

        for symbol in symbols:
            for tf in timeframes:
                try:
                    bars = self._download(
                        symbol, tf, years_back
                    )
                    total_bars += bars
                except Exception as e:
                    print(f"  Error {symbol}/{tf}: {e}")

        elapsed = time.time() - start_time
        print(
            f"\n[COLLECTOR] Done: {total_bars:,} bars "
            f"in {elapsed:.1f}s"
        )

    def _download(
        self,
        symbol: str,
        timeframe: str,
        years_back: int
    ) -> int:
        """Загрузить данные одного символа/таймфрейма"""
        mt5_tf = self.TIMEFRAME_MAP.get(timeframe)
        if mt5_tf is None:
            return 0

        # Загружаем максимум данных
        rates = mt5.copy_rates_from_pos(
            symbol, mt5_tf, 0, 99999
        )

        if rates is None or len(rates) == 0:
            print(f"  {symbol}/{timeframe}: no data")
            return 0

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "tick_volume": "volume"
        }, inplace=True)

        df = df[["open", "high", "low", "close", "volume"]]

        self.lake.write_candles(symbol, timeframe, df)

        print(f"  {symbol}/{timeframe}: {len(df):,} bars")
        return len(df)

    def collect_live(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None
    ):
        """
        Инкрементальное обновление (последние бары)
        Запускать каждую минуту
        """
        symbols = symbols or self.SYMBOLS
        timeframes = timeframes or ["M15", "H1"]

        for symbol in symbols:
            for tf in timeframes:
                try:
                    mt5_tf = self.TIMEFRAME_MAP.get(tf)
                    if mt5_tf is None:
                        continue

                    rates = mt5.copy_rates_from_pos(
                        symbol, mt5_tf, 0, 10
                    )

                    if rates is None or len(rates) == 0:
                        continue

                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(
                        df["time"], unit="s"
                    )
                    df.set_index("time", inplace=True)
                    df.rename(
                        columns={"tick_volume": "volume"},
                        inplace=True
                    )
                    df = df[["open", "high", "low", "close", "volume"]]

                    self.lake.write_candles(
                        symbol, tf, df, mode="append"
                    )

                except Exception as e:
                    pass  # Тихо пропускаем ошибки при live

    def run_collector_loop(
        self,
        interval_seconds: int = 60
    ):
        """Бесконечный цикл сбора данных"""
        print(f"[COLLECTOR] Starting live collection "
              f"every {interval_seconds}s")

        while True:
            try:
                if not self.connected:
                    self.connect()

                self.collect_live()
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[COLLECTOR] Error: {e}")
                time.sleep(5)

        self.disconnect()
