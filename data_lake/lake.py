"""
data_lake/lake.py

Data Lake для хранения 20+ лет Forex данных

Структура:
  data/lake/
    forex/
      EURUSD/
        M1/
          2020.parquet
          2021.parquet
        M5/
        M15/
        H1/
        H4/
        D1/
      GBPUSD/
        ...
    features/
      EURUSD_H1_features.parquet
    models/
      impulse_model.pkl

Формат: Parquet (в 5-10x меньше CSV, в 10-20x быстрее)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import time
import json

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False


class DataLake:
    """
    Централизованное хранилище рыночных данных

    Возможности:
    1. Хранение OHLCV в Parquet (компрессия ~5-10x)
    2. Автоматическое партиционирование по годам
    3. Инкрементальное обновление
    4. Feature Store (кэш рассчитанных фич)
    5. Быстрая загрузка любого периода
    """

    TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

    COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self, base_path: str = None):
        self.base_path = Path(
            base_path
            or os.getenv("DATA_LAKE_PATH", "data/lake")
        )
        self.forex_path = self.base_path / "forex"
        self.features_path = self.base_path / "features"
        self.models_path = self.base_path / "models"

        # Создаём структуру
        for p in [self.forex_path, self.features_path, self.models_path]:
            p.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, pd.DataFrame] = {}

    # ═══════════════════════════════════════════
    #  WRITE — запись данных
    # ═══════════════════════════════════════════

    def write_candles(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        mode: str = "append"
    ):
        """
        Записать свечные данные

        mode:
          "append" — добавить новые (не дубликаты)
          "overwrite" — перезаписать
        """
        if df is None or len(df) == 0:
            return

        symbol_path = self.forex_path / symbol / timeframe
        symbol_path.mkdir(parents=True, exist_ok=True)

        # Группируем по годам
        if hasattr(df.index, 'year'):
            years = df.index.year.unique()
        else:
            years = [datetime.now().year]

        for year in years:
            year_file = symbol_path / f"{year}.parquet"

            if hasattr(df.index, 'year'):
                year_data = df[df.index.year == year]
            else:
                year_data = df

            if mode == "append" and year_file.exists():
                existing = pd.read_parquet(year_file)
                # Убираем дубликаты по индексу
                combined = pd.concat([existing, year_data])
                combined = combined[
                    ~combined.index.duplicated(keep="last")
                ]
                combined.sort_index(inplace=True)
                year_data = combined

            # Сохраняем
            year_data.to_parquet(
                year_file,
                engine="pyarrow" if HAS_ARROW else "auto",
                compression="snappy"
            )

        print(
            f"[LAKE] Saved {symbol}/{timeframe}: "
            f"{len(df)} bars"
        )

    def write_features(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame
    ):
        """Записать фичи в Feature Store"""
        fname = f"{symbol}_{timeframe}_features.parquet"
        path = self.features_path / fname

        features.to_parquet(
            path,
            compression="snappy"
        )

        print(
            f"[LAKE] Features saved: {fname} "
            f"({len(features)} rows, "
            f"{len(features.columns)} cols)"
        )

    # ═══════════════════════════════════════════
    #  READ — чтение данных
    # ═══════════════════════════════════════════

    def read_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        last_n_bars: int = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Загрузить свечные данные

        Быстрая загрузка: только нужные годы
        """
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}_{last_n_bars}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        symbol_path = self.forex_path / symbol / timeframe

        if not symbol_path.exists():
            return None

        # Находим все файлы
        parquet_files = sorted(symbol_path.glob("*.parquet"))

        if not parquet_files:
            return None

        # Фильтруем по дате
        if start_date:
            start_year = int(start_date[:4])
            parquet_files = [
                f for f in parquet_files
                if int(f.stem) >= start_year
            ]

        if end_date:
            end_year = int(end_date[:4])
            parquet_files = [
                f for f in parquet_files
                if int(f.stem) <= end_year
            ]

        if not parquet_files:
            return None

        # Читаем и объединяем
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                print(f"[LAKE] Error reading {f}: {e}")

        if not dfs:
            return None

        result = pd.concat(dfs)
        result.sort_index(inplace=True)

        # Фильтр по дате
        if start_date:
            result = result[result.index >= start_date]
        if end_date:
            result = result[result.index <= end_date]

        # Последние N баров
        if last_n_bars:
            result = result.tail(last_n_bars)

        # Кэш
        if use_cache and len(result) < 500000:
            self._cache[cache_key] = result.copy()

        return result

    def read_features(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Загрузить фичи из Feature Store"""
        fname = f"{symbol}_{timeframe}_features.parquet"
        path = self.features_path / fname

        if not path.exists():
            return None

        return pd.read_parquet(path)

    # ═══════════════════════════════════════════
    #  INFO — информация о данных
    # ═══════════════════════════════════════════

    def get_available_data(self) -> Dict:
        """Получить информацию о доступных данных"""
        info = {}

        for symbol_dir in sorted(self.forex_path.iterdir()):
            if not symbol_dir.is_dir():
                continue

            symbol = symbol_dir.name
            info[symbol] = {}

            for tf_dir in sorted(symbol_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue

                tf = tf_dir.name
                files = sorted(tf_dir.glob("*.parquet"))

                if files:
                    years = [int(f.stem) for f in files]
                    total_size = sum(f.stat().st_size for f in files)

                    info[symbol][tf] = {
                        "years": years,
                        "files": len(files),
                        "size_mb": round(total_size / 1024 / 1024, 2),
                        "range": f"{min(years)}-{max(years)}"
                    }

        return info

    def get_stats(self) -> Dict:
        """Общая статистика Data Lake"""
        data = self.get_available_data()

        total_files = 0
        total_size = 0
        symbols = list(data.keys())

        for symbol, timeframes in data.items():
            for tf, info in timeframes.items():
                total_files += info["files"]
                total_size += info["size_mb"]

        return {
            "symbols": len(symbols),
            "symbol_list": symbols,
            "total_files": total_files,
            "total_size_mb": round(total_size, 2),
            "total_size_gb": round(total_size / 1024, 2),
        }

    def print_info(self):
        """Вывести информацию о Data Lake"""
        stats = self.get_stats()
        data = self.get_available_data()

        print(f"\n{'═' * 60}")
        print(f"  DATA LAKE INFO")
        print(f"  Path: {self.base_path}")
        print(f"{'═' * 60}")

        print(f"\n  Symbols: {stats['symbols']}")
        print(f"  Files:   {stats['total_files']}")
        print(f"  Size:    {stats['total_size_gb']:.2f} GB")

        print(f"\n  {'Symbol':<10} {'TF':<6} {'Years':<15} "
              f"{'Files':<7} {'Size MB':<10}")
        print(f"  {'─' * 48}")

        for symbol, timeframes in data.items():
            for tf, info in timeframes.items():
                print(
                    f"  {symbol:<10} {tf:<6} "
                    f"{info['range']:<15} "
                    f"{info['files']:<7} "
                    f"{info['size_mb']:<10.2f}"
                )

        print(f"{'═' * 60}\n")

    # ═══════════════════════════════════════════
    #  MAINTENANCE — обслуживание
    # ═══════════════════════════════════════════

    def compact(self, symbol: str, timeframe: str):
        """
        Сжать данные (удалить дубликаты, пересортировать)
        """
        df = self.read_candles(symbol, timeframe, use_cache=False)
        if df is None:
            return

        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        after = len(df)

        self.write_candles(symbol, timeframe, df, mode="overwrite")

        print(
            f"[LAKE] Compacted {symbol}/{timeframe}: "
            f"{before} → {after} "
            f"(removed {before - after} duplicates)"
        )

    def clear_cache(self):
        """Очистить кэш"""
        self._cache.clear()

    def validate(self, symbol: str, timeframe: str) -> Dict:
        """Проверить целостность данных"""
        df = self.read_candles(symbol, timeframe, use_cache=False)

        if df is None:
            return {"valid": False, "error": "No data"}

        issues = []

        # Проверка NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"NaN values: {nan_count}")

        # Проверка дубликатов
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            issues.append(f"Duplicate timestamps: {dup_count}")

        # Проверка сортировки
        if not df.index.is_monotonic_increasing:
            issues.append("Index not sorted")

        # Проверка нулевых значений
        zero_close = (df["close"] == 0).sum()
        if zero_close > 0:
            issues.append(f"Zero close prices: {zero_close}")

        # Проверка OHLC consistency
        invalid_ohlc = (
            (df["high"] < df["low"]).sum()
            + (df["high"] < df["close"]).sum()
            + (df["high"] < df["open"]).sum()
            + (df["low"] > df["close"]).sum()
            + (df["low"] > df["open"]).sum()
        )
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC: {invalid_ohlc}")

        return {
            "valid": len(issues) == 0,
            "rows": len(df),
            "date_range": f"{df.index[0]} — {df.index[-1]}",
            "issues": issues
              }
