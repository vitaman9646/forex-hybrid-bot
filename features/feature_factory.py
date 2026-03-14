"""
features/feature_factory.py

Feature Factory — автоматическая генерация и отбор признаков

Генерирует 150+ признаков из сырых OHLCV, затем:
1. Автоматически создаёт вариации (разные периоды)
2. Добавляет cross-features (взаимодействия)
3. Отбирает лучшие (Mutual Information + стабильность)
4. Кэширует для скорости

Категории:
 Price (15) → Trend (20) → Volatility (20) → Momentum (15)
 → Volume (10) → Structure (10) → Session (10) → Statistical (15)
 → Cross (10) → Lag (25)

Итого: ~150 raw features → 20-40 selected
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class FeatureInfo:
    """Информация о фиче"""
    name: str
    category: str
    importance: float = 0.0
    stability: float = 0.0     # % фолдов где в топ-20
    correlation_group: int = 0
    is_selected: bool = False


@dataclass
class FactoryReport:
    """Отчёт Feature Factory"""
    total_generated: int = 0
    after_variance_filter: int = 0
    after_correlation_filter: int = 0
    after_importance_filter: int = 0
    final_selected: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    top_features: List[FeatureInfo] = field(default_factory=list)
    build_time_sec: float = 0.0


class FeatureFactory:
    """
    Автоматическая фабрика признаков

    Использование:
        factory = FeatureFactory()
        features = factory.build(df)
        selected = factory.select(features, target)
    """

    # Периоды для автоматических вариаций
    EMA_PERIODS = [5, 10, 20, 50, 100, 200]
    RETURN_PERIODS = [1, 2, 3, 5, 10, 20, 50]
    VOL_WINDOWS = [5, 10, 20, 50]
    RSI_PERIODS = [7, 14, 21]
    BB_PERIODS = [10, 20, 50]

    def __init__(
        self,
        max_features: int = 35,
        max_correlation: float = 0.85,
        min_variance: float = 1e-8,
        cache_dir: str = "data/feature_cache"
    ):
        self.max_features = max_features
        self.max_correlation = max_correlation
        self.min_variance = min_variance
        self.cache_dir = cache_dir

        self.feature_info: Dict[str, FeatureInfo] = {}
        self.selected_names: List[str] = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None

    # ═══════════════════════════════════════════
    #  BUILD — генерация всех фич
    # ═══════════════════════════════════════════

    def build(
        self,
        df: pd.DataFrame,
        cross_pair_data: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Построить все признаки из сырых данных

        Args:
            df: основной DataFrame (OHLCV + индикаторы)
            cross_pair_data: данные других пар для cross-features
        """
        import time as _time
        start = _time.time()

        if len(df) < 200:
            return pd.DataFrame()

        c = df["close"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        n = len(c)

        v = (
            df["volume"].values.astype(np.float64)
            if "volume" in df.columns
            else np.ones(n)
        )

        feat = pd.DataFrame(index=df.index)

        # ═══ PRICE FEATURES ═══
        price_feats = self._build_price_features(o, h, l, c, n)
        for name, values in price_feats.items():
            feat[name] = values
            self._register(name, "price")

        # ═══ TREND FEATURES ═══
        trend_feats = self._build_trend_features(c, n)
        for name, values in trend_feats.items():
            feat[name] = values
            self._register(name, "trend")

        # ═══ VOLATILITY FEATURES ═══
        vol_feats = self._build_volatility_features(h, l, c, n)
        for name, values in vol_feats.items():
            feat[name] = values
            self._register(name, "volatility")

        # ═══ MOMENTUM FEATURES ═══
        mom_feats = self._build_momentum_features(h, l, c, n)
        for name, values in mom_feats.items():
            feat[name] = values
            self._register(name, "momentum")

        # ═══ VOLUME FEATURES ═══
        vol_f = self._build_volume_features(v, c, n)
        for name, values in vol_f.items():
            feat[name] = values
            self._register(name, "volume")

        # ═══ MARKET STRUCTURE ═══
        struct_feats = self._build_structure_features(h, l, c, n)
        for name, values in struct_feats.items():
            feat[name] = values
            self._register(name, "structure")

        # ═══ SESSION FEATURES ═══
        if hasattr(df.index, 'hour'):
            sess_feats = self._build_session_features(df.index)
            for name, values in sess_feats.items():
                feat[name] = values
                self._register(name, "session")

        # ═══ STATISTICAL FEATURES ═══
        stat_feats = self._build_statistical_features(c, n)
        for name, values in stat_feats.items():
            feat[name] = values
            self._register(name, "statistical")

        # ═══ CROSS-PAIR FEATURES ═══
        if cross_pair_data:
            cross_feats = self._build_cross_features(
                c, cross_pair_data, n
            )
            for name, values in cross_feats.items():
                feat[name] = values
                self._register(name, "cross_pair")

        # ═══ LAG FEATURES ═══
        lag_feats = self._build_lag_features(feat, n)
        for name, values in lag_feats.items():
            feat[name] = values
            self._register(name, "lag")

        # Clean
        feat = feat.replace([np.inf, -np.inf], 0)
        feat = feat.fillna(0)

        build_time = _time.time() - start

        print(
            f"[FACTORY] Built {len(feat.columns)} features "
            f"in {build_time:.2f}s"
        )

        return feat

    # ─── Price Features ──────────────────────

    def _build_price_features(self, o, h, l, c, n):
        feats = {}

        # Returns (разные периоды)
        for p in self.RETURN_PERIODS:
            ret = np.zeros(n)
            ret[p:] = (c[p:] - c[:-p]) / np.where(
                c[:-p] != 0, c[:-p], 1
            )
            feats[f"ret_{p}"] = ret

        # Log returns
        feats["log_ret_1"] = np.concatenate([
            [0],
            np.log(np.where(c[:-1] > 0, c[1:] / c[:-1], 1))
        ])

        # Gaps
        feats["gap"] = np.concatenate([
            [0],
            (o[1:] - c[:-1]) / np.where(c[:-1] != 0, c[:-1], 1)
        ])

        # Candle anatomy
        ranges = np.where(h - l > 0, h - l, 1e-10)
        feats["body_ratio"] = np.abs(c - o) / ranges
        feats["upper_wick"] = (h - np.maximum(c, o)) / ranges
        feats["lower_wick"] = (np.minimum(c, o) - l) / ranges

        # Direction
        feats["candle_dir"] = np.sign(c - o)

        return feats

    # ─── Trend Features ─────────────────────

    def _build_trend_features(self, c, n):
        feats = {}

        emas = {}
        for p in self.EMA_PERIODS:
            ema = self._ema(c, p)
            emas[p] = ema

            # Distance
            feats[f"ema{p}_dist"] = (c - ema) / np.where(
                ema != 0, ema, 1
            )

        # Slopes
        for p in [20, 50]:
            if p in emas:
                slope = np.zeros(n)
                lag = min(p // 2, 10)
                slope[lag:] = (
                    emas[p][lag:] - emas[p][:-lag]
                ) / np.where(
                    emas[p][:-lag] != 0, emas[p][:-lag], 1
                )
                feats[f"ema{p}_slope"] = slope

        # EMA crosses
        if 20 in emas and 50 in emas:
            feats["ema_cross_20_50"] = (
                emas[20] - emas[50]
            ) / np.where(emas[50] != 0, emas[50], 1)

        if 50 in emas and 200 in emas:
            feats["ema_cross_50_200"] = (
                emas[50] - emas[200]
            ) / np.where(emas[200] != 0, emas[200], 1)

        # Linear regression slope
        for w in [10, 20, 50]:
            feats[f"linreg_slope_{w}"] = self._rolling_slope(c, w)

        return feats

    # ─── Volatility Features ────────────────

    def _build_volatility_features(self, h, l, c, n):
        feats = {}

        # ATR
        atr = self._atr(h, l, c, 14)
        feats["atr_14"] = atr

        atr_sma = self._sma(atr, 20)
        feats["atr_ratio"] = np.where(
            atr_sma > 0, atr / atr_sma, 1
        )

        # ATR change
        feats["atr_change_5"] = np.concatenate([
            np.zeros(5),
            (atr[5:] - atr[:-5]) / np.where(
                atr[:-5] > 0, atr[:-5], 1
            )
        ])

        # ATR percentile
        feats["atr_pct"] = self._rolling_percentile(atr, 100)

        # Realized volatility
        for w in self.VOL_WINDOWS:
            feats[f"rvol_{w}"] = self._realized_vol(c, w)

        # Bollinger Bands
        for p in self.BB_PERIODS:
            bbw = self._bb_width(c, p)
            feats[f"bb_width_{p}"] = bbw

        # BB squeeze
        bbw_20 = feats.get("bb_width_20", np.zeros(n))
        bb_min = self._rolling_min(bbw_20, 50)
        feats["bb_squeeze"] = (
            bbw_20 <= bb_min * 1.1
        ).astype(float)

        # BB squeeze duration
        squeeze = feats["bb_squeeze"]
        duration = np.zeros(n)
        for i in range(1, n):
            if squeeze[i] > 0:
                duration[i] = duration[i - 1] + 1
        feats["squeeze_duration"] = duration

        return feats

    # ─── Momentum Features ──────────────────

    def _build_momentum_features(self, h, l, c, n):
        feats = {}

        # RSI
        for p in self.RSI_PERIODS:
            feats[f"rsi_{p}"] = self._rsi(c, p)

        # RSI change
        rsi_14 = feats.get("rsi_14", np.full(n, 50))
        feats["rsi_change_5"] = np.concatenate([
            np.zeros(5), rsi_14[5:] - rsi_14[:-5]
        ])

        # Stochastic
        feats["stoch_14"] = self._stochastic(h, l, c, 14)

        # MACD histogram
        ema12 = self._ema(c, 12)
        ema26 = self._ema(c, 26)
        macd = ema12 - ema26
        macd_sig = self._ema(macd, 9)
        hist = macd - macd_sig
        feats["macd_hist"] = hist

        feats["macd_hist_change"] = np.concatenate([
            [0], np.diff(hist)
        ])

        # Momentum acceleration
        for p in [5, 10]:
            mom = np.zeros(n)
            mom[p:] = (c[p:] - c[:-p]) / np.where(
                c[:-p] != 0, c[:-p], 1
            )
            accel = np.zeros(n)
            accel[p:] = mom[p:] - mom[:-p]
            feats[f"mom_accel_{p}"] = accel

        return feats

    # ─── Volume Features ────────────────────

    def _build_volume_features(self, v, c, n):
        feats = {}

        vol_sma20 = self._sma(v, 20)
        feats["vol_ratio"] = np.where(
            vol_sma20 > 0, v / vol_sma20, 1
        )

        vol_sma5 = self._sma(v, 5)
        feats["vol_trend"] = np.where(
            vol_sma20 > 0, vol_sma5 / vol_sma20, 1
        )

        feats["vol_spike"] = (
            feats["vol_ratio"] > 2.0
        ).astype(float)

        # OBV direction proxy
        obv_dir = np.zeros(n)
        for i in range(1, n):
            if c[i] > c[i-1]:
                obv_dir[i] = v[i]
            elif c[i] < c[i-1]:
                obv_dir[i] = -v[i]
        obv_sma = self._sma(obv_dir, 20)
        feats["obv_trend"] = np.sign(obv_sma)

        # Volume-price correlation
        feats["vol_price_corr"] = self._rolling_corr(c, v, 20)

        return feats

    # ─── Structure Features ─────────────────

    def _build_structure_features(self, h, l, c, n):
        feats = {}

        for w in [20, 50]:
            high_w = self._rolling_max(h, w)
            low_w = self._rolling_min(l, w)

            feats[f"dist_high_{w}"] = np.where(
                high_w > 0, (high_w - c) / high_w, 0
            )
            feats[f"dist_low_{w}"] = np.where(
                low_w > 0, (c - low_w) / np.where(
                    low_w != 0, low_w, 1
                ), 0
            )

            rng = high_w - low_w
            feats[f"pos_in_range_{w}"] = np.where(
                rng > 0, (c - low_w) / rng, 0.5
            )

        # New high/low
        feats["new_high_20"] = (c >= self._rolling_max(h, 20)).astype(float)
        feats["new_low_20"] = (c <= self._rolling_min(l, 20)).astype(float)

        return feats

    # ─── Session Features ───────────────────

    def _build_session_features(self, index):
        feats = {}
        hours = index.hour

        # Circular encoding
        feats["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        # Day of week
        dow = index.dayofweek
        feats["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        feats["dow_cos"] = np.cos(2 * np.pi * dow / 5)

        # Session flags
        feats["is_asia"] = ((hours >= 0) & (hours < 7)).astype(float)
        feats["is_london"] = ((hours >= 7) & (hours < 16)).astype(float)
        feats["is_ny"] = ((hours >= 12) & (hours < 21)).astype(float)
        feats["is_overlap"] = ((hours >= 12) & (hours < 16)).astype(float)

        # Killzone flags
        feats["kz_lo"] = ((hours >= 7) & (hours < 10)).astype(float)
        feats["kz_nyo"] = ((hours >= 13) & (hours < 16)).astype(float)

        return feats

    # ─── Statistical Features ───────────────

    def _build_statistical_features(self, c, n):
        feats = {}

        rets = np.zeros(n)
        rets[1:] = (c[1:] - c[:-1]) / np.where(
            c[:-1] != 0, c[:-1], 1
        )

        for w in [10, 20]:
            feats[f"skew_{w}"] = self._rolling_stat(rets, w, "skew")
            feats[f"kurt_{w}"] = self._rolling_stat(rets, w, "kurt")

        feats["autocorr_20"] = self._autocorrelation(rets, 20)
        feats["hurst_50"] = self._rolling_hurst(c, 50)

        # Entropy
        feats["entropy_20"] = self._rolling_entropy(rets, 20)

        # Z-score
        for w in [20, 50]:
            sma = self._sma(c, w)
            std = np.zeros(n)
            for i in range(w, n):
                std[i] = np.std(c[i-w:i])
            feats[f"zscore_{w}"] = np.where(
                std > 0, (c - sma) / std, 0
            )

        return feats

    # ─── Cross-Pair Features ────────────────

    def _build_cross_features(
        self,
        c: np.ndarray,
        cross_data: Dict[str, pd.DataFrame],
        n: int
    ) -> Dict:
        feats = {}

        for pair_name, pair_df in cross_data.items():
            if len(pair_df) < n:
                continue

            pair_c = pair_df["close"].values[-n:]

            # Rolling correlation
            corr = self._rolling_corr(c, pair_c, 50)
            safe_name = pair_name.replace("/", "").lower()
            feats[f"corr_{safe_name}_50"] = corr

            # Relative strength
            ret_main = np.zeros(n)
            ret_main[20:] = (c[20:] - c[:-20]) / np.where(
                c[:-20] != 0, c[:-20], 1
            )
            ret_pair = np.zeros(n)
            ret_pair[20:] = (
                pair_c[20:] - pair_c[:-20]
            ) / np.where(
                pair_c[:-20] != 0, pair_c[:-20], 1
            )
            feats[f"relstr_{safe_name}"] = ret_main - ret_pair

        return feats

    # ─── Lag Features ───────────────────────

    def _build_lag_features(
        self,
        feat: pd.DataFrame,
        n: int
    ) -> Dict:
        """Добавить лаги ключевых фич"""
        lag_feats = {}

        key_features = [
            "ret_1", "vol_ratio", "bb_squeeze",
            "rsi_14", "macd_hist"
        ]

        for fname in key_features:
            if fname not in feat.columns:
                continue

            values = feat[fname].values

            for lag in [1, 3, 5]:
                lagged = np.zeros(n)
                lagged[lag:] = values[:-lag]
                lag_feats[f"{fname}_lag{lag}"] = lagged

        return lag_feats

    # ═══════════════════════════════════════════
    #  SELECT — отбор лучших фич
    # ═══════════════════════════════════════════

    def select(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        method: str = "mutual_info"
    ) -> Tuple[List[str], FactoryReport]:
        """
        Отбор лучших признаков

        Pipeline:
        1. Variance filter (удалить константы)
        2. Correlation filter (удалить дубликаты)
        3. Importance ranking (MI или model-based)
        4. Top N selection
        """
        report = FactoryReport()
        report.total_generated = len(features.columns)

        all_feats = list(features.columns)

        # Step 1: Variance filter
        variances = features.var()
        alive = [
            f for f in all_feats
            if variances.get(f, 0) > self.min_variance
        ]
        report.after_variance_filter = len(alive)

        # Step 2: Correlation filter
        alive = self._correlation_filter(features[alive])
        report.after_correlation_filter = len(alive)

        # Step 3: Importance ranking
        if HAS_SKLEARN and method == "mutual_info":
            mi = mutual_info_classif(
                features[alive].values,
                target,
                random_state=42,
                n_neighbors=5
            )
            importance = dict(zip(alive, mi))
        else:
            # Fallback: variance-based
            importance = {
                f: float(features[f].var()) for f in alive
            }

        # Sort by importance
        ranked = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Filter low importance
        min_imp = np.percentile(
            list(importance.values()), 25
        )
        ranked = [
            (f, imp) for f, imp in ranked
            if imp > min_imp
        ]
        report.after_importance_filter = len(ranked)

        # Top N
        selected = [f for f, _ in ranked[:self.max_features]]
        report.final_selected = len(selected)

        # Update feature info
        for f, imp in ranked[:self.max_features]:
            if f in self.feature_info:
                self.feature_info[f].importance = float(imp)
                self.feature_info[f].is_selected = True

        # Category breakdown
        for f in selected:
            info = self.feature_info.get(f)
            if info:
                cat = info.category
                report.categories[cat] = (
                    report.categories.get(cat, 0) + 1
                )

        report.top_features = [
            self.feature_info[f]
            for f in selected
            if f in self.feature_info
        ]

        self.selected_names = selected

        print(
            f"[FACTORY] Selected {report.final_selected} / "
            f"{report.total_generated} features"
        )
        print(f"  Categories: {report.categories}")

        return selected, report

    def _correlation_filter(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        """Удалить высоко-коррелированные фичи"""
        cols = list(df.columns)

        if len(cols) <= 1:
            return cols

        # Быстрый расчёт корреляционной матрицы
        corr = df.corr().abs()
        to_remove = set()

        for i in range(len(cols)):
            if cols[i] in to_remove:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove:
                    continue
                if corr.iloc[i, j] > self.max_correlation:
                    # Удаляем фичу с меньшей дисперсией
                    if df[cols[i]].var() >= df[cols[j]].var():
                        to_remove.add(cols[j])
                    else:
                        to_remove.add(cols[i])
                        break

        result = [c for c in cols if c not in to_remove]

        removed = len(cols) - len(result)
        if removed > 0:
            print(f"  Correlation filter removed {removed} features")

        return result

    def _register(self, name: str, category: str):
        """Зарегистрировать фичу"""
        self.feature_info[name] = FeatureInfo(
            name=name, category=category
        )

    # ═══════════════════════════════════════════
    #  TRANSFORM — применить к новым данным
    # ═══════════════════════════════════════════

    def transform(
        self,
        df: pd.DataFrame,
        cross_pair_data: Dict[str, pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        Построить фичи и вернуть только selected

        Для использования в live-торговле
        """
        all_features = self.build(df, cross_pair_data)

        if all_features.empty:
            return None

        if not self.selected_names:
            return all_features

        available = [
            f for f in self.selected_names
            if f in all_features.columns
        ]

        return all_features[available]

    # ═══════════════════════════════════════════
    #  SAVE / LOAD
    # ═══════════════════════════════════════════

    def save(self, path: str = "data/feature_factory.pkl"):
        """Сохранить состояние"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "selected": self.selected_names,
                "info": {
                    k: {
                        "name": v.name,
                        "category": v.category,
                        "importance": v.importance,
                        "stability": v.stability,
                        "is_selected": v.is_selected
                    }
                    for k, v in self.feature_info.items()
                },
                "scaler": self.scaler,
                "time": datetime.now().isoformat()
            }, f)
        print(f"[FACTORY] Saved → {path}")

    def load(self, path: str = "data/feature_factory.pkl"):
        """Загрузить"""
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.selected_names = data["selected"]
            for k, v in data.get("info", {}).items():
                self.feature_info[k] = FeatureInfo(**v)
            self.scaler = data.get("scaler")
            print(f"[FACTORY] Loaded {len(self.selected_names)} features")
        except Exception as e:
            print(f"[FACTORY] Load error: {e}")

    # ═══════════════════════════════════════════
    #  Math helpers (same as AdvancedFeatureBuilder)
    # ═══════════════════════════════════════════

    @staticmethod
    def _ema(data, period):
        out = np.empty(len(data))
        out[0] = data[0]
        a = 2.0 / (period + 1)
        for i in range(1, len(data)):
            out[i] = a * data[i] + (1 - a) * out[i - 1]
        return out

    @staticmethod
    def _sma(data, period):
        n = len(data)
        out = np.zeros(n)
        for i in range(period - 1, n):
            out[i] = np.mean(data[i - period + 1:i + 1])
        return out

    @staticmethod
    def _atr(h, l, c, period):
        n = len(h)
        tr = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        out = np.zeros(n)
        if n >= period:
            out[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                out[i] = (out[i-1] * (period-1) + tr[i]) / period
        return out

    @staticmethod
    def _rsi(c, period):
        n = len(c)
        out = np.full(n, 50.0)
        d = np.diff(c)
        g = np.where(d > 0, d, 0)
        lo = np.where(d < 0, -d, 0)
        if len(d) < period:
            return out
        ag = np.mean(g[:period])
        al = np.mean(lo[:period])
        for i in range(period, len(d)):
            ag = (ag * (period - 1) + g[i]) / period
            al = (al * (period - 1) + lo[i]) / period
            out[i+1] = 100 - 100/(1+ag/al) if al > 0 else 100
        return out

    @staticmethod
    def _stochastic(h, l, c, period):
        n = len(c)
        out = np.full(n, 50.0)
        for i in range(period - 1, n):
            hh = np.max(h[i-period+1:i+1])
            ll = np.min(l[i-period+1:i+1])
            if hh - ll > 0:
                out[i] = (c[i] - ll) / (hh - ll) * 100
        return out

    @staticmethod
    def _bb_width(c, period):
        n = len(c)
        out = np.zeros(n)
        for i in range(period - 1, n):
            w = c[i-period+1:i+1]
            s = np.mean(w)
            out[i] = 4 * np.std(w) / s if s > 0 else 0
        return out

    @staticmethod
    def _realized_vol(c, w):
        n = len(c)
        out = np.zeros(n)
        r = np.zeros(n)
        r[1:] = np.log(np.where(c[:-1]>0, c[1:]/c[:-1], 1))
        for i in range(w, n):
            out[i] = np.std(r[i-w:i]) * np.sqrt(252)
        return out

    @staticmethod
    def _rolling_max(data, w):
        n = len(data)
        out = np.copy(data)
        for i in range(1, n):
            s = max(0, i - w + 1)
            out[i] = np.max(data[s:i+1])
        return out

    @staticmethod
    def _rolling_min(data, w):
        n = len(data)
        out = np.copy(data)
        for i in range(1, n):
            s = max(0, i - w + 1)
            out[i] = np.min(data[s:i+1])
        return out

    @staticmethod
    def _rolling_percentile(data, w):
        n = len(data)
        out = np.full(n, 50.0)
        for i in range(w, n):
            out[i] = np.sum(data[i-w:i] < data[i]) / w * 100
        return out

    @staticmethod
    def _rolling_slope(data, w):
        n = len(data)
        out = np.zeros(n)
        x = np.arange(w, dtype=np.float64)
        xm = np.mean(x)
        xv = np.sum((x - xm)**2)
        for i in range(w-1, n):
            y = data[i-w+1:i+1]
            ym = np.mean(y)
            if xv > 0:
                out[i] = np.sum((x-xm)*(y-ym)) / xv / max(abs(ym), 1e-10)
        return out

    @staticmethod
    def _rolling_corr(a, b, w):
        n = len(a)
        out = np.zeros(n)
        for i in range(w, n):
            wa, wb = a[i-w:i], b[i-w:i]
            sa, sb = np.std(wa), np.std(wb)
            if sa > 0 and sb > 0:
                out[i] = np.corrcoef(wa, wb)[0, 1]
        return out

    @staticmethod
    def _rolling_stat(data, w, stat):
        n = len(data)
        out = np.zeros(n)
        for i in range(w, n):
            d = data[i-w:i]
            m, s = np.mean(d), np.std(d)
            if s > 0:
                c = (d - m) / s
                if stat == "skew":
                    out[i] = np.mean(c**3)
                elif stat == "kurt":
                    out[i] = np.mean(c**4) - 3
        return out

    @staticmethod
    def _autocorrelation(data, w):
        n = len(data)
        out = np.zeros(n)
        for i in range(w+1, n):
            r1, r2 = data[i-w:i], data[i-w-1:i-1]
            s1, s2 = np.std(r1), np.std(r2)
            if s1 > 0 and s2 > 0:
                out[i] = np.corrcoef(r1, r2)[0, 1]
        return out

    @staticmethod
    def _rolling_hurst(data, w):
        n = len(data)
        out = np.full(n, 0.5)
        for i in range(w, n):
            d = data[i-w:i]
            r = np.diff(np.log(np.where(d > 0, d, 1)))
            if len(r) < 10:
                continue
            m = np.mean(r)
            cum = np.cumsum(r - m)
            R = np.max(cum) - np.min(cum)
            S = np.std(r)
            if S > 0 and R > 0:
                out[i] = np.log(R/S) / np.log(w)
        return out

    @staticmethod
    def _rolling_entropy(data, w):
        n = len(data)
        out = np.zeros(n)
        for i in range(w, n):
            d = data[i-w:i]
            if np.std(d) == 0:
                continue
            hist, _ = np.histogram(d, bins=10)
            p = hist / np.sum(hist)
            p = p[p > 0]
            out[i] = -np.sum(p * np.log2(p))
        return out
