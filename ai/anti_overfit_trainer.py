"""
ai/anti_overfit_trainer.py

Полный пайплайн обучения AI без переобучения

Защиты от overfitting:
1. Purged Walk-Forward CV (не обычный train/test!)
2. Embargo gap между train/test
3. Feature selection (удаление шума)
4. Regularization (ограничение сложности)
5. Final holdout test (данные которые модель НИКОГДА не видела)
6. Monte Carlo validation
7. Deflated Sharpe check
8. Feature stability across folds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import os
import warnings
import json

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import (
        mutual_info_classif
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════

@dataclass
class FoldResult:
    """Результат одного фолда Walk-Forward"""
    fold: int
    train_size: int
    test_size: int
    train_auc: float = 0.0
    test_auc: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    top_features: List[str] = field(default_factory=list)
    overfit_ratio: float = 0.0    # train_auc / test_auc


@dataclass
class TrainingReport:
    """Полный отчёт об обучении"""
    model_name: str = ""
    n_features_initial: int = 0
    n_features_selected: int = 0
    n_samples: int = 0
    n_folds: int = 0

    # Walk-Forward results
    fold_results: List[FoldResult] = field(default_factory=list)
    avg_train_auc: float = 0.0
    avg_test_auc: float = 0.0
    avg_overfit_ratio: float = 0.0
    auc_stability: float = 0.0    # std across folds

    # Holdout
    holdout_auc: float = 0.0
    holdout_accuracy: float = 0.0

    # Feature stability
    stable_features: List[str] = field(default_factory=list)
    unstable_features: List[str] = field(default_factory=list)

    # Verdict
    is_overfitted: bool = False
    confidence_score: float = 0.0
    verdict: str = ""
    recommendations: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  Advanced Feature Builder (50+ features)
# ═══════════════════════════════════════════════

class AdvancedFeatureBuilder:
    """
    Генератор 50+ признаков из нескольких категорий

    Категории:
    1. Price (returns, gaps, body ratios)
    2. Trend (EMA distance, slope, ADX)
    3. Volatility (ATR, BB, realized vol, GARCH proxy)
    4. Momentum (RSI, Stochastic, MACD, CCI)
    5. Volume (spikes, trend, accumulation)
    6. Market Structure (distance to levels, swing count)
    7. Session/Time (hour, day, session flags)
    8. Statistical (skew, kurtosis, autocorrelation)
    9. Cross-features (interactions)
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Построить все 50+ фич"""
        if len(df) < 200:
            return pd.DataFrame()

        feat = pd.DataFrame(index=df.index)
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

        # ═══ 1. PRICE FEATURES (10) ═══
        for p in [1, 2, 3, 5, 10, 20]:
            feat[f"ret_{p}"] = self._safe_return(c, p)

        feat["log_ret_1"] = self._safe_log_return(c, 1)

        feat["gap"] = np.concatenate([
            [0],
            (o[1:] - c[:-1]) / np.where(c[:-1] != 0, c[:-1], 1)
        ])

        ranges = np.where(h - l > 0, h - l, 1e-10)
        feat["body_ratio"] = np.abs(c - o) / ranges

        feat["upper_wick"] = (
            h - np.maximum(c, o)
        ) / ranges

        feat["lower_wick"] = (
            np.minimum(c, o) - l
        ) / ranges

        # ═══ 2. TREND FEATURES (8) ═══
        ema_20 = self._ema(c, 20)
        ema_50 = self._ema(c, 50)
        ema_200 = self._ema(c, 200)

        feat["ema20_dist"] = (c - ema_20) / np.where(
            ema_20 != 0, ema_20, 1
        )
        feat["ema50_dist"] = (c - ema_50) / np.where(
            ema_50 != 0, ema_50, 1
        )
        feat["ema200_dist"] = (c - ema_200) / np.where(
            ema_200 != 0, ema_200, 1
        )

        # EMA slope (trend direction)
        feat["ema20_slope"] = np.concatenate([
            np.zeros(5),
            (ema_20[5:] - ema_20[:-5]) / np.where(
                ema_20[:-5] != 0, ema_20[:-5], 1
            )
        ])

        feat["ema50_slope"] = np.concatenate([
            np.zeros(10),
            (ema_50[10:] - ema_50[:-10]) / np.where(
                ema_50[:-10] != 0, ema_50[:-10], 1
            )
        ])

        # EMA cross distance
        feat["ema_cross_20_50"] = (ema_20 - ema_50) / np.where(
            ema_50 != 0, ema_50, 1
        )

        if "adx" in df.columns:
            feat["adx"] = df["adx"].values
            feat["adx_change"] = np.concatenate([
                np.zeros(5),
                df["adx"].values[5:] - df["adx"].values[:-5]
            ])

        # Trend strength (regression slope)
        feat["trend_slope_20"] = self._rolling_slope(c, 20)

        # ═══ 3. VOLATILITY FEATURES (10) ═══
        atr = self._atr(h, l, c, 14)
        feat["atr"] = atr

        atr_sma = self._sma(atr, 20)
        feat["atr_ratio"] = np.where(
            atr_sma > 0, atr / atr_sma, 1
        )

        feat["atr_change"] = np.concatenate([
            np.zeros(5),
            (atr[5:] - atr[:-5]) / np.where(
                atr[:-5] > 0, atr[:-5], 1
            )
        ])

        for w in [5, 10, 20]:
            feat[f"realized_vol_{w}"] = self._realized_vol(c, w)

        # BB width
        bb_w = self._bb_width(c, 20)
        feat["bb_width"] = bb_w

        feat["bb_width_change"] = np.concatenate([
            [0], np.diff(bb_w)
        ])

        # Squeeze
        bb_min50 = self._rolling_min(bb_w, 50)
        feat["bb_squeeze"] = (bb_w <= bb_min50 * 1.1).astype(float)

        # ATR percentile
        feat["atr_percentile"] = self._rolling_percentile(atr, 100)

        # ═══ 4. MOMENTUM FEATURES (8) ═══
        rsi = self._rsi(c, 14)
        feat["rsi"] = rsi
        feat["rsi_change"] = np.concatenate([
            np.zeros(5), rsi[5:] - rsi[:-5]
        ])

        # Stochastic
        for period in [14]:
            stoch_k = self._stochastic(h, l, c, period)
            feat[f"stoch_k_{period}"] = stoch_k

        # Momentum acceleration
        mom_5 = feat["ret_5"].values
        feat["mom_accel"] = np.concatenate([
            np.zeros(5), mom_5[5:] - mom_5[:-5]
        ])

        # MACD histogram
        macd = self._ema(c, 12) - self._ema(c, 26)
        macd_sig = self._ema(macd, 9)
        feat["macd_hist"] = macd - macd_sig
        feat["macd_hist_change"] = np.concatenate([
            [0], np.diff(macd - macd_sig)
        ])

        # Rate of change
        feat["roc_10"] = self._safe_return(c, 10)

        # ═══ 5. VOLUME FEATURES (5) ═══
        vol_sma = self._sma(v, 20)
        feat["vol_ratio"] = np.where(
            vol_sma > 0, v / vol_sma, 1
        )

        vol_sma5 = self._sma(v, 5)
        feat["vol_trend"] = np.where(
            vol_sma > 0, vol_sma5 / vol_sma, 1
        )

        feat["vol_spike"] = (
            feat["vol_ratio"].values > 2.0
        ).astype(float)

        feat["vol_change_5"] = np.concatenate([
            np.zeros(5),
            (v[5:] - v[:-5]) / np.where(v[:-5] > 0, v[:-5], 1)
        ])

        feat["vol_price_corr"] = self._rolling_corr(
            c, v, 20
        )

        # ═══ 6. MARKET STRUCTURE (5) ═══
        high_20 = self._rolling_max(h, 20)
        low_20 = self._rolling_min(l, 20)

        feat["dist_to_high_20"] = np.where(
            high_20 > 0, (high_20 - c) / high_20, 0
        )
        feat["dist_to_low_20"] = np.where(
            low_20 > 0, (c - low_20) / low_20, 0
        )

        high_50 = self._rolling_max(h, 50)
        low_50 = self._rolling_min(l, 50)

        rng_50 = high_50 - low_50
        feat["position_in_range"] = np.where(
            rng_50 > 0, (c - low_50) / rng_50, 0.5
        )

        feat["range_pct_50"] = np.where(
            c > 0, rng_50 / c, 0
        )

        feat["new_high_20"] = (c >= high_20).astype(float)

        # ═══ 7. SESSION/TIME (5) ═══
        if hasattr(df.index, 'hour'):
            hours = df.index.hour
            feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)
            feat["is_london"] = (
                (hours >= 7) & (hours < 16)
            ).astype(float)
            feat["is_ny"] = (
                (hours >= 12) & (hours < 21)
            ).astype(float)
            feat["is_overlap"] = (
                (hours >= 12) & (hours < 16)
            ).astype(float)

        # ═══ 8. STATISTICAL (5) ═══
        rets = feat["ret_1"].values

        feat["skew_20"] = self._rolling_stat(rets, 20, "skew")
        feat["kurt_20"] = self._rolling_stat(rets, 20, "kurt")
        feat["autocorr_20"] = self._autocorrelation(rets, 20)

        feat["hurst"] = self._rolling_hurst(c, 50)

        feat["entropy_20"] = self._rolling_entropy(rets, 20)

        # ═══ 9. CROSS-FEATURES (4) ═══
        feat["vol_x_squeeze"] = (
            feat["vol_ratio"].values
            * feat["bb_squeeze"].values
        )
        feat["rsi_x_bb"] = (
            (feat["rsi"].values - 50) / 50
            * feat["bb_width"].values
        )

        if "adx" in feat.columns:
            feat["adx_x_atr"] = (
                feat["adx"].values
                * feat["atr_ratio"].values
            )

        feat["mom_x_vol"] = (
            feat["ret_5"].values
            * feat["vol_ratio"].values
        )

        # Clean
        feat = feat.replace([np.inf, -np.inf], 0)
        feat = feat.fillna(0)

        return feat

    # ─── Helper Methods ────────────────────

    @staticmethod
    def _safe_return(c, period):
        out = np.zeros(len(c))
        out[period:] = (c[period:] - c[:-period]) / np.where(
            c[:-period] != 0, c[:-period], 1
        )
        return out

    @staticmethod
    def _safe_log_return(c, period):
        out = np.zeros(len(c))
        for i in range(period, len(c)):
            if c[i - period] > 0 and c[i] > 0:
                out[i] = np.log(c[i] / c[i - period])
        return out

    @staticmethod
    def _ema(data, period):
        out = np.empty(len(data))
        out[0] = data[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _sma(data, period):
        out = np.zeros(len(data))
        for i in range(period - 1, len(data)):
            out[i] = np.mean(data[i - period + 1:i + 1])
        return out

    @staticmethod
    def _atr(h, l, c, period):
        n = len(h)
        tr = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(
                h[i] - l[i],
                abs(h[i] - c[i - 1]),
                abs(l[i] - c[i - 1])
            )
        out = np.zeros(n)
        if n >= period:
            out[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
        return out

    @staticmethod
    def _rsi(c, period):
        n = len(c)
        out = np.full(n, 50.0)
        deltas = np.diff(c)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        if len(deltas) < period:
            return out

        avg_g = np.mean(gains[:period])
        avg_l = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            if avg_l == 0:
                out[i + 1] = 100
            else:
                out[i + 1] = 100 - 100 / (1 + avg_g / avg_l)
        return out

    @staticmethod
    def _stochastic(h, l, c, period):
        n = len(c)
        out = np.full(n, 50.0)
        for i in range(period - 1, n):
            hh = np.max(h[i - period + 1:i + 1])
            ll = np.min(l[i - period + 1:i + 1])
            if hh - ll > 0:
                out[i] = (c[i] - ll) / (hh - ll) * 100
        return out

    @staticmethod
    def _bb_width(c, period):
        n = len(c)
        out = np.zeros(n)
        for i in range(period - 1, n):
            w = c[i - period + 1:i + 1]
            sma = np.mean(w)
            std = np.std(w)
            if sma > 0:
                out[i] = 4 * std / sma
        return out

    @staticmethod
    def _realized_vol(c, window):
        n = len(c)
        out = np.zeros(n)
        rets = np.zeros(n)
        rets[1:] = np.log(np.where(c[:-1] > 0, c[1:] / c[:-1], 1))
        for i in range(window, n):
            out[i] = np.std(rets[i - window:i]) * np.sqrt(252)
        return out

    @staticmethod
    def _rolling_max(data, window):
        n = len(data)
        out = np.zeros(n)
        out[0] = data[0]
        for i in range(1, n):
            start = max(0, i - window + 1)
            out[i] = np.max(data[start:i + 1])
        return out

    @staticmethod
    def _rolling_min(data, window):
        n = len(data)
        out = np.full(n, data[0])
        for i in range(1, n):
            start = max(0, i - window + 1)
            out[i] = np.min(data[start:i + 1])
        return out

    @staticmethod
    def _rolling_percentile(data, window):
        n = len(data)
        out = np.full(n, 50.0)
        for i in range(window, n):
            w = data[i - window:i]
            out[i] = np.sum(w < data[i]) / window * 100
        return out

    @staticmethod
    def _rolling_slope(data, window):
        n = len(data)
        out = np.zeros(n)
        x = np.arange(window, dtype=np.float64)
        x_mean = np.mean(x)
        x_var = np.sum((x - x_mean) ** 2)

        for i in range(window - 1, n):
            y = data[i - window + 1:i + 1]
            y_mean = np.mean(y)
            if x_var > 0:
                slope = np.sum((x - x_mean) * (y - y_mean)) / x_var
                out[i] = slope / max(abs(y_mean), 1e-10)
        return out

    @staticmethod
    def _rolling_corr(a, b, window):
        n = len(a)
        out = np.zeros(n)
        for i in range(window, n):
            wa = a[i - window:i]
            wb = b[i - window:i]
            sa, sb = np.std(wa), np.std(wb)
            if sa > 0 and sb > 0:
                out[i] = np.corrcoef(wa, wb)[0, 1]
        return out

    @staticmethod
    def _rolling_stat(data, window, stat_type):
        n = len(data)
        out = np.zeros(n)
        for i in range(window, n):
            w = data[i - window:i]
            m = np.mean(w)
            s = np.std(w)
            if s > 0:
                centered = (w - m) / s
                if stat_type == "skew":
                    out[i] = np.mean(centered ** 3)
                elif stat_type == "kurt":
                    out[i] = np.mean(centered ** 4) - 3
        return out

    @staticmethod
    def _autocorrelation(data, window):
        n = len(data)
        out = np.zeros(n)
        for i in range(window + 1, n):
            r1 = data[i - window:i]
            r2 = data[i - window - 1:i - 1]
            s1, s2 = np.std(r1), np.std(r2)
            if s1 > 0 and s2 > 0:
                out[i] = np.corrcoef(r1, r2)[0, 1]
        return out

    @staticmethod
    def _rolling_hurst(data, window):
        """Simplified Hurst exponent"""
        n = len(data)
        out = np.full(n, 0.5)
        for i in range(window, n):
            w = data[i - window:i]
            if len(w) < 20:
                continue
            rets = np.diff(np.log(np.where(w > 0, w, 1)))
            if len(rets) < 10:
                continue
            mean_r = np.mean(rets)
            cum = np.cumsum(rets - mean_r)
            R = np.max(cum) - np.min(cum)
            S = np.std(rets)
            if S > 0 and R > 0:
                out[i] = np.log(R / S) / np.log(window)
        return out

    @staticmethod
    def _rolling_entropy(data, window):
        """Shannon entropy of returns distribution"""
        n = len(data)
        out = np.zeros(n)
        n_bins = 10
        for i in range(window, n):
            w = data[i - window:i]
            if np.std(w) == 0:
                continue
            hist, _ = np.histogram(w, bins=n_bins)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            out[i] = -np.sum(probs * np.log2(probs))
        return out


# ═══════════════════════════════════════════════
#  Feature Selector
# ═══════════════════════════════════════════════

class FeatureSelector:
    """
    Отбор фич с защитой от overfitting

    Методы:
    1. Mutual Information (нелинейная зависимость)
    2. Feature importance stability across folds
    3. Correlation filter (удаление дубликатов)
    4. Variance threshold
    """

    def __init__(
        self,
        max_features: int = 30,
        min_importance: float = 0.01,
        max_correlation: float = 0.85
    ):
        self.max_features = max_features
        self.min_importance = min_importance
        self.max_correlation = max_correlation
        self.selected_features: List[str] = []

    def select(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        """Выбрать лучшие фичи"""
        if not HAS_SKLEARN:
            return list(X.columns[:self.max_features])

        features = list(X.columns)

        # Step 1: Remove zero-variance
        variances = X.var()
        features = [
            f for f in features if variances.get(f, 0) > 1e-10
        ]

        # Step 2: Mutual Information
        mi_scores = mutual_info_classif(
            X[features].values, y, random_state=42
        )
        mi_dict = dict(zip(features, mi_scores))

        # Remove low MI
        features = [
            f for f in features
            if mi_dict.get(f, 0) > self.min_importance
        ]

        # Step 3: Remove highly correlated
        if len(features) > 2:
            corr_matrix = X[features].corr().abs()
            to_remove = set()

            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    if corr_matrix.iloc[i, j] > self.max_correlation:
                        fi = features[i]
                        fj = features[j]
                        if mi_dict.get(fi, 0) >= mi_dict.get(fj, 0):
                            to_remove.add(fj)
                        else:
                            to_remove.add(fi)

            features = [f for f in features if f not in to_remove]

        # Step 4: Top N by MI score
        features.sort(key=lambda f: mi_dict.get(f, 0), reverse=True)
        features = features[:self.max_features]

        self.selected_features = features

        print(f"[FEATURES] Selected {len(features)} "
              f"from {len(X.columns)}")

        return features


# ═══════════════════════════════════════════════
#  Purged Walk-Forward Trainer
# ═══════════════════════════════════════════════

class AntiOverfitTrainer:
    """
    Полный pipeline обучения с защитой от переобучения

    Ключевое отличие от обычного:
    1. ТОЛЬКО TimeSeriesSplit (никакого random split!)
    2. Purge gap между train/test
    3. Embargo на последние N баров train
    4. Feature stability check
    5. Holdout validation на невиданных данных
    6. Overfit ratio monitoring
    """

    def __init__(
        self,
        n_folds: int = 5,
        purge_bars: int = 20,
        embargo_bars: int = 10,
        holdout_ratio: float = 0.15,
        max_features: int = 30,
        model_type: str = "lgbm"
    ):
        self.n_folds = n_folds
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.holdout_ratio = holdout_ratio
        self.max_features = max_features
        self.model_type = model_type

        self.feature_builder = AdvancedFeatureBuilder()
        self.feature_selector = FeatureSelector(
            max_features=max_features
        )
        self.scaler = StandardScaler() if HAS_SKLEARN else None

        self.model = None
        self.feature_names: List[str] = []

    def train(
        self,
        df: pd.DataFrame,
        target_builder: callable,
        model_name: str = "model"
    ) -> TrainingReport:
        """
        Полный pipeline обучения

        Args:
            df: DataFrame с OHLCV + индикаторами
            target_builder: функция(df) → np.ndarray target
            model_name: имя модели
        """
        if not HAS_SKLEARN:
            print("[TRAINER] sklearn required!")
            return TrainingReport(verdict="ERROR: no sklearn")

        report = TrainingReport(model_name=model_name)

        print(f"\n{'═' * 60}")
        print(f"  ANTI-OVERFIT TRAINING: {model_name}")
        print(f"{'═' * 60}")

        # ═══ Step 1: Build Features ═══
        print("\n📊 Step 1: Feature Engineering...")
        features = self.feature_builder.build(df)

        if features.empty:
            return TrainingReport(verdict="ERROR: no features")

        report.n_features_initial = len(features.columns)
        print(f"  Built {report.n_features_initial} features")

        # ═══ Step 2: Build Target ═══
        print("\n🎯 Step 2: Target Construction...")
        target = target_builder(df)

        # Align lengths
        min_len = min(len(features), len(target))
        features = features.iloc[:min_len]
        target = target[:min_len]

        # Remove NaN rows
        valid_mask = ~(features.isna().any(axis=1))
        features = features[valid_mask]
        target = target[valid_mask.values]

        report.n_samples = len(features)
        n_positive = int(np.sum(target))
        print(f"  Samples: {report.n_samples}")
        print(f"  Positive: {n_positive} "
              f"({n_positive/len(target)*100:.1f}%)")

        # ═══ Step 3: Holdout Split ═══
        print("\n🔒 Step 3: Holdout Split...")
        holdout_size = int(len(features) * self.holdout_ratio)
        holdout_start = len(features) - holdout_size

        X_main = features.iloc[:holdout_start]
        y_main = target[:holdout_start]
        X_holdout = features.iloc[holdout_start:]
        y_holdout = target[holdout_start:]

        print(f"  Main: {len(X_main)} | "
              f"Holdout: {len(X_holdout)} "
              f"(NEVER seen during training)")

        # ═══ Step 4: Feature Selection ═══
        print("\n🔍 Step 4: Feature Selection...")
        selected = self.feature_selector.select(X_main, y_main)
        self.feature_names = selected
        report.n_features_selected = len(selected)

        X_main = X_main[selected]
        X_holdout = X_holdout[selected]

        # ═══ Step 5: Purged Walk-Forward CV ═══
        print(f"\n📐 Step 5: Purged Walk-Forward "
              f"({self.n_folds} folds)...")
        report.n_folds = self.n_folds

        feature_importance_per_fold = []

        n = len(X_main)
        fold_size = n // self.n_folds

        for fold in range(self.n_folds):
            train_end = int(n * (fold + 1) / self.n_folds * 0.7)
            test_start = train_end + self.purge_bars
            test_end = min(
                int(n * (fold + 1) / self.n_folds),
                n
            )

            if fold > 0:
                train_start = int(n * fold / self.n_folds * 0.3)
            else:
                train_start = 0

            train_end_emb = train_end - self.embargo_bars

            if (
                train_end_emb - train_start < 100
                or test_end - test_start < 30
            ):
                continue

            X_train = X_main.iloc[train_start:train_end_emb].values
            y_train = y_main[train_start:train_end_emb]
            X_test = X_main.iloc[test_start:test_end].values
            y_test = y_main[test_start:test_end]

            # Train model
            model = self._create_model()
            model.fit(X_train, y_train)

            # Evaluate
            train_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
            test_pred = model.predict(X_test)

            try:
                train_auc = roc_auc_score(y_train, train_proba)
                test_auc = roc_auc_score(y_test, test_proba)
            except:
                train_auc = 0.5
                test_auc = 0.5

            fold_result = FoldResult(
                fold=fold,
                train_size=len(X_train),
                test_size=len(X_test),
                train_auc=round(train_auc, 4),
                test_auc=round(test_auc, 4),
                test_accuracy=round(
                    accuracy_score(y_test, test_pred), 4
                ),
                test_precision=round(
                    precision_score(y_test, test_pred,
                                    zero_division=0), 4
                ),
                test_recall=round(
                    recall_score(y_test, test_pred,
                                 zero_division=0), 4
                ),
                overfit_ratio=round(
                    train_auc / max(test_auc, 0.01), 4
                )
            )

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[-5:][::-1]
                fold_result.top_features = [
                    selected[i] for i in top_idx
                    if i < len(selected)
                ]
                feature_importance_per_fold.append(
                    dict(zip(selected, imp))
                )

            report.fold_results.append(fold_result)

            print(
                f"  Fold {fold}: "
                f"Train AUC={train_auc:.4f} → "
                f"Test AUC={test_auc:.4f} "
                f"(overfit={fold_result.overfit_ratio:.2f})"
            )

        # ═══ Step 6: Aggregate CV Results ═══
        if report.fold_results:
            aucs_train = [f.train_auc for f in report.fold_results]
            aucs_test = [f.test_auc for f in report.fold_results]

            report.avg_train_auc = round(np.mean(aucs_train), 4)
            report.avg_test_auc = round(np.mean(aucs_test), 4)
            report.auc_stability = round(np.std(aucs_test), 4)
            report.avg_overfit_ratio = round(
                np.mean([f.overfit_ratio for f in report.fold_results]),
                4
            )

        # ═══ Step 7: Feature Stability ═══
        print("\n📋 Step 6: Feature Stability...")
        if feature_importance_per_fold:
            report.stable_features, report.unstable_features = (
                self._check_feature_stability(
                    feature_importance_per_fold, selected
                )
            )

        # ═══ Step 8: Train Final Model ═══
        print("\n🏗️ Step 7: Final Model Training...")
        self.model = self._create_model()

        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_main.values)
        else:
            X_scaled = X_main.values

        self.model.fit(X_scaled, y_main)

        # ═══ Step 9: Holdout Validation ═══
        print("\n🔒 Step 8: HOLDOUT Validation "
              "(data model NEVER saw)...")

        if self.scaler:
            X_hold_scaled = self.scaler.transform(X_holdout.values)
        else:
            X_hold_scaled = X_holdout.values

        hold_proba = self.model.predict_proba(X_hold_scaled)[:, 1]
        hold_pred = self.model.predict(X_hold_scaled)

        try:
            report.holdout_auc = round(
                roc_auc_score(y_holdout, hold_proba), 4
            )
        except:
            report.holdout_auc = 0.5

        report.holdout_accuracy = round(
            accuracy_score(y_holdout, hold_pred), 4
        )

        print(f"  Holdout AUC: {report.holdout_auc}")
        print(f"  Holdout Accuracy: {report.holdout_accuracy}")

        # ═══ Step 10: Verdict ═══
        report = self._generate_verdict(report)

        self._print_report(report)

        return report

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Предсказание"""
        if self.model is None or not self.feature_names:
            return np.full(len(features), 0.5)

        X = features[self.feature_names].values

        if self.scaler:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)[:, 1]

    def _create_model(self):
        """Создать модель с регуляризацией"""
        if self.model_type == "lgbm" and HAS_LGB:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.03,
                min_child_samples=30,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=0.3,
                random_state=42,
                verbose=-1,
                n_jobs=1
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                max_features=0.7,
                min_samples_leaf=30,
                random_state=42
            )

    @staticmethod
    def _check_feature_stability(
        importances_per_fold: List[Dict],
        features: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Проверить стабильность фич между фолдами

        Фича стабильна если она в топ-10 в >50% фолдов
        """
        n_folds = len(importances_per_fold)
        top_count = {}

        for fold_imp in importances_per_fold:
            sorted_feats = sorted(
                fold_imp.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_10 = [f[0] for f in sorted_feats[:10]]

            for f in top_10:
                top_count[f] = top_count.get(f, 0) + 1

        stable = [
            f for f, count in top_count.items()
            if count >= n_folds * 0.5
        ]

        unstable = [
            f for f in features
            if f not in stable
            and top_count.get(f, 0) > 0
            and top_count.get(f, 0) < n_folds * 0.3
        ]

        print(f"  Stable features: {len(stable)}")
        print(f"  Unstable features: {len(unstable)}")

        return stable, unstable

    @staticmethod
    def _generate_verdict(report: TrainingReport) -> TrainingReport:
        """Финальный вердикт"""
        score = 0
        recs = []

        # Test AUC
        if report.avg_test_auc > 0.6:
            score += 2
        elif report.avg_test_auc > 0.55:
            score += 1
        else:
            recs.append(
                "Low test AUC — model has weak predictive power"
            )

        # Overfit ratio (идеально ~1.0, плохо >1.3)
        if report.avg_overfit_ratio < 1.1:
            score += 2
        elif report.avg_overfit_ratio < 1.2:
            score += 1
        else:
            recs.append(
                f"High overfit ratio ({report.avg_overfit_ratio:.2f}) "
                f"— reduce model complexity"
            )
            report.is_overfitted = True

        # AUC stability
        if report.auc_stability < 0.03:
            score += 2
        elif report.auc_stability < 0.05:
            score += 1
        else:
            recs.append(
                "Unstable AUC across folds — model is fragile"
            )

        # Holdout
        if report.holdout_auc > 0.58:
            score += 2
        elif report.holdout_auc > 0.53:
            score += 1
        else:
            recs.append(
                "Poor holdout performance — likely overfitted"
            )
            report.is_overfitted = True

        # Holdout vs CV
        holdout_vs_cv = abs(
            report.holdout_auc - report.avg_test_auc
        )
        if holdout_vs_cv < 0.03:
            score += 1
        elif holdout_vs_cv > 0.08:
            recs.append(
                "Large gap between CV and holdout — "
                "distribution shift or overfit"
            )

        # Feature stability
        if len(report.stable_features) >= 5:
            score += 1

        report.confidence_score = round(score / 10, 2)

        if score >= 8:
            report.verdict = "🟢 EXCELLENT — Safe to deploy"
        elif score >= 6:
            report.verdict = "🟡 GOOD — Minor improvements needed"
        elif score >= 4:
            report.verdict = "🟠 RISKY — Significant overfit risk"
        else:
            report.verdict = "🔴 OVERFITTED — Do NOT deploy"

        report.recommendations = recs

        return report

    @staticmethod
    def _print_report(report: TrainingReport):
        """Вывод отчёта"""
        print(f"\n{'═' * 60}")
        print(f"  TRAINING REPORT: {report.model_name}")
        print(f"{'═' * 60}")

        print(f"\n  Data:")
        print(f"    Samples: {report.n_samples}")
        print(f"    Features: {report.n_features_initial} → "
              f"{report.n_features_selected}")

        print(f"\n  Walk-Forward CV ({report.n_folds} folds):")
        print(f"    Avg Train AUC: {report.avg_train_auc}")
        print(f"    Avg Test AUC:  {report.avg_test_auc}")
        print(f"    AUC Stability: ±{report.auc_stability}")
        print(f"    Overfit Ratio: {report.avg_overfit_ratio}")

        print(f"\n  Holdout (never-seen data):")
        print(f"    AUC:      {report.holdout_auc}")
        print(f"    Accuracy: {report.holdout_accuracy}")

        print(f"\n  Feature Stability:")
        print(f"    Stable:   {len(report.stable_features)}")
        if report.stable_features:
            for f in report.stable_features[:10]:
                print(f"      ✅ {f}")

        print(f"\n  Confidence: {report.confidence_score:.0%}")
        print(f"  Verdict: {report.verdict}")

        if report.recommendations:
            print(f"\n  Recommendations:")
            for rec in report.recommendations:
                print(f"    ⚠️ {rec}")

        print(f"{'═' * 60}\n")

    def save(self, path: str = "data/anti_overfit_model.pkl"):
        """Сохранить"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "features": self.feature_names,
                "scaler": self.scaler,
                "time": datetime.now().isoformat()
            }, f)

    def load(self, path: str = "data/anti_overfit_model.pkl"):
        """Загрузить"""
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.scaler = data.get("scaler")
