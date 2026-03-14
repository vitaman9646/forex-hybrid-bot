"""
ai/impulse_predictor.py

AI предсказание рыночных импульсов за 1-3 свечи

Что считается импульсом:
- Движение > 1 ATR за 1-3 свечи
- Или return > определённого порога

Модель: LightGBM/XGBoost ensemble
Фичи: 40+ технических и статистических
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import pickle
import os
import warnings

warnings.filterwarnings("ignore")


@dataclass
class ImpulsePrediction:
    """Результат предсказания импульса"""
    probability: float           # вероятность импульса
    predicted_direction: int     # 1=up, -1=down, 0=unknown
    magnitude: float             # ожидаемый размер в ATR
    confidence: float            # уверенность модели
    horizon_bars: int            # горизонт предсказания
    top_features: Dict[str, float] = None  # важнейшие фичи
    recommendation: str = ""     # trade / wait / prepare


class FeatureEngine:
    """
    Генерация фич для предсказания импульсов

    40+ фич из нескольких категорий:
    1. Price action (returns, gaps, patterns)
    2. Volatility (ATR, BB, realized vol)
    3. Momentum (RSI, ADX, MACD divergence)
    4. Volume (spikes, trend, ratio)
    5. Microstructure (spread, candle anatomy)
    6. Time (session, day of week)
    7. Statistical (skew, kurtosis, autocorr)
    """

    @staticmethod
    def build_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Построить все фичи"""
        if df is None or len(df) < 100:
            return None

        feat = pd.DataFrame(index=df.index)

        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        o = df["open"].values
        v = df["volume"].values if "volume" in df.columns else np.ones(len(c))

        n = len(c)

        # ═══ 1. Price Action ═══
        # Returns
        for p in [1, 3, 5, 10, 20]:
            ret = np.zeros(n)
            ret[p:] = (c[p:] - c[:-p]) / np.where(
                c[:-p] != 0, c[:-p], 1
            )
            feat[f"return_{p}"] = ret

        # Gaps
        gaps = np.zeros(n)
        gaps[1:] = (o[1:] - c[:-1]) / np.where(
            c[:-1] != 0, c[:-1], 1
        )
        feat["gap"] = gaps

        # Candle body ratio
        ranges = h - l
        bodies = np.abs(c - o)
        feat["body_ratio"] = np.where(
            ranges > 0, bodies / ranges, 0
        )

        # Upper/lower wick
        feat["upper_wick"] = np.where(
            ranges > 0,
            (h - np.maximum(c, o)) / ranges,
            0
        )
        feat["lower_wick"] = np.where(
            ranges > 0,
            (np.minimum(c, o) - l) / ranges,
            0
        )

        # ═══ 2. Volatility ═══
        # ATR-based
        if "atr" in df.columns:
            atr = df["atr"].values
            feat["atr"] = atr

            atr_sma20 = np.zeros(n)
            for i in range(20, n):
                atr_sma20[i] = np.mean(atr[i-20:i])
            feat["atr_ratio"] = np.where(
                atr_sma20 > 0, atr / atr_sma20, 1
            )
        else:
            atr = np.zeros(n)

        # Realized volatility
        for w in [5, 10, 20]:
            rv = np.zeros(n)
            rets = np.zeros(n)
            rets[1:] = np.log(
                np.where(c[:-1] > 0, c[1:] / c[:-1], 1)
            )
            for i in range(w, n):
                rv[i] = np.std(rets[i-w:i]) * np.sqrt(252)
            feat[f"realized_vol_{w}"] = rv

        # Bollinger width
        if "bb_width" in df.columns:
            feat["bb_width"] = df["bb_width"].values
        else:
            bb_w = np.zeros(n)
            for i in range(20, n):
                window = c[i-20:i]
                sma = np.mean(window)
                std = np.std(window)
                if sma > 0:
                    bb_w[i] = (2 * 2 * std) / sma
            feat["bb_width"] = bb_w

        # BB width change
        bbw = feat["bb_width"].values
        feat["bb_width_change"] = np.concatenate([
            [0], np.diff(bbw)
        ])

        # Squeeze detection
        bb_min = np.zeros(n)
        for i in range(50, n):
            bb_min[i] = np.min(bbw[i-50:i])
        feat["bb_squeeze"] = (
            bbw <= bb_min * 1.1
        ).astype(float)

        # ═══ 3. Momentum ═══
        if "rsi" in df.columns:
            feat["rsi"] = df["rsi"].values

        if "adx" in df.columns:
            feat["adx"] = df["adx"].values

        # ADX change
        if "adx" in df.columns:
            adx = df["adx"].values
            feat["adx_change_5"] = np.concatenate([
                np.zeros(5), adx[5:] - adx[:-5]
            ])

        # Momentum acceleration
        mom_5 = feat["return_5"].values
        feat["momentum_accel"] = np.concatenate([
            np.zeros(5), mom_5[5:] - mom_5[:-5]
        ])

        # ═══ 4. Volume ═══
        vol_sma = np.zeros(n)
        for i in range(20, n):
            vol_sma[i] = np.mean(v[i-20:i])

        feat["volume_ratio"] = np.where(
            vol_sma > 0, v / vol_sma, 1
        )

        # Volume trend
        vol_sma5 = np.zeros(n)
        for i in range(5, n):
            vol_sma5[i] = np.mean(v[i-5:i])
        feat["volume_trend"] = np.where(
            vol_sma > 0, vol_sma5 / vol_sma, 1
        )

        # Volume spike
        feat["volume_spike"] = (
            feat["volume_ratio"].values > 2.0
        ).astype(float)

        # ═══ 5. Statistical ═══
        for w in [10, 20]:
            skew = np.zeros(n)
            kurt = np.zeros(n)
            rets = feat["return_1"].values

            for i in range(w, n):
                window = rets[i-w:i]
                m = np.mean(window)
                s = np.std(window)
                if s > 0:
                    skew[i] = np.mean(((window - m) / s) ** 3)
                    kurt[i] = np.mean(((window - m) / s) ** 4) - 3

            feat[f"skewness_{w}"] = skew
            feat[f"kurtosis_{w}"] = kurt

        # Autocorrelation
        rets = feat["return_1"].values
        autocorr = np.zeros(n)
        for i in range(21, n):
            r1 = rets[i-20:i]
            r2 = rets[i-21:i-1]
            if np.std(r1) > 0 and np.std(r2) > 0:
                autocorr[i] = np.corrcoef(r1, r2)[0, 1]
        feat["autocorrelation"] = autocorr

        # ═══ 6. Time ═══
        if hasattr(df.index, 'hour'):
            feat["hour"] = df.index.hour
            feat["day_of_week"] = df.index.dayofweek

            # Session encoding
            hours = df.index.hour
            feat["is_london"] = (
                (hours >= 7) & (hours < 16)
            ).astype(float)
            feat["is_ny"] = (
                (hours >= 12) & (hours < 21)
            ).astype(float)
            feat["is_overlap"] = (
                (hours >= 12) & (hours < 16)
            ).astype(float)

        # ═══ 7. Cross-features ═══
        feat["vol_x_squeeze"] = (
            feat["volume_ratio"].values
            * feat["bb_squeeze"].values
        )

        if "adx" in df.columns:
            feat["adx_x_atr_ratio"] = (
                feat["adx"].values
                * feat.get("atr_ratio", np.ones(n))
            )

        # Убираем NaN
        feat = feat.fillna(0)
        feat = feat.replace([np.inf, -np.inf], 0)

        return feat

    @staticmethod
    def build_target(
        df: pd.DataFrame,
        horizon: int = 3,
        threshold_atr: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Построить target variable

        impulse = 1 если за следующие horizon баров
        движение > threshold × ATR

        Returns:
            target_class: (n,) — 0 или 1
            target_direction: (n,) — 1 (up) или -1 (down)
        """
        n = len(df)
        target_class = np.zeros(n)
        target_dir = np.zeros(n)

        c = df["close"].values
        atr = df["atr"].values if "atr" in df.columns else np.ones(n) * 0.001

        for i in range(n - horizon):
            future = c[i + 1:i + 1 + horizon]
            current = c[i]
            threshold = atr[i] * threshold_atr

            if threshold <= 0:
                continue

            max_up = np.max(future) - current
            max_down = current - np.min(future)

            if max_up > threshold or max_down > threshold:
                target_class[i] = 1

                if max_up > max_down:
                    target_dir[i] = 1
                else:
                    target_dir[i] = -1

        return target_class, target_dir


class ImpulsePredictor:
    """
    AI-предсказатель рыночных импульсов

    Ансамбль из двух моделей:
    1. Classifier: будет ли импульс (0/1)
    2. Regressor: если да, то в какую сторону (+1/-1)
    """

    def __init__(self):
        self.impulse_model = None
        self.direction_model = None
        self.feature_engine = FeatureEngine()
        self.feature_names: List[str] = []
        self.model_path = "data/impulse_model.pkl"
        self.is_trained = False
        self._load()

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 3
    ) -> ImpulsePrediction:
        """
        Предсказать импульс на следующие horizon баров
        """
        if not self.is_trained:
            return ImpulsePrediction(
                probability=0.5,
                predicted_direction=0,
                magnitude=0,
                confidence=0,
                horizon_bars=horizon,
                recommendation="WAIT — model not trained"
            )

        features = self.feature_engine.build_features(df)
        if features is None:
            return ImpulsePrediction(
                probability=0.5,
                predicted_direction=0,
                magnitude=0,
                confidence=0,
                horizon_bars=horizon,
                recommendation="WAIT — insufficient data"
            )

        # Берём последнюю строку
        X = features[self.feature_names].iloc[-1:].values

        # Предсказание импульса
        impulse_prob = float(
            self.impulse_model.predict_proba(X)[0][1]
        )

        # Предсказание направления
        direction = 0
        if impulse_prob > 0.5:
            dir_pred = self.direction_model.predict(X)[0]
            dir_proba = self.direction_model.predict_proba(X)[0]
            direction = int(dir_pred)

            if direction == 0:
                direction = 1 if dir_proba[1] > dir_proba[0] else -1

        # Magnitude (примерная оценка)
        atr = df["atr"].iloc[-1] if "atr" in df.columns else 0
        magnitude = atr * impulse_prob * 2

        # Confidence
        confidence = self._calc_confidence(
            impulse_prob, features, df
        )

        # Feature importance
        top_feats = self._get_top_features(X)

        # Recommendation
        recommendation = self._get_recommendation(
            impulse_prob, confidence
        )

        return ImpulsePrediction(
            probability=round(impulse_prob, 4),
            predicted_direction=direction,
            magnitude=round(magnitude, 6),
            confidence=round(confidence, 3),
            horizon_bars=horizon,
            top_features=top_feats,
            recommendation=recommendation
        )

    def train(
        self,
        df: pd.DataFrame,
        horizon: int = 3,
        threshold_atr: float = 1.0,
        test_size: float = 0.2
    ) -> Dict:
        """Обучить модели"""
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier
            )
            from sklearn.model_selection import (
                train_test_split, cross_val_score
            )
            from sklearn.metrics import (
                classification_report, roc_auc_score
            )
        except ImportError:
            print("[IMPULSE] sklearn не установлен")
            return {"error": "sklearn required"}

        try:
            import lightgbm as lgb
            USE_LGB = True
        except ImportError:
            USE_LGB = False

        print(f"\n[IMPULSE] Training on {len(df)} bars...")

        # Features
        features = self.feature_engine.build_features(df)
        if features is None:
            return {"error": "Failed to build features"}

        # Target
        target_class, target_dir = self.feature_engine.build_target(
            df, horizon, threshold_atr
        )

        # Убираем последние horizon баров (нет target)
        features = features.iloc[:-horizon]
        target_class = target_class[:-horizon]
        target_dir = target_dir[:-horizon]

        # Feature names
        self.feature_names = list(features.columns)
        X = features.values
        y_impulse = target_class
        y_direction = target_dir

        # Class balance
        n_impulse = int(np.sum(y_impulse))
        n_total = len(y_impulse)
        print(
            f"  Impulses: {n_impulse}/{n_total} "
            f"({n_impulse/n_total*100:.1f}%)"
        )

        if n_impulse < 50:
            return {"error": "Too few impulses for training"}

        # Split (temporal, not random!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_impulse[:split_idx], y_impulse[split_idx:]
        yd_train = y_direction[:split_idx]
        yd_test = y_direction[split_idx:]

        # ═══ Model 1: Impulse Classifier ═══
        if USE_LGB:
            model1 = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            )
        else:
            model1 = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )

        model1.fit(X_train, y_train)
        self.impulse_model = model1

        # Evaluate
        y_pred = model1.predict(X_test)
        y_proba = model1.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5

        print(f"\n  Impulse Model:")
        print(f"    AUC: {auc:.4f}")

        accuracy = np.mean(y_pred == y_test)
        print(f"    Accuracy: {accuracy:.4f}")

        # ═══ Model 2: Direction Classifier ═══
        # Только на импульсных барах
        mask_train = y_train == 1
        mask_test = y_test == 1

        if np.sum(mask_train) > 20:
            X_dir_train = X_train[mask_train]
            y_dir_train = (yd_train[mask_train] > 0).astype(int)

            model2 = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
            model2.fit(X_dir_train, y_dir_train)
            self.direction_model = model2

            if np.sum(mask_test) > 5:
                X_dir_test = X_test[mask_test]
                y_dir_test = (yd_test[mask_test] > 0).astype(int)
                dir_acc = model2.score(X_dir_test, y_dir_test)
                print(f"\n  Direction Model:")
                print(f"    Accuracy: {dir_acc:.4f}")
        else:
            # Fallback
            self.direction_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )
            y_dir_all = (yd_train > 0).astype(int)
            self.direction_model.fit(X_train, y_dir_all)

        self.is_trained = True
        self._save()

        # Feature importance
        importances = model1.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]

        print(f"\n  Top 10 features:")
        for idx in top_idx:
            print(
                f"    {self.feature_names[idx]:<25} "
                f"{importances[idx]:.4f}"
            )

        return {
            "auc": round(auc, 4),
            "accuracy": round(accuracy, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(self.feature_names),
            "n_impulses_train": int(np.sum(y_train)),
            "top_features": {
                self.feature_names[i]: round(float(importances[i]), 4)
                for i in top_idx
            }
        }

    def _calc_confidence(
        self,
        impulse_prob: float,
        features: pd.DataFrame,
        df: pd.DataFrame
    ) -> float:
        """Уверенность в предсказании"""
        conf = 0.3

        # Высокая вероятность
        if impulse_prob > 0.8:
            conf += 0.3
        elif impulse_prob > 0.65:
            conf += 0.2

        # BB squeeze подтверждение
        if "bb_squeeze" in features.columns:
            if features["bb_squeeze"].iloc[-1] > 0:
                conf += 0.15

        # Volume подтверждение
        if "volume_ratio" in features.columns:
            if features["volume_ratio"].iloc[-1] > 1.5:
                conf += 0.1

        return min(conf, 0.95)

    def _get_top_features(self, X: np.ndarray) -> Dict[str, float]:
        """Получить значения топ-фич"""
        if not self.is_trained or not self.feature_names:
            return {}

        importances = self.impulse_model.feature_importances_
        top_idx = np.argsort(importances)[-5:][::-1]

        return {
            self.feature_names[i]: round(float(X[0, i]), 6)
            for i in top_idx
            if i < X.shape[1]
        }

    @staticmethod
    def _get_recommendation(
        prob: float,
        confidence: float
    ) -> str:
        if prob > 0.75 and confidence > 0.6:
            return "TRADE — high impulse probability"
        elif prob > 0.65:
            return "PREPARE — moderate impulse expected"
        elif prob > 0.5:
            return "MONITOR — slight impulse signal"
        else:
            return "WAIT — no impulse expected"

    def _save(self):
        """Сохранить модели"""
        os.makedirs(
            os.path.dirname(self.model_path), exist_ok=True
        )
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "impulse_model": self.impulse_model,
                "direction_model": self.direction_model,
                "feature_names": self.feature_names,
                "trained": self.is_trained,
                "time": datetime.now().isoformat()
            }, f)
        print("[IMPULSE] Models saved")

    def _load(self):
        """Загрузить модели"""
        if not os.path.exists(self.model_path):
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)

            self.impulse_model = data["impulse_model"]
            self.direction_model = data["direction_model"]
            self.feature_names = data["feature_names"]
            self.is_trained = data["trained"]
            print("[IMPULSE] Models loaded")
        except Exception as e:
            print(f"[IMPULSE] Load error: {e}")
