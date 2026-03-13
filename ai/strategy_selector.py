"""
AI Strategy Selector
Машинное обучение для выбора лучшей стратегии
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
from config import ai_config


class AIStrategySelector:
    """
    ML модель которая предсказывает лучшую стратегию
    на основе текущих рыночных условий
    """

    def __init__(self):
        self.model = None
        self.feature_columns = [
            "adx", "atr", "rsi", "volatility",
            "bb_width", "momentum", "volume_ratio",
            "ema_50_vs_200",     # расстояние EMA
            "stoch_k",
            "macd_histogram"
        ]
        self.strategy_labels = {
            0: "trend_strategy",
            1: "range_strategy",
            2: "breakout_strategy",
            3: "scalping_strategy"
        }
        self.last_train_time = None
        self._load_model()

    def predict_best_strategy(
        self,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Предсказать лучшую стратегию
        Возвращает: {"strategy": "...", "confidence": 0.X}
        """
        if self.model is None:
            return None

        features = self._extract_features(df)
        if features is None:
            return None

        # Предсказание
        features_array = np.array([features])
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]

        strategy = self.strategy_labels.get(
            prediction, "trend_strategy"
        )
        confidence = float(max(probabilities))

        return {
            "strategy": strategy,
            "confidence": round(confidence, 3),
            "probabilities": {
                self.strategy_labels[i]: round(float(p), 3)
                for i, p in enumerate(probabilities)
            }
        }

    def _extract_features(
        self,
        df: pd.DataFrame
    ) -> Optional[list]:
        """Извлечь фичи из текущих данных"""
        if df is None or len(df) < 5:
            return None

        row = df.iloc[-1]

        try:
            features = [
                row.get("adx", 0),
                row.get("atr", 0),
                row.get("rsi", 50),
                row.get("volatility", 0),
                row.get("bb_width", 0),
                row.get("momentum", 0),
                row.get("volume_ratio", 1),
                (
                    row.get("ema_50", 0)
                    - row.get("ema_200", 0)
                ),
                row.get("stoch_k", 50),
                row.get("macd_histogram", 0),
            ]
            return features
        except Exception as e:
            print(f"[AI] Ошибка извлечения фич: {e}")
            return None

    def train(self, trade_log_path: str = "data/trade_log.csv"):
        """
        Обучить модель на исторических данных
        Target: стратегия которая дала лучший результат
        """
        if not os.path.exists(trade_log_path):
            print("[AI] Нет данных для обучения")
            return

        df = pd.read_csv(trade_log_path)

        if len(df) < ai_config.min_training_samples:
            print(
                f"[AI] Мало данных: {len(df)} "
                f"(нужно {ai_config.min_training_samples})"
            )
            return

        # Подготовка данных
        feature_cols = [
            "adx", "atr", "rsi", "volatility"
        ]

        # Проверяем наличие колонок
        available_cols = [
            c for c in feature_cols if c in df.columns
        ]

        if len(available_cols) < 3:
            print("[AI] Недостаточно feature колонок")
            return

        # Target: маппинг стратегий
        label_map = {v: k for k, v in self.strategy_labels.items()}
        df["target"] = df["strategy"].map(label_map)
        df = df.dropna(subset=["target"])

        if len(df) < 20:
            print("[AI] Мало размеченных данных")
            return

        X = df[available_cols].fillna(0).values
        y = df["target"].astype(int).values

        # Обучение
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Кросс-валидация
            scores = cross_val_score(model, X, y, cv=3)
            print(
                f"[AI] CV accuracy: {scores.mean():.3f} "
                f"(±{scores.std():.3f})"
            )

            # Финальное обучение
            model.fit(X, y)
            self.model = model
            self.last_train_time = datetime.now()

            # Сохраняем
            self._save_model()
            print("[AI] ✅ Модель обучена и сохранена")

        except ImportError:
            print("[AI] sklearn не установлен")
        except Exception as e:
            print(f"[AI] Ошибка обучения: {e}")

    def should_retrain(self) -> bool:
        """Пора ли переобучить модель"""
        if self.last_train_time is None:
            return True

        hours_since = (
            datetime.now() - self.last_train_time
        ).total_seconds() / 3600

        return hours_since >= ai_config.retrain_interval_hours

    def _save_model(self):
        """Сохранить модель"""
        os.makedirs(
            os.path.dirname(ai_config.model_path), exist_ok=True
        )
        with open(ai_config.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "train_time": self.last_train_time,
                "features": self.feature_columns
            }, f)

    def _load_model(self):
        """Загрузить модель"""
        if os.path.exists(ai_config.model_path):
            try:
                with open(ai_config.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.last_train_time = data.get("train_time")
                    print("[AI] Модель загружена")
            except Exception as e:
                print(f"[AI] Ошибка загрузки модели: {e}")
