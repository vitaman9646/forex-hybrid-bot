"""
AI Signal Filter
Дополнительная фильтрация сигналов через ML
Решает: стоит ли входить в эту конкретную сделку
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict
from strategies.base_strategy import TradeSignal, SignalType
from config import ai_config


class AISignalFilter:
    """
    Бинарный классификатор: WIN или LOSS
    Фильтрует сигналы с низкой вероятностью успеха
    """

    def __init__(self):
        self.model = None
        self.model_path = "data/signal_filter_model.pkl"
        self._load_model()

    def should_trade(
        self,
        signal: TradeSignal,
        df: pd.DataFrame,
        market_regime: str
    ) -> Dict:
        """
        Решает, стоит ли входить в сделку
        Returns: {"approved": bool, "win_probability": float}
        """
        if not ai_config.enabled or self.model is None:
            # AI выключен — пропускаем все сигналы
            return {
                "approved": True,
                "win_probability": signal.confidence,
                "reason": "AI фильтр неактивен"
            }

        features = self._build_features(signal, df, market_regime)

        if features is None:
            return {
                "approved": True,
                "win_probability": 0.5,
                "reason": "Не удалось извлечь фичи"
            }

        # Предсказание
        features_array = np.array([features])
        win_prob = self.model.predict_proba(features_array)[0][1]

        approved = win_prob >= ai_config.confidence_threshold

        result = {
            "approved": approved,
            "win_probability": round(float(win_prob), 3),
            "reason": (
                f"AI: вероятность win={win_prob:.1%}"
                if approved
                else f"AI: низкая вероятность win={win_prob:.1%}"
            )
        }

        if not approved:
            print(
                f"[AI FILTER] ❌ Сигнал отклонён: "
                f"{signal.strategy_name} | "
                f"win_prob={win_prob:.1%}"
            )

        return result

    def _build_features(
        self,
        signal: TradeSignal,
        df: pd.DataFrame,
        market_regime: str
    ) -> Optional[list]:
        """Формирование фич для фильтра"""
        if df is None or len(df) < 5:
            return None

        row = df.iloc[-1]

        try:
            # Маппинг режимов
            regime_map = {
                "trend_up": 1,
                "trend_down": 2,
                "range": 3,
                "volatile": 4,
                "squeeze": 5,
                "unknown": 0
            }

            # Маппинг стратегий
            strategy_map = {
                "trend_strategy": 1,
                "range_strategy": 2,
                "breakout_strategy": 3,
                "scalping_strategy": 4
            }

            features = [
                row.get("adx", 0),
                row.get("atr", 0),
                row.get("rsi", 50),
                row.get("volatility", 0),
                row.get("bb_width", 0),
                row.get("momentum", 0),
                row.get("volume_ratio", 1),
                signal.confidence,
                regime_map.get(market_regime, 0),
                strategy_map.get(signal.strategy_name, 0),
                1 if signal.signal_type == SignalType.BUY else 0,
                abs(signal.entry_price - signal.stop_loss),
                abs(signal.take_profit - signal.entry_price),
            ]

            return features

        except Exception as e:
            print(f"[AI FILTER] Ошибка фич: {e}")
            return None

    def train(self, trade_log_path: str = "data/trade_log.csv"):
        """Обучить фильтр на истории сделок"""
        if not os.path.exists(trade_log_path):
            print("[AI FILTER] Нет данных")
            return

        df = pd.read_csv(trade_log_path)

        if len(df) < 50:
            print(f"[AI FILTER] Мало данных: {len(df)}")
            return

        # Target: 1 = win, 0 = loss
        df["target"] = (df["result"] == "win").astype(int)

        # Фичи из лога
        feature_cols = [
            "adx", "atr", "rsi", "volatility", "confidence"
        ]
        available = [c for c in feature_cols if c in df.columns]

        if len(available) < 3:
            print("[AI FILTER] Мало feature колонок")
            return

        X = df[available].fillna(0).values
        y = df["target"].values

        try:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            model.fit(X, y)
            self.model = model

            # Сохраняем
            os.makedirs(
                os.path.dirname(self.model_path), exist_ok=True
            )
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)

            print("[AI FILTER] ✅ Фильтр обучен")

        except ImportError:
            print("[AI FILTER] sklearn не установлен")
        except Exception as e:
            print(f"[AI FILTER] Ошибка: {e}")

    def _load_model(self):
        """Загрузить модель"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                print("[AI FILTER] Модель загружена")
            except Exception as e:
                print(f"[AI FILTER] Ошибка загрузки: {e}")
