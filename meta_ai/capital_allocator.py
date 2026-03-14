"""
meta_ai/capital_allocator.py

Meta-AI: динамическое распределение капитала между стратегиями
Использует Reinforcement Learning подход (Multi-Armed Bandit)
+ анализ корреляций между стратегиями
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import os


@dataclass
class StrategyPerformance:
    """Метрики производительности стратегии"""
    name: str
    recent_returns: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    consecutive_losses: int = 0
    last_trade_time: datetime = None

    @property
    def winrate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_return(self) -> float:
        if not self.recent_returns:
            return 0.0
        return np.mean(self.recent_returns)

    @property
    def return_std(self) -> float:
        if len(self.recent_returns) < 2:
            return 1.0
        return max(np.std(self.recent_returns), 0.0001)


class ThompsonSamplingBandit:
    """
    Thompson Sampling для выбора стратегий
    Байесовский подход: каждая стратегия имеет
    распределение Beta(alpha, beta)
    """

    def __init__(self, n_strategies: int):
        self.n = n_strategies
        # Параметры Beta-распределения
        self.alpha = np.ones(n_strategies)  # успехи + 1
        self.beta = np.ones(n_strategies)   # неудачи + 1

    def select(self) -> int:
        """Выбрать стратегию через семплирование"""
        samples = np.array([
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n)
        ])
        return int(np.argmax(samples))

    def update(self, strategy_idx: int, reward: float):
        """Обновить после получения результата"""
        if reward > 0:
            self.alpha[strategy_idx] += 1
        else:
            self.beta[strategy_idx] += 1

    def get_probabilities(self) -> np.ndarray:
        """Ожидаемая вероятность успеха каждой стратегии"""
        return self.alpha / (self.alpha + self.beta)


class MetaAICapitalAllocator:
    """
    Meta-AI: распределяет капитал между стратегиями

    Методы аллокации:
    1. Thompson Sampling (exploration vs exploitation)
    2. Risk Parity (равный риск)
    3. Mean-Variance Optimization (Markowitz)
    4. Momentum-based (больше денег победителям)

    Финальные веса = ансамбль всех методов
    """

    STRATEGY_NAMES = [
        "trend_strategy",
        "range_strategy",
        "breakout_strategy",
        "scalping_strategy"
    ]

    def __init__(self):
        self.n_strategies = len(self.STRATEGY_NAMES)

        # Производительность каждой стратегии
        self.performance: Dict[str, StrategyPerformance] = {
            name: StrategyPerformance(name=name)
            for name in self.STRATEGY_NAMES
        }

        # Thompson Sampling
        self.bandit = ThompsonSamplingBandit(self.n_strategies)

        # Текущие веса
        self.weights: Dict[str, float] = {
            name: 1.0 / self.n_strategies
            for name in self.STRATEGY_NAMES
        }

        # История весов
        self.weight_history: List[Dict] = []

        # Матрица корреляций доходностей
        self.correlation_matrix: Optional[np.ndarray] = None

        # Настройки
        self.min_weight = 0.05          # минимум 5%
        self.max_weight = 0.50          # максимум 50%
        self.lookback_period = 50       # окно для расчётов
        self.rebalance_interval = 24    # часов между ребалансами
        self.last_rebalance: Optional[datetime] = None

        # Веса ансамбля методов
        self.method_weights = {
            "thompson": 0.25,
            "risk_parity": 0.25,
            "mean_variance": 0.25,
            "momentum": 0.25
        }

    def update_performance(
        self,
        strategy_name: str,
        pnl: float,
        is_win: bool
    ):
        """
        Обновить метрики после закрытия сделки

        Args:
            strategy_name: имя стратегии
            pnl: прибыль/убыток в долларах
            is_win: выиграна ли сделка
        """
        if strategy_name not in self.performance:
            return

        perf = self.performance[strategy_name]
        perf.recent_returns.append(pnl)
        perf.total_trades += 1
        perf.total_pnl += pnl
        perf.last_trade_time = datetime.now()

        if is_win:
            perf.wins += 1
            perf.consecutive_losses = 0
        else:
            perf.losses += 1
            perf.consecutive_losses += 1

        # Обновляем drawdown
        perf.peak_equity = max(perf.peak_equity, perf.total_pnl)
        perf.current_drawdown = perf.peak_equity - perf.total_pnl
        perf.max_drawdown = max(
            perf.max_drawdown, perf.current_drawdown
        )

        # Sharpe ratio
        if len(perf.recent_returns) >= 10:
            returns = np.array(perf.recent_returns)
            if returns.std() > 0:
                perf.sharpe_ratio = (
                    returns.mean() / returns.std()
                    * np.sqrt(252)
                )

        # Calmar ratio
        if perf.max_drawdown > 0:
            perf.calmar_ratio = perf.total_pnl / perf.max_drawdown

        # Thompson Sampling обновление
        idx = self.STRATEGY_NAMES.index(strategy_name)
        self.bandit.update(idx, pnl)

    def get_optimal_weights(self) -> Dict[str, float]:
        """
        Рассчитать оптимальные веса
        Ансамбль из 4 методов
        """
        # Проверяем нужна ли ребалансировка
        if not self._should_rebalance():
            return self.weights

        # Собираем веса от каждого метода
        w_thompson = self._thompson_weights()
        w_risk_parity = self._risk_parity_weights()
        w_mean_var = self._mean_variance_weights()
        w_momentum = self._momentum_weights()

        # Ансамбль
        final_weights = {}
        for name in self.STRATEGY_NAMES:
            w = (
                self.method_weights["thompson"]
                * w_thompson.get(name, 0.25)
                + self.method_weights["risk_parity"]
                * w_risk_parity.get(name, 0.25)
                + self.method_weights["mean_variance"]
                * w_mean_var.get(name, 0.25)
                + self.method_weights["momentum"]
                * w_momentum.get(name, 0.25)
            )
            final_weights[name] = w

        # Нормализация + ограничения
        final_weights = self._apply_constraints(final_weights)

        # Штраф за серию убытков
        final_weights = self._apply_drawdown_penalty(final_weights)

        # Финальная нормализация
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {
                k: round(v / total, 4)
                for k, v in final_weights.items()
            }

        self.weights = final_weights
        self.last_rebalance = datetime.now()

        # Сохраняем историю
        self.weight_history.append({
            "time": datetime.now().isoformat(),
            "weights": dict(final_weights),
            "method_contributions": {
                "thompson": w_thompson,
                "risk_parity": w_risk_parity,
                "mean_variance": w_mean_var,
                "momentum": w_momentum
            }
        })

        return final_weights

    # ─── Метод 1: Thompson Sampling ──────────────

    def _thompson_weights(self) -> Dict[str, float]:
        """Веса на основе Thompson Sampling"""
        probs = self.bandit.get_probabilities()
        total = probs.sum()

        if total == 0:
            return {
                name: 1.0 / self.n_strategies
                for name in self.STRATEGY_NAMES
            }

        return {
            name: float(probs[i] / total)
            for i, name in enumerate(self.STRATEGY_NAMES)
        }

    # ─── Метод 2: Risk Parity ───────────────────

    def _risk_parity_weights(self) -> Dict[str, float]:
        """
        Risk Parity: распределяем так, чтобы
        каждая стратегия вносила РАВНЫЙ риск
        Вес обратно пропорционален волатильности
        """
        inv_vols = {}

        for name in self.STRATEGY_NAMES:
            perf = self.performance[name]
            vol = perf.return_std

            if vol == 0:
                vol = 1.0

            inv_vols[name] = 1.0 / vol

        total = sum(inv_vols.values())

        if total == 0:
            return {
                name: 1.0 / self.n_strategies
                for name in self.STRATEGY_NAMES
            }

        return {
            name: inv_vols[name] / total
            for name in self.STRATEGY_NAMES
        }

    # ─── Метод 3: Mean-Variance (Markowitz) ─────

    def _mean_variance_weights(self) -> Dict[str, float]:
        """
        Упрощённая оптимизация Markowitz
        Максимизируем Sharpe Ratio портфеля
        """
        n = self.n_strategies
        returns_data = []

        for name in self.STRATEGY_NAMES:
            perf = self.performance[name]
            if len(perf.recent_returns) >= 5:
                returns_data.append(list(perf.recent_returns))
            else:
                returns_data.append([0.0] * 5)

        # Выравниваем длины
        min_len = min(len(r) for r in returns_data)
        if min_len < 3:
            return {
                name: 1.0 / n
                for name in self.STRATEGY_NAMES
            }

        returns_matrix = np.array([
            r[-min_len:] for r in returns_data
        ])

        # Средние доходности
        mean_returns = returns_matrix.mean(axis=1)

        # Ковариационная матрица
        cov_matrix = np.cov(returns_matrix)

        if cov_matrix.ndim == 0:
            return {
                name: 1.0 / n
                for name in self.STRATEGY_NAMES
            }

        # Сохраняем корреляционную матрицу
        std_devs = np.sqrt(np.diag(cov_matrix))
        std_devs = np.where(std_devs == 0, 1, std_devs)
        self.correlation_matrix = (
            cov_matrix
            / np.outer(std_devs, std_devs)
        )

        # Простая оптимизация: вес ~ Sharpe
        sharpe_scores = []
        for i in range(n):
            std = std_devs[i] if std_devs[i] > 0 else 1
            sharpe_scores.append(mean_returns[i] / std)

        # Сдвигаем чтобы все были положительные
        min_score = min(sharpe_scores)
        if min_score < 0:
            sharpe_scores = [
                s - min_score + 0.01 for s in sharpe_scores
            ]

        total = sum(sharpe_scores)
        if total == 0:
            return {
                name: 1.0 / n
                for name in self.STRATEGY_NAMES
            }

        return {
            name: sharpe_scores[i] / total
            for i, name in enumerate(self.STRATEGY_NAMES)
        }

    # ─── Метод 4: Momentum ──────────────────────

    def _momentum_weights(self) -> Dict[str, float]:
        """
        Momentum: больше капитала стратегиям,
        которые показали лучшие результаты за последний период
        """
        scores = {}

        for name in self.STRATEGY_NAMES:
            perf = self.performance[name]

            if len(perf.recent_returns) < 5:
                scores[name] = 1.0
                continue

            recent = list(perf.recent_returns)[-20:]

            # Комбинация метрик
            avg_return = np.mean(recent) if recent else 0
            winrate = perf.winrate
            sharpe = perf.sharpe_ratio

            # Взвешенный скор
            score = (
                avg_return * 0.4
                + winrate * 0.3
                + max(sharpe, 0) * 0.1
                + 0.2  # базовый
            )

            scores[name] = max(score, 0.01)

        total = sum(scores.values())

        return {
            name: scores[name] / total
            for name in self.STRATEGY_NAMES
        }

    # ─── Ограничения и штрафы ────────────────────

    def _apply_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Применить min/max ограничения"""
        constrained = {}

        for name, w in weights.items():
            constrained[name] = max(
                self.min_weight,
                min(self.max_weight, w)
            )

        return constrained

    def _apply_drawdown_penalty(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Штраф за серию убытков
        Если стратегия потеряла 5+ сделок подряд — уменьшаем вес
        """
        penalized = {}

        for name, w in weights.items():
            perf = self.performance[name]
            consecutive = perf.consecutive_losses

            if consecutive >= 5:
                penalty = 0.5     # -50%
            elif consecutive >= 3:
                penalty = 0.75    # -25%
            else:
                penalty = 1.0

            penalized[name] = w * penalty

        return penalized

    def _should_rebalance(self) -> bool:
        """Нужна ли ребалансировка"""
        if self.last_rebalance is None:
            return True

        hours = (
            datetime.now() - self.last_rebalance
        ).total_seconds() / 3600

        return hours >= self.rebalance_interval

    # ─── Отчётность ─────────────────────────────

    def get_allocation_report(self) -> Dict:
        """Полный отчёт по аллокации"""
        report = {
            "current_weights": dict(self.weights),
            "strategies": {}
        }

        for name in self.STRATEGY_NAMES:
            perf = self.performance[name]
            report["strategies"][name] = {
                "weight": self.weights.get(name, 0),
                "total_trades": perf.total_trades,
                "winrate": round(perf.winrate * 100, 1),
                "total_pnl": round(perf.total_pnl, 2),
                "sharpe": round(perf.sharpe_ratio, 2),
                "max_dd": round(perf.max_drawdown, 2),
                "consecutive_losses": perf.consecutive_losses,
                "avg_return": round(perf.avg_return, 2)
            }

        if self.correlation_matrix is not None:
            report["correlation_matrix"] = (
                self.correlation_matrix.tolist()
            )

        report["bandit_probs"] = {
            name: round(float(p), 3)
            for name, p in zip(
                self.STRATEGY_NAMES,
                self.bandit.get_probabilities()
            )
        }

        return report

    def save_state(self, path: str = "data/meta_ai_state.json"):
        """Сохранить состояние"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "weights": self.weights,
            "bandit_alpha": self.bandit.alpha.tolist(),
            "bandit_beta": self.bandit.beta.tolist(),
            "weight_history": self.weight_history[-50:],
            "timestamp": datetime.now().isoformat()
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str = "data/meta_ai_state.json"):
        """Загрузить состояние"""
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                state = json.load(f)

            self.weights = state.get("weights", self.weights)
            self.bandit.alpha = np.array(
                state.get("bandit_alpha", [1, 1, 1, 1])
            )
            self.bandit.beta = np.array(
                state.get("bandit_beta", [1, 1, 1, 1])
            )

            print("[META-AI] Состояние загружено")
        except Exception as e:
            print(f"[META-AI] Ошибка загрузки: {e}")
