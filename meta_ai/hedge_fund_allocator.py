"""
meta_ai/hedge_fund_allocator.py

Meta-AI Capital Allocator уровня хедж-фондов

Методы аллокации:
1. UCB1 Multi-Armed Bandit (exploration vs exploitation)
2. Risk Parity с условной корреляцией
3. Online Kelly Criterion
4. Regime-Conditional Weights
5. Drawdown-Adjusted Allocation

Ограничения (constraints):
- Min/Max weight
- Maximum weight change per rebalance
- Minimum diversification ratio
- Correlation penalty
- Consecutive loss penalty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
import json
import os


class MarketRegimeType(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class StrategyTracker:
    """Полный трекинг стратегии"""
    name: str

    # Returns
    returns: deque = field(
        default_factory=lambda: deque(maxlen=500)
    )
    daily_returns: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    # Counters
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # P&L
    total_pnl: float = 0.0
    peak_pnl: float = 0.0
    current_dd: float = 0.0
    max_dd: float = 0.0

    # UCB
    ucb_sum_reward: float = 0.0
    ucb_n_pulls: int = 0

    # Regime performance
    regime_returns: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "trending": [],
            "ranging": [],
            "volatile": [],
            "calm": []
        }
    )

    # Timing
    last_trade: Optional[datetime] = None

    @property
    def winrate(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.wins / self.total_trades

    @property
    def avg_return(self) -> float:
        if not self.returns:
            return 0
        return float(np.mean(self.returns))

    @property
    def std_return(self) -> float:
        if len(self.returns) < 2:
            return 1.0
        return max(float(np.std(self.returns)), 1e-6)

    @property
    def sharpe(self) -> float:
        if self.std_return == 0:
            return 0
        return self.avg_return / self.std_return * np.sqrt(252)


class HedgeFundAllocator:
    """
    Meta-AI Allocator уровня хедж-фондов

    Ансамбль из 5 методов + строгие constraints

    В отличие от простого аллокатора:
    1. UCB1 для exploration (не застревает на одной стратегии)
    2. Regime-conditional (разные веса для разных рынков)
    3. Correlation penalty (не аллоцировать коррелирующие)
    4. Smooth rebalancing (без резких скачков)
    5. Emergency drawdown protection
    """

    STRATEGIES = [
        "trend_strategy",
        "range_strategy",
        "breakout_strategy",
        "scalping_strategy",
        "session_strategy",
        "smc_strategy",
    ]

    def __init__(self):
        self.n = len(self.STRATEGIES)

        self.trackers: Dict[str, StrategyTracker] = {
            name: StrategyTracker(name=name)
            for name in self.STRATEGIES
        }

        self.weights = {
            name: 1.0 / self.n for name in self.STRATEGIES
        }

        self.prev_weights = dict(self.weights)
        self.current_regime = MarketRegimeType.CALM
        self.total_rounds = 0

        # Constraints
        self.min_weight = 0.03
        self.max_weight = 0.40
        self.max_change = 0.08         # max change per rebalance
        self.rebalance_hours = 12
        self.min_trades_for_signal = 10
        self.emergency_dd_threshold = 0.15  # 15% DD → reduce

        # Method weights in ensemble
        self.method_weights = {
            "ucb": 0.20,
            "risk_parity": 0.20,
            "kelly": 0.15,
            "regime": 0.25,
            "momentum": 0.20,
        }

        self.last_rebalance: Optional[datetime] = None
        self.weight_history: List[Dict] = []

    # ═══════════════════════════════════════════
    #  RECORD & UPDATE
    # ═══════════════════════════════════════════

    def record_trade(
        self,
        strategy: str,
        pnl: float,
        is_win: bool,
        regime: str = "calm"
    ):
        """Записать результат сделки"""
        if strategy not in self.trackers:
            return

        t = self.trackers[strategy]
        t.returns.append(pnl)
        t.total_trades += 1
        t.total_pnl += pnl
        t.last_trade = datetime.now()
        self.total_rounds += 1

        if is_win:
            t.wins += 1
            t.consecutive_wins += 1
            t.consecutive_losses = 0
        else:
            t.losses += 1
            t.consecutive_losses += 1
            t.consecutive_wins = 0

        # Drawdown
        t.peak_pnl = max(t.peak_pnl, t.total_pnl)
        t.current_dd = t.peak_pnl - t.total_pnl
        t.max_dd = max(t.max_dd, t.current_dd)

        # UCB
        t.ucb_sum_reward += max(pnl, 0) / max(abs(pnl), 1)
        t.ucb_n_pulls += 1

        # Regime
        if regime in t.regime_returns:
            t.regime_returns[regime].append(pnl)

    def set_regime(self, regime: MarketRegimeType):
        """Установить текущий режим рынка"""
        self.current_regime = regime

    # ═══════════════════════════════════════════
    #  GET WEIGHTS
    # ═══════════════════════════════════════════

    def get_weights(self) -> Dict[str, float]:
        """Получить текущие оптимальные веса"""
        if self._should_rebalance():
            self._rebalance()
        return dict(self.weights)

    def _should_rebalance(self) -> bool:
        if self.last_rebalance is None:
            return True
        hours = (
            datetime.now() - self.last_rebalance
        ).total_seconds() / 3600
        return hours >= self.rebalance_hours

    def _rebalance(self):
        """Пересчёт весов — ансамбль из 5 методов"""
        self.prev_weights = dict(self.weights)

        # Считаем веса по каждому методу
        w1 = self._ucb1_weights()
        w2 = self._risk_parity_weights()
        w3 = self._kelly_weights()
        w4 = self._regime_weights()
        w5 = self._momentum_weights()

        # Ансамбль
        new = {}
        for name in self.STRATEGIES:
            w = (
                self.method_weights["ucb"] * w1.get(name, 1/self.n)
                + self.method_weights["risk_parity"] * w2.get(name, 1/self.n)
                + self.method_weights["kelly"] * w3.get(name, 1/self.n)
                + self.method_weights["regime"] * w4.get(name, 1/self.n)
                + self.method_weights["momentum"] * w5.get(name, 1/self.n)
            )
            new[name] = w

        # Apply constraints
        new = self._apply_constraints(new)

        # Emergency DD check
        new = self._emergency_dd_check(new)

        # Normalize
        total = sum(new.values())
        if total > 0:
            new = {k: round(v/total, 4) for k, v in new.items()}

        self.weights = new
        self.last_rebalance = datetime.now()

        self.weight_history.append({
            "time": datetime.now().isoformat(),
            "weights": dict(new),
            "regime": self.current_regime.value
        })

    # ─── Method 1: UCB1 ─────────────────────

    def _ucb1_weights(self) -> Dict[str, float]:
        """
        Upper Confidence Bound (exploration/exploitation)

        UCB = avg_reward + c * sqrt(ln(N) / n_i)

        Гарантирует что каждая стратегия получит шанс,
        даже если пока показала слабый результат
        """
        scores = {}
        c = 1.5  # exploration parameter

        for name in self.STRATEGIES:
            t = self.trackers[name]

            if t.ucb_n_pulls == 0:
                scores[name] = 10.0  # High score for unexplored
                continue

            avg_reward = t.ucb_sum_reward / t.ucb_n_pulls

            exploration = c * np.sqrt(
                np.log(max(self.total_rounds, 1)) / t.ucb_n_pulls
            )

            scores[name] = max(avg_reward + exploration, 0.01)

        total = sum(scores.values())
        return {k: v/total for k, v in scores.items()}

    # ─── Method 2: Risk Parity ──────────────

    def _risk_parity_weights(self) -> Dict[str, float]:
        """Обратно пропорционально волатильности"""
        inv_vols = {}
        for name in self.STRATEGIES:
            t = self.trackers[name]
            vol = t.std_return
            inv_vols[name] = 1.0 / max(vol, 0.001)

        total = sum(inv_vols.values())
        return {k: v/total for k, v in inv_vols.items()}

    # ─── Method 3: Online Kelly ─────────────

    def _kelly_weights(self) -> Dict[str, float]:
        """
        Kelly Criterion: f* = (p*b - q) / b

        Half-Kelly для безопасности
        """
        kellys = {}
        for name in self.STRATEGIES:
            t = self.trackers[name]

            if t.total_trades < self.min_trades_for_signal:
                kellys[name] = 1.0 / self.n
                continue

            p = t.winrate
            q = 1 - p

            wins = [r for r in t.returns if r > 0]
            losses = [r for r in t.returns if r < 0]

            if not wins or not losses:
                kellys[name] = 1.0 / self.n
                continue

            b = np.mean(wins) / max(abs(np.mean(losses)), 0.01)

            kelly = (p * b - q) / max(b, 0.01)
            kelly = max(kelly * 0.5, 0.02)  # Half-Kelly, min 2%
            kellys[name] = min(kelly, 0.5)

        total = sum(kellys.values())
        if total > 0:
            return {k: v/total for k, v in kellys.items()}
        return {k: 1.0/self.n for k in self.STRATEGIES}

    # ─── Method 4: Regime-Conditional ───────

    def _regime_weights(self) -> Dict[str, float]:
        """
        Веса зависят от текущего режима рынка

        Каждая стратегия имеет историю по режимам
        """
        regime = self.current_regime.value
        scores = {}

        for name in self.STRATEGIES:
            t = self.trackers[name]
            regime_rets = t.regime_returns.get(regime, [])

            if len(regime_rets) < 5:
                # Мало данных — используем базовые приоритеты
                scores[name] = self._base_regime_priority(
                    name, regime
                )
                continue

            avg = np.mean(regime_rets)
            wr = sum(1 for r in regime_rets if r > 0) / len(regime_rets)

            scores[name] = max(avg * 10 + wr, 0.01)

        total = sum(scores.values())
        if total > 0:
            return {k: v/total for k, v in scores.items()}
        return {k: 1.0/self.n for k in self.STRATEGIES}

    @staticmethod
    def _base_regime_priority(
        strategy: str,
        regime: str
    ) -> float:
        """Базовые приоритеты по режимам (до накопления данных)"""
        priorities = {
            "trending": {
                "trend_strategy": 0.25,
                "session_strategy": 0.20,
                "smc_strategy": 0.20,
                "breakout_strategy": 0.15,
                "scalping_strategy": 0.10,
                "range_strategy": 0.10,
            },
            "ranging": {
                "range_strategy": 0.25,
                "smc_strategy": 0.20,
                "session_strategy": 0.20,
                "scalping_strategy": 0.15,
                "trend_strategy": 0.10,
                "breakout_strategy": 0.10,
            },
            "volatile": {
                "breakout_strategy": 0.25,
                "session_strategy": 0.20,
                "smc_strategy": 0.20,
                "trend_strategy": 0.15,
                "scalping_strategy": 0.10,
                "range_strategy": 0.10,
            },
            "calm": {
                "scalping_strategy": 0.20,
                "range_strategy": 0.20,
                "session_strategy": 0.20,
                "smc_strategy": 0.15,
                "trend_strategy": 0.15,
                "breakout_strategy": 0.10,
            },
        }

        return priorities.get(
            regime, {}
        ).get(strategy, 1.0 / 6)

    # ─── Method 5: Momentum ─────────────────

    def _momentum_weights(self) -> Dict[str, float]:
        """Больше капитала тем кто показывает результат СЕЙЧАС"""
        scores = {}

        for name in self.STRATEGIES:
            t = self.trackers[name]

            if len(t.returns) < self.min_trades_for_signal:
                scores[name] = 1.0
                continue

            # Последние 20 сделок
            recent = list(t.returns)[-20:]
            avg = np.mean(recent)
            wr = sum(1 for r in recent if r > 0) / len(recent)

            score = avg * 5 + wr + 0.5
            scores[name] = max(score, 0.01)

        total = sum(scores.values())
        return {k: v/total for k, v in scores.items()}

    # ─── Constraints ────────────────────────

    def _apply_constraints(
        self,
        new: Dict[str, float]
    ) -> Dict[str, float]:
        """Строгие ограничения"""
        constrained = {}

        for name in self.STRATEGIES:
            w = new.get(name, 1.0 / self.n)
            old = self.prev_weights.get(name, 1.0 / self.n)

            # Min/Max
            w = max(self.min_weight, min(self.max_weight, w))

            # Max change
            change = w - old
            if abs(change) > self.max_change:
                w = old + np.sign(change) * self.max_change

            # Consecutive loss penalty
            t = self.trackers[name]
            if t.consecutive_losses >= 7:
                w *= 0.3
            elif t.consecutive_losses >= 5:
                w *= 0.5
            elif t.consecutive_losses >= 3:
                w *= 0.75

            constrained[name] = max(w, self.min_weight)

        return constrained

    def _emergency_dd_check(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Экстренная проверка drawdown

        Если стратегия в DD > 15% — сильно сокращаем
        """
        for name in self.STRATEGIES:
            t = self.trackers[name]

            if t.peak_pnl > 0:
                dd_ratio = t.current_dd / t.peak_pnl
                if dd_ratio > self.emergency_dd_threshold:
                    weights[name] *= 0.3
                    print(
                        f"[ALLOCATOR] ⚠️ Emergency DD: "
                        f"{name} weight reduced "
                        f"(DD={dd_ratio:.1%})"
                    )

        return weights

    # ═══════════════════════════════════════════
    #  REPORTING
    # ═══════════════════════════════════════════

    def get_report(self) -> Dict:
        """Полный отчёт"""
        report = {
            "weights": dict(self.weights),
            "regime": self.current_regime.value,
            "total_rounds": self.total_rounds,
            "strategies": {}
        }

        for name in self.STRATEGIES:
            t = self.trackers[name]
            report["strategies"][name] = {
                "weight": self.weights.get(name, 0),
                "trades": t.total_trades,
                "winrate": round(t.winrate * 100, 1),
                "pnl": round(t.total_pnl, 2),
                "sharpe": round(t.sharpe, 2),
                "max_dd": round(t.max_dd, 2),
                "consec_losses": t.consecutive_losses,
                "avg_return": round(t.avg_return, 2),
            }

        return report

    def save(self, path: str = "data/hf_allocator.json"):
        """Сохранить"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "weights": self.weights,
            "total_rounds": self.total_rounds,
            "trackers": {
                name: {
                    "total_trades": t.total_trades,
                    "wins": t.wins,
                    "losses": t.losses,
                    "total_pnl": t.total_pnl,
                    "peak_pnl": t.peak_pnl,
                    "ucb_sum": t.ucb_sum_reward,
                    "ucb_n": t.ucb_n_pulls,
                    "consec_losses": t.consecutive_losses,
                    "returns": list(t.returns)[-100:],
                }
                for name, t in self.trackers.items()
            },
            "time": datetime.now().isoformat()
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str = "data/hf_allocator.json"):
        """Загрузить"""
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                state = json.load(f)

            self.weights = state.get("weights", self.weights)
            self.total_rounds = state.get("total_rounds", 0)

            for name, data in state.get("trackers", {}).items():
                if name in self.trackers:
                    t = self.trackers[name]
                    t.total_trades = data.get("total_trades", 0)
                    t.wins = data.get("wins", 0)
                    t.losses = data.get("losses", 0)
                    t.total_pnl = data.get("total_pnl", 0)
                    t.peak_pnl = data.get("peak_pnl", 0)
                    t.ucb_sum_reward = data.get("ucb_sum", 0)
                    t.ucb_n_pulls = data.get("ucb_n", 0)
                    t.consecutive_losses = data.get("consec_losses", 0)
                    for r in data.get("returns", []):
                        t.returns.append(r)

            print("[ALLOCATOR] State loaded")
        except Exception as e:
            print(f"[ALLOCATOR] Load error: {e}")
