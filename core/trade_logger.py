"""
Trade Logger
Логирование всех сделок для анализа и обучения AI
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import json


class TradeLogger:
    """Сохранение и загрузка истории сделок"""

    def __init__(self, filepath: str = "data/trade_log.csv"):
        self.filepath = filepath
        self.columns = [
            "timestamp",
            "symbol",
            "signal_type",
            "strategy",
            "entry_price",
            "exit_price",
            "stop_loss",
            "take_profit",
            "lot_size",
            "profit",
            "pnl_pips",
            "confidence",
            "market_regime",
            "adx",
            "atr",
            "rsi",
            "volatility",
            "spread",
            "duration_minutes",
            "reason",
            "result",        # win / loss / breakeven
            "metadata"
        ]
        self._ensure_file()

    def _ensure_file(self):
        """Создать файл с заголовками если не существует"""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log_trade(
        self,
        trade_info: Dict,
        market_info: Dict = None,
        exit_info: Dict = None
    ):
        """Записать сделку в лог"""
        market_info = market_info or {}
        exit_info = exit_info or {}

        # Определяем результат
        profit = exit_info.get("profit", 0)
        if profit > 0:
            result = "win"
        elif profit < 0:
            result = "loss"
        else:
            result = "breakeven"

        row = {
            "timestamp": datetime.now().isoformat(),
            "symbol": trade_info.get("symbol", ""),
            "signal_type": trade_info.get("type", ""),
            "strategy": trade_info.get("strategy", ""),
            "entry_price": trade_info.get("price", 0),
            "exit_price": exit_info.get("exit_price", 0),
            "stop_loss": trade_info.get("sl", 0),
            "take_profit": trade_info.get("tp", 0),
            "lot_size": trade_info.get("volume", 0),
            "profit": profit,
            "pnl_pips": exit_info.get("pnl_pips", 0),
            "confidence": trade_info.get("confidence", 0),
            "market_regime": market_info.get("regime", ""),
            "adx": market_info.get("adx", 0),
            "atr": market_info.get("atr", 0),
            "rsi": market_info.get("rsi", 0),
            "volatility": market_info.get("volatility", 0),
            "spread": market_info.get("spread", 0),
            "duration_minutes": exit_info.get("duration", 0),
            "reason": trade_info.get("reason", ""),
            "result": result,
            "metadata": json.dumps(
                trade_info.get("metadata", {}),
                default=str
            )
        }

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)

        print(f"[LOG] Сделка записана: {result} | "
              f"profit={profit:.2f}")

    def get_all_trades(self) -> List[Dict]:
        """Получить все сделки"""
        trades = []

        if not os.path.exists(self.filepath):
            return trades

        with open(self.filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(dict(row))

        return trades

    def get_strategy_stats(self, strategy_name: str) -> Dict:
        """Статистика по конкретной стратегии"""
        trades = self.get_all_trades()
        strategy_trades = [
            t for t in trades
            if t["strategy"] == strategy_name
        ]

        if not strategy_trades:
            return {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "winrate": 0,
                "total_profit": 0,
                "avg_profit": 0
            }

        wins = sum(1 for t in strategy_trades if t["result"] == "win")
        losses = sum(1 for t in strategy_trades if t["result"] == "loss")
        total = len(strategy_trades)
        total_profit = sum(float(t["profit"]) for t in strategy_trades)

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winrate": round(wins / total * 100, 1) if total > 0 else 0,
            "total_profit": round(total_profit, 2),
            "avg_profit": round(
                total_profit / total, 2
            ) if total > 0 else 0
        }

    def get_overall_stats(self) -> Dict:
        """Общая статистика"""
        trades = self.get_all_trades()

        if not trades:
            return {"total": 0, "message": "Нет сделок"}

        total = len(trades)
        wins = sum(1 for t in trades if t["result"] == "win")
        losses = sum(1 for t in trades if t["result"] == "loss")
        profits = [float(t["profit"]) for t in trades]
        total_profit = sum(profits)

        # Profit Factor
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )

        # Max Drawdown (упрощённый)
        cumulative = []
        running = 0
        for p in profits:
            running += p
            cumulative.append(running)

        peak = 0
        max_dd = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "winrate": round(wins / total * 100, 1),
            "total_profit": round(total_profit, 2),
            "avg_profit": round(total_profit / total, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_dd, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2)
      }
