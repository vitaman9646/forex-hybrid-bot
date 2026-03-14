"""
portfolio/multi_currency_bot.py

Portfolio-бот для торговли 20+ валютными парами
Учитывает корреляции, диверсификацию, экспозицию по валютам
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class CurrencyExposure:
    """Экспозиция по отдельной валюте"""
    currency: str
    long_exposure: float = 0.0     # суммарная длинная
    short_exposure: float = 0.0    # суммарная короткая
    net_exposure: float = 0.0      # чистая
    pairs_involved: List[str] = field(default_factory=list)


@dataclass
class PairAnalysis:
    """Анализ торговой пары"""
    symbol: str
    spread_cost: float          # стоимость спреда
    avg_daily_range: float      # средний дневной диапазон
    liquidity_score: float      # ликвидность 0-1
    correlation_cluster: int    # кластер корреляции
    tradeable: bool             # можно ли торговать
    priority: int               # приоритет (ниже = лучше)


class CorrelationEngine:
    """
    Анализ корреляций между валютными парами
    Предотвращает открытие коррелированных позиций
    """

    # Известные высокие корреляции
    HIGH_CORRELATION_PAIRS = {
        ("EURUSD", "GBPUSD"): 0.85,
        ("EURUSD", "USDCHF"): -0.90,
        ("GBPUSD", "GBPJPY"): 0.80,
        ("AUDUSD", "NZDUSD"): 0.90,
        ("USDJPY", "USDCHF"): 0.75,
        ("EURJPY", "GBPJPY"): 0.85,
        ("EURUSD", "EURGBP"): 0.60,
    }

    def __init__(self):
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None

    def calculate_correlations(
        self,
        price_data: Dict[str, pd.DataFrame],
        period: int = 100
    ) -> pd.DataFrame:
        """
        Рассчитать матрицу корреляций из реальных данных

        Args:
            price_data: {symbol: DataFrame with 'close'}
            period: период для расчёта
        """
        returns = {}

        for symbol, df in price_data.items():
            if df is not None and len(df) >= period:
                ret = df["close"].pct_change().dropna().tail(period)
                returns[symbol] = ret

        if len(returns) < 2:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns)
        returns_df = returns_df.dropna()

        if len(returns_df) < 20:
            return pd.DataFrame()

        self.correlation_matrix = returns_df.corr()
        self.last_update = datetime.now()

        return self.correlation_matrix

    def get_correlation(
        self,
        pair1: str,
        pair2: str
    ) -> float:
        """Получить корреляцию между двумя парами"""
        # Сначала из рассчитанной матрицы
        if (
            self.correlation_matrix is not None
            and pair1 in self.correlation_matrix.columns
            and pair2 in self.correlation_matrix.columns
        ):
            return float(
                self.correlation_matrix.loc[pair1, pair2]
            )

        # Fallback: из известных корреляций
        key1 = (pair1, pair2)
        key2 = (pair2, pair1)

        if key1 in self.HIGH_CORRELATION_PAIRS:
            return self.HIGH_CORRELATION_PAIRS[key1]
        if key2 in self.HIGH_CORRELATION_PAIRS:
            return self.HIGH_CORRELATION_PAIRS[key2]

        return 0.0

    def find_correlated_pairs(
        self,
        symbol: str,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Найти пары, сильно коррелированные с данной"""
        correlated = []

        if self.correlation_matrix is not None:
            if symbol in self.correlation_matrix.columns:
                corrs = self.correlation_matrix[symbol]
                for other, corr in corrs.items():
                    if (
                        other != symbol
                        and abs(corr) >= threshold
                    ):
                        correlated.append((other, float(corr)))

        # Добавляем из known correlations
        for (p1, p2), corr in self.HIGH_CORRELATION_PAIRS.items():
            if p1 == symbol and abs(corr) >= threshold:
                correlated.append((p2, corr))
            elif p2 == symbol and abs(corr) >= threshold:
                correlated.append((p1, corr))

        return correlated


class MultiCurrencyPortfolio:
    """
    Мультивалютный портфель-бот

    Функции:
    1. Торговля 20+ парами одновременно
    2. Контроль корреляций
    3. Контроль экспозиции по валютам
    4. Оптимальное распределение капитала
    5. Приоритизация пар по качеству сигнала
    """

    # Все торгуемые пары (можно расширять)
    ALL_PAIRS = {
        # Мажоры
        "major": [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "USDCAD", "NZDUSD"
        ],
        # Кроссы
        "cross": [
            "EURGBP", "EURJPY", "GBPJPY", "EURAUD",
            "EURNZD", "GBPAUD", "AUDNZD", "AUDJPY",
            "NZDJPY", "CADJPY", "CHFJPY", "GBPCAD"
        ],
        # Экзотики (опционально)
        "exotic": [
            "USDMXN", "USDZAR", "USDTRY",
            "EURTRY", "USDSEK", "USDNOK"
        ]
    }

    def __init__(self, max_pairs: int = 20):
        self.max_pairs = max_pairs
        self.correlation_engine = CorrelationEngine()

        # Активные пары
        self.active_pairs: List[str] = []

        # Открытые позиции
        self.open_positions: Dict[str, Dict] = {}

        # Лимиты
        self.max_positions = 8               # макс открытых
        self.max_per_currency = 3            # макс позиций с одной валютой
        self.max_correlated = 2              # макс коррелированных
        self.correlation_threshold = 0.7     # порог корреляции
        self.max_exposure_pct = 0.15         # макс 15% на одну валюту

    def select_trading_pairs(
        self,
        pair_analyses: Dict[str, PairAnalysis]
    ) -> List[str]:
        """
        Выбрать оптимальный набор пар для торговли
        Учитывает ликвидность, спред, корреляции
        """
        candidates = []

        for symbol, analysis in pair_analyses.items():
            if analysis.tradeable:
                candidates.append(analysis)

        # Сортируем по приоритету (спред + ликвидность)
        candidates.sort(key=lambda x: x.priority)

        selected = []
        selected_currencies: Set[str] = set()

        for candidate in candidates:
            if len(selected) >= self.max_pairs:
                break

            symbol = candidate.symbol

            # Проверяем корреляцию с уже выбранными
            too_correlated = False
            for existing in selected:
                corr = abs(
                    self.correlation_engine.get_correlation(
                        symbol, existing
                    )
                )
                if corr >= self.correlation_threshold:
                    too_correlated = True
                    break

            if too_correlated:
                continue

            # Проверяем экспозицию валют
            base, quote = self._split_pair(symbol)
            base_count = sum(
                1 for c in selected_currencies if c == base
            )
            quote_count = sum(
                1 for c in selected_currencies if c == quote
            )

            if (
                base_count >= self.max_per_currency
                or quote_count >= self.max_per_currency
            ):
                continue

            selected.append(symbol)
            selected_currencies.add(base)
            selected_currencies.add(quote)

        self.active_pairs = selected
        return selected

    def can_open_position(
        self,
        symbol: str,
        direction: str,     # "buy" or "sell"
        lot_size: float
    ) -> Tuple[bool, str]:
        """
        Проверка: можно ли открыть новую позицию

        Returns:
            (approved, reason)
        """
        # 1. Макс позиций
        if len(self.open_positions) >= self.max_positions:
            return False, (
                f"Макс позиций ({self.max_positions}) достигнуто"
            )

        # 2. Уже есть позиция по этой паре
        if symbol in self.open_positions:
            return False, f"Уже есть позиция по {symbol}"

        # 3. Корреляционная проверка
        correlated_count = 0
        for open_symbol in self.open_positions:
            corr = abs(
                self.correlation_engine.get_correlation(
                    symbol, open_symbol
                )
            )
            if corr >= self.correlation_threshold:
                correlated_count += 1

        if correlated_count >= self.max_correlated:
            return False, (
                f"Слишком много коррелированных позиций "
                f"({correlated_count})"
            )

        # 4. Валютная экспозиция
        exposure = self.calculate_currency_exposure()
        base, quote = self._split_pair(symbol)

        base_exp = exposure.get(base, CurrencyExposure(base))
        quote_exp = exposure.get(quote, CurrencyExposure(quote))

        if direction == "buy":
            # Покупка: long base, short quote
            new_base_net = abs(
                base_exp.net_exposure + lot_size
            )
            new_quote_net = abs(
                quote_exp.net_exposure - lot_size
            )
        else:
            new_base_net = abs(
                base_exp.net_exposure - lot_size
            )
            new_quote_net = abs(
                quote_exp.net_exposure + lot_size
            )

        total_exposure = sum(
            abs(e.net_exposure)
            for e in exposure.values()
        ) + lot_size

        if total_exposure > 0:
            if (
                new_base_net / total_exposure
                > self.max_exposure_pct
            ):
                return False, (
                    f"Экспозиция {base} "
                    f"превышает {self.max_exposure_pct:.0%}"
                )

        # 5. Проверка хеджирования
        hedge_warning = self._check_hedge(
            symbol, direction
        )
        if hedge_warning:
            return False, hedge_warning

        return True, "OK"

    def calculate_currency_exposure(
        self
    ) -> Dict[str, CurrencyExposure]:
        """Рассчитать экспозицию по каждой валюте"""
        exposure: Dict[str, CurrencyExposure] = {}

        for symbol, pos in self.open_positions.items():
            base, quote = self._split_pair(symbol)
            lot = pos.get("volume", 0)
            direction = pos.get("type", "buy")

            # Инициализация
            if base not in exposure:
                exposure[base] = CurrencyExposure(currency=base)
            if quote not in exposure:
                exposure[quote] = CurrencyExposure(currency=quote)

            if direction == "buy":
                # Long base, Short quote
                exposure[base].long_exposure += lot
                exposure[quote].short_exposure += lot
            else:
                # Short base, Long quote
                exposure[base].short_exposure += lot
                exposure[quote].long_exposure += lot

            exposure[base].pairs_involved.append(symbol)
            exposure[quote].pairs_involved.append(symbol)

        # Рассчитываем net
        for curr, exp in exposure.items():
            exp.net_exposure = (
                exp.long_exposure - exp.short_exposure
            )

        return exposure

    def allocate_capital_per_pair(
        self,
        total_capital: float,
        pair_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Распределение капитала между парами
        На основе скоров (качество сигнала × ликвидность)

        Returns:
            {symbol: allocated_capital}
        """
        if not pair_scores:
            return {}

        # Нормализуем скоры
        total_score = sum(pair_scores.values())
        if total_score == 0:
            equal = total_capital / len(pair_scores)
            return {s: equal for s in pair_scores}

        allocation = {}
        max_per_pair = total_capital * 0.15  # макс 15% на пару

        for symbol, score in pair_scores.items():
            raw_alloc = total_capital * (score / total_score)
            allocation[symbol] = min(raw_alloc, max_per_pair)

        # Перераспределяем обрезанное
        allocated = sum(allocation.values())
        if allocated < total_capital * 0.95:
            remaining = total_capital - allocated
            n_pairs = len(allocation)
            bonus = remaining / n_pairs

            for symbol in allocation:
                allocation[symbol] = min(
                    allocation[symbol] + bonus,
                    max_per_pair
                )

        return allocation

    def register_position(
        self,
        symbol: str,
        position_info: Dict
    ):
        """Зарегистрировать открытую позицию"""
        self.open_positions[symbol] = position_info

    def remove_position(self, symbol: str):
        """Удалить закрытую позицию"""
        self.open_positions.pop(symbol, None)

    def _check_hedge(
        self,
        symbol: str,
        direction: str
    ) -> Optional[str]:
        """
        Проверка на неэффективное хеджирование
        Например: BUY EURUSD + SELL EURUSD (через корреляцию)
        """
        base, quote = self._split_pair(symbol)

        for open_symbol, pos in self.open_positions.items():
            open_base, open_quote = self._split_pair(open_symbol)
            open_dir = pos.get("type", "buy")

            # Прямой хедж
            if open_symbol == symbol:
                continue

            # Проверяем обратные пары
            # Например: BUY EURUSD и SELL USDCHF
            # (высокая отрицательная корреляция)
            corr = self.correlation_engine.get_correlation(
                symbol, open_symbol
            )

            if corr < -0.8:
                # Высокая отрицательная корреляция
                if direction == open_dir:
                    return (
                        f"Потенциальный хедж: "
                        f"{direction} {symbol} "
                        f"vs {open_dir} {open_symbol} "
                        f"(corr={corr:.2f})"
                    )

            if corr > 0.8:
                # Высокая положительная корреляция
                if direction != open_dir:
                    return (
                        f"Потенциальный хедж: "
                        f"{direction} {symbol} "
                        f"vs {open_dir} {open_symbol} "
                        f"(corr={corr:.2f})"
                    )

        return None

    @staticmethod
    def _split_pair(symbol: str) -> Tuple[str, str]:
        """Разделить пару на base/quote"""
        symbol = symbol.upper().replace(".", "")

        if len(symbol) == 6:
            return symbol[:3], symbol[3:]

        return symbol, "USD"

    def get_portfolio_summary(self) -> Dict:
        """Сводка по портфелю"""
        exposure = self.calculate_currency_exposure()

        return {
            "active_pairs": len(self.active_pairs),
            "open_positions": len(self.open_positions),
            "positions": dict(self.open_positions),
            "currency_exposure": {
                curr: {
                    "long": exp.long_exposure,
                    "short": exp.short_exposure,
                    "net": exp.net_exposure,
                    "pairs": exp.pairs_involved
                }
                for curr, exp in exposure.items()
            },
            "total_exposure": sum(
                abs(e.net_exposure) for e in exposure.values()
            )
        }

    def analyze_pair(
        self,
        symbol: str,
        df: pd.DataFrame,
        spread: float
    ) -> PairAnalysis:
        """Анализ отдельной пары"""
        if df is None or len(df) < 20:
            return PairAnalysis(
                symbol=symbol,
                spread_cost=999,
                avg_daily_range=0,
                liquidity_score=0,
                correlation_cluster=0,
                tradeable=False,
                priority=999
            )

        # Средний дневной диапазон
        daily_range = (df["high"] - df["low"]).mean()

        # Стоимость спреда относительно ATR
        atr = df["atr"].iloc[-1] if "atr" in df.columns else daily_range
        spread_ratio = spread / atr if atr > 0 else 999

        # Ликвидность (из объёма)
        if "volume" in df.columns:
            vol_score = min(
                df["volume"].mean() / 10000, 1.0
            )
        else:
            vol_score = 0.5

        # Tradeable: спред не слишком большой
        tradeable = spread_ratio < 0.3 and vol_score > 0.2

        # Приоритет (ниже = лучше)
        priority = int(spread_ratio * 100 - vol_score * 50)

        return PairAnalysis(
            symbol=symbol,
            spread_cost=round(spread_ratio, 4),
            avg_daily_range=round(daily_range, 6),
            liquidity_score=round(vol_score, 2),
            correlation_cluster=0,
            tradeable=tradeable,
            priority=max(1, priority)
  )
