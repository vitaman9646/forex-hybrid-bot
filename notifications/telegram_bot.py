"""
Telegram Notifications
Отправка уведомлений о сделках и статистике
"""

import requests
from typing import Dict, Optional
from config import telegram_config


class TelegramNotifier:
    """Отправка уведомлений в Telegram"""

    def __init__(self):
        self.enabled = telegram_config.enabled
        self.token = telegram_config.bot_token
        self.chat_id = telegram_config.chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, text: str) -> bool:
        """Отправить текстовое сообщение"""
        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200

        except Exception as e:
            print(f"[TELEGRAM] Ошибка отправки: {e}")
            return False

    def notify_trade_open(self, trade_info: Dict):
        """Уведомление об открытии сделки"""
        msg = (
            "🟢 <b>Новая сделка</b>\n\n"
            f"📊 Символ: <b>{trade_info.get('symbol', '?')}</b>\n"
            f"📈 Тип: <b>{trade_info.get('type', '?')}</b>\n"
            f"💰 Цена: {trade_info.get('price', 0)}\n"
            f"🛑 SL: {trade_info.get('sl', 0)}\n"
            f"🎯 TP: {trade_info.get('tp', 0)}\n"
            f"📦 Лот: {trade_info.get('volume', 0)}\n"
            f"🤖 Стратегия: {trade_info.get('strategy', '?')}\n"
            f"📝 Причина: {trade_info.get('reason', '?')}\n"
            f"🎯 Уверенность: {trade_info.get('confidence', 0)}"
        )
        self.send_message(msg)

    def notify_trade_close(
        self,
        trade_info: Dict,
        profit: float
    ):
        """Уведомление о закрытии сделки"""
        emoji = "✅" if profit >= 0 else "❌"
        msg = (
            f"{emoji} <b>Сделка закрыта</b>\n\n"
            f"📊 Символ: <b>{trade_info.get('symbol', '?')}</b>\n"
            f"📈 Тип: {trade_info.get('type', '?')}\n"
            f"💰 Прибыль: <b>{profit:.2f}</b>\n"
            f"🤖 Стратегия: {trade_info.get('strategy', '?')}"
        )
        self.send_message(msg)

    def notify_daily_stats(self, stats: Dict):
        """Ежедневная статистика"""
        msg = (
            "📊 <b>Дневная статистика</b>\n\n"
            f"💰 Баланс: {stats.get('balance', 0):.2f}\n"
            f"📈 Дневной P&L: {stats.get('daily_pnl', 0):.2f}\n"
            f"📊 Сделок: {stats.get('daily_trades', 0)}\n"
            f"✅ Win: {stats.get('daily_wins', 0)}\n"
            f"❌ Loss: {stats.get('daily_losses', 0)}\n"
            f"📦 Открыто: {stats.get('open_positions', 0)}"
        )
        self.send_message(msg)

    def notify_error(self, error_msg: str):
        """Уведомление об ошибке"""
        msg = f"⚠️ <b>ОШИБКА</b>\n\n{error_msg}"
        self.send_message(msg)

    def notify_regime_change(
        self,
        old_regime: str,
        new_regime: str
    ):
        """Уведомление о смене режима рынка"""
        msg = (
            "🔄 <b>Смена режима рынка</b>\n\n"
            f"Было: {old_regime}\n"
            f"Стало: <b>{new_regime}</b>"
        )
        self.send_message(msg)
