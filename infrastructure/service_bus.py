"""
infrastructure/service_bus.py

Message Bus для коммуникации между сервисами
Использует Redis Pub/Sub + Streams

Каналы:
  signals     — торговые сигналы
  trades      — открытые/закрытые сделки
  models      — обновления ML моделей
  data        — новые данные
  alerts      — ошибки и предупреждения
  metrics     — метрики для мониторинга
  commands    — команды управления
"""

import json
import time
from typing import Dict, Optional, Callable, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import os

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@dataclass
class Message:
    """Сообщение шины"""
    channel: str
    event: str
    data: Dict
    source: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class ServiceBus:
    """
    Шина сообщений между сервисами

    Паттерны:
    1. Pub/Sub (один-ко-многим)
    2. Request/Reply (запрос-ответ)
    3. Stream (очередь с историей)
    """

    # Стандартные каналы
    CHANNELS = {
        "signals": "forex:signals",
        "trades": "forex:trades",
        "models": "forex:models",
        "data": "forex:data",
        "alerts": "forex:alerts",
        "metrics": "forex:metrics",
        "commands": "forex:commands",
    }

    def __init__(
        self,
        service_name: str = "unknown",
        redis_host: str = None,
        redis_port: int = 6379
    ):
        self.service_name = service_name
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port

        self.redis_client = None
        self.pubsub = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self._listener_thread = None
        self._running = False

        self._connect()

    def _connect(self):
        """Подключение к Redis"""
        if not HAS_REDIS:
            print(f"[BUS:{self.service_name}] Redis not installed, "
                  f"using local mode")
            return

        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            print(
                f"[BUS:{self.service_name}] Connected to "
                f"Redis {self.redis_host}:{self.redis_port}"
            )
        except Exception as e:
            print(f"[BUS:{self.service_name}] Redis error: {e}")
            self.redis_client = None

    # ─── Publish ──────────────────────────────

    def publish(
        self,
        channel: str,
        event: str,
        data: Dict
    ) -> bool:
        """Опубликовать сообщение"""
        msg = Message(
            channel=channel,
            event=event,
            data=data,
            source=self.service_name
        )

        if self.redis_client is None:
            # Local mode: вызываем подписчиков напрямую
            self._local_dispatch(msg)
            return True

        try:
            redis_channel = self.CHANNELS.get(channel, channel)
            payload = json.dumps(asdict(msg))
            self.redis_client.publish(redis_channel, payload)
            return True
        except Exception as e:
            print(f"[BUS] Publish error: {e}")
            return False

    def publish_signal(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        **kwargs
    ):
        """Опубликовать торговый сигнал"""
        self.publish("signals", "new_signal", {
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy,
            "confidence": confidence,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            **kwargs
        })

    def publish_trade(
        self,
        action: str,      # "open" / "close"
        trade_info: Dict
    ):
        """Опубликовать событие сделки"""
        self.publish("trades", action, trade_info)

    def publish_model_update(
        self,
        model_name: str,
        metrics: Dict
    ):
        """Сообщить об обновлении модели"""
        self.publish("models", "updated", {
            "model": model_name,
            "metrics": metrics
        })

    def publish_alert(
        self,
        level: str,       # "info" / "warning" / "error" / "critical"
        message: str,
        details: Dict = None
    ):
        """Опубликовать алерт"""
        self.publish("alerts", level, {
            "message": message,
            "details": details or {}
        })

    def publish_metric(
        self,
        name: str,
        value: float,
        labels: Dict = None
    ):
        """Опубликовать метрику"""
        self.publish("metrics", "gauge", {
            "name": name,
            "value": value,
            "labels": labels or {}
        })

    # ─── Subscribe ────────────────────────────

    def subscribe(
        self,
        channel: str,
        callback: Callable[[Message], None]
    ):
        """Подписаться на канал"""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(callback)

        if self.pubsub:
            redis_channel = self.CHANNELS.get(channel, channel)
            self.pubsub.subscribe(redis_channel)

    def start_listening(self):
        """Запустить прослушивание в фоне"""
        if self.pubsub is None:
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self._listener_thread.start()

    def stop_listening(self):
        """Остановить прослушивание"""
        self._running = False

    def _listen_loop(self):
        """Фоновый цикл прослушивания"""
        while self._running:
            try:
                message = self.pubsub.get_message(timeout=1)
                if message and message["type"] == "message":
                    payload = json.loads(message["data"])
                    msg = Message(**payload)
                    self._dispatch(msg)
            except Exception as e:
                print(f"[BUS] Listen error: {e}")
                time.sleep(1)

    def _dispatch(self, msg: Message):
        """Отправить сообщение подписчикам"""
        callbacks = self.subscribers.get(msg.channel, [])
        for cb in callbacks:
            try:
                cb(msg)
            except Exception as e:
                print(f"[BUS] Callback error: {e}")

    def _local_dispatch(self, msg: Message):
        """Локальная диспетчеризация без Redis"""
        self._dispatch(msg)

    # ─── Key-Value Store ──────────────────────

    def set_state(self, key: str, value: Any, ttl: int = 0):
        """Сохранить состояние в Redis"""
        if self.redis_client is None:
            return

        full_key = f"forex:state:{self.service_name}:{key}"
        payload = json.dumps(value, default=str)

        if ttl > 0:
            self.redis_client.setex(full_key, ttl, payload)
        else:
            self.redis_client.set(full_key, payload)

    def get_state(self, key: str, service: str = None) -> Any:
        """Получить состояние"""
        if self.redis_client is None:
            return None

        svc = service or self.service_name
        full_key = f"forex:state:{svc}:{key}"

        data = self.redis_client.get(full_key)
        if data:
            return json.loads(data)
        return None

    # ─── Shared Data ──────────────────────────

    def set_shared(self, key: str, value: Any):
        """Общие данные между сервисами"""
        if self.redis_client is None:
            return
        self.redis_client.set(
            f"forex:shared:{key}",
            json.dumps(value, default=str)
        )

    def get_shared(self, key: str) -> Any:
        """Получить общие данные"""
        if self.redis_client is None:
            return None
        data = self.redis_client.get(f"forex:shared:{key}")
        return json.loads(data) if data else None
