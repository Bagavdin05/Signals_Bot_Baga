import ccxt
import asyncio
from telegram import Bot, Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler
)
from telegram.error import TelegramError
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import html
import re
import json
import os
import time
import numpy as np

# Общая конфигурация
TELEGRAM_TOKEN = "7952768185:AAGuhybXaGPJqtlGPd1-O4nc6_FpUL2rOgw"
TELEGRAM_CHAT_IDS = ["1167694150", "7916502470", "5381553894", "1111230981"]

# Конфигурация спотового арбитража (по умолчанию)
DEFAULT_SPOT_SETTINGS = {
    "THRESHOLD_PERCENT": 0.5,
    "MAX_THRESHOLD_PERCENT": 40,
    "CHECK_INTERVAL": 30,
    "MIN_EXCHANGES_FOR_PAIR": 2,
    "MIN_VOLUME_USD": 500000,
    "MIN_ENTRY_AMOUNT_USDT": 5,
    "MAX_ENTRY_AMOUNT_USDT": 300,
    "MAX_IMPACT_PERCENT": 0.5,
    "ORDER_BOOK_DEPTH": 10,
    "MIN_NET_PROFIT_USD": 5,
    "ENABLED": True,
    "PRICE_CONVERGENCE_THRESHOLD": 0.5,
    "PRICE_CONVERGENCE_ENABLED": True,
    "VOLATILITY_THRESHOLD": 10.0,
    "MIN_ORDER_BOOK_VOLUME": 100,
    "MAX_VOLATILITY_PERCENT": 15.0
}

# Конфигурация фьючерсного арбитража (по умолчанию)
DEFAULT_FUTURES_SETTINGS = {
    "THRESHOLD_PERCENT": 0.5,
    "MAX_THRESHOLD_PERCENT": 20,
    "CHECK_INTERVAL": 30,
    "MIN_VOLUME_USD": 200000,
    "MIN_EXCHANGES_FOR_PAIR": 2,
    "MIN_ENTRY_AMOUNT_USDT": 5,
    "MAX_ENTRY_AMOUNT_USDT": 150,
    "MIN_NET_PROFIT_USD": 3,
    "ENABLED": True,
    "PRICE_CONVERGENCE_THRESHOLD": 0.5,
    "PRICE_CONVERGENCE_ENABLED": True,
    "VOLATILITY_THRESHOLD": 10.0,
    "MIN_ORDER_BOOK_VOLUME": 1000,
    "FUNDING_RATE_THRESHOLD": 0.01,
    "MIN_FUNDING_RATE_TO_RECEIVE": -0.005,
    "IDEAL_FUNDING_SCENARIO": -0.01,
    "FUNDING_CHECK_INTERVAL": 3600,
    "MAX_HOLDING_HOURS": 24,
    "MAX_IMPACT_PERCENT": 0.5,
    "MAX_VOLATILITY_PERCENT": 15.0,
    "RED_FUNDING_THRESHOLD": 0.005
}

# Конфигурация спот-фьючерсного арбитража (по умолчанию)
DEFAULT_SPOT_FUTURES_SETTINGS = {
    "THRESHOLD_PERCENT": 0.5,
    "MAX_THRESHOLD_PERCENT": 20,
    "CHECK_INTERVAL": 30,
    "MIN_VOLUME_USD": 300000,
    "MIN_EXCHANGES_FOR_PAIR": 2,
    "MIN_ENTRY_AMOUNT_USDT": 5,
    "MAX_ENTRY_AMOUNT_USDT": 150,
    "MIN_NET_PROFIT_USD": 3,
    "ENABLED": True,
    "PRICE_CONVERGENCE_THRESHOLD": 0.5,
    "PRICE_CONVERGENCE_ENABLED": True,
    "VOLATILITY_THRESHOLD": 10.0,
    "MIN_ORDER_BOOK_VOLUME": 1000,
    "MAX_IMPACT_PERCENT": 0.5,
    "MAX_VOLATILITY_PERCENT": 15.0
}

# Настройки бирж
EXCHANGE_SETTINGS = {
    "bybit": {"ENABLED": True},
    "mexc": {"ENABLED": True},
    "okx": {"ENABLED": True},
    "gate": {"ENABLED": True},
    "bitget": {"ENABLED": True},
    "kucoin": {"ENABLED": True},
    "htx": {"ENABLED": True},
    "bingx": {"ENABLED": True},
    "phemex": {"ENABLED": True},
    "coinex": {"ENABLED": True},
    "blofin": {"ENABLED": True}
}

# Состояния для ConversationHandler
SETTINGS_MENU, SPOT_SETTINGS, FUTURES_SETTINGS, SPOT_FUTURES_SETTINGS, EXCHANGE_SETTINGS_MENU, SETTING_VALUE, COIN_SELECTION = range(7)

# Состояния для пагинации
ARBITRAGE_LIST, ARBITRAGE_PAGE = range(8, 10)

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("CryptoArbBot")

# Глобальные переменные для отслеживания истории уведомлений и длительности арбитража
price_convergence_history = defaultdict(dict)
last_convergence_notification = defaultdict(dict)
arbitrage_start_times = defaultdict(dict)
current_arbitrage_opportunities = defaultdict(dict)
previous_arbitrage_opportunities = defaultdict(dict)
sent_arbitrage_opportunities = defaultdict(dict)

# Глобальные переменные для хранения последних настроек бирж
LAST_EXCHANGE_SETTINGS = None

# Глобальные переменные для отслеживания волатильности
price_history = defaultdict(list)
VOLATILITY_WINDOW = 10

# Глобальные переменные для отслеживания ставок финансирования
funding_rates_cache = {}
last_funding_check = 0

# Глобальные переменные для отслеживания волатильности монет
coin_volatility_history = defaultdict(list)
COIN_VOLATILITY_WINDOW = 20

# Глобальные переменные для пагинации
user_pagination_data = defaultdict(dict)

# Загрузка сохраненных настроек
def load_settings():
    try:
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки настроек: {e}")

    return {
        "SPOT": DEFAULT_SPOT_SETTINGS.copy(),
        "FUTURES": DEFAULT_FUTURES_SETTINGS.copy(),
        "SPOT_FUTURES": DEFAULT_SPOT_FUTURES_SETTINGS.copy(),
        "EXCHANGES": EXCHANGE_SETTINGS.copy()
    }

# Сохранение настроек
def save_settings(settings):
    try:
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")

# Глобальные переменные
SHARED_BOT = None
SPOT_EXCHANGES_LOADED = {}
FUTURES_EXCHANGES_LOADED = {}
SETTINGS = load_settings()

# Конфигурация бирж для спота
SPOT_EXCHANGES = {
    "bybit": {
        "api": ccxt.bybit({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://www.bybit.com/trade/spot/{s.replace('/', '')}",
        "withdraw_url": lambda c: f"https://www.bybit.com/user/assets/withdraw",
        "deposit_url": lambda c: f"https://www.bybit.com/user/assets/deposit",
        "emoji": "🏛",
        "blacklist": []
    },
    "mexc": {
        "api": ccxt.mexc({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.002,
        "maker_fee": 0.002,
        "url_format": lambda s: f"https://www.mexc.com/exchange/{s.replace('/', '_')}",
        "withdraw_url": lambda c: f"https://www.mexc.com/ru-RU/assets/withdraw/{c}",
        "deposit_url": lambda c: f"https://www.mexc.com/ru-RU/assets/deposit/{c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "okx": {
        "api": ccxt.okx({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.0008,
        "url_format": lambda s: f"https://www.okx.com/trade-spot/{s.replace('/', '-').lower()}",
        "withdraw_url": lambda c: f"https://www.okx.com/ru/balance/withdrawal/{c.lower()}-chain",
        "deposit_url": lambda c: f"https://www.okx.com/ru/balance/recharge/{c.lower()}",
        "emoji": "🏛",
        "blacklist": ["BTC"]
    },
    "gate": {
        "api": ccxt.gateio({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.002,
        "maker_fee": 0.002,
        "url_format": lambda s: f"https://www.gate.io/trade/{s.replace('/', '_')}",
        "withdraw_url": lambda c: f"https://www.gate.io/myaccount/withdraw/{c}",
        "deposit_url": lambda c: f"https://www.gate.io/myaccount/deposit/{c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "bitget": {
        "api": ccxt.bitget({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://www.bitget.com/spot/{s.replace('/', '')}_SPBL",
        "withdraw_url": lambda c: f"https://www.bitget.com/ru/asset/withdraw?coinId={c}",
        "deposit_url": lambda c: f"https://www.bitget.com/ru/asset/recharge?coinId={c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "kucoin": {
        "api": ccxt.kucoin({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://www.kucoin.com/trade/{s.replace('/', '-')}",
        "withdraw_url": lambda c: f"https://www.kucoin.com/ru/assets/withdraw/{c}",
        "deposit_url": lambda c: f"https://www.kucoin.com/ru/assets/coin/{c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "htx": {
        "api": ccxt.htx({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.002,
        "maker_fee": 0.002,
        "url_format": lambda s: f"https://www.htx.com/trade/{s.replace('/', '_').lower()}",
        "withdraw_url": lambda c: f"https://www.htx.com/ru-ru/finance/withdraw/{c.lower()}",
        "deposit_url": lambda c: f"https://www.htx.com/ru-ru/finance/deposit/{c.lower()}",
        "emoji": "🏛",
        "blacklist": []
    },
    "bingx": {
        "api": ccxt.bingx({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://bingx.com/en-us/spot/{s.replace('/', '')}",
        "withdraw_url": lambda c: f"https://bingx.com/en-us/assets/withdraw/{c}",
        "deposit_url": lambda c: f"https://bingx.com/en-us/assets/deposit/{c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "phemex": {
        "api": ccxt.phemex({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://phemex.com/spot/trade/{s.replace('/', '')}",
        "withdraw_url": lambda c: f"https://phemex.com/assets/withdraw?asset={c}",
        "deposit_url": lambda c: f"https://phemex.com/assets/deposit?asset={c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "coinex": {
        "api": ccxt.coinex({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: m.get('spot', False) and m['quote'] == 'USDT',
        "taker_fee": 0.002,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://www.coinex.com/exchange/{s.replace('/', '-')}",
        "withdraw_url": lambda c: f"https://www.coinex.com/asset/withdraw/{c}",
        "deposit_url": lambda c: f"https://www.coinex.com/asset/deposit/{c}",
        "emoji": "🏛",
        "blacklist": []
    },
    "blofin": {
        "api": ccxt.blofin({
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot"
            }
        }),
        "symbol_format": lambda s: f"{s}/USDT",
        "is_spot": lambda m: (
                m.get('type') == 'spot' and
                m['quote'] == 'USDT'
        ),
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://www.blofin.com/spot/{s.replace('/', '-')}",
        "withdraw_url": lambda c: f"https://www.blofin.com/assets/withdraw/{c}",
        "deposit_url": lambda c: f"https://www.blofin.com/assets/deposit/{c}",
        "emoji": "🏛",
        "blacklist": []
    }
}

# Конфигурация бирж для фьючерсов
FUTURES_EXCHANGES = {
    "bybit": {
        "api": ccxt.bybit({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: (m.get('swap', False) or m.get('future', False)) and m['settle'] == 'USDT',
        "taker_fee": 0.0006,
        "maker_fee": 0.0001,
        "url_format": lambda s: f"https://www.bybit.com/trade/usdt/{s.replace('/', '').replace(':USDT', '')}",
        "blacklist": ["BTC", "ETH"],
        "emoji": "📊",
        "supports_funding": True
    },
    "mexc": {
        "api": ccxt.mexc({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and 'USDT' in m['id'],
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://futures.mexc.com/exchange/{s.replace('/', '_').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "okx": {
        "api": ccxt.okx({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: (m.get('swap', False) or m.get('future', False)) and m['settle'] == 'USDT',
        "taker_fee": 0.0005,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://www.okx.com/trade-swap/{s.replace('/', '-').replace(':USDT', '').lower()}",
        "blacklist": ["BTC", "ETH"],
        "emoji": "📊",
        "supports_funding": True
    },
    "gate": {
        "api": ccxt.gateio({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and '_USDT' in m['id'],
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://www.gate.io/futures_trade/{s.replace('/', '_').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "bitget": {
        "api": ccxt.bitget({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and 'USDT' in m['id'],
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://www.bitget.com/ru/futures/{s.replace('/', '').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "kucoin": {
        "api": ccxt.kucoin({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and 'USDT' in m['id'],
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://www.kucoin.com/futures/trade/{s.replace('/', '-').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "htx": {
        "api": ccxt.htx({
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "fetchMarkets": ["swap"]
            }
        }),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and m.get('linear', False),
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://www.htx.com/futures/exchange/{s.split(':')[0].replace('/', '_').lower()}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "bingx": {
        "api": ccxt.bingx({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and 'USDT' in m['id'],
        "taker_fee": 0.0005,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://bingx.com/en-us/futures/{s.replace('/', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "phemex": {
        "api": ccxt.phemex({
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
            }
        }),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: m.get('swap', False) and m['settle'] == 'USDT',
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://phemex.com/futures/trade/{s.replace('/', '').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "coinex": {
        "api": ccxt.coinex({"enableRateLimit": True}),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: (m.get('swap', False) or m.get('future', False)) and m['settle'] == 'USDT',
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "url_format": lambda s: f"https://www.coinex.com/perpetual/{s.replace('/', '-').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    },
    "blofin": {
        "api": ccxt.blofin({
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap"
            }
        }),
        "symbol_format": lambda s: f"{s}/USDT:USDT",
        "is_futures": lambda m: (
                m.get('type') in ['swap', 'future'] and
                m.get('settle') == 'USDT' and
                m.get('linear', False)
        ),
        "taker_fee": 0.0006,
        "maker_fee": 0.0002,
        "url_format": lambda s: f"https://www.blofin.com/futures/{s.replace('/', '-').replace(':USDT', '')}",
        "blacklist": [],
        "emoji": "📊",
        "supports_funding": True
    }
}

# Reply-клавиатуры
def get_main_keyboard():
    return ReplyKeyboardMarkup([
        [KeyboardButton("📈 Актуальные связки")], [KeyboardButton("🔧 Настройки")],
        [KeyboardButton("📊 Статус бота"), KeyboardButton("ℹ️ Помощь")]
    ], resize_keyboard=True)

def get_settings_keyboard():
    return ReplyKeyboardMarkup([
        [KeyboardButton("🚀️ Спот"), KeyboardButton("📊 Фьючерсы"), KeyboardButton("↔️ Спот-Фьючерсы")],
        [KeyboardButton("🏛 Биржи"), KeyboardButton("🔄 Сброс")],
        [KeyboardButton("🔙 Главное меню")]
    ], resize_keyboard=True)

def get_spot_settings_keyboard():
    spot = SETTINGS['SPOT']
    return ReplyKeyboardMarkup([
        [KeyboardButton(f"Порог: {spot['THRESHOLD_PERCENT']}%"),
         KeyboardButton(f"Макс. порог: {spot['MAX_THRESHOLD_PERCENT']}%")],
        [KeyboardButton(f"Интервал: {spot['CHECK_INTERVAL']}с"),
         KeyboardButton(f"Объем: ${spot['MIN_VOLUME_USD'] / 1000:.0f}K")],
        [KeyboardButton(f"Мин. сумма: ${spot['MIN_ENTRY_AMOUNT_USDT']}"),
         KeyboardButton(f"Макс. сумма: ${spot['MAX_ENTRY_AMOUNT_USDT']}")],
        [KeyboardButton(f"Влияние: {spot['MAX_IMPACT_PERCENT']}%"),
         KeyboardButton(f"Стакан: {spot['ORDER_BOOK_DEPTH']}")],
        [KeyboardButton(f"Прибыль: ${spot['MIN_NET_PROFIT_USD']}"),
         KeyboardButton(f"Статус: {'ВКЛ' if spot['ENABLED'] else 'ВЫКЛ'}")],
        [KeyboardButton(f"Сходимость: {spot['PRICE_CONVERGENCE_THRESHOLD']}%"),
         KeyboardButton(f"Увед. сравн.: {'🔔' if spot['PRICE_CONVERGENCE_ENABLED'] else '🔕'}")],
        [KeyboardButton(f"Волатильность: {spot['VOLATILITY_THRESHOLD']}%"),
         KeyboardButton(f"Мин. объем стакана: ${spot['MIN_ORDER_BOOK_VOLUME']}")],
        [KeyboardButton(f"Макс. волатильность: {spot['MAX_VOLATILITY_PERCENT']}%")],
        [KeyboardButton("🔙 Назад в настройки")]
    ], resize_keyboard=True)

def get_futures_settings_keyboard():
    futures = SETTINGS['FUTURES']
    return ReplyKeyboardMarkup([
        [KeyboardButton(f"Порог: {futures['THRESHOLD_PERCENT']}%"),
         KeyboardButton(f"Макс. порог: {futures['MAX_THRESHOLD_PERCENT']}%")],
        [KeyboardButton(f"Интервал: {futures['CHECK_INTERVAL']}с"),
         KeyboardButton(f"Объем: ${futures['MIN_VOLUME_USD'] / 1000:.0f}K")],
        [KeyboardButton(f"Мин. сумма: ${futures['MIN_ENTRY_AMOUNT_USDT']}"),
         KeyboardButton(f"Макс. сумма: ${futures['MAX_ENTRY_AMOUNT_USDT']}")],
        [KeyboardButton(f"Прибыль: ${futures['MIN_NET_PROFIT_USD']}"),
         KeyboardButton(f"Статус: {'ВКЛ' if futures['ENABLED'] else 'ВЫКЛ'}")],
        [KeyboardButton(f"Сходимость: {futures['PRICE_CONVERGENCE_THRESHOLD']}%"),
         KeyboardButton(f"Увед. сравн.: {'🔔' if futures['PRICE_CONVERGENCE_ENABLED'] else '🔕'}")],
        [KeyboardButton(f"Волатильность: {futures['VOLATILITY_THRESHOLD']}%"),
         KeyboardButton(f"Мин. объем стакана: ${futures['MIN_ORDER_BOOK_VOLUME']}")],
        [KeyboardButton(f"Макс. фандинг: {futures['FUNDING_RATE_THRESHOLD']}%"),
         KeyboardButton(f"Мин. фандинг: {futures['MIN_FUNDING_RATE_TO_RECEIVE']}%")],
        [KeyboardButton(f"Красный фандинг: {futures['RED_FUNDING_THRESHOLD']}%"),
         KeyboardButton(f"Макс. волатильность: {futures['MAX_VOLATILITY_PERCENT']}%")],
        [KeyboardButton("🔙 Назад в настройки")]
    ], resize_keyboard=True)

def get_spot_futures_settings_keyboard():
    spot_futures = SETTINGS['SPOT_FUTURES']
    return ReplyKeyboardMarkup([
        [KeyboardButton(f"Порог: {spot_futures['THRESHOLD_PERCENT']}%"),
         KeyboardButton(f"Макс. порог: {spot_futures['MAX_THRESHOLD_PERCENT']}%")],
        [KeyboardButton(f"Интервал: {spot_futures['CHECK_INTERVAL']}с"),
         KeyboardButton(f"Объем: ${spot_futures['MIN_VOLUME_USD'] / 1000:.0f}K")],
        [KeyboardButton(f"Мин. сумма: ${spot_futures['MIN_ENTRY_AMOUNT_USDT']}"),
         KeyboardButton(f"Макс. сумма: ${spot_futures['MAX_ENTRY_AMOUNT_USDT']}")],
        [KeyboardButton(f"Прибыль: ${spot_futures['MIN_NET_PROFIT_USD']}"),
         KeyboardButton(f"Статус: {'ВКЛ' if spot_futures['ENABLED'] else 'ВЫКЛ'}")],
        [KeyboardButton(f"Сходимость: {spot_futures['PRICE_CONVERGENCE_THRESHOLD']}%"),
         KeyboardButton(f"Увед. сравн.: {'🔔' if spot_futures['PRICE_CONVERGENCE_ENABLED'] else '🔕'}")],
        [KeyboardButton(f"Волатильность: {spot_futures['VOLATILITY_THRESHOLD']}%"),
         KeyboardButton(f"Мин. объем стакана: ${spot_futures['MIN_ORDER_BOOK_VOLUME']}")],
        [KeyboardButton(f"Макс. волатильность: {spot_futures['MAX_VOLATILITY_PERCENT']}%")],
        [KeyboardButton("🔙 Назад в настройки")]
    ], resize_keyboard=True)

def get_exchange_settings_keyboard():
    keyboard = []
    row = []
    for i, (exchange, config) in enumerate(SETTINGS['EXCHANGES'].items()):
        status = "✅" if config['ENABLED'] else "❌"
        row.append(KeyboardButton(f"{exchange}: {status}"))
        if (i + 1) % 2 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([KeyboardButton("🔙 Назад в настройки")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_arbitrage_list_keyboard(page, total_pages):
    """Клавиатура для навигации по страницам арбитражных связок"""
    keyboard = []
    if total_pages > 1:
        nav_buttons = []
        if page > 0:
            nav_buttons.append(KeyboardButton("⬅️ Предыдущая"))
        nav_buttons.append(KeyboardButton(f"{page + 1}/{total_pages}"))
        if page < total_pages - 1:
            nav_buttons.append(KeyboardButton("Следующая ➡️"))
        keyboard.append(nav_buttons)
    keyboard.append([KeyboardButton("🔙 Главное меню")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def send_telegram_message(message: str, chat_id: str = None, reply_markup: ReplyKeyboardMarkup = None):
    global SHARED_BOT
    if not SHARED_BOT:
        SHARED_BOT = Bot(token=TELEGRAM_TOKEN)

    targets = [chat_id] if chat_id else TELEGRAM_CHAT_IDS

    for target_id in targets:
        try:
            await SHARED_BOT.send_message(
                chat_id=target_id,
                text=message,
                parse_mode="HTML",
                disable_web_page_preview=True,
                reply_markup=reply_markup
            )
            logger.info(f"Сообщение отправлено в чат {target_id}")
        except TelegramError as e:
            logger.error(f"Ошибка отправки в {target_id}: {e}")

def format_duration(seconds):
    """Форматирует длительность в читаемый вид"""
    if seconds < 60:
        return f"{int(seconds)} сек"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds_remaining = int(seconds % 60)
        return f"{minutes} мин {seconds_remaining} сек"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} ч {minutes} мин"

def add_opportunity_to_sent(arb_type: str, base: str, exchange1: str, exchange2: str, spread: float,
                            price1: float, price2: float, volume1: float = None, volume2: float = None,
                            min_entry_amount: float = None, max_entry_amount: float = None,
                            profit_min: dict = None, profit_max: dict = None,
                            available_volume: float = None, order_book_volume: float = None,
                            long_funding: float = None, short_funding: float = None):
    """Добавляет связку в отправленные возможности"""
    key = f"{arb_type}_{base}_{exchange1}_{exchange2}"
    current_time = time.time()

    sent_arbitrage_opportunities[key] = {
        'arb_type': arb_type,
        'base': base,
        'exchange1': exchange1,
        'exchange2': exchange2,
        'spread': spread,
        'price1': price1,
        'price2': price2,
        'volume1': volume1,
        'volume2': volume2,
        'min_entry_amount': min_entry_amount,
        'max_entry_amount': max_entry_amount,
        'profit_min': profit_min,
        'profit_max': profit_max,
        'available_volume': available_volume,
        'order_book_volume': order_book_volume,
        'long_funding': long_funding,
        'short_funding': short_funding,
        'start_time': current_time,
        'last_updated': current_time
    }

    # Также добавляем в current_arbitrage_opportunities для отображения в актуальных связках
    current_arbitrage_opportunities[key] = sent_arbitrage_opportunities[key].copy()

    # Запускаем отсчет времени для этой связки
    arbitrage_start_times[key] = current_time
    previous_arbitrage_opportunities[key] = True

    logger.info(f"Связка добавлена в отправленные: {key}")

async def send_price_convergence_notification(arb_type: str, base: str, exchange1: str, exchange2: str,
                                              price1: float, price2: float, spread: float, volume1: float = None,
                                              volume2: float = None, duration: float = None):
    """Отправляет уведомление о сравнении цен с длительностью арбитража и удаляет связку из актуальных"""

    if not SETTINGS[arb_type]['PRICE_CONVERGENCE_ENABLED']:
        return

    convergence_threshold = SETTINGS[arb_type]['PRICE_CONVERGENCE_THRESHOLD']

    if abs(spread) > convergence_threshold:
        return

    # Проверяем, была ли эта связка ранее отправленной арбитражной возможностью
    previous_key = f"{arb_type}_{base}_{exchange1}_{exchange2}"
    if previous_key not in sent_arbitrage_opportunities:
        return

    # Проверяем, не отправляли ли мы уже уведомление для этой связки
    current_time = time.time()
    notification_key = f"{arb_type}_{base}_{exchange1}_{exchange2}"

    # Проверяем, прошло ли достаточно времени с последнего уведомления (5 минут)
    if (notification_key in last_convergence_notification and
            current_time - last_convergence_notification[notification_key] < 300):
        return

    # Обновляем время последнего уведомления
    last_convergence_notification[notification_key] = current_time

    # Определяем тип арбитража для заголовка
    if arb_type == 'SPOT':
        arb_type_name = "Спотовый"
        emoji = "🚀"
    elif arb_type == 'FUTURES':
        arb_type_name = "Фьючерсный"
        emoji = "📊"
    else:
        arb_type_name = "Спот-Фьючерсный"
        emoji = "↔️"

    utc_plus_3 = timezone(timedelta(hours=3))
    current_time_str = datetime.now(utc_plus_3).strftime('%H:%M:%S')

    # Форматируем объемы
    def format_volume(vol):
        if vol is None:
            return "N/A"
        if vol >= 1_000_000:
            return f"${vol / 1_000_000:.1f}M"
        if vol >= 1_000:
            return f"${vol / 1_000:.1f}K"
        return f"${vol:.1f}"

    volume1_str = format_volume(volume1)
    volume2_str = format_volume(volume2)

    # Форматируем длительность
    duration_str = format_duration(duration) if duration is not None else "N/A"

    # Получаем URL для бирж
    if arb_type == 'SPOT':
        exchange1_config = SPOT_EXCHANGES[exchange1]
        exchange2_config = SPOT_EXCHANGES[exchange2]
        symbol1 = exchange1_config["symbol_format"](base)
        symbol2 = exchange2_config["symbol_format"](base)
        url1 = exchange1_config["url_format"](symbol1)
        url2 = exchange2_config["url_format"](symbol2)
    else:
        exchange1_config = FUTURES_EXCHANGES[exchange1]
        exchange2_config = FUTURES_EXCHANGES[exchange2]
        symbol1 = exchange1_config["symbol_format"](base)
        symbol2 = exchange2_config["symbol_format"](base)
        url1 = exchange1_config["url_format"](symbol1.replace(':USDT', ''))
        url2 = exchange2_config["url_format"](symbol2.replace(':USDT', ''))

    safe_base = html.escape(base)

    # Создаем красивое сообщение с информацией о длительности
    message = (
        f"🎯 <b>ЦЕНЫ СРАВНИЛИСЬ!</b> {emoji}\n\n"
        f"▫️ <b>Тип:</b> {arb_type_name} арбитраж\n"
        f"▫️ <b>Монета:</b> <code>{safe_base}</code>\n"
        f"▫️ <b>Разница цен:</b> <code>{spread:.2f}%</code>\n"
        f"▫️ <b>Длительность арбитража:</b> {duration_str}\n\n"

        f"🟢 <b><a href='{url1}'>{exchange1.upper()}</a>:</b>\n"
        f"   💰 Цена: <code>${price1:.8f}</code>\n"
        f"   📊 Объем: {volume1_str}\n\n"

        f"🔵 <b><a href='{url2}'>{exchange2.upper()}</a>:</b>\n"
        f"   💰 Цена: <code>${price2:.8f}</code>\n"
        f"   📊 Объем: {volume2_str}\n\n"

        f"⏰ <i>{current_time_str}</i>\n"
        f"🔔 <i>Уведомление о сходимости цен</i>"
    )

    await send_telegram_message(message)
    logger.info(
        f"Отправлено уведомление о сходимости цен для {base} ({arb_type}): {spread:.4f}%, длительность: {duration_str}")

    # Удаляем связку из всех словарей, чтобы она не отображалась в актуальных
    key = f"{arb_type}_{base}_{exchange1}_{exchange2}"
    if key in sent_arbitrage_opportunities:
        del sent_arbitrage_opportunities[key]
    if key in current_arbitrage_opportunities:
        del current_arbitrage_opportunities[key]
    if key in arbitrage_start_times:
        del arbitrage_start_times[key]
    if key in previous_arbitrage_opportunities:
        del previous_arbitrage_opportunities[key]

    logger.info(f"Связка удалена из актуальных после сходимости цен: {key}")

def update_arbitrage_duration(arb_type: str, base: str, exchange1: str, exchange2: str, spread: float):
    """Обновляет время длительности арбитражной возможности"""
    key = f"{arb_type}_{base}_{exchange1}_{exchange2}"
    current_time = time.time()

    # Если связка была отправлена в Telegram и спред превышает порог арбитража - начинаем отсчет
    if (key in sent_arbitrage_opportunities and
            SETTINGS[arb_type]['THRESHOLD_PERCENT'] <= spread <= SETTINGS[arb_type]['MAX_THRESHOLD_PERCENT'] and
            key not in arbitrage_start_times):
        arbitrage_start_times[key] = current_time
        previous_arbitrage_opportunities[key] = True
        logger.debug(f"Начало арбитража для {key}")

    # Если спред упал ниже порога сходимости - вычисляем длительность и очищаем
    elif (spread <= SETTINGS[arb_type]['PRICE_CONVERGENCE_THRESHOLD'] and
          key in arbitrage_start_times):
        start_time = arbitrage_start_times.pop(key)
        duration = current_time - start_time
        logger.debug(f"Завершение арбитража для {key}, длительность: {duration:.0f} сек")
        return duration

    return None

def update_current_arbitrage_opportunities(arb_type: str, base: str, exchange1: str, exchange2: str, spread: float,
                                           price1: float, price2: float, volume1: float = None, volume2: float = None,
                                           min_entry_amount: float = None, max_entry_amount: float = None,
                                           profit_min: dict = None, profit_max: dict = None,
                                           available_volume: float = None, order_book_volume: float = None,
                                           long_funding: float = None, short_funding: float = None):
    """Обновляет информацию о текущих арбитражных возможностях (только для отправленных связок)"""
    key = f"{arb_type}_{base}_{exchange1}_{exchange2}"
    current_time = time.time()

    # Обновляем только связки, которые были отправлены в Telegram
    if key in sent_arbitrage_opportunities:
        sent_arbitrage_opportunities[key].update({
            'spread': spread,
            'price1': price1,
            'price2': price2,
            'volume1': volume1,
            'volume2': volume2,
            'min_entry_amount': min_entry_amount,
            'max_entry_amount': max_entry_amount,
            'profit_min': profit_min,
            'profit_max': profit_max,
            'available_volume': available_volume,
            'order_book_volume': order_book_volume,
            'long_funding': long_funding,
            'short_funding': short_funding,
            'last_updated': current_time
        })

        current_arbitrage_opportunities[key] = sent_arbitrage_opportunities[key].copy()

def calculate_volatility(prices):
    """Рассчитывает волатильность на основе истории цен"""
    if len(prices) < 2:
        return 0.0

    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

    if not returns:
        return 0.0

    return np.std(returns) * 100  # в процентах

def update_price_history(arb_type: str, base: str, exchange: str, price: float):
    """Обновляет историю цен для расчета волатильности"""
    key = f"{arb_type}_{base}_{exchange}"

    # Добавляем новую цену
    price_history[key].append(price)

    # Ограничиваем размер истории
    if len(price_history[key]) > VOLATILITY_WINDOW:
        price_history[key] = price_history[key][-VOLATILITY_WINDOW:]

def check_volatility(arb_type: str, base: str, exchange: str, price: float) -> bool:
    """Проверяет, не превышает ли волатильность допустимый порог"""
    key = f"{arb_type}_{base}_{exchange}"

    if key not in price_history:
        return True

    volatility = calculate_volatility(price_history[key])
    threshold = SETTINGS[arb_type]['VOLATILITY_THRESHOLD']

    logger.debug(f"Волатильность для {key}: {volatility:.2f}% (порог: {threshold}%)")

    return volatility <= threshold

def update_coin_volatility_history(base: str, price: float):
    """Обновляет историю цен для расчета общей волатильности монеты"""
    # Добавляем новую цену
    coin_volatility_history[base].append(price)

    # Ограничиваем размер истории
    if len(coin_volatility_history[base]) > COIN_VOLATILITY_WINDOW:
        coin_volatility_history[base] = coin_volatility_history[base][-COIN_VOLATILITY_WINDOW:]

def check_coin_volatility(base: str, arb_type: str) -> bool:
    """Проверяет общую волатильность монеты"""
    if base not in coin_volatility_history or len(coin_volatility_history[base]) < 2:
        return True

    volatility = calculate_volatility(coin_volatility_history[base])
    max_volatility = SETTINGS[arb_type].get('MAX_VOLATILITY_PERCENT', 15.0)

    logger.debug(f"Общая волатильность монеты {base}: {volatility:.2f}% (макс. порог: {max_volatility}%)")

    if volatility > max_volatility:
        logger.info(f"Монета {base} отфильтрована из-за высокой волатильности: {volatility:.2f}% > {max_volatility}%")
        return False

    return True

async def get_current_funding_rates():
    """Получает текущие ставки финансирования со всех бирж"""
    global funding_rates_cache, last_funding_check

    current_time = time.time()
    # Проверяем ставки финансирования раз в час
    if current_time - last_funding_check < SETTINGS['FUTURES']['FUNDING_CHECK_INTERVAL']:
        return funding_rates_cache

    funding_data = {}
    logger.info("Обновление ставок финансирования...")

    for name, data in FUTURES_EXCHANGES_LOADED.items():
        if not SETTINGS['EXCHANGES'][name]['ENABLED']:
            continue

        exchange = data["api"]
        config = data["config"]

        if not config.get("supports_funding", False):
            continue

        try:
            # Получаем все рынки фьючерсов
            markets = exchange.markets
            for symbol, market in markets.items():
                try:
                    if config["is_futures"](market):
                        base = market['base']

                        # Получаем ставку финансирования
                        funding_rate = await asyncio.get_event_loop().run_in_executor(
                            None, exchange.fetch_funding_rate, symbol
                        )

                        if funding_rate and 'fundingRate' in funding_rate:
                            rate = float(funding_rate['fundingRate']) * 100  # Конвертируем в проценты

                            if base not in funding_data:
                                funding_data[base] = {}
                            funding_data[base][name] = rate

                            logger.debug(f"Ставка финансирования для {base} на {name}: {rate:.4f}%")
                except Exception as e:
                    logger.debug(f"Ошибка получения ставки финансирования для {symbol} на {name}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Ошибка получения ставок финансирования для {name}: {e}")

    funding_rates_cache = funding_data
    last_funding_check = current_time
    logger.info(f"Обновлены ставки финансирования для {len(funding_data)} монет")
    return funding_data

def calculate_funding_score(long_funding: float, short_funding: float) -> float:
    """Рассчитывает оценку выгодности финансирования"""
    # Для лонга: отрицательное финансирование выгодно (мы получаем)
    # Для шорта: положительное финансирование выгодно (мы получаем)
    long_score = -long_funding  # Инвертируем для лонга
    short_score = short_funding

    total_score = long_score + short_score
    return total_score

def is_favorable_funding(long_funding: float, short_funding: float) -> bool:
    """Проверяет, является ли финансирование выгодным"""
    funding_score = calculate_funding_score(long_funding, short_funding)

    # Проверяем максимальную ставку для оплаты
    if funding_score > SETTINGS['FUTURES']['FUNDING_RATE_THRESHOLD']:
        return False

    # Проверяем минимальную ставку для получения (если установлена)
    if (SETTINGS['FUTURES']['MIN_FUNDING_RATE_TO_RECEIVE'] is not None and
            funding_score < SETTINGS['FUTURES']['MIN_FUNDING_RATE_TO_RECEIVE']):
        return False

    return True

def has_red_funding(long_funding: float, short_funding: float) -> bool:
    """Проверяет, является ли финансирование 'красным' (очень невыгодным)"""
    funding_score = calculate_funding_score(long_funding, short_funding)
    red_threshold = SETTINGS['FUTURES']['RED_FUNDING_THRESHOLD']
    
    # Если общий score превышает порог красного фандинга - считаем связку невыгодной
    return funding_score > red_threshold

def calculate_effective_profit_with_funding(base_profit: float, entry_amount: float,
                                            long_funding: float, short_funding: float,
                                            holding_hours: int = 8) -> float:
    """Рассчитывает эффективную прибыль с учетом финансирования"""
    # Рассчитываем влияние финансирования
    funding_impact = (short_funding - long_funding) * (holding_hours / 8) * entry_amount / 100

    # Эффективная прибыль с учетом финансирования
    effective_profit = base_profit + funding_impact
    return effective_profit

async def get_current_arbitrage_opportunities_page(page: int = 0, opportunities_per_page: int = 5):
    """Возвращает форматированное сообщение с текущими арбитражными возможностями для указанной страницы"""
    
    # Используем только отправленные связки
    filtered_opportunities = {}
    current_time = time.time()

    for key, opportunity in sent_arbitrage_opportunities.items():
        # Проверяем, что связка не устарела (теперь связки не удаляются по времени, только при сходимости)
        # Но все же удаляем очень старые связки (24 часа) на всякий случай
        if (current_time - opportunity['last_updated']) <= 86400:  # 24 часа
            filtered_opportunities[key] = opportunity

    if not filtered_opportunities:
        return "📊 <b>Актуальные арбитражные связки</b>\n\n" \
               "⏳ В данный момент активных арбитражных возможностей не обнаружено.", 0, 0

    # Группируем по типу арбитража
    spot_opportunities = []
    futures_opportunities = []
    spot_futures_opportunities = []

    for key, opportunity in filtered_opportunities.items():
        arb_type = opportunity['arb_type']
        duration = time.time() - opportunity['start_time']

        opportunity_info = {
            'base': opportunity['base'],
            'exchange1': opportunity['exchange1'],
            'exchange2': opportunity['exchange2'],
            'spread': opportunity['spread'],
            'price1': opportunity['price1'],
            'price2': opportunity['price2'],
            'min_entry_amount': opportunity.get('min_entry_amount'),
            'max_entry_amount': opportunity.get('max_entry_amount'),
            'profit_min': opportunity.get('profit_min'),
            'profit_max': opportunity.get('profit_max'),
            'available_volume': opportunity.get('available_volume'),
            'order_book_volume': opportunity.get('order_book_volume'),
            'long_funding': opportunity.get('long_funding'),
            'short_funding': opportunity.get('short_funding'),
            'duration': duration
        }

        if arb_type == 'SPOT':
            spot_opportunities.append(opportunity_info)
        elif arb_type == 'FUTURES':
            futures_opportunities.append(opportunity_info)
        else:
            spot_futures_opportunities.append(opportunity_info)

    # Сортируем по спреду (по убыванию)
    spot_opportunities.sort(key=lambda x: x['spread'], reverse=True)
    futures_opportunities.sort(key=lambda x: x['spread'], reverse=True)
    spot_futures_opportunities.sort(key=lambda x: x['spread'], reverse=True)

    # Объединяем все возможности в один список для пагинации
    all_opportunities = spot_opportunities + futures_opportunities + spot_futures_opportunities
    
    # Рассчитываем общее количество страниц
    total_opportunities = len(all_opportunities)
    total_pages = (total_opportunities + opportunities_per_page - 1) // opportunities_per_page
    
    # Если запрошенная страница превышает общее количество страниц, возвращаем последнюю
    if page >= total_pages:
        page = total_pages - 1
    if page < 0:
        page = 0

    # Получаем возможности для текущей страницы
    start_idx = page * opportunities_per_page
    end_idx = start_idx + opportunities_per_page
    page_opportunities = all_opportunities[start_idx:end_idx]

    utc_plus_3 = timezone(timedelta(hours=3))
    current_time_str = datetime.now(utc_plus_3).strftime('%H:%M:%S')

    message = f"📊 <b>Актуальные арбитражные связки</b> (Страница {page + 1}/{total_pages})\n\n"

    if not page_opportunities:
        message += "⏳ На этой странице нет активных арбитражных возможностей."
    else:
        for i, opp in enumerate(page_opportunities, start=1):
            duration_str = format_duration(opp['duration'])

            # Определяем тип арбитража
            if opp in spot_opportunities:
                arb_type_emoji = "🚀"
                arb_type_name = "Спот"
            elif opp in futures_opportunities:
                arb_type_emoji = "📊"
                arb_type_name = "Фьючерсы"
            else:
                arb_type_emoji = "↔️"
                arb_type_name = "Спот-Фьючерсы"

            # Форматируем сумму входа и прибыль
            entry_amount_str = f"${opp['min_entry_amount']:.2f}-${opp['max_entry_amount']:.2f}" if opp.get(
                'min_entry_amount') and opp.get('max_entry_amount') else "N/A"

            profit_str = "N/A"
            if opp.get('profit_min') and opp.get('profit_max'):
                profit_min_net = opp['profit_min'].get('net', 0)
                profit_max_net = opp['profit_max'].get('net', 0)
                profit_str = f"${profit_min_net:.2f}-${profit_max_net:.2f}"

            # Форматируем доступный объем
            available_volume_str = f"{opp['available_volume']:.6f} {opp['base']}" if opp.get(
                'available_volume') else "N/A"
            order_book_volume_str = f"${opp['order_book_volume']:.2f}" if opp.get('order_book_volume') else "N/A"

            # Добавляем информацию о финансировании для фьючерсов
            funding_info = ""
            if opp.get('long_funding') is not None and opp.get('short_funding') is not None:
                funding_score = calculate_funding_score(opp['long_funding'], opp['short_funding'])
                funding_emoji = "🟢"
                if has_red_funding(opp['long_funding'], opp['short_funding']):
                    funding_emoji = "🔴"
                funding_info = f"\n      {funding_emoji} Фандинг: {funding_score:.4f}%"

            message += (
                f"{arb_type_emoji} <b>{arb_type_name}</b> | <code>{opp['base']}</code>\n"
                f"   📈 Разница: {opp['spread']:.2f}%\n"
                f"   🟢 {opp['exchange1'].upper()} → 🔴 {opp['exchange2'].upper()}\n"
                f"   💰 Сумма входа: {entry_amount_str}\n"
                f"   💵 Прибыль: {profit_str}"
                f"{funding_info}\n"
                f"   📊 Объем: {available_volume_str}\n"
                f"   ⏱ Длительность: {duration_str}\n\n"
            )

    message += f"⏰ <i>Обновлено: {current_time_str}</i>\n"
    message += f"📈 <i>Всего активных связок: {total_opportunities}</i>"

    return message, total_pages, page

def cleanup_old_opportunities():
    """Очищает устаревшие арбитражные возможности (старше 24 часов)"""
    current_time = time.time()
    keys_to_remove = []

    for key, opportunity in sent_arbitrage_opportunities.items():
        # Удаляем если связка устарела (старше 24 часов)
        if current_time - opportunity['last_updated'] > 86400:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del sent_arbitrage_opportunities[key]
        if key in current_arbitrage_opportunities:
            del current_arbitrage_opportunities[key]
        if key in arbitrage_start_times:
            del arbitrage_start_times[key]
        logger.debug(f"Удалена устаревшая связка: {key}")

def load_markets_sync(exchange):
    try:
        exchange.load_markets()
        logger.info(f"Рынки загружены для {exchange.id}")
        return exchange
    except Exception as e:
        logger.error(f"Ошибка загрузки {exchange.id}: {e}")
        return None

async def fetch_ticker_data(exchange, symbol: str):
    try:
        ticker = await asyncio.get_event_loop().run_in_executor(
            None, exchange.fetch_ticker, symbol
        )

        if ticker:
            price = float(ticker['last']) if ticker.get('last') else None

            # Пытаемся получить объем из разных источников
            volume = None
            if ticker.get('quoteVolume') is not None:
                volume = float(ticker['quoteVolume'])
            elif ticker.get('baseVolume') is not None and price:
                volume = float(ticker['baseVolume']) * price

            logger.debug(f"Данные тикера {symbol} на {exchange.id}: цена={price}, объем={volume}")

            return {
                'price': price,
                'volume': volume,
                'symbol': symbol
            }
        return None
    except Exception as e:
        logger.warning(f"Ошибка данных {symbol} на {exchange.id}: {e}")
        return None

async def fetch_order_book(exchange, symbol: str, depth: int = None):
    if depth is None:
        depth = SETTINGS['SPOT']['ORDER_BOOK_DEPTH']

    try:
        order_book = await asyncio.get_event_loop().run_in_executor(
            None, exchange.fetch_order_book, symbol, depth)
        logger.debug(f"Стакан загружен для {symbol} на {exchange.id}")
        return order_book
    except Exception as e:
        logger.warning(f"Ошибка стакана {symbol} на {exchange.id}: {e}")
        return None

def calculate_available_volume(order_book, side: str, max_impact_percent: float):
    if not order_book:
        return 0, 0

    if side == 'buy':
        asks = order_book['asks']
        if not asks:
            return 0, 0
        best_ask = asks[0][0]
        max_allowed_price = best_ask * (1 + max_impact_percent / 100)
        total_volume = 0
        total_value = 0
        for price, volume in asks:
            if price > max_allowed_price:
                break
            total_volume += volume
            total_value += volume * price
        return total_volume, total_value
    elif side == 'sell':
        bids = order_book['bids']
        if not bids:
            return 0, 0
        best_bid = bids[0][0]
        min_allowed_price = best_bid * (1 - max_impact_percent / 100)
        total_volume = 0
        total_value = 0
        for price, volume in bids:
            if price < min_allowed_price:
                break
            total_volume += volume
            total_value += volume * price
        return total_volume, total_value
    return 0, 0

async def check_deposit_withdrawal_status(exchange, currency: str, check_type: str = 'deposit'):
    try:
        try:
            currencies = await asyncio.get_event_loop().run_in_executor(
                None, exchange.fetch_currencies)
            if currency in currencies:
                currency_info = currencies[currency]
                if check_type == 'deposit':
                    status = currency_info.get('deposit', False)
                else:
                    status = currency_info.get('withdraw', False)
                logger.debug(
                    f"Статус {check_type} для {currency} на {exchange.id}: {status} (через fetch_currencies)"
                )
                return status
        except (ccxt.NotSupported, AttributeError) as e:
            logger.debug(
                f"fetch_currencies не поддерживается на {exchange.id}: {e}")

        try:
            symbol = f"{currency}/USDT"
            market = exchange.market(symbol)
            if market:
                if check_type == 'deposit':
                    status = market.get('deposit', True)
                else:
                    status = market.get('withdraw', True)
                logger.debug(
                    f"Статус {check_type} для {currency} на {exchange.id}: {status} (через market)"
                )
                return status
        except (ccxt.BadSymbol, KeyError) as e:
            logger.debug(
                f"Ошибка проверки market для {currency} на {exchange.id}: {e}")

        try:
            currency_info = exchange.currency(currency)
            if check_type == 'deposit':
                status = currency_info.get(
                    'active', False) and currency_info.get('deposit', True)
            else:
                status = currency_info.get(
                    'active', False) and currency_info.get('withdraw', True)
            logger.debug(
                f"Статус {check_type} для {currency} на {exchange.id}: {status} (через currency)"
            )
            return status
        except (KeyError, AttributeError) as e:
            logger.debug(
                f"Ошибка проверки currency для {currency} на {exchange.id}: {e}"
            )

        logger.debug(
            f"Не удалось проверить статус {check_type} для {currency} на {exchange.id}, предполагаем True"
        )
        return True
    except Exception as e:
        logger.warning(
            f"Ошибка проверки {check_type} {currency} на {exchange.id}: {e}")
        return True

def calculate_min_entry_amount(buy_price: float, sell_price: float, min_profit: float, buy_fee_percent: float,
                               sell_fee_percent: float) -> float:
    profit_per_unit = sell_price * (1 - sell_fee_percent) - buy_price * (1 + buy_fee_percent)
    if profit_per_unit <= 0:
        return 0
    min_amount = min_profit / profit_per_unit
    return min_amount * buy_price

def calculate_profit(buy_price: float, sell_price: float, amount: float, buy_fee_percent: float,
                     sell_fee_percent: float) -> dict:
    buy_cost = amount * buy_price * (1 + buy_fee_percent)
    sell_revenue = amount * sell_price * (1 - sell_fee_percent)
    net_profit = sell_revenue - buy_cost
    profit_percent = (net_profit / buy_cost) * 100 if buy_cost > 0 else 0

    return {
        "net": net_profit,
        "percent": profit_percent,
        "entry_amount": amount * buy_price
    }

async def load_spot_exchanges():
    """Загружает спотовые биржи на основе текущих настроек"""
    global SPOT_EXCHANGES_LOADED, LAST_EXCHANGE_SETTINGS

    exchanges = {}
    for name, config in SPOT_EXCHANGES.items():
        if not SETTINGS['EXCHANGES'][name]['ENABLED']:
            continue

        try:
            # Для BloFin устанавливаем правильный тип рынка
            if name == "blofin":
                config["api"].options['defaultType'] = 'spot'

            exchange = await asyncio.get_event_loop().run_in_executor(
                None, load_markets_sync, config["api"])
            if exchange:
                exchanges[name] = {"api": exchange, "config": config}
                logger.info(f"{name.upper()} успешно загружена")

                # Дополнительная проверка для BloFin
                if name == "blofin":
                    spot_markets = [m for m in exchange.markets.values() if config["is_spot"](m)]
                    logger.info(f"BloFin спотовые рынки: {len(spot_markets)}")
                    for market in spot_markets[:5]:  # Показать первые 5 рынков для проверки
                        logger.info(f"BloFin рынок: {market['symbol']}")
        except Exception as e:
            logger.error(f"Ошибка инициализации {name}: {e}")

    SPOT_EXCHANGES_LOADED = exchanges
    LAST_EXCHANGE_SETTINGS = SETTINGS['EXCHANGES'].copy()
    return exchanges

async def load_futures_exchanges():
    """Загружает фьючерсные биржи на основе текущих настроек"""
    global FUTURES_EXCHANGES_LOADED, LAST_EXCHANGE_SETTINGS

    exchanges = {}
    for name, config in FUTURES_EXCHANGES.items():
        if not SETTINGS['EXCHANGES'][name]['ENABLED']:
            continue

        try:
            # Для BloFin устанавливаем правильный тип рынка
            if name == "blofin":
                config["api"].options['defaultType'] = 'swap'

            exchange = await asyncio.get_event_loop().run_in_executor(
                None, load_markets_sync, config["api"]
            )
            if exchange:
                exchanges[name] = {
                    "api": exchange,
                    "config": config
                }
                logger.info(f"{name.upper()} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка инициализации {name}: {e}")

    FUTURES_EXCHANGES_LOADED = exchanges
    LAST_EXCHANGE_SETTINGS = SETTINGS['EXCHANGES'].copy()
    return exchanges

async def check_spot_arbitrage():
    logger.info("Запуск проверки спотового арбитража")

    if not SETTINGS['SPOT']['ENABLED']:
        logger.info("Спотовый арбитраж отключен в настройках")
        return

    # Инициализация бирж
    await load_spot_exchanges()

    if len(SPOT_EXCHANGES_LOADED) < SETTINGS['SPOT']['MIN_EXCHANGES_FOR_PAIR']:
        logger.error(
            f"Недостаточно бирж (нужно минимум {SETTINGS['SPOT']['MIN_EXCHANGES_FOR_PAIR']})")
        return

    # Сбор всех торговых пар
    all_pairs = defaultdict(set)
    for name, data in SPOT_EXCHANGES_LOADED.items():
        exchange = data["api"]
        config = data["config"]
        for symbol, market in exchange.markets.items():
            try:
                if config["is_spot"](market):
                    base = market['base']
                    # Пропускаем монеты из черного списка
                    if base in config.get("blacklist", []):
                        continue
                    all_pairs[base].add((name, symbol))
            except Exception as e:
                logger.warning(
                    f"Ошибка обработки пары {symbol} на {name}: {e}")

    valid_pairs = {
        base: list(pairs)
        for base, pairs in all_pairs.items()
        if len(pairs) >= SETTINGS['SPOT']['MIN_EXCHANGES_FOR_PAIR']
    }

    if not valid_pairs:
        logger.error("Нет пар, торгуемых хотя бы на двух биржах")
        return

    logger.info(f"Найдено {len(valid_pairs)} пар для анализа")

    while SETTINGS['SPOT']['ENABLED']:
        try:
            # Проверяем, изменились ли настройки бирж
            if LAST_EXCHANGE_SETTINGS != SETTINGS['EXCHANGES']:
                logger.info("Обнаружено изменение настроек бирж. Перезагружаем спотовые биржи...")
                await load_spot_exchanges()

                # Перестраиваем список пар после перезагрузки бирж
                all_pairs = defaultdict(set)
                for name, data in SPOT_EXCHANGES_LOADED.items():
                    exchange = data["api"]
                    config = data["config"]
                    for symbol, market in exchange.markets.items():
                        try:
                            if config["is_spot"](market):
                                base = market['base']
                                if base in config.get("blacklist", []):
                                    continue
                                all_pairs[base].add((name, symbol))
                        except Exception as e:
                            logger.warning(f"Ошибка обработки пары {symbol} на {name}: {e}")

                valid_pairs = {
                    base: list(pairs)
                    for base, pairs in all_pairs.items()
                    if len(pairs) >= SETTINGS['SPOT']['MIN_EXCHANGES_FOR_PAIR']
                }

                if not valid_pairs:
                    logger.error("Нет пар, торгуемых хотя бы на двух биржах после перезагрузки")
                    await asyncio.sleep(SETTINGS['SPOT']['CHECK_INTERVAL'])
                    continue

            found_opportunities = 0
            for base, exchange_symbols in valid_pairs.items():
                try:
                    # Проверяем общую волатильность монеты
                    if not check_coin_volatility(base, 'SPOT'):
                        continue

                    ticker_data = {}

                    # Получаем данные тикеров для всех бирж
                    for name, symbol in exchange_symbols:
                        try:
                            data = await fetch_ticker_data(
                                SPOT_EXCHANGES_LOADED[name]["api"], symbol)
                            if data and data['price'] is not None:
                                # Обновляем историю цен для расчета волатильности
                                update_price_history('SPOT', base, name, data['price'])
                                update_coin_volatility_history(base, data['price'])

                                # Проверяем волатильность
                                if not check_volatility('SPOT', base, name, data['price']):
                                    logger.debug(f"Пропускаем {base} на {name} из-за высокой волатильности")
                                    continue

                                # Если объем известен, проверяем минимальный объем
                                if data['volume'] is None:
                                    logger.debug(f"Объем неизвестен для {symbol} на {name}, но продолжаем обработку")
                                    ticker_data[name] = data
                                elif data['volume'] >= SETTINGS['SPOT']['MIN_VOLUME_USD']:
                                    ticker_data[name] = data
                                else:
                                    logger.debug(
                                        f"Объем {symbol} на {name} слишком мал: {data['volume']}"
                                    )
                            else:
                                logger.debug(
                                    f"Нет данных для {symbol} на {name}")
                        except Exception as e:
                            logger.warning(
                                f"Ошибка получения данных {base} на {name}: {e}"
                            )

                    if len(ticker_data) < SETTINGS['SPOT']['MIN_EXCHANGES_FOR_PAIR']:
                        continue

                    # Сортируем биржи по цене
                    sorted_data = sorted(ticker_data.items(),
                                         key=lambda x: x[1]['price'])
                    min_ex = sorted_data[0]  # Самая низкая цена (покупка)
                    max_ex = sorted_data[-1]  # Самая высокая цена (продажа)

                    # Рассчитываем спред
                    spread = (max_ex[1]['price'] -
                              min_ex[1]['price']) / min_ex[1]['price'] * 100

                    logger.debug(
                        f"Пара {base}: спред {spread:.2f}% (min: {min_ex[0]} {min_ex[1]['price']}, max: {max_ex[0]} {max_ex[1]['price']})"
                    )

                    # Обновляем информацию о текущих арбитражных возможностях (только для отправленных связок)
                    update_current_arbitrage_opportunities(
                        'SPOT', base, min_ex[0], max_ex[0], spread,
                        min_ex[1]['price'], max_ex[1]['price'],
                        min_ex[1]['volume'], max_ex[1]['volume']
                    )

                    # Проверяем сходимость цен для уведомления (только для отправленных связок)
                    duration = update_arbitrage_duration('SPOT', base, min_ex[0], max_ex[0], spread)
                    if duration is not None:
                        await send_price_convergence_notification(
                            'SPOT', base, min_ex[0], max_ex[0],
                            min_ex[1]['price'], max_ex[1]['price'], spread,
                            min_ex[1]['volume'], max_ex[1]['volume'], duration
                        )

                    if SETTINGS['SPOT']['THRESHOLD_PERCENT'] <= spread <= SETTINGS['SPOT']['MAX_THRESHOLD_PERCENT']:
                        # Проверяем доступность депозита и вывода
                        deposit_available = await check_deposit_withdrawal_status(
                            SPOT_EXCHANGES_LOADED[max_ex[0]]["api"], base, 'deposit')
                        withdrawal_available = await check_deposit_withdrawal_status(
                            SPOT_EXCHANGES_LOADED[min_ex[0]]["api"], base, 'withdrawal')

                        logger.debug(
                            f"Пара {base}: депозит={deposit_available}, вывод={withdrawal_available}"
                        )

                        if not (deposit_available and withdrawal_available):
                            logger.debug(
                                f"Пропускаем {base}: депозит или вывод недоступен"
                            )
                            continue

                        # Получаем стаканы ордеров
                        buy_exchange = SPOT_EXCHANGES_LOADED[min_ex[0]]["api"]
                        sell_exchange = SPOT_EXCHANGES_LOADED[max_ex[0]]["api"]
                        buy_symbol = min_ex[1]['symbol']
                        sell_symbol = max_ex[1]['symbol']

                        buy_order_book, sell_order_book = await asyncio.gather(
                            fetch_order_book(buy_exchange, buy_symbol),
                            fetch_order_book(sell_exchange, sell_symbol))

                        if not buy_order_book or not sell_order_book:
                            logger.debug(
                                f"Пропускаем {base}: нет данных стакана")
                            continue

                        # Рассчитываем доступный объем из стакана
                        buy_volume, buy_value = calculate_available_volume(
                            buy_order_book, 'buy', SETTINGS['SPOT']['MAX_IMPACT_PERCENT'])
                        sell_volume, sell_value = calculate_available_volume(
                            sell_order_book, 'sell', SETTINGS['SPOT']['MAX_IMPACT_PERCENT'])
                        available_volume = min(buy_volume, sell_volume)
                        order_book_volume = min(buy_value, sell_value)

                        logger.debug(
                            f"Пара {base}: доступный объем {available_volume}, объем стакана ${order_book_volume:.2f}"
                        )

                        # Проверяем минимальный объем стакана
                        if order_book_volume < SETTINGS['SPOT']['MIN_ORDER_BOOK_VOLUME']:
                            logger.debug(f"Пропускаем {base}: объем стакана слишком мал: ${order_book_volume:.2f}")
                            continue

                        if available_volume <= 0:
                            continue

                        # Получаем комиссии
                        buy_fee = SPOT_EXCHANGES_LOADED[min_ex[0]]["config"]["taker_fee"]
                        sell_fee = SPOT_EXCHANGES_LOADED[max_ex[0]]["config"]["taker_fee"]

                        # Рассчитываем минимальную сумму для MIN_NET_PROFIT_USD
                        min_amount_for_profit = calculate_min_entry_amount(
                            buy_price=min_ex[1]['price'],
                            sell_price=max_ex[1]['price'],
                            min_profit=SETTINGS['SPOT']['MIN_NET_PROFIT_USD'],
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee)

                        if min_amount_for_profit <= 0:
                            logger.debug(
                                f"Пропускаем {base}: недостаточная прибыль")
                            continue

                        # Рассчитываем максимально возможную сумму входа на основе доступного объема
                        max_possible_amount = min(
                            available_volume * min_ex[1]['price'],
                            SETTINGS['SPOT']['MAX_ENTRY_AMOUNT_USDT'],
                            order_book_volume)

                        max_entry_amount = max_possible_amount
                        min_entry_amount = max(min_amount_for_profit,
                                               SETTINGS['SPOT']['MIN_ENTRY_AMOUNT_USDT'])

                        if min_entry_amount > max_entry_amount:
                            logger.debug(
                                f"Пропускаем {base}: min_entry_amount > max_entry_amount"
                            )
                            continue

                        # Рассчитываем прибыль
                        profit_min = calculate_profit(
                            buy_price=min_ex[1]['price'],
                            sell_price=max_ex[1]['price'],
                            amount=min_entry_amount / min_ex[1]['price'],
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee)

                        profit_max = calculate_profit(
                            buy_price=min_ex[1]['price'],
                            sell_price=max_ex[1]['price'],
                            amount=max_possible_amount / min_ex[1]['price'],
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee)

                        # Форматируем сообщение
                        utc_plus_3 = timezone(timedelta(hours=3))
                        current_time = datetime.now(utc_plus_3).strftime(
                            '%H:%M:%S')

                        def format_volume(vol):
                            if vol is None:
                                return "N/A"
                            if vol >= 1_000_000:
                                return f"${vol / 1_000_000:.1f}M"
                            if vol >= 1_000:
                                return f"${vol / 1_000:.1f}K"
                            return f"${vol:.1f}"

                        min_volume = format_volume(min_ex[1]['volume'])
                        max_volume = format_volume(max_ex[1]['volume'])

                        safe_base = html.escape(base)
                        buy_exchange_config = SPOT_EXCHANGES[min_ex[0]]
                        sell_exchange_config = SPOT_EXCHANGES[max_ex[0]]

                        buy_url = buy_exchange_config["url_format"](buy_symbol)
                        sell_url = sell_exchange_config["url_format"](
                            sell_symbol)
                        withdraw_url = buy_exchange_config["withdraw_url"](
                            base)
                        deposit_url = sell_exchange_config["deposit_url"](base)

                        message = (
                            f"🚀 <b>Спотовый арбитраж:</b> <code>{safe_base}</code>\n"
                            f"▫️ <b>Разница цен:</b> {spread:.2f}%\n"
                            f"▫️ <b>Доступный объем:</b> {available_volume:.6f} {safe_base}\n"
                            f"▫️ <b>Объем стакана:</b> ${order_book_volume:.2f}\n"
                            f"▫️ <b>Сумма входа:</b> ${min_entry_amount:.2f}-${max_entry_amount:.2f}\n\n"
                            f"🟢 <b>Покупка на <a href='{buy_url}'>{min_ex[0].upper()}</a>:</b> ${min_ex[1]['price']:.8f}\n"
                            f"   <b>Объём:</b> {min_volume}\n"
                            f"   <b>Комиссия:</b> {buy_fee * 100:.2f}%\n"
                            f"   <b><a href='{withdraw_url}'>Вывод</a></b>\n\n"
                            f"🔴 <b>Продажа на <a href='{sell_url}'>{max_ex[0].upper()}</a>:</b> ${max_ex[1]['price']:.8f}\n"
                            f"   <b>Объём:</b> {max_volume}\n"
                            f"   <b>Комиссия:</b> {sell_fee * 100:.2f}%\n"
                            f"   <b><a href='{deposit_url}'>Депозит</a></b>\n\n"
                            f"💰️ <b>Чистая прибыль:</b> ${profit_min['net']:.2f}-${profit_max['net']:.2f} ({profit_max['percent']:.2f}%)\n\n"
                            f"⏱ {current_time}\n")

                        logger.info(
                            f"Найдена арбитражная возможность: {base} ({spread:.2f}%)"
                        )

                        # Отправляем сообщение в Telegram
                        await send_telegram_message(message)

                        # Добавляем связку в отправленные возможности
                        add_opportunity_to_sent(
                            'SPOT', base, min_ex[0], max_ex[0], spread,
                            min_ex[1]['price'], max_ex[1]['price'],
                            min_ex[1]['volume'], max_ex[1]['volume'],
                            min_entry_amount, max_entry_amount, profit_min, profit_max,
                            available_volume, order_book_volume
                        )

                        # Обновляем текущие возможности с новой информацией
                        update_current_arbitrage_opportunities(
                            'SPOT', base, min_ex[0], max_ex[0], spread,
                            min_ex[1]['price'], max_ex[1]['price'],
                            min_ex[1]['volume'], max_ex[1]['volume'],
                            min_entry_amount, max_entry_amount, profit_min, profit_max,
                            available_volume, order_book_volume
                        )

                        found_opportunities += 1

                except Exception as e:
                    logger.error(f"Ошибка обработки пары {base}: {e}")

            # Очищаем устаревшие возможности (только старше 24 часов)
            cleanup_old_opportunities()

            logger.info(
                f"Цикл спотового арбитража завершен. Найдено возможностей: {found_opportunities}")
            await asyncio.sleep(SETTINGS['SPOT']['CHECK_INTERVAL'])

        except Exception as e:
            logger.error(f"Ошибка в основном цикле спотового арбитража: {e}")
            await asyncio.sleep(60)

async def check_futures_arbitrage():
    logger.info("Запуск проверки фьючерсного арбитража")

    if not SETTINGS['FUTURES']['ENABLED']:
        logger.info("Фьючерсный арбитраж отключен в настройках")
        return

    # Инициализация бирж
    await load_futures_exchanges()

    if len(FUTURES_EXCHANGES_LOADED) < SETTINGS['FUTURES']['MIN_EXCHANGES_FOR_PAIR']:
        logger.error(f"Недостаточно бирж (нужно минимум {SETTINGS['FUTURES']['MIN_EXCHANGES_FOR_PAIR']})")
        return

    # Получаем актуальные ставки финансирования
    funding_rates = await get_current_funding_rates()

    # Сбор всех торговых пар USDT
    all_pairs = defaultdict(set)
    for name, data in FUTURES_EXCHANGES_LOADED.items():
        exchange = data["api"]
        config = data["config"]
        for symbol, market in exchange.markets.items():
            try:
                if config["is_futures"](market):
                    base = market['base']
                    # Пропускаем монеты из черного списка
                    if base in config.get("blacklist", []):
                        continue
                    all_pairs[base].add((name, symbol))
            except Exception as e:
                logger.warning(f"Ошибка обработки пары {symbol} на {name}: {e}")

    valid_pairs = {
        base: list(pairs) for base, pairs in all_pairs.items()
        if len(pairs) >= SETTINGS['FUTURES']['MIN_EXCHANGES_FOR_PAIR']
    }

    if not valid_pairs:
        logger.error("Нет фьючерсных USDT пар, торгуемых хотя бы на двух биржах")
        return

    logger.info(f"Найдено {len(valid_pairs)} фьючерсных USDT пар для анализа")

    while SETTINGS['FUTURES']['ENABLED']:
        try:
            # Проверяем, изменились ли настройки бирж
            if LAST_EXCHANGE_SETTINGS != SETTINGS['EXCHANGES']:
                logger.info("Обнаружено изменение настроек бирж. Перезагружаем фьючерсные биржи...")
                await load_futures_exchanges()

                # Перестраиваем список пар после перезагрузки бирж
                all_pairs = defaultdict(set)
                for name, data in FUTURES_EXCHANGES_LOADED.items():
                    exchange = data["api"]
                    config = data["config"]
                    for symbol, market in exchange.markets.items():
                        try:
                            if config["is_futures"](market):
                                base = market['base']
                                if base in config.get("blacklist", []):
                                    continue
                                all_pairs[base].add((name, symbol))
                        except Exception as e:
                            logger.warning(f"Ошибка обработки пары {symbol} на {name}: {e}")

                valid_pairs = {
                    base: list(pairs) for base, pairs in all_pairs.items()
                    if len(pairs) >= SETTINGS['FUTURES']['MIN_EXCHANGES_FOR_PAIR']
                }

                if not valid_pairs:
                    logger.error("Нет фьючерсных USDT пар, торгуемых хотя бы на двух биржах после перезагрузки")
                    await asyncio.sleep(SETTINGS['FUTURES']['CHECK_INTERVAL'])
                    continue

            found_opportunities = 0
            for base, exchange_symbols in valid_pairs.items():
                try:
                    # Проверяем общую волатильность монеты
                    if not check_coin_volatility(base, 'FUTURES'):
                        continue

                    ticker_data = {}

                    # Получаем данные тикеров для всех бирж
                    for name, symbol in exchange_symbols:
                        try:
                            data = await fetch_ticker_data(FUTURES_EXCHANGES_LOADED[name]["api"], symbol)
                            if data and data['price'] is not None:
                                # Обновляем историю цен для расчета волатильности
                                update_price_history('FUTURES', base, name, data['price'])
                                update_coin_volatility_history(base, data['price'])

                                # Проверяем волатильность
                                if not check_volatility('FUTURES', base, name, data['price']):
                                    logger.debug(f"Пропускаем {base} на {name} из-за высокой волатильности")
                                    continue

                                # Если объем известен, проверяем минимальный объем
                                if data['volume'] is None:
                                    logger.debug(f"Объем неизвестен для {symbol} на {name}, но продолжаем обработку")
                                    ticker_data[name] = data
                                elif data['volume'] >= SETTINGS['FUTURES']['MIN_VOLUME_USD']:
                                    ticker_data[name] = data
                                else:
                                    logger.debug(f"Объем {symbol} на {name} слишком мал: {data['volume']}")
                            else:
                                logger.debug(f"Нет данных для {symbol} на {name}")
                        except Exception as e:
                            logger.warning(f"Ошибка получения данных {base} на {name}: {e}")

                    if len(ticker_data) < SETTINGS['FUTURES']['MIN_EXCHANGES_FOR_PAIR']:
                        continue

                    # Сортируем биржи по цене
                    sorted_data = sorted(ticker_data.items(), key=lambda x: x[1]['price'])
                    min_ex = sorted_data[0]  # Самая низкая цена (покупка)
                    max_ex = sorted_data[-1]  # Самая высокая цена (продажа)

                    # Рассчитываем спред
                    spread = (max_ex[1]['price'] - min_ex[1]['price']) / min_ex[1]['price'] * 100

                    logger.debug(
                        f"Пара {base}: спред {spread:.2f}% (min: {min_ex[0]} {min_ex[1]['price']}, max: {max_ex[0]} {max_ex[1]['price']})")

                    # Обновляем информацию о текущих арбитражных возможностях (только для отправленных связок)
                    update_current_arbitrage_opportunities(
                        'FUTURES', base, min_ex[0], max_ex[0], spread,
                        min_ex[1]['price'], max_ex[1]['price'],
                        min_ex[1]['volume'], max_ex[1]['volume']
                    )

                    # Проверяем сходимость цен для уведомления (только для отправленных связок)
                    duration = update_arbitrage_duration('FUTURES', base, min_ex[0], max_ex[0], spread)
                    if duration is not None:
                        await send_price_convergence_notification(
                            'FUTURES', base, min_ex[0], max_ex[0],
                            min_ex[1]['price'], max_ex[1]['price'], spread,
                            min_ex[1]['volume'], max_ex[1]['volume'], duration
                        )

                    if SETTINGS['FUTURES']['THRESHOLD_PERCENT'] <= spread <= SETTINGS['FUTURES'][
                        'MAX_THRESHOLD_PERCENT']:

                        # Получаем ставки финансирования для обеих бирж
                        long_funding = funding_rates.get(base, {}).get(min_ex[0], 0)
                        short_funding = funding_rates.get(base, {}).get(max_ex[0], 0)

                        # Проверяем выгодность финансирования
                        if not is_favorable_funding(long_funding, short_funding):
                            logger.debug(
                                f"Пропускаем {base} из-за невыгодного финансирования: лонг {long_funding:.4f}%, шорт {short_funding:.4f}%")
                            continue

                        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: пропускаем связки с "красным" фандингом
                        if has_red_funding(long_funding, short_funding):
                            logger.debug(
                                f"Пропускаем {base} из-за красного фандинга: лонг {long_funding:.4f}%, шорт {short_funding:.4f}%")
                            continue

                        # Получаем стаканы ордеров для фьючерсов
                        buy_exchange = FUTURES_EXCHANGES_LOADED[min_ex[0]]["api"]
                        sell_exchange = FUTURES_EXCHANGES_LOADED[max_ex[0]]["api"]
                        buy_symbol = min_ex[1]['symbol']
                        sell_symbol = max_ex[1]['symbol']

                        buy_order_book, sell_order_book = await asyncio.gather(
                            fetch_order_book(buy_exchange, buy_symbol, depth=10),
                            fetch_order_book(sell_exchange, sell_symbol, depth=10))

                        if not buy_order_book or not sell_order_book:
                            logger.debug(f"Пропускаем {base}: нет данных стакана")
                            continue

                        # Рассчитываем доступный объем из стакана
                        buy_volume, buy_value = calculate_available_volume(
                            buy_order_book, 'buy', SETTINGS['FUTURES']['MAX_IMPACT_PERCENT'])
                        sell_volume, sell_value = calculate_available_volume(
                            sell_order_book, 'sell', SETTINGS['FUTURES']['MAX_IMPACT_PERCENT'])
                        available_volume = min(buy_volume, sell_volume)
                        order_book_volume = min(buy_value, sell_value)

                        # Проверяем минимальный объем стакана
                        if order_book_volume < SETTINGS['FUTURES']['MIN_ORDER_BOOK_VOLUME']:
                            logger.debug(f"Пропускаем {base}: объем стакана слишком мал: ${order_book_volume:.2f}")
                            continue

                        # Получаем комиссии
                        buy_fee = FUTURES_EXCHANGES_LOADED[min_ex[0]]["config"]["taker_fee"]
                        sell_fee = FUTURES_EXCHANGES_LOADED[max_ex[0]]["config"]["taker_fee"]

                        # Рассчитываем минимальную сумму для MIN_NET_PROFIT_USD
                        min_amount_for_profit = calculate_min_entry_amount(
                            buy_price=min_ex[1]['price'],
                            sell_price=max_ex[1]['price'],
                            min_profit=SETTINGS['FUTURES']['MIN_NET_PROFIT_USD'],
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee
                        )

                        if min_amount_for_profit <= 0:
                            logger.debug(f"Пропускаем {base}: недостаточная прибыль")
                            continue

                        # Рассчитываем максимально возможную сумму входа
                        max_possible_amount = min(
                            available_volume * min_ex[1]['price'],
                            SETTINGS['FUTURES']['MAX_ENTRY_AMOUNT_USDT'],
                            order_book_volume)

                        max_entry_amount = max_possible_amount
                        min_entry_amount = max(min_amount_for_profit, SETTINGS['FUTURES']['MIN_ENTRY_AMOUNT_USDT'])

                        if min_entry_amount > max_entry_amount:
                            logger.debug(f"Пропускаем {base}: min_entry_amount > max_entry_amount")
                            continue

                        # Рассчитываем базовую прибыль
                        profit_min = calculate_profit(
                            buy_price=min_ex[1]['price'],
                            sell_price=max_ex[1]['price'],
                            amount=min_entry_amount / min_ex[1]['price'],
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee
                        )

                        profit_max = calculate_profit(
                            buy_price=min_ex[1]['price'],
                            sell_price=max_ex[1]['price'],
                            amount=max_possible_amount / min_ex[1]['price'],
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee
                        )

                        # Рассчитываем эффективную прибыль с учетом финансирования
                        effective_profit_min = calculate_effective_profit_with_funding(
                            profit_min['net'], min_entry_amount, long_funding, short_funding)
                        effective_profit_max = calculate_effective_profit_with_funding(
                            profit_max['net'], max_entry_amount, long_funding, short_funding)

                        # Форматируем сообщение
                        utc_plus_3 = timezone(timedelta(hours=3))
                        current_time = datetime.now(utc_plus_3).strftime('%H:%M:%S')

                        def format_volume(vol):
                            if vol is None:
                                return "N/A"
                            if vol >= 1_000_000:
                                return f"${vol / 1_000_000:.1f}M"
                            if vol >= 1_000:
                                return f"${vol / 1_000:.1f}K"
                            return f"${vol:.1f}"

                        min_volume = format_volume(min_ex[1]['volume'])
                        max_volume = format_volume(max_ex[1]['volume'])

                        safe_base = html.escape(base)
                        buy_exchange_config = FUTURES_EXCHANGES[min_ex[0]]
                        sell_exchange_config = FUTURES_EXCHANGES[max_ex[0]]

                        buy_url = buy_exchange_config["url_format"](min_ex[1]['symbol'].replace(':USDT', ''))
                        sell_url = sell_exchange_config["url_format"](max_ex[1]['symbol'].replace(':USDT', ''))

                        # Рассчитываем оценку финансирования
                        funding_score = calculate_funding_score(long_funding, short_funding)
                        funding_emoji = "🟢" 
                        if has_red_funding(long_funding, short_funding):
                            funding_emoji = "🔴"
                        elif funding_score > 0:
                            funding_emoji = "🟡"

                        message = (
                            f"📊 <b>Фьючерсный арбитраж:</b> <code>{safe_base}</code>\n"
                            f"▫️ <b>Разница цен:</b> {spread:.2f}%\n"
                            f"▫️ <b>Доступный объем:</b> {available_volume:.6f} {safe_base}\n"
                            f"▫️ <b>Объем стакана:</b> ${order_book_volume:.2f}\n"
                            f"▫️ <b>Сумма входа:</b> ${min_entry_amount:.2f}-${max_entry_amount:.2f}\n"
                            f"▫️ {funding_emoji} <b>Фандинг:</b> лонг {long_funding:.4f}% | шорт {short_funding:.4f}% | общий {funding_score:.4f}%\n\n"
                            f"🟢 <b>Лонг на <a href='{buy_url}'>{min_ex[0].upper()}</a>:</b> ${min_ex[1]['price']:.8f}\n"
                            f"   <b>Объём:</b> {min_volume}\n"
                            f"   <b>Комиссия:</b> {buy_fee * 100:.3f}%\n\n"
                            f"🔴 <b>Шорт на <a href='{sell_url}'>{max_ex[0].upper()}</a>:</b> ${max_ex[1]['price']:.8f}\n"
                            f"   <b>Объём:</b> {max_volume}\n"
                            f"   <b>Комиссия:</b> {sell_fee * 100:.3f}%\n\n"
                            f"💰 <b>Базовая прибыль:</b> ${profit_min['net']:.2f}-${profit_max['net']:.2f}\n"
                            f"💎 <b>С учетом фандинга:</b> ${effective_profit_min:.2f}-${effective_profit_max:.2f} ({profit_max['percent']:.2f}%)\n\n"
                            f"⏱ {current_time}\n"
                        )

                        logger.info(f"Найдена арбитражная возможность: {base} ({spread:.2f}%)")

                        # Отправляем сообщение в Telegram
                        await send_telegram_message(message)

                        # Добавляем связку в отправленные возможности
                        add_opportunity_to_sent(
                            'FUTURES', base, min_ex[0], max_ex[0], spread,
                            min_ex[1]['price'], max_ex[1]['price'],
                            min_ex[1]['volume'], max_ex[1]['volume'],
                            min_entry_amount, max_entry_amount, profit_min, profit_max,
                            available_volume, order_book_volume,
                            long_funding, short_funding
                        )

                        # Обновляем текущие возможности с новой информацией
                        update_current_arbitrage_opportunities(
                            'FUTURES', base, min_ex[0], max_ex[0], spread,
                            min_ex[1]['price'], max_ex[1]['price'],
                            min_ex[1]['volume'], max_ex[1]['volume'],
                            min_entry_amount, max_entry_amount, profit_min, profit_max,
                            available_volume, order_book_volume,
                            long_funding, short_funding
                        )

                        found_opportunities += 1

                except Exception as e:
                    logger.error(f"Ошибка обработки пары {base}: {e}")

            # Очищаем устаревшие возможности (только старше 24 часов)
            cleanup_old_opportunities()

            logger.info(f"Цикл фьючерсного арбитража завершен. Найдено возможностей: {found_opportunities}")
            await asyncio.sleep(SETTINGS['FUTURES']['CHECK_INTERVAL'])

        except Exception as e:
            logger.error(f"Ошибка в основном цикле фьючерсного арбитража: {e}")
            await asyncio.sleep(60)

async def check_spot_futures_arbitrage():
    """Проверка спот-фьючерсного арбитража"""
    logger.info("Запуск проверки спот-фьючерсного арбитража")

    if not SETTINGS['SPOT_FUTURES']['ENABLED']:
        logger.info("Спот-фьючерсный арбитраж отключен в настройки")
        return

    # Инициализация бирж
    global SPOT_EXCHANGES_LOADED, FUTURES_EXCHANGES_LOADED

    # Загружаем спотовые и фьючерсные биржи
    await load_spot_exchanges()
    await load_futures_exchanges()

    if len(SPOT_EXCHANGES_LOADED) < 1 or len(FUTURES_EXCHANGES_LOADED) < 1:
        logger.error("Недостаточно бирж для спот-фьючерсного арбитража")
        return

    # Собираем все торговые пары
    spot_pairs = defaultdict(set)
    futures_pairs = defaultdict(set)

    # Собираем спотовые пары
    for name, data in SPOT_EXCHANGES_LOADED.items():
        exchange = data["api"]
        config = data["config"]
        for symbol, market in exchange.markets.items():
            try:
                if config["is_spot"](market):
                    base = market['base']
                    # Пропускаем монеты из черного списка
                    if base in config.get("blacklist", []):
                        continue
                    spot_pairs[base].add((name, symbol))
            except Exception as e:
                logger.warning(f"Ошибка обработки спотовой пары {symbol} на {name}: {e}")

    # Собираем фьючерсные пары
    for name, data in FUTURES_EXCHANGES_LOADED.items():
        exchange = data["api"]
        config = data["config"]
        for symbol, market in exchange.markets.items():
            try:
                if config["is_futures"](market):
                    base = market['base']
                    # Пропускаем монеты из черного списка
                    if base in config.get("blacklist", []):
                        continue
                    futures_pairs[base].add((name, symbol))
            except Exception as e:
                logger.warning(f"Ошибка обработки фьючерсной пары {symbol} на {name}: {e}")

    # Находим общие пары
    common_pairs = set(spot_pairs.keys()) & set(futures_pairs.keys())

    if not common_pairs:
        logger.error("Нет общих пар для спот-фьючерсного арбитража")
        return

    logger.info(f"Найдено {len(common_pairs)} общих пар для анализа")

    while SETTINGS['SPOT_FUTURES']['ENABLED']:
        try:
            # Проверяем, изменились ли настройки бирж
            if LAST_EXCHANGE_SETTINGS != SETTINGS['EXCHANGES']:
                logger.info("Обнаружено изменение настроек бирж. Перезагружаем спотовые и фьючерсные биржи...")
                await load_spot_exchanges()
                await load_futures_exchanges()

                # Перестраиваем списки пар после перезагрузки бирж
                spot_pairs = defaultdict(set)
                futures_pairs = defaultdict(set)

                for name, data in SPOT_EXCHANGES_LOADED.items():
                    exchange = data["api"]
                    config = data["config"]
                    for symbol, market in exchange.markets.items():
                        try:
                            if config["is_spot"](market):
                                base = market['base']
                                if base in config.get("blacklist", []):
                                    continue
                                spot_pairs[base].add((name, symbol))
                        except Exception as e:
                            logger.warning(f"Ошибка обработки спотовой пары {symbol} на {name}: {e}")

                for name, data in FUTURES_EXCHANGES_LOADED.items():
                    exchange = data["api"]
                    config = data["config"]
                    for symbol, market in exchange.markets.items():
                        try:
                            if config["is_futures"](market):
                                base = market['base']
                                if base in config.get("blacklist", []):
                                    continue
                                futures_pairs[base].add((name, symbol))
                        except Exception as e:
                            logger.warning(f"Ошибка обработки фьючерсной пары {symbol} на {name}: {e}")

                common_pairs = set(spot_pairs.keys()) & set(futures_pairs.keys())

                if not common_pairs:
                    logger.error("Нет общих пар для спот-фьючерсного арбитража после перезагрузки")
                    await asyncio.sleep(SETTINGS['SPOT_FUTURES']['CHECK_INTERVAL'])
                    continue

            found_opportunities = 0
            for base in common_pairs:
                try:
                    # Проверяем общую волатильность монеты
                    if not check_coin_volatility(base, 'SPOT_FUTURES'):
                        continue

                    spot_ticker_data = {}
                    futures_ticker_data = {}

                    # Получаем данные тикеров для спотовых бирж
                    for name, symbol in spot_pairs[base]:
                        try:
                            data = await fetch_ticker_data(SPOT_EXCHANGES_LOADED[name]["api"], symbol)
                            if data and data['price'] is not None:
                                # Обновляем историю цен для расчета волатильности
                                update_price_history('SPOT_FUTURES', base, f"{name}_spot", data['price'])
                                update_coin_volatility_history(base, data['price'])

                                # Проверяем волатильность
                                if not check_volatility('SPOT_FUTURES', base, f"{name}_spot", data['price']):
                                    logger.debug(f"Пропускаем {base} на {name} (спот) из-за высокой волатильности")
                                    continue

                                if data['volume'] is None or data['volume'] >= SETTINGS['SPOT_FUTURES'][
                                    'MIN_VOLUME_USD']:
                                    spot_ticker_data[name] = data
                        except Exception as e:
                            logger.warning(f"Ошибка получения спотовых данных {base} на {name}: {e}")

                    # Получаем данные тикеров для фьючерсных бирж
                    for name, symbol in futures_pairs[base]:
                        try:
                            data = await fetch_ticker_data(FUTURES_EXCHANGES_LOADED[name]["api"], symbol)
                            if data and data['price'] is not None:
                                # Обновляем историю цен для расчета волатильности
                                update_price_history('SPOT_FUTURES', base, f"{name}_futures", data['price'])
                                update_coin_volatility_history(base, data['price'])

                                # Проверяем волатильность
                                if not check_volatility('SPOT_FUTURES', base, f"{name}_futures", data['price']):
                                    logger.debug(f"Пропускаем {base} на {name} (фьючерсы) из-за высокой волатильности")
                                    continue

                                if data['volume'] is None or data['volume'] >= SETTINGS['SPOT_FUTURES'][
                                    'MIN_VOLUME_USD']:
                                    futures_ticker_data[name] = data
                        except Exception as e:
                            logger.warning(f"Ошибка получения фьючерсных данных {base} на {name}: {e}")

                    if not spot_ticker_data or not futures_ticker_data:
                        continue

                    # Находим лучшие цены
                    min_spot = min(spot_ticker_data.items(), key=lambda x: x[1]['price'])
                    max_futures = max(futures_ticker_data.items(), key=lambda x: x[1]['price'])

                    # Рассчитываем спред
                    spread = (max_futures[1]['price'] - min_spot[1]['price']) / min_spot[1]['price'] * 100

                    logger.debug(
                        f"Пара {base}: спред {spread:.2f}% (spot: {min_spot[0]} {min_spot[1]['price']}, futures: {max_futures[0]} {max_futures[1]['price']})")

                    # Обновляем информацию о текущих арбитражных возможностях (только для отправленных связок)
                    update_current_arbitrage_opportunities(
                        'SPOT_FUTURES', base, min_spot[0], max_futures[0], spread,
                        min_spot[1]['price'], max_futures[1]['price'],
                        min_spot[1]['volume'], max_futures[1]['volume']
                    )

                    # Проверяем сходимость цен для уведомления (только для отправленных связок)
                    duration = update_arbitrage_duration('SPOT_FUTURES', base, min_spot[0], max_futures[0], spread)
                    if duration is not None:
                        await send_price_convergence_notification(
                            'SPOT_FUTURES', base, min_spot[0], max_futures[0],
                            min_spot[1]['price'], max_futures[1]['price'], spread,
                            min_spot[1]['volume'], max_futures[1]['volume'], duration
                        )

                    if SETTINGS['SPOT_FUTURES']['THRESHOLD_PERCENT'] <= spread <= SETTINGS['SPOT_FUTURES'][
                        'MAX_THRESHOLD_PERCENT']:
                        # Проверяем доступность депозита и вывода для спота
                        deposit_available = await check_deposit_withdrawal_status(
                            SPOT_EXCHANGES_LOADED[min_spot[0]]["api"], base, 'deposit')
                        withdrawal_available = await check_deposit_withdrawal_status(
                            SPOT_EXCHANGES_LOADED[min_spot[0]]["api"], base, 'withdrawal')

                        if not (deposit_available and withdrawal_available):
                            logger.debug(f"Пропускаем {base}: депозит или вывод недоступен")
                            continue

                        # Получаем стаканы ордеров
                        spot_exchange = SPOT_EXCHANGES_LOADED[min_spot[0]]["api"]
                        futures_exchange = FUTURES_EXCHANGES_LOADED[max_futures[0]]["api"]
                        spot_symbol = min_spot[1]['symbol']
                        futures_symbol = max_futures[1]['symbol']

                        spot_order_book, futures_order_book = await asyncio.gather(
                            fetch_order_book(spot_exchange, spot_symbol),
                            fetch_order_book(futures_exchange, futures_symbol, depth=10))

                        if not spot_order_book or not futures_order_book:
                            logger.debug(f"Пропускаем {base}: нет данных стакана")
                            continue

                        # Рассчитываем доступный объем из стакана
                        spot_volume, spot_value = calculate_available_volume(
                            spot_order_book, 'buy', SETTINGS['SPOT_FUTURES']['MAX_IMPACT_PERCENT'])
                        futures_volume, futures_value = calculate_available_volume(
                            futures_order_book, 'sell', SETTINGS['SPOT_FUTURES']['MAX_IMPACT_PERCENT'])
                        available_volume = min(spot_volume, futures_volume)
                        order_book_volume = min(spot_value, futures_value)

                        # Проверяем минимальный объем стакана
                        if order_book_volume < SETTINGS['SPOT_FUTURES']['MIN_ORDER_BOOK_VOLUME']:
                            logger.debug(f"Пропускаем {base}: объем стакана слишком мал: ${order_book_volume:.2f}")
                            continue

                        # Получаем комиссии
                        spot_fee = SPOT_EXCHANGES_LOADED[min_spot[0]]["config"]["taker_fee"]
                        futures_fee = FUTURES_EXCHANGES_LOADED[max_futures[0]]["config"]["taker_fee"]

                        # Рассчитываем минимальную сумму для MIN_NET_PROFIT_USD
                        min_amount_for_profit = calculate_min_entry_amount(
                            buy_price=min_spot[1]['price'],
                            sell_price=max_futures[1]['price'],
                            min_profit=SETTINGS['SPOT_FUTURES']['MIN_NET_PROFIT_USD'],
                            buy_fee_percent=spot_fee,
                            sell_fee_percent=futures_fee
                        )

                        if min_amount_for_profit <= 0:
                            logger.debug(f"Пропускаем {base}: недостаточная прибыль")
                            continue

                        # Рассчитываем максимально возможную сумму входа
                        max_possible_amount = min(
                            available_volume * min_spot[1]['price'],
                            SETTINGS['SPOT_FUTURES']['MAX_ENTRY_AMOUNT_USDT'],
                            order_book_volume)

                        max_entry_amount = max_possible_amount
                        min_entry_amount = max(min_amount_for_profit, SETTINGS['SPOT_FUTURES']['MIN_ENTRY_AMOUNT_USDT'])

                        if min_entry_amount > max_entry_amount:
                            logger.debug(f"Пропускаем {base}: min_entry_amount > max_entry_amount")
                            continue

                        # Рассчитываем прибыль
                        profit_min = calculate_profit(
                            buy_price=min_spot[1]['price'],
                            sell_price=max_futures[1]['price'],
                            amount=min_entry_amount / min_spot[1]['price'],
                            buy_fee_percent=spot_fee,
                            sell_fee_percent=futures_fee
                        )

                        profit_max = calculate_profit(
                            buy_price=min_spot[1]['price'],
                            sell_price=max_futures[1]['price'],
                            amount=max_possible_amount / min_spot[1]['price'],
                            buy_fee_percent=spot_fee,
                            sell_fee_percent=futures_fee
                        )

                        # Форматируем сообщение
                        utc_plus_3 = timezone(timedelta(hours=3))
                        current_time = datetime.now(utc_plus_3).strftime('%H:%M:%S')

                        def format_volume(vol):
                            if vol is None:
                                return "N/A"
                            if vol >= 1_000_000:
                                return f"${vol / 1_000_000:.1f}M"
                            if vol >= 1_000:
                                return f"${vol / 1_000:.1f}K"
                            return f"${vol:.1f}"

                        spot_volume_str = format_volume(min_spot[1]['volume'])
                        futures_volume_str = format_volume(max_futures[1]['volume'])

                        safe_base = html.escape(base)
                        spot_exchange_config = SPOT_EXCHANGES[min_spot[0]]
                        futures_exchange_config = FUTURES_EXCHANGES[max_futures[0]]

                        spot_url = spot_exchange_config["url_format"](min_spot[1]['symbol'])
                        futures_url = futures_exchange_config["url_format"](
                            max_futures[1]['symbol'].replace(':USDT', ''))
                        withdraw_url = spot_exchange_config["withdraw_url"](base)
                        deposit_url = spot_exchange_config["deposit_url"](base)

                        message = (
                            f"↔️ <b>Спот-Фьючерсный арбитраж:</b> <code>{safe_base}</code>\n"
                            f"▫️ <b>Разница цен:</b> {spread:.2f}%\n"
                            f"▫️ <b>Доступный объем:</b> {available_volume:.6f} {safe_base}\n"
                            f"▫️ <b>Объем стакана:</b> ${order_book_volume:.2f}\n"
                            f"▫️ <b>Сумма входа:</b> ${min_entry_amount:.2f}-${max_entry_amount:.2f}\n\n"
                            f"🟢 <b>Покупка на споте <a href='{spot_url}'>{min_spot[0].upper()}</a>:</b> ${min_spot[1]['price']:.8f}\n"
                            f"   <b>Объём:</b> {spot_volume_str}\n"
                            f"   <b>Комиссия:</b> {spot_fee * 100:.2f}%\n"
                            f"   <b><a href='{withdraw_url}'>Вывод</a> | <a href='{deposit_url}'>Депозит</a></b>\n\n"
                            f"🔴 <b>Шорт на фьючерсах <a href='{futures_url}'>{max_futures[0].upper()}</a>:</b> ${max_futures[1]['price']:.8f}\n"
                            f"   <b>Объём:</b> {futures_volume_str}\n"
                            f"   <b>Комиссия:</b> {futures_fee * 100:.3f}%\n\n"
                            f"💰 <b>Чистая прибыль:</b> ${profit_min['net']:.2f}-${profit_max['net']:.2f} ({profit_max['percent']:.2f}%)\n\n"
                            f"⏱ {current_time}\n"
                        )

                        logger.info(f"Найдена спот-фьючерсная арбитражная возможность: {base} ({spread:.2f}%)")

                        # Отправляем сообщение в Telegram
                        await send_telegram_message(message)

                        # Добавляем связку в отправленные возможности
                        add_opportunity_to_sent(
                            'SPOT_FUTURES', base, min_spot[0], max_futures[0], spread,
                            min_spot[1]['price'], max_futures[1]['price'],
                            min_spot[1]['volume'], max_futures[1]['volume'],
                            min_entry_amount, max_entry_amount, profit_min, profit_max,
                            available_volume, order_book_volume
                        )

                        # Обновляем текущие возможности с новой информацией
                        update_current_arbitrage_opportunities(
                            'SPOT_FUTURES', base, min_spot[0], max_futures[0], spread,
                            min_spot[1]['price'], max_futures[1]['price'],
                            min_spot[1]['volume'], max_futures[1]['volume'],
                            min_entry_amount, max_entry_amount, profit_min, profit_max,
                            available_volume, order_book_volume
                        )

                        found_opportunities += 1

                except Exception as e:
                    logger.error(f"Ошибка обработки пары {base}: {e}")

            # Очищаем устаревшие возможности (только старше 24 часов)
            cleanup_old_opportunities()

            logger.info(f"Цикл спот-фьючерсного арбитража завершен. Найдено возможностей: {found_opportunities}")
            await asyncio.sleep(SETTINGS['SPOT_FUTURES']['CHECK_INTERVAL'])

        except Exception as e:
            logger.error(f"Ошибка в основном цикле спот-фьючерсного арбитража: {e}")
            await asyncio.sleep(60)

def format_price(price: float) -> str:
    """Форматирует цену для красивого отображения"""
    if price is None:
        return "N/A"

    # Для цен > 1000 используем запятые как разделители тысяч
    if price >= 1000:
        return f"$<code>{price:.2f}</code>"

    # Для цен > 1 используем 4 знака после запятой
    if price >= 1:
        return f"$<code>{price:.4f}</code>"

    # Для цен < 1 используем 8 знаков после запятой
    return f"$<code>{price:.8f}</code>"

def format_volume(vol: float) -> str:
    """Форматирует объем для красивого отображения"""
    if vol is None:
        return "N/A"

    # Для объемов > 1 миллиона
    if vol >= 1_000_000:
        return f"${vol / 1_000_000:.1f}M"

    # Для объемов > 1000
    if vol >= 1_000:
        return f"${vol / 1_000:.1f}K"

    # Для объемов < 1000
    return f"${vol:.0f}"

async def get_coin_prices(coin: str, market_type: str):
    """Получает цены монеты на всех биржах для указанного рынка с фильтрацией по объему"""
    coin = coin.upper()

    # Перезагружаем биржи если настройки изменились
    if LAST_EXCHANGE_SETTINGS != SETTINGS['EXCHANGES']:
        if market_type == "spot":
            await load_spot_exchanges()
            exchanges = SPOT_EXCHANGES_LOADED
        else:
            await load_futures_exchanges()
            exchanges = FUTURES_EXCHANGES_LOADED
    else:
        exchanges = SPOT_EXCHANGES_LOADED if market_type == "spot" else FUTURES_EXCHANGES_LOADED

    if not exchanges:
        return "❌ Биржи еще не загружены. Попробуйте позже."

    results = []
    found_on = 0
    filtered_out = 0

    # Определяем минимальный объем в зависимости от типа рынка
    if market_type == "spot":
        min_volume = SETTINGS['SPOT']['MIN_VOLUME_USD']
        min_entry = SETTINGS['SPOT']['MIN_ENTRY_AMOUNT_USDT']
        max_entry = SETTINGS['SPOT']['MAX_ENTRY_AMOUNT_USDT']
    else:
        min_volume = SETTINGS['FUTURES']['MIN_VOLUME_USD']
        min_entry = SETTINGS['FUTURES']['MIN_ENTRY_AMOUNT_USDT']
        max_entry = SETTINGS['FUTURES']['MAX_ENTRY_AMOUNT_USDT']

    for name, data in exchanges.items():
        exchange = data["api"]
        config = data["config"]

        # Формируем символ в зависимости от типа рынка
        symbol = config["symbol_format"](coin)

        try:
            market = exchange.market(symbol)
            if (market_type == "spot" and config["is_spot"](market)) or \
                    (market_type == "futures" and config["is_futures"](market)):

                ticker = await fetch_ticker_data(exchange, symbol)
                if ticker and ticker['price']:
                    # Проверяем объем - фильтруем по минимальному объему из настроек
                    if ticker.get('volume') is not None and ticker['volume'] < min_volume:
                        filtered_out += 1
                        logger.debug(f"Биржа {name} отфильтрована по объему: {ticker['volume']} < {min_volume}")
                        continue

                    found_on += 1
                    price = ticker['price']
                    volume = ticker.get('volume')

                    # Получаем URL для биржи
                    url = config["url_format"](symbol)

                    # Добавляем данные для сортировки
                    results.append({
                        "price": price,
                        "name": name.upper(),
                        "volume": volume,
                        "url": url,
                        "emoji": config.get("emoji", "🏛")
                    })
        except Exception as e:
            logger.warning(f"Ошибка получения цены {symbol} на {name}: {e}")

    # Сортируем результаты по цене (от низкой к высокой)
    results.sort(key=lambda x: x["price"])

    utc_plus_3 = timezone(timedelta(hours=3))
    current_time = datetime.now(utc_plus_3).strftime('%H:%M:%S')

    market_name = "Спот" if market_type == "spot" else "Фьючерсы"
    market_color = "market_color = 🚀" if market_type == "spot" else "📊"

    if results:
        # Рассчитываем разницу в процентах между самой низкой и высокой ценой
        min_price = results[0]["price"]
        max_price = results[-1]["price"]
        price_diff_percent = ((max_price - min_price) / min_price) * 100

        # Формируем заголовок с информацией о фильтрации
        response = f"{market_color} <b>{market_name} рынки для <code>{coin}</code>:</b>\n\n"
        response += f"<i>Минимальный объем: ${min_volume:,.0f}</i>\n"
        response += f"<i>Отфильтровано бирж: {filtered_out}</i>\n\n"

        # Добавляем данные по каждой бирже
        for idx, item in enumerate(results, 1):
            # Сделаем название биржи кликабельной ссылкой
            response += (
                f"{item['emoji']} <a href='{item['url']}'><b>{item['name']}</b></a>\n"
                f"▫️ Цена: {format_price(item['price'])}\n"
                f"▫️ Объем: {format_volume(item['volume'])}\n"
            )

            # Добавляем разделитель, если это не последний элемент
            if idx < len(results):
                response += "\n"

        # Добавляем информацию о возможной арбитражной прибыли
        if len(results) >= 2 and min_price < max_price:
            # Находим биржи с минимальной и максимальной ценой
            min_exchange = results[0]
            max_exchange = results[-1]

            # Получаем комиссии для этих бирж
            if market_type == "spot":
                buy_fee = SPOT_EXCHANGES[min_exchange['name'].lower()]["taker_fee"]
                sell_fee = SPOT_EXCHANGES[max_exchange['name'].lower()]["taker_fee"]
            else:
                buy_fee = FUTURES_EXCHANGES[min_exchange['name'].lower()]["taker_fee"]
                sell_fee = FUTURES_EXCHANGES[max_exchange['name'].lower()]["taker_fee"]

            # Рассчитываем прибыль для минимальной и максимальной суммы входа
            profit_min = calculate_profit(
                buy_price=min_price,
                sell_price=max_price,
                amount=min_entry / min_price,
                buy_fee_percent=buy_fee,
                sell_fee_percent=sell_fee
            )

            profit_max = calculate_profit(
                buy_price=min_price,
                sell_price=max_price,
                amount=max_entry / min_price,
                buy_fee_percent=buy_fee,
                sell_fee_percent=sell_fee
            )

            # Добавляем информацию о возможной арбитражной прибыли
            response += f"\n💼 <b>Возможный арбитраж:</b>\n"
            response += f"🟢 Покупка на {min_exchange['name']}: {format_price(min_price)}\n"
            response += f"🔴 Продажа на {max_exchange['name']}: {format_price(max_price)}\n"
            response += f"💰 Сумма входа: ${min_entry:.2f}-${max_entry:.2f}\n"
            response += f"💵 Чистая прибыль: ${profit_min['net']:.2f}-${profit_max['net']:.2f}\n"

        # Добавляем разницу цен и время
        response += f"\n📈 <b>Разница цен:</b> {price_diff_percent:.2f}%\n"
        response += f"⏱ {current_time} | Бирж: {found_on}"
    else:
        if filtered_out > 0:
            response = f"❌ Монета {coin} найдена на {filtered_out} биржах, но объем меньше ${min_volume:,.0f}"
        else:
            response = f"❌ Монета {coin} не найдена на {market_name} рынке"

    return response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = str(update.effective_user.id)
    if user_id not in TELEGRAM_CHAT_IDS:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return

    await update.message.reply_text(
        "🤖 <b>Crypto Arbitrage Bot</b>\n\n"
        "Используйте кнопки ниже для взаимодействия с ботом:",
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user_id = str(update.effective_user.id)
    if user_id not in TELEGRAM_CHAT_IDS:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return

    text = update.message.text

    if text == "🔧 Настройки":
        await update.message.reply_text(
            "⚙️ <b>Настройки бота</b>\n\nВыберите категорию:",
            parse_mode="HTML",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    elif text == "📈 Актуальные связки":
        # Устанавливаем начальную страницу
        context.user_data['arbitrage_page'] = 0
        context.user_data['arbitrage_per_page'] = 5
        
        # Получаем первую страницу
        message, total_pages, current_page = await get_current_arbitrage_opportunities_page(
            page=0, opportunities_per_page=5
        )
        
        # Сохраняем данные пагинации
        context.user_data['arbitrage_total_pages'] = total_pages
        context.user_data['arbitrage_current_page'] = current_page
        
        # Отправляем сообщение с клавиатурой пагинации
        await update.message.reply_text(
            text=message,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=get_arbitrage_list_keyboard(current_page, total_pages)
        )
        return ARBITRAGE_LIST

    elif text == "📊 Статус бота":
        try:
            spot_status = "✅ ВКЛ" if SETTINGS['SPOT']['ENABLED'] else "❌ ВЫКЛ"
            futures_status = "✅ ВКЛ" if SETTINGS['FUTURES']['ENABLED'] else "❌ ВЫКЛ"
            spot_futures_status = "✅ ВКЛ" if SETTINGS['SPOT_FUTURES']['ENABLED'] else "❌ ВЫКЛ"

            enabled_exchanges = [name for name, config in SETTINGS['EXCHANGES'].items() if config['ENABLED']]
            exchanges_status = ", ".join(enabled_exchanges) if enabled_exchanges else "Нет активных бирж"

            # Получаем информацию о финансировании
            funding_info = ""
            if SETTINGS['FUTURES']['ENABLED']:
                try:
                    funding_rates = await get_current_funding_rates()
                    funding_info = f"\n💰 Активных ставок фандинга: {len(funding_rates)}"
                except Exception as e:
                    logger.error(f"Ошибка при получении ставок финансирования: {e}")
                    funding_info = f"\n💰 Не удалось загрузить ставки фандинга"

            status_message = (
                f"🤖 <b>Статус бота</b>\n\n"
                f"🚀 Спотовый арбитраж: {spot_status}\n"
                f"📊 Фьючерсный арбитраж: {futures_status}\n"
                f"↔️ Спот-Фьючерсный арбитраж: {spot_futures_status}\n"
                f"🏛 Активные биржи: {exchanges_status}\n"
                f"📈 Активных связок: {len(sent_arbitrage_opportunities)}"
                f"{funding_info}"
            )

            await update.message.reply_text(
                status_message,
                parse_mode="HTML",
                reply_markup=get_main_keyboard()
            )
        except Exception as e:
            logger.error(f"Ошибка в обработчике статуса бота: {e}")
            await update.message.reply_text(
                "❌ Произошла ошибка при получении статуса бота.",
                reply_markup=get_main_keyboard()
            )
        return ConversationHandler.END

    elif text == "ℹ️ Помощь":
        await update.message.reply_text(
            "🤖 <b>Crypto Arbitrage Bot</b>\n\n"
            "🔍 <b>Поиск монеты</b> - показывает цены на разных биржах, просто введите название монеты (BTC, ETH...)\n"
            "🔧 <b>Настройки</b> - позволяет настроить параметры арбитража\n"
            "📊 <b>Статус бота</b> - показывает текущее состояние бота\n"
            "📈 <b>Актуальные связки</b> - показывает текущие арбитражные возможности и их длительность\n\n"
            "<b>Особенности фьючерсного арбитража:</b>\n"
            "• Учитывает ставки финансирования (funding rate)\n"
            "• Фильтрует невыгодные связки с высокими платежами\n"
            "• Показывает эффективную прибыль с учетом фандинга\n\n"
            "Бот автоматически ищет арбитражные возможности и присылает уведомления.",
            parse_mode="HTML",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END

    # Если это не команда, предполагаем, что это название монеты
    if not text.startswith('/'):
        # Проверяем, что введен допустимый символ (только буквы и цифры)
        if re.match(r'^[A-Z0-9]{1,15}$', text.upper()):
            # Сохраняем монету в контексте и предлагаем выбрать тип рынка
            context.user_data['coin'] = text.upper()
            await update.message.reply_text(
                f"🔍 Выберите тип рынка для <b><code>{text.upper()}</code></b>:",
                parse_mode="HTML",
                reply_markup=ReplyKeyboardMarkup([
                    [KeyboardButton(f"🚀 {text.upper()} Спот"), KeyboardButton(f"📊 {text.upper()} Фьючерсы")],
                    [KeyboardButton("🔙 Главное меню")]
                ], resize_keyboard=True)
            )
            return COIN_SELECTION
        else:
            await update.message.reply_text(
                "⚠️ Неверный формат названия монеты. Используйте только буквы и цифры (например BTC или ETH)",
                reply_markup=get_main_keyboard()
            )
            return ConversationHandler.END

    await update.message.reply_text(
        "Неизвестная команда. Используйте кнопки меню.",
        reply_markup=get_main_keyboard()
    )
    return ConversationHandler.END

async def handle_coin_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора типа рынка для монеты"""
    text = update.message.text
    coin = context.user_data.get('coin')

    if text == "🔙 Главное меню":
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END

    if not coin:
        await update.message.reply_text(
            "Не удалось определить монету. Попробуйте снова.",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END

    if "Спот" in text:
        market_type = "spot"
    elif "Фьючерсы" в text:
        market_type = "futures"
    else:
        await update.message.reply_text(
            "Пожалуйста, выберите тип рынка с помощью кнопок.",
            reply_markup=ReplyKeyboardMarkup([
                [KeyboardButton(f"🚀 {coin} Спот"), KeyboardButton(f"📊 {coin} Фьючерсы")],
                [KeyboardButton("🔙 Главное меню")]
            ], resize_keyboard=True)
        )
        return COIN_SELECTION

    # Показываем "Загрузка..."
    await update.message.reply_text(
        f"⏳ Загружаем данные для <b><code>{coin}</code></b> на {'споте' if market_type == 'spot' else 'фьючерсах'}...",
        parse_mode="HTML"
    )

    # Получаем данные
    response = await get_coin_prices(coin, market_type)

    # Отправляем результаты
    await update.message.reply_text(
        text=response,
        parse_mode="HTML",
        disable_web_page_preview=True,
        reply_markup=get_main_keyboard()
    )
    return ConversationHandler.END

async def handle_arbitrage_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка навигации по списку арбитражных связок"""
    text = update.message.text
    user_id = str(update.effective_user.id)
    
    # Получаем текущие данные пагинации
    current_page = context.user_data.get('arbitrage_page', 0)
    total_pages = context.user_data.get('arbitrage_total_pages', 1)
    per_page = context.user_data.get('arbitrage_per_page', 5)

    if text == "⬅️ Предыдущая":
        if current_page > 0:
            current_page -= 1
    elif text == "Следующая ➡️":
        if current_page < total_pages - 1:
            current_page += 1
    elif text == "🔙 Главное меню":
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END
    else:
        # Если нажата другая кнопка, остаемся на текущей странице
        pass

    # Обновляем данные пагинации
    context.user_data['arbitrage_page'] = current_page
    context.user_data['arbitrage_current_page'] = current_page

    # Получаем данные для текущей страницы
    message, total_pages, current_page = await get_current_arbitrage_opportunities_page(
        page=current_page, opportunities_per_page=per_page
    )
    
    # Обновляем общее количество страниц
    context.user_data['arbitrage_total_pages'] = total_pages

    # Отправляем обновленное сообщение
    await update.message.reply_text(
        text=message,
        parse_mode="HTML",
        disable_web_page_preview=True,
        reply_markup=get_arbitrage_list_keyboard(current_page, total_pages)
    )
    return ARBITRAGE_LIST

async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка меню настроек"""
    text = update.message.text

    if text == "🚀️ Спот":
        await update.message.reply_text(
            "🚀️ <b>Настройки спотового арбитража</b>\n\nВыберите параметр для изменения:",
            parse_mode="HTML",
            reply_markup=get_spot_settings_keyboard()
        )
        return SPOT_SETTINGS

    elif text == "📊 Фьючерсы":
        await update.message.reply_text(
            "📊 <b>Настройки фьючерсного арбитража</b>\n\nВыберите параметр для изменения:",
            parse_mode="HTML",
            reply_markup=get_futures_settings_keyboard()
        )
        return FUTURES_SETTINGS

    elif text == "↔️ Спот-Фьючерсы":
        await update.message.reply_text(
            "↔️ <b>Настройки спот-фьючерсного арбитража</b>\n\nВыберите параметр для изменения:",
            parse_mode="HTML",
            reply_markup=get_spot_futures_settings_keyboard()
        )
        return SPOT_FUTURES_SETTINGS

    elif text == "🏛 Биржи":
        await update.message.reply_text(
            "🏛 <b>Настройки бирж</b>\n\nВыберите биржу для включения/выключения:",
            parse_mode="HTML",
            reply_markup=get_exchange_settings_keyboard()
        )
        return EXCHANGE_SETTINGS_MENU

    elif text == "🔄 Сброс":
        global SETTINGS, LAST_EXCHANGE_SETTINGS
        SETTINGS = {
            "SPOT": DEFAULT_SPOT_SETTINGS.copy(),
            "FUTURES": DEFAULT_FUTURES_SETTINGS.copy(),
            "SPOT_FUTURES": DEFAULT_SPOT_FUTURES_SETTINGS.copy(),
            "EXCHANGES": EXCHANGE_SETTINGS.copy()
        }
        save_settings(SETTINGS)
        LAST_EXCHANGE_SETTINGS = None  # Сбрасываем, чтобы принудительно перезагрузить биржи
        await update.message.reply_text(
            "✅ Настройки сброшены к значениям по умолчанию",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    elif text == "🔙 Главное меню":
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Неизвестная команда. Используйте кнопки меню.",
        reply_markup=get_settings_keyboard()
    )
    return SETTINGS_MENU

async def handle_spot_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка настроек спота"""
    text = update.message.text

    if text == "🔙 Назад в настройки":
        await update.message.reply_text(
            "⚙️ <b>Настройки бота</b>\n\nВыберите категорию:",
            parse_mode="HTML",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    # Обработка изменения параметров
    if text.startswith("Порог:"):
        context.user_data['setting'] = ('SPOT', 'THRESHOLD_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для порога арбитража (текущее: {SETTINGS['SPOT']['THRESHOLD_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. порог:"):
        context.user_data['setting'] = ('SPOT', 'MAX_THRESHOLD_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимального порога (текущее: {SETTINGS['SPOT']['MAX_THRESHOLD_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Интервал:"):
        context.user_data['setting'] = ('SPOT', 'CHECK_INTERVAL')
        await update.message.reply_text(
            f"Введите новое значение для интервала проверки (текущее: {SETTINGS['SPOT']['CHECK_INTERVAL']} сек):"
        )
        return SETTING_VALUE

    elif text.startswith("Объем:"):
        context.user_data['setting'] = ('SPOT', 'MIN_VOLUME_USD')
        await update.message.reply_text(
            f"Введите новое значение для минимального объема (текущее: ${SETTINGS['SPOT']['MIN_VOLUME_USD']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. сумма:"):
        context.user_data['setting'] = ('SPOT', 'MIN_ENTRY_AMOUNT_USDT')
        await update.message.reply_text(
            f"Введите новое значение для минимальной суммы входа (текущее: ${SETTINGS['SPOT']['MIN_ENTRY_AMOUNT_USDT']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. сумма:"):
        context.user_data['setting'] = ('SPOT', 'MAX_ENTRY_AMOUNT_USDT')
        await update.message.reply_text(
            f"Введите новое значение для максимальной суммы входа (текущее: ${SETTINGS['SPOT']['MAX_ENTRY_AMOUNT_USDT']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Влияние:"):
        context.user_data['setting'] = ('SPOT', 'MAX_IMPACT_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимального влияния (текущее: {SETTINGS['SPOT']['MAX_IMPACT_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Стакан:"):
        context.user_data['setting'] = ('SPOT', 'ORDER_BOOK_DEPTH')
        await update.message.reply_text(
            f"Введите новое значение для глубины стакана (текущее: {SETTINGS['SPOT']['ORDER_BOOK_DEPTH']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Прибыль:"):
        context.user_data['setting'] = ('SPOT', 'MIN_NET_PROFIT_USD')
        await update.message.reply_text(
            f"Введите новое значение для минимальной прибыли (текущее: ${SETTINGS['SPOT']['MIN_NET_PROFIT_USD']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Сходимость:"):
        context.user_data['setting'] = ('SPOT', 'PRICE_CONVERGENCE_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога сходимости цен (текущее: {SETTINGS['SPOT']['PRICE_CONVERGENCE_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Увед. сравн.:"):
        SETTINGS['SPOT']['PRICE_CONVERGENCE_ENABLED'] = not SETTINGS['SPOT']['PRICE_CONVERGENCE_ENABLED']
        save_settings(SETTINGS)
        status = "🔔 ВКЛ" if SETTINGS['SPOT']['PRICE_CONVERGENCE_ENABLED'] else "🔕 ВЫКЛ"
        await update.message.reply_text(
            f"✅ Уведомления о сравнении цен {status}",
            reply_markup=get_spot_settings_keyboard()
        )
        return SPOT_SETTINGS

    elif text.startswith("Волатильность:"):
        context.user_data['setting'] = ('SPOT', 'VOLATILITY_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога волатильности (текущее: {SETTINGS['SPOT']['VOLATILITY_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. объем стакана:"):
        context.user_data['setting'] = ('SPOT', 'MIN_ORDER_BOOK_VOLUME')
        await update.message.reply_text(
            f"Введите новое значение для минимального объема стакана (текущее: ${SETTINGS['SPOT']['MIN_ORDER_BOOK_VOLUME']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. волатильность:"):
        context.user_data['setting'] = ('SPOT', 'MAX_VOLATILITY_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимальной волатильности монеты (текущее: {SETTINGS['SPOT']['MAX_VOLATILITY_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Статус:"):
        SETTINGS['SPOT']['ENABLED'] = not SETTINGS['SPOT']['ENABLED']
        save_settings(SETTINGS)
        status = "ВКЛ" if SETTINGS['SPOT']['ENABLED'] else "ВЫКЛ"
        await update.message.reply_text(
            f"✅ Спотовый арбитраж {status}",
            reply_markup=get_spot_settings_keyboard()
        )
        return SPOT_SETTINGS

    await update.message.reply_text(
        "Неизвестная команда. Используйте кнопки меню.",
        reply_markup=get_spot_settings_keyboard()
    )
    return SPOT_SETTINGS

async def handle_futures_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка настроек фьючерсов"""
    text = update.message.text

    if text == "🔙 Назад в настройки":
        await update.message.reply_text(
            "⚙️ <b>Настройки бота</b>\n\nВыберите категориу:",
            parse_mode="HTML",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    # Обработка изменения параметров
    if text.startswith("Порог:"):
        context.user_data['setting'] = ('FUTURES', 'THRESHOLD_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для порога арбитража (текущее: {SETTINGS['FUTURES']['THRESHOLD_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. порог:"):
        context.user_data['setting'] = ('FUTURES', 'MAX_THRESHOLD_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимального порога (текущее: {SETTINGS['FUTURES']['MAX_THRESHOLD_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Интервал:"):
        context.user_data['setting'] = ('FUTURES', 'CHECK_INTERVAL')
        await update.message.reply_text(
            f"Введите новое значение для интервала проверки (текущее: {SETTINGS['FUTURES']['CHECK_INTERVAL']} сек):"
        )
        return SETTING_VALUE

    elif text.startswith("Объем:"):
        context.user_data['setting'] = ('FUTURES', 'MIN_VOLUME_USD')
        await update.message.reply_text(
            f"Введите новое значение для минимального объема (текущее: ${SETTINGS['FUTURES']['MIN_VOLUME_USD']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. сумма:"):
        context.user_data['setting'] = ('FUTURES', 'MIN_ENTRY_AMOUNT_USDT')
        await update.message.reply_text(
            f"Введите новое значение для минимальной суммы входа (текущее: ${SETTINGS['FUTURES']['MIN_ENTRY_AMOUNT_USDT']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. сумма:"):
        context.user_data['setting'] = ('FUTURES', 'MAX_ENTRY_AMOUNT_USDT')
        await update.message.reply_text(
            f"Введите новое значение для максимальной суммы входа (текущее: ${SETTINGS['FUTURES']['MAX_ENTRY_AMOUNT_USDT']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Прибыль:"):
        context.user_data['setting'] = ('FUTURES', 'MIN_NET_PROFIT_USD')
        await update.message.reply_text(
            f"Введите новое значение для минимальной прибыли (текущее: ${SETTINGS['FUTURES']['MIN_NET_PROFIT_USD']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Сходимость:"):
        context.user_data['setting'] = ('FUTURES', 'PRICE_CONVERGENCE_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога сходимости цен (текущее: {SETTINGS['FUTURES']['PRICE_CONVERGENCE_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Увед. сравн.:"):
        SETTINGS['FUTURES']['PRICE_CONVERGENCE_ENABLED'] = not SETTINGS['FUTURES']['PRICE_CONVERGENCE_ENABLED']
        save_settings(SETTINGS)
        status = "🔔 ВКЛ" if SETTINGS['FUTURES']['PRICE_CONVERGENCE_ENABLED'] else "🔕 ВЫКЛ"
        await update.message.reply_text(
            f"✅ Уведомления о сравнении цен {status}",
            reply_markup=get_futures_settings_keyboard()
        )
        return FUTURES_SETTINGS

    elif text.startswith("Волатильность:"):
        context.user_data['setting'] = ('FUTURES', 'VOLATILITY_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога волатильности (текущее: {SETTINGS['FUTURES']['VOLATILITY_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. объем стакана:"):
        context.user_data['setting'] = ('FUTURES', 'MIN_ORDER_BOOK_VOLUME')
        await update.message.reply_text(
            f"Введите новое значение для минимального объема стакана (текущее: ${SETTINGS['FUTURES']['MIN_ORDER_BOOK_VOLUME']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. фандинг:"):
        context.user_data['setting'] = ('FUTURES', 'FUNDING_RATE_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для максимальной ставки финансирования (текущее: {SETTINGS['FUTURES']['FUNDING_RATE_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. фандинг:"):
        context.user_data['setting'] = ('FUTURES', 'MIN_FUNDING_RATE_TO_RECEIVE')
        await update.message.reply_text(
            f"Введите новое значение для минимальной ставки финансирования (текущее: {SETTINGS['FUTURES']['MIN_FUNDING_RATE_TO_RECEIVE']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Красный фандинг:"):
        context.user_data['setting'] = ('FUTURES', 'RED_FUNDING_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога красного фандинга (текущее: {SETTINGS['FUTURES']['RED_FUNDING_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. волатильность:"):
        context.user_data['setting'] = ('FUTURES', 'MAX_VOLATILITY_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимальной волатильности монеты (текущее: {SETTINGS['FUTURES']['MAX_VOLATILITY_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Статус:"):
        SETTINGS['FUTURES']['ENABLED'] = not SETTINGS['FUTURES']['ENABLED']
        save_settings(SETTINGS)
        status = "ВКЛ" if SETTINGS['FUTURES']['ENABLED'] else "ВЫКЛ"
        await update.message.reply_text(
            f"✅ Фьючерсный арбитраж {status}",
            reply_markup=get_futures_settings_keyboard()
        )
        return FUTURES_SETTINGS

    await update.message.reply_text(
        "Неизвестная команда. Используйте кнопки меню.",
        reply_markup=get_futures_settings_keyboard()
    )
    return FUTURES_SETTINGS

async def handle_spot_futures_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка настроек спот-фьючерсного арбитража"""
    text = update.message.text

    if text == "🔙 Назад в настройки":
        await update.message.reply_text(
            "⚙️ <b>Настройки бота</b>\n\nВыберите категорию:",
            parse_mode="HTML",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    # Обработка изменения параметров
    if text.startswith("Порог:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'THRESHOLD_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для порога арбитража (текущее: {SETTINGS['SPOT_FUTURES']['THRESHOLD_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. порог:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MAX_THRESHOLD_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимального порога (текущее: {SETTINGS['SPOT_FUTURES']['MAX_THRESHOLD_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Интервал:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'CHECK_INTERVAL')
        await update.message.reply_text(
            f"Введите новое значение для интервала проверки (текущее: {SETTINGS['SPOT_FUTURES']['CHECK_INTERVAL']} сек):"
        )
        return SETTING_VALUE

    elif text.startswith("Объем:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MIN_VOLUME_USD')
        await update.message.reply_text(
            f"Введите новое значение для минимального объема (текущее: ${SETTINGS['SPOT_FUTURES']['MIN_VOLUME_USD']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. сумма:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MIN_ENTRY_AMOUNT_USDT')
        await update.message.reply_text(
            f"Введите новое значение для минимальной суммы входа (текущее: ${SETTINGS['SPOT_FUTURES']['MIN_ENTRY_AMOUNT_USDT']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. сумма:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MAX_ENTRY_AMOUNT_USDT')
        await update.message.reply_text(
            f"Введите новое значение для максимальной суммы входа (текущее: ${SETTINGS['SPOT_FUTURES']['MAX_ENTRY_AMOUNT_USDT']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Прибыль:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MIN_NET_PROFIT_USD')
        await update.message.reply_text(
            f"Введите новое значение для минимальной прибыли (текущее: ${SETTINGS['SPOT_FUTURES']['MIN_NET_PROFIT_USD']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Сходимость:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'PRICE_CONVERGENCE_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога сходимости цен (текущее: {SETTINGS['SPOT_FUTURES']['PRICE_CONVERGENCE_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Увед. сравн.:"):
        SETTINGS['SPOT_FUTURES']['PRICE_CONVERGENCE_ENABLED'] = not SETTINGS['SPOT_FUTURES'][
            'PRICE_CONVERGENCE_ENABLED']
        save_settings(SETTINGS)
        status = "🔔 ВКЛ" if SETTINGS['SPOT_FUTURES']['PRICE_CONVERGENCE_ENABLED'] else "🔕 ВЫКЛ"
        await update.message.reply_text(
            f"✅ Уведомления о сравнении цен {status}",
            reply_markup=get_spot_futures_settings_keyboard()
        )
        return SPOT_FUTURES_SETTINGS

    elif text.startswith("Волатильность:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'VOLATILITY_THRESHOLD')
        await update.message.reply_text(
            f"Введите новое значение для порога волатильности (текущее: {SETTINGS['SPOT_FUTURES']['VOLATILITY_THRESHOLD']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Мин. объем стакана:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MIN_ORDER_BOOK_VOLUME')
        await update.message.reply_text(
            f"Введите новое значение для минимального объема стакана (текущее: ${SETTINGS['SPOT_FUTURES']['MIN_ORDER_BOOK_VOLUME']}):"
        )
        return SETTING_VALUE

    elif text.startswith("Макс. волатильность:"):
        context.user_data['setting'] = ('SPOT_FUTURES', 'MAX_VOLATILITY_PERCENT')
        await update.message.reply_text(
            f"Введите новое значение для максимальной волатильности монеты (текущее: {SETTINGS['SPOT_FUTURES']['MAX_VOLATILITY_PERCENT']}%):"
        )
        return SETTING_VALUE

    elif text.startswith("Статус:"):
        SETTINGS['SPOT_FUTURES']['ENABLED'] = not SETTINGS['SPOT_FUTURES']['ENABLED']
        save_settings(SETTINGS)
        status = "ВКЛ" if SETTINGS['SPOT_FUTURES']['ENABLED'] else "ВЫКЛ"
        await update.message.reply_text(
            f"✅ Спот-фьючерсный арбитраж {status}",
            reply_markup=get_spot_futures_settings_keyboard()
        )
        return SPOT_FUTURES_SETTINGS

    await update.message.reply_text(
        "Неизвестная команда. Используйте кнопки меню.",
        reply_markup=get_spot_futures_settings_keyboard()
    )
    return SPOT_FUTURES_SETTINGS

async def handle_exchange_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка настроек бирж"""
    text = update.message.text

    if text == "🔙 Назад в настройки":
        await update.message.reply_text(
            "⚙️ <b>Настройки бота</b>\n\nВыберите категорию:",
            parse_mode="HTML",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    # Обработка включения/выключения бирж
    for exchange in SETTINGS['EXCHANGES'].keys():
        if text.startswith(f"{exchange}:"):
            SETTINGS['EXCHANGES'][exchange]['ENABLED'] = not SETTINGS['EXCHANGES'][exchange]['ENABLED']
            save_settings(SETTINGS)

            status = "✅ ВКЛ" if SETTINGS['EXCHANGES'][exchange]['ENABLED'] else "❌ ВЫКЛ"
            await update.message.reply_text(
                f"✅ Биржа {exchange.upper()} {status}",
                reply_markup=get_exchange_settings_keyboard()
            )
            return EXCHANGE_SETTINGS_MENU

    await update.message.reply_text(
        "Неизвестная команда. Используйте кнопки меню.",
        reply_markup=get_exchange_settings_keyboard()
    )
    return EXCHANGE_SETTINGS_MENU

async def handle_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ввода значения настройки"""
    text = update.message.text
    setting_info = context.user_data.get('setting')

    if not setting_info:
        await update.message.reply_text(
            "Ошибка: не удалось определить настройку. Попробуйте снова.",
            reply_markup=get_settings_keyboard()
        )
        return SETTINGS_MENU

    arb_type, setting_key = setting_info

    try:
        # Обработка числовых значений
        if setting_key in ['THRESHOLD_PERCENT', 'MAX_THRESHOLD_PERCENT', 'MAX_IMPACT_PERCENT',
                           'PRICE_CONVERGENCE_THRESHOLD', 'VOLATILITY_THRESHOLD',
                           'FUNDING_RATE_THRESHOLD', 'MIN_FUNDING_RATE_TO_RECEIVE', 'IDEAL_FUNDING_SCENARIO',
                           'RED_FUNDING_THRESHOLD', 'MAX_VOLATILITY_PERCENT']:
            value = float(text)
        elif setting_key in ['CHECK_INTERVAL', 'ORDER_BOOK_DEPTH', 'FUNDING_CHECK_INTERVAL', 'MAX_HOLDING_HOURS']:
            value = int(text)
        elif setting_key in ['MIN_VOLUME_USD', 'MIN_ENTRY_AMOUNT_USDT', 'MAX_ENTRY_AMOUNT_USDT', 'MIN_NET_PROFIT_USD',
                             'MIN_ORDER_BOOK_VOLUME']:
            value = float(text)
        else:
            value = text

        # Устанавливаем новое значение
        SETTINGS[arb_type][setting_key] = value
        save_settings(SETTINGS)

        await update.message.reply_text(
            f"✅ Настройка {setting_key} изменена на {text}",
            reply_markup=get_spot_settings_keyboard() if arb_type == 'SPOT' else
            get_futures_settings_keyboard() if arb_type == 'FUTURES' else
            get_spot_futures_settings_keyboard()
        )

        return SPOT_SETTINGS if arb_type == 'SPOT' else \
            FUTURES_SETTINGS if arb_type == 'FUTURES' else \
                SPOT_FUTURES_SETTINGS

    except ValueError:
        await update.message.reply_text(
            "❌ Неверный формат. Введите число.",
            reply_markup=get_spot_settings_keyboard() if arb_type == 'SPOT' else
            get_futures_settings_keyboard() if arb_type == 'FUTURES' else
            get_spot_futures_settings_keyboard()
        )
        return SETTING_VALUE

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена диалога"""
    await update.message.reply_text(
        "Операция отменена.",
        reply_markup=get_main_keyboard()
    )
    return ConversationHandler.END

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}", exc_info=context.error)

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла ошибка. Попробуйте позже.",
            reply_markup=get_main_keyboard()
        )

def main():
    """Основная функция запуска бота"""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Conversation handler для настроек
    conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        ],
        states={
            SETTINGS_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_settings)
            ],
            SPOT_SETTINGS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_spot_settings)
            ],
            FUTURES_SETTINGS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_futures_settings)
            ],
            SPOT_FUTURES_SETTINGS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_spot_futures_settings)
            ],
            EXCHANGE_SETTINGS_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_exchange_settings)
            ],
            SETTING_VALUE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_value)
            ],
            COIN_SELECTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_coin_selection)
            ],
            ARBITRAGE_LIST: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_arbitrage_list)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)

    # Запускаем арбитражные задачи в фоне
    loop = asyncio.get_event_loop()
    loop.create_task(check_spot_arbitrage())
    loop.create_task(check_futures_arbitrage())
    loop.create_task(check_spot_futures_arbitrage())

    logger.info("Бот запущен")

    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main()
