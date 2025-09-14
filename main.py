import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import logging
import asyncio
import talib
import warnings
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest
import html
from collections import defaultdict
import pytz
from scipy import stats
import math
import requests  # Добавим для получения дополнительных данных

warnings.filterwarnings('ignore')

# Настройка часового пояса Москвы (UTC+3)
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# Настройка логирования только в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('MT5Bot')

# Конфигурация Telegram бота
TELEGRAM_BOT_TOKEN = "7952768185:AAGuhybXaGPJqtlGPd1-O4nc6_FpUL2rOgw"
TELEGRAM_CHAT_IDS = ["1167694150", "7916502470"]

# Попытка импорта MetaTrader5
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 не установлен. Будет использован демо-режим.")
    MT5_AVAILABLE = False


class MT5TradingBot:
    def __init__(self):
        self.telegram_app = None
        self.last_analysis_start_time = None
        self.last_signals = []
        self.is_analyzing = False
        self.analysis_lock = asyncio.Lock()
        self.telegram_queue = asyncio.Queue()
        self.telegram_worker_task = None
        self.signal_history = defaultdict(list)
        self.symbol_24h_volume = {}
        self.demo_mode = not MT5_AVAILABLE
        self.last_prediction = None
        self.prediction_history = []
        self.market_state = {
            'trend': 'neutral',
            'volatility': 'medium',
            'momentum': 'neutral',
            'last_update': None
        }
        self.account_balance = 10.0  # Депозит $10
        self.positions = []
        self.liquidation_levels = defaultdict(list)
        self.whale_activity = defaultdict(list)
        self.long_short_ratio = defaultdict(dict)

        # Специализированные настройки для скальпинга
        self.config = {
            'timeframes': {
                'M15': 15,
                'M5': 5,
                'M1': 1
            },
            'min_volume_24h': 50000,
            'max_symbols': 5,  # Увеличили для включения GOLD
            'analysis_interval': 10,
            'risk_per_trade': 0.02,  # 2% от депозита ($0.20)
            'virtual_balance': self.account_balance,
            'min_confidence': 0.65,
            'risk_reward_ratio': 1.5,
            'atr_multiplier_sl': 1.2,
            'atr_multiplier_tp': 1.8,
            'scalping_symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'GOLD'],  # Добавили GOLD
            'signal_validity_seconds': 120,
            'required_indicators': 2,
            'min_price_change': 0.0005,
            'volume_spike_threshold': 2.0,
            'trend_strength_threshold': 0.2,
            'correction_filter': True,
            'multi_timeframe_confirmation': True,
            'market_trend_filter': True,
            'volume_confirmation': True,
            'volatility_filter': True,
            'price_action_filter': True,
            'required_timeframes': 2,
            'pin_bar_threshold': 0.6,
            'trend_confirmation': True,
            'lot_size': 0.01,  # Уменьшили лот до 0.01
            'max_spread_pips': 3.0,
            'demo_mode': self.demo_mode,
            'prediction_history_size': 50,
            'pattern_weight': 0.7,
            'indicator_weight': 0.3,
            'market_phases': {
                'accumulation': 0.2,
                'bullish': 0.8,
                'distribution': -0.2,
                'bearish': -0.8
            },
            'liquidity_filter': True,
            'volatility_min': 0.0003,
            'volatility_max': 0.0015,
            'liquidation_threshold': 0.7,  # Порог для определения ликвидаций
            'whale_volume_threshold': 5.0, # Порог объема для определения китов
        }

        self.top_symbols = []
        self.signals = []
        self.analysis_stats = {'total_analyzed': 0, 'signals_found': 0}

        if self.demo_mode:
            logger.info("Торговый бот инициализирован в ДЕМО-РЕЖИМЕ (MT5 недоступен)")
        else:
            logger.info("Торговый бот для MT5 инициализирован")

    def get_moscow_time(self, dt=None):
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(MOSCOW_TZ)

    def format_moscow_time(self, dt=None, format_str='%Y-%m-%d %H:%M:%S'):
        moscow_time = self.get_moscow_time(dt)
        return moscow_time.strftime(format_str)

    def initialize_mt5(self):
        """Инициализация подключения к MT5"""
        if not MT5_AVAILABLE:
            logger.warning("MT5 недоступен, работа в демо-режиме")
            return False

        try:
            if not mt5.initialize():
                logger.error(f"Ошибка инициализации MT5: {mt5.last_error()}")
                return False

            account = 513707711
            password = "!DXmj0CgYS"
            server = "FxPro-MT5"

            authorized = mt5.login(account, password=password, server=server)
            if not authorized:
                logger.error(f"Ошибка авторизации на MT5: {mt5.last_error()}")
                return False

            logger.info(f"Успешное подключение к MT5, счет: {account}")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации MT5: {e}")
            return False

    def get_symbols_list(self):
        """Получение списка символов - основные валютные пары для скальпинга"""
        if not MT5_AVAILABLE or self.demo_mode:
            demo_symbols = self.config['scalping_symbols']
            logger.info(f"Используется демо-список: {demo_symbols}")
            return demo_symbols

        try:
            symbols = mt5.symbols_get()
            symbol_names = [s.name for s in symbols]

            scalping_symbols = []
            for symbol in self.config['scalping_symbols']:
                if symbol in symbol_names:
                    scalping_symbols.append(symbol)

            if not scalping_symbols:
                logger.warning("Не найдены символы для скальпинга, используем демо-символы")
                return self.config['scalping_symbols']

            logger.info(f"Найдены символы для скальпинга: {scalping_symbols}")
            return scalping_symbols[:self.config['max_symbols']]
        except Exception as e:
            logger.error(f"Ошибка получения символов из MT5: {e}")
            return self.config['scalping_symbols']

    async def initialize_telegram(self):
        try:
            request = HTTPXRequest(
                connection_pool_size=10,
                connect_timeout=30.0,
                read_timeout=30.0,
                write_timeout=30.0
            )

            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
            self.telegram_app.add_handler(CommandHandler("prediction", self.telegram_prediction))
            self.telegram_app.add_handler(CommandHandler("signal", self.telegram_signal))
            self.telegram_app.add_handler(CommandHandler("balance", self.telegram_balance))
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.upstream.start_polling()
            self.telegram_worker_task = asyncio.create_task(self.telegram_worker())
            logger.info("Telegram бот инициализирован")

            mode = "ДЕМО-РЕЖИМЕ (MT5 недоступен)" if self.demo_mode else "режиме реального времени"
            startup_message = (
                f"🤖 <b>СКАЛЬПИНГ БОТ ДЛЯ MT5 ЗАПУЩЕН!</b>\n\n"
                f"📊 Бот начал анализ валютных пар в {mode}\n"
                f"⏰ Время запуска: {self.format_moscow_time()}\n"
                f"💰 Начальный депозит: ${self.account_balance}\n"
                f"📈 Размер лота: {self.config['lot_size']}\n"
                "🌍 Часовой пояс: Москва (UTC+3)\n"
                "📈 Брокер: FXPRO\n\n"
                "⚡ <b>СПЕЦИАЛИЗИРОВАННАЯ ВЕРСИЯ ДЛЯ СКАЛЬПИНГА</b>"
            )
            await self.send_telegram_message(startup_message)

        except Exception as e:
            logger.error(f"Ошибка инициализации Telegram бота: {e}")

    async def telegram_worker(self):
        logger.info("Telegram worker запущен")
        while True:
            try:
                chat_id, message = await self.telegram_queue.get()
                logger.info(f"Получено сообщение для отправки в Telegram (chat_id: {chat_id})")
                if chat_id and message:
                    try:
                        for attempt in range(3):
                            try:
                                await self.telegram_app.bot.send_message(
                                    chat_id=chat_id,
                                    text=message,
                                    parse_mode='HTML',
                                    disable_web_page_preview=True,
                                    read_timeout=30,
                                    write_timeout=30,
                                    connect_timeout=30
                                )
                                logger.info(f"Сообщение успешно отправлено в Telegram (chat_id: {chat_id})")
                                break
                            except Exception as e:
                                if attempt < 2:
                                    logger.warning(f"Попытка {attempt + 1} не удалась, повтор через 5 секунд: {e}")
                                    await asyncio.sleep(5)
                                else:
                                    logger.error(f"Не удалось отправить сообщение в Telegram после 3 попыток: {e}")
                    except Exception as e:
                        logger.error(f"Ошибка при отправке сообщения в Telegram: {e}")
                self.telegram_queue.task_done()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в telegram_worker: {e}")
                await asyncio.sleep(1)

    async def send_telegram_message(self, message: str, chat_ids: list = None):
        if chat_ids is None:
            chat_ids = TELEGRAM_CHAT_IDS

        if len(message) > 4096:
            parts = [message[i:i + 4090] for i in range(0, len(message), 4090)]
            for part in parts:
                await self.telegram_queue.put((chat_ids[0], part))
                await asyncio.sleep(0.5)
        else:
            for chat_id in chat_ids:
                await self.telegram_queue.put((chat_id, message))
                logger.info(f"Сообщение добавлено в очередь для chat_id: {chat_id}")

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current_time = self.format_moscow_time()
        mode = "ДЕМО-РЕЖИМЕ (MT5 недоступен)" if self.demo_mode else "режиме реального времени"

        welcome_text = (
            f"🚀 <b>СКАЛЬПИНГ БОТ ДЛЯ ФОРЕКС</b>\n\n"
            f"📊 <b>Режим работы:</b> {mode}\n"
            f"💰 <b>Депозит:</b> ${self.account_balance}\n"
            f"📊 <b>Размер лота:</b> {self.config['lot_size']}\n"
            f"📊 <b>Брокер:</b> FXPRO\n\n"
            "⚡ <b>Мультитаймфрейм анализ</b> (M15, M5, M1)\n"
            "📈 <b>Технические индикаторы:</b> RSI, MACD, Bollinger Bands, EMA, ATR, Stochastic\n\n"
            "🔮 <b>СПЕЦИАЛИЗИРОВАННАЯ ВЕРСИЯ ДЛЯ СКАЛЬПИНГА</b>\n\n"
            "⚙️ <b>Настройки:</b>\n"
            f"• Мин. уверенность: {self.config['min_confidence'] * 100}%\n"
            f"• Риск/вознаграждение: 1:{self.config['risk_reward_ratio']}\n"
            f"• Интервал анализа: {self.config['analysis_interval']} сек\n"
            f"• Часовой пояс: Москва (UTC+3)\n"
            f"• Размер лота: {self.config['lot_size']}\n"
            f"• Макс. спред: {self.config['max_spread_pips']} пипсов\n\n"
            f"🕐 <b>Текущее время:</b> {current_time}\n\n"
            "📊 Сигналы отправляются автоматически после каждого анализа\n"
            "📈 Прогноз направления цены отображается непрерывно\n\n"
            "🔍 <b>Команды:</b>\n"
            "/start - информация о боте\n"
            "/prediction - текущий прогноз цены\n"
            "/signal - последний торговый сигнал\n"
            "/balance - информация о балансе"
        )
        await self.send_telegram_message(welcome_text, [update.effective_chat.id])

    async def telegram_prediction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /prediction"""
        if not self.last_prediction:
            await self.send_telegram_message("📊 Прогноз еще не сформирован. Подождите немного.",
                                             [update.effective_chat.id])
            return

        prediction_text = self.format_prediction_message()
        await self.send_telegram_message(prediction_text, [update.effective_chat.id])

    async def telegram_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /signal"""
        if not self.signals:
            await self.send_telegram_message("📊 Сигналы еще не сформированы. Подождите немного.",
                                             [update.effective_chat.id])
            return

        signal_text = self.format_signal_message()
        await self.send_telegram_message(signal_text, [update.effective_chat.id])

    async def telegram_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /balance"""
        balance_text = (
            f"💰 <b>ИНФОРМАЦИЯ О БАЛАНСЕ</b>\n\n"
            f"📊 <b>Текущий баланс:</b> ${self.account_balance:.2f}\n"
            f"📈 <b>Размер лота:</b> {self.config['lot_size']}\n"
            f"⚖️ <b>Риск на сделку:</b> {self.config['risk_per_trade'] * 100}%\n"
            f"🔢 <b>Открыто позиций:</b> {len(self.positions)}\n\n"
            f"📊 <b>Последние сделки:</b>\n"
        )

        if self.positions:
            for i, pos in enumerate(self.positions[-5:]):  # Последние 5 сделок
                profit_color = "🟢" if pos['profit'] >= 0 else "🔴"
                balance_text += f"{profit_color} {pos['symbol']} {pos['type']} | P&L: ${pos['profit']:.2f}\n"
        else:
            balance_text += "Нет открытых позиций\n"

        balance_text += f"\n⏰ <b>Обновлено:</b> {self.format_moscow_time()}"

        await self.send_telegram_message(balance_text, [update.effective_chat.id])

    def format_prediction_message(self):
        """Форматирование сообщения с прогнозом"""
        if not self.last_prediction:
            return "📊 Прогноз еще не сформирован"

        symbol = self.last_prediction['symbol']
        direction = self.last_prediction['direction']
        confidence = self.last_prediction['confidence'] * 100
        price = self.last_prediction['price']
        target = self.last_prediction['target_price']
        timeframe = self.last_prediction['timeframe']
        timestamp = self.format_moscow_time(self.last_prediction['timestamp'])

        emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "🟡"
        trend_emoji = "📈" if direction == "BUY" else "📉" if direction == "SELL" else "↔️"

        message = (
            f"{emoji} <b>ПРОГНОЗ ДЛЯ {symbol}</b> {trend_emoji}\n\n"
            f"<b>Направление:</b> <code>{direction}</code>\n"
            f"<b>Уверенность:</b> <code>{confidence:.1f}%</code>\n"
            f"<b>Текущая цена:</b> <code>{price:.5f}</code>\n"
            f"<b>Целевая цена:</b> <code>{target:.5f}</code>\n"
            f"<b>Таймфрейм:</b> <code>{timeframe}</code>\n"
            f"<b>Время анализа:</b> <code>{timestamp}</code>\n\n"
        )

        # Добавляем историю прогнозов
        if self.prediction_history:
            message += "<b>📊 ИСТОРИЯ ПРОГНОЗОВ:</b>\n"
            for i, pred in enumerate(self.prediction_history[-5:]):  # Последние 5 прогнозов
                pred_direction = pred['direction']
                pred_confidence = pred['confidence'] * 100
                pred_emoji = "🟢" if pred_direction == "BUY" else "🔴" if pred_direction == "SELL" else "🟡"
                pred_time = self.format_moscow_time(pred['timestamp'], '%H:%M:%S')
                message += f"{pred_emoji} {pred_time}: {pred_direction} ({pred_confidence:.1f}%)\n"

        message += f"\n<b>⚡ Следующий анализ через {self.config['analysis_interval']} сек</b>"

        return message

    def format_signal_message(self):
        """Форматирование сообщения с сигналом"""
        if not self.signals:
            return "📊 Сигналы еще не сформированы"

        signal = self.signals[0]  # Берем самый сильный сигнал
        symbol = signal['symbol']
        signal_type = signal['signal']
        confidence = signal['confidence'] * 100
        price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        reasons = signal.get('detailed_reasons', ['Нет подробной информации'])

        # Расчет потенциальной прибыли
        price_diff = abs(take_profit - price)
        potential_profit = (price_diff / price) * 100

        emoji = "🟢" if signal_type == 'BUY' else "🔴"
        trend_emoji = "📈" if signal_type == 'BUY' else "📉"

        # Добавляем информацию о ликвидациях и китах
        extra_info = ""
        if symbol in self.liquidation_levels:
            long_liq = len([l for l in self.liquidation_levels[symbol] if l['type'] == 'long'])
            short_liq = len([l for l in self.liquidation_levels[symbol] if l['type'] == 'short'])
            extra_info += f"📊 Ликвидации: LONG {long_liq} | SHORT {short_liq}\n"

        if symbol in self.whale_activity:
            whale_buy = len([w for w in self.whale_activity[symbol] if w['type'] == 'buy'])
            whale_sell = len([w for w in self.whale_activity[symbol] if w['type'] == 'sell'])
            extra_info += f"🐋 Активность китов: BUY {whale_buy} | SELL {whale_sell}\n"

        if symbol in self.long_short_ratio:
            ratio = self.long_short_ratio[symbol]
            extra_info += f"⚖️ Соотношение LONG/SHORT: {ratio.get('long', 0):.2f}/{ratio.get('short', 0):.2f}\n"

        message = (
            f"{emoji} <b>СКАЛЬПИНГ СИГНАЛ ДЛЯ {symbol}</b> {trend_emoji}\n\n"
            f"<b>Тип сигнала:</b> <code>{signal_type}</code>\n"
            f"<b>Уверенность:</b> <code>{confidence:.1f}%</code>\n"
            f"<b>Текущая цена:</b> <code>{price:.5f}</code>\n"
            f"<b>Стоп-лосс:</b> <code>{stop_loss:.5f}</code>\n"
            f"<b>Тейк-профит:</b> <code>{take_profit:.5f}</code>\n"
            f"<b>Потенциальная прибыль:</b> <code>{potential_profit:.2f}%</code>\n\n"
            f"{extra_info}\n"
            f"<b>📊 ПОДРОБНЫЙ АНАЛИЗ:</b>\n"
        )

        for i, reason in enumerate(reasons[:5]):  # Ограничиваем 5 причинами
            message += f"• {reason}\n"

        message += f"\n<b>⚡ Рекомендуемый размер позиции: {self.config['lot_size']} лота</b>"

        return message

    def update_signal_history(self, symbol, signal_type, confidence):
        now = self.get_moscow_time()
        self.signal_history[symbol] = [sig for sig in self.signal_history[symbol]
                                       if now - sig['time'] < timedelta(hours=24)]
        self.signal_history[symbol].append({
            'time': now,
            'signal': signal_type,
            'confidence': confidence
        })

    def get_signal_count_last_24h(self, symbol):
        if symbol not in self.signal_history:
            return 0
        now = self.get_moscow_time()
        recent_signals = [sig for sig in self.signal_history[symbol]
                          if now - sig['time'] < timedelta(hours=24)]
        return len(recent_signals)

    async def send_automatic_signals(self):
        if not self.signals:
            logger.info("Нет сигналов для отправки в Telegram")
            return

        try:
            analysis_time_str = self.format_moscow_time(self.last_analysis_start_time)
            mode = "ДЕМО-РЕЖИМ" if self.demo_mode else "РЕЖИМ РЕАЛЬНОГО ВРЕМЕНИ"

            message = f"🚀 <b>НОВЫЕ СКАЛЬПИНГ СИГНАЛЫ ({mode})</b>\n\n"
            message += "<i>Сигналы основаны на углубленном техническом анализе</i>\n\n"

            for i, signal in enumerate(self.signals[:3]):  # Ограничиваем 3 сигналами
                symbol_name = signal['symbol']
                confidence_percent = signal['confidence'] * 100
                signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴"
                signal_count = self.get_signal_count_last_24h(signal['symbol'])

                # Расчет потенциальной прибыли
                price_diff = abs(signal['take_profit'] - signal['price'])
                potential_profit = (price_diff / signal['price']) * 100

                # Добавляем информацию о ликвидациях и китах
                extra_info = ""
                if symbol_name in self.liquidation_levels:
                    long_liq = len([l for l in self.liquidation_levels[symbol_name] if l['type'] == 'long'])
                    short_liq = len([l for l in self.liquidation_levels[symbol_name] if l['type'] == 'short'])
                    extra_info += f"📊 Ликвидации: LONG {long_liq} | SHORT {short_liq}\n"

                if symbol_name in self.whale_activity:
                    whale_buy = len([w for w in self.whale_activity[symbol_name] if w['type'] == 'buy'])
                    whale_sell = len([w for w in self.whale_activity[symbol_name] if w['type'] == 'sell'])
                    extra_info += f"🐋 Активность китов: BUY {whale_buy} | SELL {whale_sell}\n"

                message += (
                    f"{signal_emoji} <b>{html.escape(symbol_name)}</b>\n"
                    f"<b>📊 Сигнал:</b> <code>{html.escape(signal['signal'])}</code> <code>(Сила: {confidence_percent:.0f}%)</code>\n"
                    f"<b>💰 Текущая цена:</b> <code>{signal['price']:.5f}</code>\n"
                    f"<b>🎯 Тейк-профит:</b> <code>{signal['take_profit']:.5f}</code>\n"
                    f"<b>🛑 Стоп-лосс:</b> <code>{signal['stop_loss']:.5f}</code>\n"
                    f"<b>📈 Потенциальная прибыль:</b> <code>{potential_profit:.2f}%</code>\n"
                    f"<b>⚖️ Размер лота:</b> <code>{self.config['lot_size']}</code>\n"
                    f"<b>🔢 Сигналов за 24ч:</b> <code>{signal_count}</code>\n"
                    f"{extra_info}\n"
                )

                # Добавляем причины сигнала
                if signal.get('detailed_reasons'):
                    message += f"<b>🔍 Причины:</b>\n"
                    for reason in signal['detailed_reasons'][:3]:
                        message += f"• {html.escape(reason)}\n"
                    message += "\n"

            message += f"<b>⏱️ Время начала анализа:</b> {html.escape(analysis_time_str)}\n"
            message += "<b>🌍 Часовой пояс:</b> Москва (UTC+3)\n"
            message += f"<b>⚡ Следующий анализ через {self.config['analysis_interval']} секунд</b>"

            logger.info(f"Отправка {len(self.signals)} сигналов в Telegram")
            await self.send_telegram_message(message)

        except Exception as e:
            logger.error(f"Ошибка при автоматической отправке сигналов: {e}")

    async def send_prediction_update(self):
        """Отправка обновления прогноза"""
        if not self.last_prediction:
            return

        try:
            prediction_text = self.format_prediction_message()
            await self.send_telegram_message(prediction_text)
        except Exception as e:
            logger.error(f"Ошибка при отправке прогноза: {e}")

    def get_symbol_info(self, symbol):
        """Получение информации о символе"""
        if self.demo_mode:
            return type('obj', (object,), {
                'spread': 15
            })()

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Не удалось получить информацию о символе {symbol}")
                return None
            return symbol_info
        except Exception as e:
            logger.error(f"Ошибка получения информации о символе {symbol}: {e}")
            return None

    def get_spread_pips(self, symbol):
        """Получение спреда в пипсах"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return 0

            # Для валютных пар пипс = 0.0001, кроме JPY пар где пипс = 0.01
            pip_value = 0.0001 if not symbol.endswith('JPY') else 0.01
            spread = symbol_info.spread * pip_value
            return spread
        except Exception as e:
            logger.error(f"Ошибка расчета спреда для {symbol}: {e}")
            return 0

    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Получение OHLCV данных из MT5 или генерация демо-данных"""
        if self.demo_mode:
            return self.generate_forex_demo_data(symbol, timeframe, limit)

        try:
            timeframe_map = {
                'M15': mt5.TIMEFRAME_M15,
                'M5': mt5.TIMEFRAME_M5,
                'M1': mt5.TIMEFRAME_M1
            }

            mt5_timeframe = timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"Неизвестный таймфрейм: {timeframe}")
                return None

            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, limit)
            if rates is None:
                logger.error(f"Не удалось получить данные для {symbol} на {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)

            if df.isnull().values.any():
                return None

            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol} на {timeframe}: {e}")
            return None

    def generate_forex_demo_data(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Генерация реалистичных демо-данных для Forex"""
        try:
            # Базовые цены для разных валютных пар
            base_prices = {
                'EURUSD': 1.0850,
                'GBPUSD': 1.2650,
                'USDJPY': 147.50,
                'USDCHF': 0.8800,
                'USDCAD': 1.3500,
                'AUDUSD': 0.6550,
                'NZDUSD': 0.6100,
                'XAUUSD': 1980.0  # Добавили золото
            }

            base_price = base_prices.get(symbol, 1.1000)

            volatility_multiplier = {
                'M1': 0.3,
                'M5': 0.5,
                'M15': 0.8
            }.get(timeframe, 0.5)

            # Увеличим волатильность для золота
            if symbol == 'XAUUSD':
                volatility_multiplier *= 2.0

            now = datetime.now()
            timeframe_minutes = self.config['timeframes'][timeframe]
            timestamps = [now - timedelta(minutes=timeframe_minutes * i) for i in range(limit)]
            timestamps.reverse()

            np.random.seed(int(time.time()))

            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            # Начальная цена с небольшим случайным отклонением
            current_price = base_price * np.random.uniform(0.999, 1.001)

            # Добавляем трендовый компонент
            trend_direction = np.random.choice([-1, 1])
            trend_strength = np.random.uniform(0.0001, 0.0003)

            for i in range(limit):
                # Базовое изменение цены с трендом
                trend_component = current_price * trend_strength * trend_direction

                # Случайный шум
                random_component = np.random.normal(0, 0.0003 * volatility_multiplier) * current_price

                price_change = trend_component + random_component

                open_price = current_price
                close_price = current_price + price_change

                # Высокий и низкий уровень с учетом волатильности
                high_price = max(open_price, close_price) + abs(
                    np.random.normal(0, 0.0002 * volatility_multiplier)) * current_price
                low_price = min(open_price, close_price) - abs(
                    np.random.normal(0, 0.0002 * volatility_multiplier)) * current_price

                # Объем зависит от волатильности
                volume = int(np.random.randint(500, 2000) * (1 + volatility_multiplier * 0.5))

                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)

                current_price = close_price

            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=timestamps)

            return df

        except Exception as e:
            logger.error(f"Ошибка генерации демо-данных для {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 100:
            return df

        try:
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_percent'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            # EMA
            df['ema_8'] = talib.EMA(df['close'], timeperiod=8)
            df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            # Stochastic
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                       fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            # ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            # Momentum
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            # OBV
            df['obv'] = talib.OBV(df['close'], df['volume'])
            # Price Rate of Change
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            # Commodity Channel Index
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            # TRIX
            df['trix'] = talib.TRIX(df['close'], timeperiod=14)
            # Parabolic SAR
            df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            # Rate of Change
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            # Average Directional Index
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            # Plus Directional Indicator
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            # Minus Directional Indicator
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            # Linear Regression Slope
            df['linreg_slope'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)

            # Price trends
            df['price_trend'] = self.calculate_price_trend(df)

            # Pin Bar detection
            df['pin_bar'] = self.detect_pin_bars(df)

            # Volume indicators
            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Liquidation detection (добавили анализ ликвидаций)
            df['liquidation_signal'] = self.detect_liquidations(df)

        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
        return df

    def detect_pin_bars(self, df: pd.DataFrame) -> pd.Series:
        """Обнаружение пин-баров"""
        try:
            pin_bars = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]

                if total_range == 0:
                    continue

                body_to_range_ratio = body_size / total_range

                # Bullish pin bar (hammer)
                if (body_to_range_ratio < self.config['pin_bar_threshold'] and
                        (df['close'].iloc[i] - df['low'].iloc[i]) / total_range > 0.6 and
                        df['close'].iloc[i] > df['open'].iloc[i] and
                        df['close'].iloc[i] > df['close'].iloc[i - 1]):
                    pin_bars.iloc[i] = 1

                # Bearish pin bar (shooting star)
                elif (body_to_range_ratio < self.config['pin_bar_threshold'] and
                      (df['high'].iloc[i] - df['close'].iloc[i]) / total_range > 0.6 and
                      df['close'].iloc[i] < df['open'].iloc[i] and
                      df['close'].iloc[i] < df['close'].iloc[i - 1]):
                    pin_bars.iloc[i] = -1

            return pin_bars
        except Exception as e:
            logger.error(f"Ошибка обнаружения пин-баров: {e}")
            return pd.Series(0, index=df.index)

    def detect_liquidations(self, df: pd.DataFrame) -> pd.Series:
        """Обнаружение потенциальных ликвидаций"""
        try:
            liquidation_signals = pd.Series(0, index=df.index)

            for i in range(1, len(df)):
                # Резкие движения цены с высоким объемом могут указывать на ликвидации
                price_change = abs(df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
                volume_ratio = df['volume'].iloc[i] / df['volume_ma'].iloc[i] if i > 0 else 1

                if price_change > self.config['liquidation_threshold'] and volume_ratio > 2.0:
                    # Определяем направление движения
                    if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                        # Резкий рост - возможны ликвидации шортистов
                        liquidation_signals.iloc[i] = 1
                    else:
                        # Резкое падение - возможны ликвидации лонгистов
                        liquidation_signals.iloc[i] = -1

            return liquidation_signals
        except Exception as e:
            logger.error(f"Ошибка обнаружения ликвидаций: {e}")
            return pd.Series(0, index=df.index)

    def calculate_price_trend(self, df, period=20):
        try:
            x = np.arange(len(df))
            y = df['close'].values
            if len(df) > period:
                x = x[-period:]
                y = y[-period:]
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * r_value ** 2
        except Exception:
            return 0

    def analyze_multiple_timeframes(self, dfs: dict) -> dict:
        timeframe_weights = {'M15': 0.40, 'M5': 0.35, 'M1': 0.25}
        analysis_results = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 50:
                continue

            df = df.dropna()
            if len(df) < 20:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]
            tf_analysis = {
                'trend': 'neutral', 'momentum': 'neutral', 'volume': 'normal',
                'volatility': 'normal', 'signals': [], 'strength': 0,
                'price_action': 'neutral', 'market_condition': 'neutral',
                'detailed_signals': []
            }

            # Анализ тренда по EMA
            ema_trend_score = 0
            if last['ema_21'] > last['ema_50']:
                ema_trend_score += 1
                tf_analysis['detailed_signals'].append(('EMA21 > EMA50', 0.3))
            if last['ema_50'] > last['ema_200']:
                ema_trend_score += 1
                tf_analysis['detailed_signals'].append(('EMA50 > EMA200', 0.3))
            if last['close'] > last['ema_200']:
                ema_trend_score += 1
                tf_analysis['detailed_signals'].append(('Price > EMA200', 0.4))

            if ema_trend_score >= 2:
                tf_analysis['trend'] = 'bullish'
                tf_analysis['strength'] += ema_trend_score * 0.2
                tf_analysis['detailed_signals'].append(('Bullish EMA Trend', 0.6))
            elif ema_trend_score <= 1:
                tf_analysis['trend'] = 'bearish'
                tf_analysis['strength'] += (3 - ema_trend_score) * 0.2
                tf_analysis['detailed_signals'].append(('Bearish EMA Trend', 0.6))

            # Анализ импульса
            momentum_score = 0
            if last['rsi'] > 50:
                momentum_score += 1
                tf_analysis['detailed_signals'].append(('RSI > 50', 0.2))
            if last['macd'] > last['macd_signal']:
                momentum_score += 1
                tf_analysis['detailed_signals'].append(('MACD Bullish', 0.3))
            if last['stoch_k'] > 50:
                momentum_score += 1
                tf_analysis['detailed_signals'].append(('Stochastic > 50', 0.2))
            if last['close'] > last['ema_21']:
                momentum_score += 1
                tf_analysis['detailed_signals'].append(('Price > EMA21', 0.2))
            if last['trix'] > 0:
                momentum_score += 1
                tf_analysis['detailed_signals'].append(('TRIX Positive', 0.3))
            if last['roc'] > 0:
                momentum_score += 1
                tf_analysis['detailed_signals'].append(('ROC Positive', 0.2))

            if momentum_score >= 4:
                tf_analysis['momentum'] = 'bullish'
                tf_analysis['strength'] += momentum_score * 0.15
                tf_analysis['detailed_signals'].append(('Bullish Momentum', 0.5))
            elif momentum_score <= 2:
                tf_analysis['momentum'] = 'bearish'
                tf_analysis['strength'] += (6 - momentum_score) * 0.15
                tf_analysis['detailed_signals'].append(('Bearish Momentum', 0.5))

            # Анализ объема
            volume_ratio = last['volume_ratio'] if 'volume_ratio' in last else 1.0
            if volume_ratio > self.config['volume_spike_threshold']:
                tf_analysis['volume'] = 'high'
                tf_analysis['strength'] += 0.3
                tf_analysis['detailed_signals'].append(('High Volume', 0.4))
            elif volume_ratio < 0.5:
                tf_analysis['volume'] = 'low'
                tf_analysis['strength'] -= 0.15
                tf_analysis['detailed_signals'].append(('Low Volume', -0.3))

            # Анализ волатильности
            if 'bb_width' in last and last['bb_width'] > df['bb_width'].mean() * 1.5:
                tf_analysis['volatility'] = 'high'
                tf_analysis['detailed_signals'].append(('High Volatility', 0.2))
            elif 'bb_width' in last and last['bb_width'] < df['bb_width'].mean() * 0.5:
                tf_analysis['volatility'] = 'low'
                tf_analysis['detailed_signals'].append(('Low Volatility', -0.2))

            # Анализ ценового действия
            price_action_score = 0
            is_bullish_candle = last['close'] > last['open']
            is_bearish_candle = last['close'] < last['open']

            if last['pin_bar'] == 1:
                price_action_score += 2
                tf_analysis['signals'].append(('bullish_pin_bar', 0.8))
                tf_analysis['detailed_signals'].append(('Bullish Pin Bar', 0.8))
            elif last['pin_bar'] == -1:
                price_action_score -= 2
                tf_analysis['signals'].append(('bearish_pin_bar', 0.8))
                tf_analysis['detailed_signals'].append(('Bearish Pin Bar', 0.8))

            if is_bullish_candle and last['close'] > prev['high'] and last['open'] < prev['low']:
                price_action_score += 2
                tf_analysis['signals'].append(('bullish_engulfing', 0.8))
                tf_analysis['detailed_signals'].append(('Bullish Engulfing', 0.8))
            elif is_bearish_candle and last['close'] < prev['low'] and last['open'] > prev['high']:
                price_action_score -= 2
                tf_analysis['signals'].append(('bearish_engulfing', 0.8))
                tf_analysis['detailed_signals'].append(('Bearish Engulfing', 0.8))

            if is_bullish_candle and (last['close'] - last['open']) / (last['high'] - last['low']) > 0.7:
                price_action_score += 1
                tf_analysis['signals'].append(('hammer', 0.6))
                tf_analysis['detailed_signals'].append(('Hammer Pattern', 0.6))
            elif is_bearish_candle and (last['open'] - last['close']) / (last['high'] - last['low']) > 0.7:
                price_action_score -= 1
                tf_analysis['signals'].append(('shooting_star', 0.6))
                tf_analysis['detailed_signals'].append(('Shooting Star', 0.6))

            if price_action_score >= 1:
                tf_analysis['price_action'] = 'bullish'
            elif price_action_score <= -1:
                tf_analysis['price_action'] = 'bearish'

            # Дополнительные сигналы
            if last['rsi'] < 30:
                tf_analysis['signals'].append(('oversold', 0.6))
                tf_analysis['detailed_signals'].append(('RSI Oversold', 0.6))
            elif last['rsi'] > 70:
                tf_analysis['signals'].append(('overbought', 0.6))
                tf_analysis['detailed_signals'].append(('RSI Overbought', 0.6))

            if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bullish', 0.7))
                tf_analysis['detailed_signals'].append(('MACD Crossover Bullish', 0.7))
            elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bearish', 0.7))
                tf_analysis['detailed_signals'].append(('MACD Crossover Bearish', 0.7))

            if last['stoch_k'] < 20:
                tf_analysis['signals'].append(('stoch_oversold', 0.5))
                tf_analysis['detailed_signals'].append(('Stochastic Oversold', 0.5))
            elif last['stoch_k'] > 80:
                tf_analysis['signals'].append(('stoch_overbought', 0.5))
                tf_analysis['detailed_signals'].append(('Stochastic Overbought', 0.5))

            if last['adx'] > 25:
                tf_analysis['signals'].append(('strong_trend', 0.5))
                tf_analysis['detailed_signals'].append(('Strong Trend (ADX > 25)', 0.5))

            # Анализ ликвидаций
            if 'liquidation_signal' in last and last['liquidation_signal'] != 0:
                if last['liquidation_signal'] == 1:
                    tf_analysis['signals'].append(('short_liquidation', 0.7))
                    tf_analysis['detailed_signals'].append(('Short Liquidation', 0.7))
                else:
                    tf_analysis['signals'].append(('long_liquidation', 0.7))
                    tf_analysis['detailed_signals'].append(('Long Liquidation', 0.7))

            analysis_results[tf] = tf_analysis

        return analysis_results

    def calculate_confidence_from_analysis(self, analysis_results: dict) -> float:
        total_confidence = 0
        total_weight = 0
        signals_count = 0
        trend_alignment = 0
        confirmed_timeframes = 0

        for tf, analysis in analysis_results.items():
            weight = {'M15': 0.40, 'M5': 0.35, 'M1': 0.25}.get(tf, 0.3)

            if analysis['trend'] != 'neutral':
                confirmed_timeframes += 1

            if analysis['trend'] == 'bullish':
                total_confidence += analysis['strength'] * weight
            elif analysis['trend'] == 'bearish':
                total_confidence -= analysis['strength'] * weight

            if analysis['volume'] == 'high':
                if analysis['trend'] == 'bullish':
                    total_confidence += 0.3 * weight
                elif analysis['trend'] == 'bearish':
                    total_confidence -= 0.3 * weight

            if analysis['price_action'] == 'bullish':
                total_confidence += 0.4 * weight
            elif analysis['price_action'] == 'bearish':
                total_confidence -= 0.4 * weight

            for signal_name, signal_strength in analysis['signals']:
                if 'bull' in signal_name or 'overbought' in signal_name:
                    total_confidence += signal_strength * weight
                elif 'bear' in signal_name or 'oversold' in signal_name:
                    total_confidence -= signal_strength * weight
                signals_count += 1

            if tf in ['M15', 'M5']:
                if analysis['trend'] == 'bullish':
                    trend_alignment += weight
                elif analysis['trend'] == 'bearish':
                    trend_alignment -= weight

            total_weight += weight

        if confirmed_timeframes < self.config['required_timeframes']:
            return 0

        if total_weight > 0:
            confidence = total_confidence / total_weight
        else:
            confidence = 0

        if signals_count >= self.config['required_indicators'] and abs(confidence) > 0.4:
            confidence *= 1.5

        if abs(trend_alignment) > 0.4:
            confidence *= 1.5

        return min(max(confidence, -1), 1)

    def calculate_price_prediction(self, df: pd.DataFrame, signal_type: str, price: float) -> float:
        """Расчет прогноза цены на основе текущего тренда и волатильности"""
        try:
            atr = df['atr'].iloc[-1]

            if signal_type == 'BUY':
                # Прогноз роста цены
                prediction = price + (atr * self.config['atr_multiplier_tp'] * 1.5)
            else:
                # Прогноз падения цены
                prediction = price - (atr * self.config['atr_multiplier_tp'] * 1.5)

            return prediction
        except Exception as e:
            logger.error(f"Ошибка расчета прогноза цены: {e}")
            return price

    def calculate_stop_loss_take_profit(self, df: pd.DataFrame, signal_type: str, price: float) -> tuple:
        try:
            atr = df['atr'].iloc[-1]
            min_sl_percent = 0.001  # Уменьшили для скальпинга
            max_sl_percent = 0.005  # Уменьшили для скальпинга

            if signal_type == 'BUY':
                base_sl = price - (atr * self.config['atr_multiplier_sl'])
                base_tp = price + (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                support_levels = self.find_support_levels(df)
                if support_levels:
                    closest_support = max([level for level in support_levels if level < price], default=None)
                    if closest_support and closest_support > base_sl:
                        base_sl = closest_support * 0.995

                min_sl_price = price * (1 - max_sl_percent)
                max_sl_price = price * (1 - min_sl_percent)
                base_sl = max(base_sl, min_sl_price)
                base_sl = min(base_sl, max_sl_price)

                risk = price - base_sl
                if risk > 0:
                    min_tp = price + risk * self.config['risk_reward_ratio']
                    base_tp = max(base_tp, min_tp)

                    resistance_levels = self.find_resistance_levels(df)
                    if resistance_levels:
                        closest_resistance = min([level for level in resistance_levels if level > price], default=None)
                        if closest_resistance and closest_resistance < base_tp:
                            base_tp = closest_resistance * 0.995
            else:
                base_sl = price + (atr * self.config['atr_multiplier_sl'])
                base_tp = price - (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                resistance_levels = self.find_resistance_levels(df)
                if resistance_levels:
                    closest_resistance = min([level for level in resistance_levels if level > price], default=None)
                    if closest_resistance and closest_resistance < base_sl:
                        base_sl = closest_resistance * 1.005

                min_sl_price = price * (1 + min_sl_percent)
                max_sl_price = price * (1 + max_sl_percent)
                base_sl = min(base_sl, max_sl_price)
                base_sl = max(base_sl, min_sl_price)

                risk = base_sl - price
                if risk > 0:
                    min_tp = price - risk * self.config['risk_reward_ratio']
                    base_tp = min(base_tp, min_tp)

                    support_levels = self.find_support_levels(df)
                    if support_levels:
                        closest_support = max([level for level in support_levels if level < price], default=None)
                        if closest_support and closest_support > base_tp:
                            base_tp = closest_support * 1.005

            return base_sl, base_tp, None

        except Exception as e:
            logger.error(f"Ошибка расчета стоп-лосса и тейк-профита: {e}")
            return None, None, None

    def find_support_levels(self, df: pd.DataFrame, lookback_period: int = 100) -> list:
        try:
            support_levels = []

            for i in range(5, len(df) - 5):
                if (df['low'].iloc[i] < df['low'].iloc[i - 1] and
                        df['low'].iloc[i] < df['low'].iloc[i - 2] and
                        df['low'].iloc[i] < df['low'].iloc[i - 3] and
                        df['low'].iloc[i] < df['low'].iloc[i - 4] and
                        df['low'].iloc[i] < df['low'].iloc[i - 5] and
                        df['low'].iloc[i] < df['low'].iloc[i + 1] and
                        df['low'].iloc[i] < df['low'].iloc[i + 2] and
                        df['low'].iloc[i] < df['low'].iloc[i + 3] and
                        df['low'].iloc[i] < df['low'].iloc[i + 4] and
                        df['low'].iloc[i] < df['low'].iloc[i + 5]):
                    support_levels.append(df['low'].iloc[i])

            # Добавляем технические уровни поддержки
            support_levels.append(df['ema_21'].iloc[-1])
            support_levels.append(df['ema_50'].iloc[-1])
            support_levels.append(df['ema_200'].iloc[-1])
            support_levels.append(df['bb_lower'].iloc[-1])

            # Фильтруем и сортируем уровни
            valid_levels = [level for level in support_levels if level < df['close'].iloc[-1]]
            if not valid_levels:
                # Если нет уровней ниже текущей цены, используем минимальную цену из истории
                valid_levels = [df['low'].min()]

            return sorted(set(valid_levels), reverse=True)[:5]
        except Exception as e:
            logger.error(f"Ошибка поиска уровней поддержки: {e}")
            return []

    def find_resistance_levels(self, df: pd.DataFrame, lookback_period: int = 100) -> list:
        try:
            resistance_levels = []

            for i in range(5, len(df) - 5):
                if (df['high'].iloc[i] > df['high'].iloc[i - 1] and
                        df['high'].iloc[i] > df['high'].iloc[i - 2] and
                        df['high'].iloc[i] > df['high'].iloc[i - 3] and
                        df['high'].iloc[i] > df['high'].iloc[i - 4] and
                        df['high'].iloc[i] > df['high'].iloc[i - 5] and
                        df['high'].iloc[i] > df['high'].iloc[i + 1] and
                        df['high'].iloc[i] > df['high'].iloc[i + 2] and
                        df['high'].iloc[i] > df['high'].iloc[i + 3] and
                        df['high'].iloc[i] > df['high'].iloc[i + 4] and
                        df['high'].iloc[i] > df['high'].iloc[i + 5]):
                    resistance_levels.append(df['high'].iloc[i])

            # Добавляем технические уровни сопротивления
            resistance_levels.append(df['ema_21'].iloc[-1])
            resistance_levels.append(df['ema_50'].iloc[-1])
            resistance_levels.append(df['ema_200'].iloc[-1])
            resistance_levels.append(df['bb_upper'].iloc[-1])

            # Фильтруем и сортируем уровни
            valid_levels = [level for level in resistance_levels if level > df['close'].iloc[-1]]
            if not valid_levels:
                # Если нет уровней выше текущей цены, используем максимальную цену из истории
                valid_levels = [df['high'].max()]

            return sorted(set(valid_levels))[:5]
        except Exception as e:
            logger.error(f"Ошибка поиска уровней сопротивления: {e}")
            return []

    def generate_detailed_reasons(self, analysis_results: dict) -> list:
        """Генерация подробных причин для сигнала"""
        reasons = []

        for tf, analysis in analysis_results.items():
            # Добавляем основные сигналы
            for signal_name, signal_strength in analysis.get('detailed_signals', []):
                if abs(signal_strength) > 0.3:  # Только значимые сигналы
                    direction = "бычий" if signal_strength > 0 else "медвежий"
                    reasons.append(f"{tf}: {signal_name} ({direction})")

            # Добавляем информацию о тренде
            if analysis['trend'] != 'neutral':
                reasons.append(f"{tf}: {analysis['trend']} тренд")

            # Добавляем информацию об объеме
            if analysis['volume'] != 'normal':
                reasons.append(f"{tf}: {analysis['volume']} объем")

        # Ограничиваем количество причин
        return reasons[:10]

    def check_liquidity_and_volatility(self, df: pd.DataFrame) -> bool:
        """Проверка ликвидности и волатильности для скальпинга"""
        try:
            # Проверяем волатильность
            atr = df['atr'].iloc[-1]
            if atr < self.config['volatility_min'] or atr > self.config['volatility_max']:
                return False

            # Проверяем объем
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            if current_volume < volume_ma * 0.7:  # Низкий объем относительно среднего
                return False

            return True
        except Exception as e:
            logger.error(f"Ошибка проверки ликвидности и волатильности: {e}")
            return False

    async def detect_whale_activity(self, symbol: str, df: pd.DataFrame):
        """Обнаружение активности китов (больших ордеров)"""
        try:
            # Анализируем объемы на последних 5 свечах
            recent_volumes = df['volume'].tail(5).values
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]

            whale_detected = False
            whale_type = None

            # Проверяем наличие аномально высоких объемов
            for i, volume in enumerate(recent_volumes):
                if volume > volume_ma * self.config['whale_volume_threshold']:
                    whale_detected = True
                    # Определяем тип активности по движению цены
                    price_change = df['close'].iloc[-5 + i] - df['open'].iloc[-5 + i]
                    whale_type = 'buy' if price_change > 0 else 'sell'
                    break

            if whale_detected and whale_type:
                self.whale_activity[symbol].append({
                    'time': self.get_moscow_time(),
                    'type': whale_type,
                    'volume_ratio': volume / volume_ma,
                    'price': df['close'].iloc[-1]
                })

                # Ограничиваем историю активности китов
                if len(self.whale_activity[symbol]) > 20:
                    self.whale_activity[symbol] = self.whale_activity[symbol][-20:]

        except Exception as e:
            logger.error(f"Ошибка обнаружения активности китов для {symbol}: {e}")

    async def update_liquidation_levels(self, symbol: str, df: pd.DataFrame):
        """Обновление уровней ликвидаций"""
        try:
            # Этот метод имитирует обнаружение ликвидаций
            # В реальной торговле эту информацию нужно получать от брокера или из внешних источников

            # Имитация ликвидаций на основе резких движений цены
            price_change = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]

            if price_change > self.config['liquidation_threshold']:
                liquidation_type = 'long' if df['close'].iloc[-1] < df['open'].iloc[-1] else 'short'

                self.liquidation_levels[symbol].append({
                    'time': self.get_moscow_time(),
                    'type': liquidation_type,
                    'price': df['close'].iloc[-1],
                    'change_percent': price_change * 100
                })

                # Ограничиваем историю ликвидаций
                if len(self.liquidation_levels[symbol]) > 20:
                    self.liquidation_levels[symbol] = self.liquidation_levels[symbol][-20:]

        except Exception as e:
            logger.error(f"Ошибка обновления уровней ликвидаций для {symbol}: {e}")

    async def update_long_short_ratio(self, symbol: str, df: pd.DataFrame):
        """Обновление соотношения long/short"""
        try:
            # Этот метод имитирует расчет соотношения long/short
            # В реальной торговле эту информацию нужно получать от брокера

            # Простая имитация на основе технических индикаторов
            rsi = df['rsi'].iloc[-1]
            trend = self.calculate_price_trend(df.tail(20))

            # Чем выше RSI и тренд, тем больше long позиций
            long_ratio = min(1.0, max(0.0, (rsi - 30) / 40 + trend * 0.5))
            short_ratio = 1.0 - long_ratio

            self.long_short_ratio[symbol] = {
                'long': long_ratio,
                'short': short_ratio,
                'time': self.get_moscow_time()
            }

        except Exception as e:
            logger.error(f"Ошибка обновления соотношения long/short для {symbol}: {e}")

    def generate_trading_signal(self, dfs: dict, symbol: str, analysis_start_time) -> dict:
        if not dfs:
            return None

        try:
            spread = self.get_spread_pips(symbol)
            if spread > self.config['max_spread_pips']:
                logger.info(f"Спред для {symbol} слишком высок: {spread:.2f} пипсов")
                return None

            # Проверяем ликвидность и волатильность для скальпинга
            main_df = dfs.get('M5', next(iter(dfs.values())))
            if not self.check_liquidity_and_volatility(main_df):
                logger.info(f"Низкая ликвидность/волатильность для {symbol}, пропускаем")
                return None

            # Обновляем информацию о ликвидациях, китах и соотношении long/short
            asyncio.create_task(self.detect_whale_activity(symbol, main_df))
            asyncio.create_task(self.update_liquidation_levels(symbol, main_df))
            asyncio.create_task(self.update_long_short_ratio(symbol, main_df))

            analysis_results = self.analyze_multiple_timeframes(dfs)
            if not analysis_results:
                return None

            confidence = self.calculate_confidence_from_analysis(analysis_results)
            last = main_df.iloc[-1]

            signal = {
                'symbol': symbol,
                'timestamp': analysis_start_time,
                'price': last['close'],
                'signal': 'HOLD',
                'confidence': abs(confidence),
                'reasons': [],
                'detailed_reasons': [],
                'stop_loss': 0,
                'take_profit': 0,
                'signal_count_24h': self.get_signal_count_last_24h(symbol)
            }

            # Генерируем подробные причины
            detailed_reasons = self.generate_detailed_reasons(analysis_results)
            signal['detailed_reasons'] = detailed_reasons

            # Добавляем основные причины
            for tf, analysis in analysis_results.items():
                if analysis['signals']:
                    signal['reasons'].append(f"{tf}: {', '.join([s[0] for s in analysis['signals']])}")
                if analysis['trend'] != 'neutral':
                    signal['reasons'].append(f"{tf} trend: {analysis['trend']}")

            if abs(confidence) < self.config['min_confidence']:
                signal['signal'] = 'HOLD'
                return signal

            price_change = abs((last['close'] - last['open']) / last['open'])
            if price_change < self.config['min_price_change']:
                signal['signal'] = 'HOLD'
                return signal

            if confidence > 0:
                signal['signal'] = 'BUY'
            else:
                signal['signal'] = 'SELL'

            stop_loss, take_profit, _ = self.calculate_stop_loss_take_profit(
                main_df, signal['signal'], signal['price']
            )

            if stop_loss is None or take_profit is None:
                signal['signal'] = 'HOLD'
                return signal

            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit

            # Сохраняем прогноз
            target_price = self.calculate_price_prediction(main_df, signal['signal'], signal['price'])
            self.last_prediction = {
                'symbol': symbol,
                'direction': signal['signal'],
                'confidence': abs(confidence),
                'price': signal['price'],
                'target_price': target_price,
                'timeframe': 'M5',
                'timestamp': analysis_start_time
            }

            # Добавляем в историю прогнозов
            self.prediction_history.append(self.last_prediction)
            if len(self.prediction_history) > self.config['prediction_history_size']:
                self.prediction_history.pop(0)

            self.update_signal_history(symbol, signal['signal'], signal['confidence'])
            return signal

        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str, analysis_start_time) -> dict:
        try:
            dfs = {}
            for timeframe in self.config['timeframes']:
                df = await self.fetch_ohlcv_data(symbol, timeframe, limit=200)
                if df is not None:
                    df = self.calculate_technical_indicators(df)
                    dfs[timeframe] = df
                await asyncio.sleep(0.02)

            if not dfs:
                return None

            signal = self.generate_trading_signal(dfs, symbol, analysis_start_time)
            return signal
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

    async def run_analysis(self):
        logger.info("Начало анализа для скальпинга...")
        start_time = time.time()
        analysis_start_time = self.get_moscow_time()
        self.last_analysis_start_time = analysis_start_time

        self.top_symbols = self.get_symbols_list()
        self.analysis_stats['total_analyzed'] = len(self.top_symbols)
        if not self.top_symbols:
            logger.warning("Не найдено символов для анализа")
            return []

        tasks = []
        for symbol in self.top_symbols:
            task = asyncio.create_task(self.analyze_symbol(symbol, analysis_start_time))
            tasks.append(task)

        all_signals = []

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    continue
                elif result is not None:
                    all_signals.append(result)
        except Exception as e:
            logger.error(f"Ошибка обработки батча: {e}")

        # Фильтруем только сигналы BUY/SELL
        trading_signals = [s for s in all_signals if s['signal'] in ['BUY', 'SELL']]
        self.signals = sorted(trading_signals, key=lambda x: x['confidence'], reverse=True)
        self.analysis_stats['signals_found'] = len(self.signals)
        analysis_time = time.time() - start_time
        logger.info(f"Анализ завершен за {analysis_time:.1f} сек. Найдено {len(self.signals)} сигналов")

        # Отправляем сигналы и прогноз
        if self.signals:
            await self.send_automatic_signals()
        await self.send_prediction_update()

        return self.signals

    def print_signals(self, max_signals: int = 5):
        if not self.signals:
            print("🚫 Нет торговых сигналов")
            return

        mode = "ДЕМО-РЕЖИМ" if self.demo_mode else "РЕЖИМ РЕАЛЬНОГО ВРЕМЕНИ"
        print("\n" + "=" * 120)
        print(f"🎯 СКАЛЬПИНГ СИГНАЛЫ - {mode}")
        print(f"⏰ Время начала анализа: {self.format_moscow_time(self.last_analysis_start_time)}")
        print("=" * 120)
        print(
            f"{'Ранг':<4} {'Пара':<10} {'Сигнал':<8} {'Уверенность':<10} {'Цена':<10} {'R/R':<6} {'Вх.24ч':<6} {'Причины'}")
        print("-" * 120)

        for i, signal in enumerate(self.signals[:max_signals]):
            rank = f"{i + 1}"
            symbol = signal['symbol'][:10]
            signal_type = signal['signal'][:8]
            confidence = f"{signal['confidence'] * 100:.0f}%"
            price = f"{signal['price']:.5f}"
            rr_ratio = f"{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}"
            signal_count = f"{signal['signal_count_24h']}"
            reasons = ', '.join(signal['reasons'][:2]) if signal['reasons'] else 'N/A'

            print(
                f"{rank:<4} {symbol:<10} {signal_type:<8} {confidence:<10} {price:<10} {rr_ratio:<6} {signal_count:<6} {reasons}")

        print("=" * 120)

        for i, signal in enumerate(self.signals[:3]):
            print(
                f"\n🔥 ТОП-{i + 1}: {signal['symbol']}")
            print(f"📊 Сигнал: {signal['signal']} ({signal['confidence'] * 100:.0f}% уверенности)")
            print(f"💰 Цена: {signal['price']:.5f}")
            print(f"🛑 Стоп-лосс: {signal['stop_loss']:.5f}")
            print(f"🎯 Тейк-профит: {signal['take_profit']:.5f}")
            print(
                f"📈 R/R соотношение: 1:{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}")
            print(f"🔢 Сигналов за 24ч: {signal['signal_count_24h']}")

            # Добавляем информацию о ликвидациях и китах
            if signal['symbol'] in self.liquidation_levels:
                long_liq = len([l for l in self.liquidation_levels[signal['symbol']] if l['type'] == 'long'])
                short_liq = len([l for l in self.liquidation_levels[signal['symbol']] if l['type'] == 'short'])
                print(f"📊 Ликвидации: LONG {long_liq} | SHORT {short_liq}")

            if signal['symbol'] in self.whale_activity:
                whale_buy = len([w for w in self.whale_activity[signal['symbol']] if w['type'] == 'buy'])
                whale_sell = len([w for w in self.whale_activity[signal['symbol']] if w['type'] == 'sell'])
                print(f"🐋 Активность китов: BUY {whale_buy} | SELL {whale_sell}")

            if signal['symbol'] in self.long_short_ratio:
                ratio = self.long_short_ratio[signal['symbol']]
                print(f"⚖️ Соотношение LONG/SHORT: {ratio.get('long', 0):.2f}/{ratio.get('short', 0):.2f}")

            if signal.get('detailed_reasons'):
                print(f"🔍 Подробные причины:")
                for reason in signal['detailed_reasons'][:5]:
                    print(f"   • {reason}")

    async def run_continuous(self):
        analysis_count = 0
        while True:
            try:
                analysis_count += 1
                current_time = self.format_moscow_time()
                mode = "ДЕМО-РЕЖИМ" if self.demo_mode else "РЕЖИМ РЕАЛЬНОГО ВРЕМЕНИ"
                print(f"\n{'=' * 80}")
                print(f"📊 СКАЛЬПИНГ АНАЛИЗ #{analysis_count} - {current_time} (МСК) - {mode}")
                print(f"{'=' * 80}")
                start_time = time.time()
                await self.run_analysis()
                self.last_signals = self.signals.copy()
                if self.signals:
                    self.print_signals()
                else:
                    print("🚫 Сигналов не найдено")
                execution_time = time.time() - start_time
                print(f"\n⏱️ Время выполнения анализа: {execution_time:.1f} секунд")
                print(
                    f"📈 Статистика: {self.analysis_stats['total_analyzed']} пар, {self.analysis_stats['signals_found']} сигналов")
                print(f"🔄 Следующий анализ через {self.config['analysis_interval']} секунд...")
                await asyncio.sleep(self.config['analysis_interval'])
            except KeyboardInterrupt:
                print("\n\n🛑 Бот остановлен пользователем")
                break
            except Exception as e:
                print(f"\n❌ Ошибка в основном цикле: {e}")
                print("🔄 Повторная попытка через 5 секунд...")
                await asyncio.sleep(5)


async def main():
    bot = MT5TradingBot()
    try:
        current_time = bot.format_moscow_time()
        mode = "ДЕМО-РЕЖИМЕ (MT5 недоступен)" if bot.demo_mode else "режиме реального времени"

        print("🚀 Запуск скальпинг бота для Forex!")
        print(f"📊 Специализированный анализ для скальпинга в {mode}")
        print("⚡ Мультитаймфрейм анализ (M15, M5, M1)")
        print(
            "📈 Технические индикаторы: RSI, MACD, Bollinger Bands, EMA, ATR, Stochastic")
        print(
            f"⚙️ Настройки: мин. уверенность {bot.config['min_confidence'] * 100}%, R/R=1:{bot.config['risk_reward_ratio']}")
        print(f"💰 Депозит: ${bot.account_balance}")
        print(f"📊 Размер лота: {bot.config['lot_size']}")
        print(f"🌍 Часовой пояс: Москва (UTC+3)")
        print(f"🕐 Текущее время: {current_time}")

        if not bot.demo_mode:
            if not bot.initialize_mt5():
                print("❌ Не удалось подключиться к MT5. Переход в демо-режим.")
                bot.demo_mode = True
                bot.config['demo_mode'] = True

        print("⏸️ Для остановки нажмите Ctrl+C\n")
        await bot.initialize_telegram()
        print("📈 Выполняю первоначальный анализ...")
        await bot.run_analysis()
        bot.last_signals = bot.signals.copy()
        if bot.signals:
            bot.print_signals()
        else:
            print("📊 Сигналов не найдено")
        await bot.run_continuous()
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
    finally:
        if bot.telegram_app:
            await bot.telegram_app.updater.stop()
            await bot.telegram_app.stop()
            await bot.telegram_app.shutdown()
        if bot.telegram_worker_task:
            bot.telegram_worker_task.cancel()
        if MT5_AVAILABLE:
            mt5.shutdown()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
    except:
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Программа завершена")
    except Exception as e:
        print(f"🔄 Перезапуск после критической ошибки: {e}")
        time.sleep(10)
