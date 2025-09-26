
import ccxt
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
import aiohttp
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns

warnings.filterwarnings('ignore')

# Настройка часового пояса Москвы (UTC+3)
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# Настройка логирования только в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FuturesBot')

# Конфигурация Telegram бота
TELEGRAM_BOT_TOKEN = "7952768185:AAGuhybXaGPJqtlGPd1-O4nc6_FpUL2rOgw"
TELEGRAM_CHAT_IDS = ["1167694150", "7916502470"]


class FuturesTradingBot:
    def __init__(self):
        self.exchanges = self.initialize_exchanges()
        self.telegram_app = None
        self.last_analysis_start_time = None
        self.last_signals = []
        self.is_analyzing = False
        self.analysis_lock = asyncio.Lock()
        self.telegram_queue = asyncio.Queue()
        self.telegram_worker_task = None
        self.signal_history = defaultdict(list)
        self.session = None
        self.symbol_24h_volume = {}
        self.symbol_leverage_info = {}
        self.liquidation_data = defaultdict(list)
        self.oi_data = defaultdict(dict)

        self.config = {
            'timeframes': ['1w', '1d', '4h', '1h', '15m', '5m'],  # Добавлены 1д и 1нед
            'min_volume_24h': 5000000,
            'max_symbols_per_exchange': 30,
            'analysis_interval': 60,
            'risk_per_trade': 0.02,
            'virtual_balance': 10,
            'min_confidence': 0.95,
            'risk_reward_ratio': 2.0,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 1.5,
            'blacklist': ['USDC/USDT', 'USDC/USD', 'USDCE/USDT', 'USDCB/USDT', 'BUSD/USDT'],
            'signal_validity_seconds': 300,
            'priority_exchanges': ['bybit', 'mexc', 'okx', 'gateio', 'bitget', 'kucoin', 'htx', 'bingx', 'phemex',
                                   'coinex', 'xt', 'ascendex', 'bitrue', 'blofin'],
            'required_indicators': 4,
            'min_price_change': 0.01,
            'max_slippage_percent': 0.001,
            'volume_spike_threshold': 2.0,
            'trend_strength_threshold': 0.7,
            'correction_filter': True,
            'multi_timeframe_confirmation': True,
            'market_trend_filter': True,
            'volume_confirmation': True,
            'volatility_filter': True,
            'price_action_filter': True,
            'min_leverage': 5,
            'required_timeframes': 3,
            'pin_bar_threshold': 0.6,
            'trend_confirmation': True,
            'liquidation_analysis': True,
            'oi_analysis': True,
            'whale_alert_threshold': 100000,
            'entry_point_strategy': 'moderate',
            'default_leverage': 10,
            'position_size_calculation': 'risk_based',
            'max_position_size_percent': 0.1,
            'timeframe_weights': {  # Веса для разных таймфреймов
                '1w': 0.25,  # Высший приоритет - недельный тренд
                '1d': 0.20,  # Дневной тренд
                '4h': 0.18,  # 4-часовой
                '1h': 0.15,  # Часовой
                '15m': 0.12,  # 15-минутный
                '5m': 0.10  # 5-минутный (входной)
            }
        }

        self.top_symbols = []
        self.signals = []
        self.analysis_stats = {'total_analyzed': 0, 'signals_found': 0}

        logger.info("Торговый бот инициализирован")

    async def initialize_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    def get_moscow_time(self, dt=None):
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(MOSCOW_TZ)

    def format_moscow_time(self, dt=None, format_str='%Y-%m-%d %H:%M:%S'):
        moscow_time = self.get_moscow_time(dt)
        return moscow_time.strftime(format_str)

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

    async def initialize_telegram(self):
        try:
            request = HTTPXRequest(connection_pool_size=10, read_timeout=30, write_timeout=30, connect_timeout=30)
            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()
            self.telegram_worker_task = asyncio.create_task(self.telegram_worker())
            logger.info("Telegram бот инициализирован")

            startup_message = (
                "🤖 <b>ТОРГОВЫЙ БОТ ЗАПУЩЕН!</b>\n\n"
                "📊 Бот начал анализ рынка и будет присылать ТОПОВЫЕ сигналы\n"
                f"⏰ Время запуска: {self.format_moscow_time()}\n"
                "🌍 Часовой пояс: Москва (UTC+3)\n\n"
                "⚡ <b>УЛУЧШЕННАЯ ВЕРСИЯ С АНАЛИЗОМ ЛИКВИДАЦИЙ И КИТОВЫХ ОРДЕРОВ</b>\n"
                "📈 <b>Добавлены таймфреймы 1 день и 1 неделя</b>"
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
                        await self.telegram_app.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                        logger.info(f"Сообщение успешно отправлено в Telegram (chat_id: {chat_id})")
                    except Exception as e:
                        logger.error(f"Не удалось отправить сообщение в Telegram: {e}")
                self.telegram_queue.task_done()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в telegram_worker: {e}")
                await asyncio.sleep(1)

    async def send_telegram_message(self, message: str, chat_ids: list = None):
        if chat_ids is None:
            chat_ids = TELEGRAM_CHAT_IDS

        if len(message) > 4096:
            message = message[:4090] + "..."

        for chat_id in chat_ids:
            await self.telegram_queue.put((chat_id, message))
            logger.info(f"Сообщение добавлено в очередь для chat_id: {chat_id}")

    async def send_telegram_image(self, image_buffer, caption: str = "", chat_ids: list = None):
        if chat_ids is None:
            chat_ids = TELEGRAM_CHAT_IDS

        for chat_id in chat_ids:
            try:
                image_buffer.seek(0)
                await self.telegram_app.bot.send_photo(
                    chat_id=chat_id,
                    photo=image_buffer,
                    caption=caption,
                    parse_mode='HTML'
                )
                logger.info(f"Изображение отправлено в Telegram (chat_id: {chat_id})")
            except Exception as e:
                logger.error(f"Ошибка отправки изображения в Telegram: {e}")

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current_time = self.format_moscow_time()
        welcome_text = (
            "🚀 <b>ТОРГОВЫЙ БОТ ДЛЯ ФЬЮЧЕРСОВ</b>\n\n"
            "📊 <b>Поддерживаемые биржи:</b> Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, HTX, BingX, Phemex, CoinEx, XT, AscendEX, Bitrue, Blofin\n\n"
            "⚡ <b>Мультитаймфрейм анализ</b> (1W, 1D, 4h, 1h, 15m, 5m)\n"
            "📈 <b>Технические индикаторы:</b> RSI, MACD, Bollinger Bands, EMA, Volume, ATR, Stochastic, ADX, OBV, VWAP, Ichimoku\n"
            "🐋 <b>Анализ китовых ордеров и ликвидаций</b>\n\n"
            "🔮 <b>УЛУЧШЕННАЯ ВЕРСИЯ С АНАЛИЗОМ ЛИКВИДАЦИЙ И КИТОВЫХ ОРДЕРОВ</b>\n"
            "📈 <b>Добавлены таймфреймы 1 день и 1 неделя</b>\n\n"
            "⚙️ <b>Настройки:</b>\n"
            f"• Мин. уверенность: {self.config['min_confidence'] * 100}%\n"
            f"• Риск/вознаграждение: 1:{self.config['risk_reward_ratio']}\n"
            f"• Интервал анализа: {self.config['analysis_interval']} сек\n"
            f"• Виртуальный баланс: ${self.config['virtual_balance']}\n"
            f"• Плечо по умолчанию: {self.config['default_leverage']}x\n"
            f"• Часовой пояс: Москва (UTC+3)\n\n"
            f"🕐 <b>Текущее время:</b> {current_time}\n\n"
            "📊 Сигналы отправляются автоматически после каждого анализа"
        )
        await self.send_telegram_message(welcome_text, [update.effective_chat.id])

    async def send_automatic_signals(self):
        if not self.signals:
            logger.info("Нет сигналов для отправки в Telegram")
            return

        try:
            analysis_time_str = self.format_moscow_time(self.last_analysis_start_time)
            message = "🚀 <b>НОВЫЕ ТОРГОВЫЕ СИГНАЛЫ</b>\n\n"
            message += "<i>Нажмите на данные, чтобы скопировать. Нажмите на биржу, чтобы перейти к торговле.</i>\n\n"

            for i, signal in enumerate(self.signals[:5]):
                symbol_name = signal['symbol'].replace('/USDT', '')
                exchange_url = self.get_exchange_url(signal['exchange'], signal['symbol'])
                confidence_percent = signal['confidence'] * 100
                signal_emoji = "🟢" if signal['signal'] == 'LONG' else "🔴"
                formatted_exchange = self.format_exchange_name(signal['exchange'])
                signal_count = self.get_signal_count_last_24h(signal['symbol'])
                volume_24h = self.symbol_24h_volume.get(signal['symbol'], 0)
                volume_str = f"{volume_24h / 1e6:.2f}M" if volume_24h >= 1e6 else f"{volume_24h / 1e3:.1f}K"

                profit_usd = signal['potential_profit_usd']
                loss_usd = signal['potential_loss_usd']

                liquidation_info = ""
                if 'liquidation_zones' in signal:
                    liq_zones = signal['liquidation_zones']
                    if liq_zones['above'] and liq_zones['below']:
                        liquidation_info = f"📊 Ликвидации: ↑{liq_zones['above'] / 1e6:.2f}M / ↓{liq_zones['below'] / 1e6:.2f}M\n"

                whale_info = ""
                if 'whale_orders' in signal:
                    whale_orders = signal['whale_orders']
                    if whale_orders['buy'] or whale_orders['sell']:
                        whale_info = f"🐋 Киты: 🟢{whale_orders['buy']} | 🔴{whale_orders['sell']}\n"

                message += (
                    f"{signal_emoji} <b>#{i + 1}: <a href='{exchange_url}'>{html.escape(formatted_exchange)}</a></b>\n"
                    f"<b>🪙 Монета:</b> <code>{html.escape(symbol_name)}</code>\n"
                    f"<b>📊 Сигнал:</b> <code>{html.escape(signal['signal'])}</code>\n"
                    f"<b>🎯 Точка входа:</b> <code>{signal['recommended_entry']:.6f}</code>\n"
                    f"<b>⚖️ Размер позиции:</b> <code>{signal['recommended_size']:.4f}</code>\n"
                    f"<b>💰 Текущая цена:</b> <code>{signal['current_price']:.6f}</code>\n"
                    f"<b>📈 Объем 24ч:</b> <code>{volume_str}</code>\n"
                    f"{liquidation_info}{whale_info}"
                    f"<b>🎯 Тейк-профит:</b> <code>{signal['take_profit']:.6f}</code> <code>(+${profit_usd:.2f})</code>\n"
                    f"<b>🛑 Стоп-лосс:</b> <code>{signal['stop_loss']:.6f}</code> <code>(-${loss_usd:.2f})</code>\n"
                    f"<b>💸 Цена ликвидации ({self.config['default_leverage']}X):</b> <code>{signal.get('liquidation_price', 'N/A'):.6f}</code>\n"
                    f"<b>🔢 Сигналов за 24ч:</b> <code>{signal_count}</code>\n\n"
                )

            message += f"<b>⏱️ Время начала анализа:</b> {html.escape(analysis_time_str)}\n"
            message += f"<b>💰 Виртуальный баланс:</b> ${self.config['virtual_balance']}\n"
            message += f"<b>⚖️ Плечо:</b> {self.config['default_leverage']}x\n"
            message += "<b>🌍 Часовой пояс:</b> Москва (UTC+3)\n"
            message += "<b>⚡ Автоматическое обновление</b>"

            logger.info(f"Отправка {len(self.signals)} сигналов в Telegram")
            await self.send_telegram_message(message)

            if self.signals:
                best_signal = self.signals[0]
                await self.generate_and_send_chart(best_signal)

        except Exception as e:
            logger.error(f"Ошибка при автоматической отправке сигналов: {e}")

    async def generate_and_send_chart(self, signal):
        try:
            exchange_name = signal['exchange']
            symbol = signal['symbol']
            exchange = self.exchanges.get(exchange_name)

            if not exchange:
                return

            df = await self.fetch_ohlcv_data(exchange_name, symbol, '15m', 100)
            if df is None or len(df) < 50:
                return

            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

            ax1.plot(df.index, df['close'], label='Price', color='white', linewidth=1.5)
            ax1.plot(df.index, df['ema_21'], label='EMA 21', color='orange', linewidth=1)
            ax1.plot(df.index, df['ema_50'], label='EMA 50', color='red', linewidth=1)
            ax1.plot(df.index, df['ema_200'], label='EMA 200', color='purple', linewidth=1)

            support_levels = self.find_support_levels(df)
            resistance_levels = self.find_resistance_levels(df)

            for level in support_levels[:3]:
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.7, linewidth=0.7)
            for level in resistance_levels[:3]:
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.7, linewidth=0.7)

            current_price = signal['current_price']
            entry_price = signal['recommended_entry']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            ax1.axhline(y=entry_price, color='blue', linestyle='-', alpha=0.8, linewidth=1.5, label='Entry')
            ax1.axhline(y=stop_loss, color='red', linestyle='-', alpha=0.8, linewidth=1.5, label='Stop Loss')
            ax1.axhline(y=take_profit, color='green', linestyle='-', alpha=0.8, linewidth=1.5, label='Take Profit')

            ax2.bar(df.index, df['volume'], color=np.where(df['close'] > df['open'], 'green', 'red'), alpha=0.7)
            ax2.plot(df.index, df['volume_ma'], color='yellow', linewidth=1, label='Volume MA')

            ax1.set_title(f'{symbol} - {exchange_name}', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.set_title('Volume', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()

            caption = (f"<b>📈 График для {symbol}</b>\n"
                       f"<b>Сигнал:</b> {signal['signal']} | <b>Уверенность:</b> {signal['confidence'] * 100:.1f}%\n"
                       f"<b>Точка входа:</b> {entry_price:.6f} | <b>Текущая цена:</b> {current_price:.6f}")

            await self.send_telegram_image(buffer, caption)

        except Exception as e:
            logger.error(f"Ошибка генерации графика: {e}")

    def get_exchange_url(self, exchange_name: str, symbol: str) -> str:
        base_symbol = symbol.replace('/USDT', '').replace(':', '').replace('-', '')
        urls = {
            'bybit': f'https://www.bybit.com/trade-uspt/{base_symbol}USDT',
            'okx': f'https://www.okx.com/trade-swap/{base_symbol}-USDT-SWAP',
            'mexc': f'https://futures.mexc.com/exchange/{base_symbol}_USDT',
            'gateio': f'https://www.gate.io/futures_trade/{base_symbol}_USDT',
            'bitget': f'https://www.bitget.com/futures/{base_symbol}USDT',
            'kucoin': f'https://futures.kucoin.com/trade/{base_symbol}USDT',
            'htx': f'https://www.htx.com/futures/{base_symbol}_USDT',
            'bingx': f'https://bingx.com/swap/{base_symbol}-USDT',
            'phemex': f'https://phemex.com/contracts/{base_symbol}USDT',
            'coinex': f'https://www.coinex.com/futures/{base_symbol}USDT',
            'xt': f'https://futures.xt.com/trade/{base_symbol}_USDT',
            'ascendex': f'https://ascendex.com/futures/{base_symbol}/USDT',
            'bitrue': f'https://www.bitrue.com/future/{base_symbol}_USDT',
            'blofin': f'https://www.blofin.com/trade/{base_symbol}-USDT'
        }
        return urls.get(exchange_name, f'https://www.{exchange_name}.com')

    def format_exchange_name(self, exchange_name: str) -> str:
        exchange_names = {
            'bybit': 'Bybit',
            'mexc': 'MEXC',
            'okx': 'OKX',
            'gateio': 'Gate.io',
            'bitget': 'Bitget',
            'kucoin': 'KuCoin',
            'htx': 'HTX',
            'bingx': 'BingX',
            'phemex': 'Phemex',
            'coinex': 'CoinEx',
            'xt': 'XT',
            'ascendex': 'AscendEX',
            'bitrue': 'Bitrue',
            'blofin': 'Blofin'
        }
        return exchange_names.get(exchange_name, exchange_name.upper())

    def initialize_exchanges(self) -> dict:
        exchanges = {}
        exchange_configs = {
            'bybit': {'options': {'defaultType': 'swap'}},
            'mexc': {'options': {'defaultType': 'swap'}},
            'okx': {'options': {'defaultType': 'swap'}},
            'gateio': {'options': {'defaultType': 'swap'}},
            'bitget': {'options': {'defaultType': 'swap'}},
            'kucoin': {'options': {'defaultType': 'swap'}},
            'htx': {'options': {'defaultType': 'swap'}},
            'bingx': {'options': {'defaultType': 'swap'}},
            'phemex': {'options': {'defaultType': 'swap'}},
            'coinex': {'options': {'defaultType': 'swap'}},
            'xt': {'options': {'defaultType': 'swap'}},
            'ascendex': {'options': {'defaultType': 'swap'}},
            'bitrue': {'options': {'defaultType': 'swap'}},
            'blofin': {'options': {'defaultType': 'swap'}}
        }

        for exchange_name, config in exchange_configs.items():
            try:
                exchange_class = getattr(ccxt, exchange_name)
                exchange_instance = exchange_class({
                    'enableRateLimit': True,
                    'options': config['options']
                })
                exchange_instance.load_markets()
                exchanges[exchange_name] = exchange_instance
                logger.info(f"Успешное подключение к {exchange_name}")
            except Exception as e:
                logger.error(f"Ошибка подключения к {exchange_name}: {e}")
                exchanges[exchange_name] = None
        return exchanges

    def is_blacklisted(self, symbol: str) -> bool:
        symbol_upper = symbol.upper()
        for blacklisted_symbol in self.config['blacklist']:
            if blacklisted_symbol.upper() in symbol_upper:
                return True
        return False

    def get_futures_symbols(self, exchange, exchange_name: str) -> list:
        futures_symbols = []
        try:
            markets = exchange.load_markets()
            for symbol, market in markets.items():
                try:
                    if self.is_blacklisted(symbol):
                        continue
                    if (market.get('swap', False) or market.get('future', False) or
                            'swap' in symbol.lower() or 'future' in symbol.lower() or
                            '/USDT:' in symbol or symbol.endswith('/USDT') or
                            'USDT' in symbol and ('PERP' in symbol or 'SWAP' in symbol)):
                        if 'USDT' in symbol and market.get('active', False):
                            futures_symbols.append(symbol)
                except Exception:
                    continue
            logger.info(f"С {exchange_name} найдено {len(futures_symbols)} фьючерсных пар")
        except Exception as e:
            logger.error(f"Ошибка получения пар с {exchange_name}: {e}")
        return futures_symbols[:self.config['max_symbols_per_exchange']]

    async def fetch_exchange_volume_data(self, exchange, exchange_name: str, symbols: list) -> dict:
        volume_map = {}
        try:
            tickers = exchange.fetch_tickers(symbols)
            for symbol, ticker in tickers.items():
                try:
                    if self.is_blacklisted(symbol):
                        continue
                    volume = ticker.get('quoteVolume', 0)
                    if volume and volume > self.config['min_volume_24h']:
                        normalized_symbol = symbol.replace(':', '/').replace('-', '/')
                        volume_map[normalized_symbol] = volume
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Ошибка получения данных объема с {exchange_name}: {e}")
        return volume_map

    async def fetch_leverage_info(self, exchange, symbol: str) -> dict:
        try:
            exchange_default_leverage = {
                'bybit': 50, 'mexc': 50, 'okx': 50, 'gateio': 50,
                'bitget': 50, 'kucoin': 50, 'htx': 50, 'bingx': 50,
                'phemex': 50, 'coinex': 50, 'xt': 50, 'ascendex': 50,
                'bitrue': 50, 'blofin': 50
            }
            default_leverage = exchange_default_leverage.get(exchange.name, 10)
            leverage_info = {
                'min_leverage': 1,
                'max_leverage': default_leverage,
                'leverage_available': True
            }
            return leverage_info
        except Exception as e:
            logger.error(f"Ошибка получения информации о плече для {symbol}: {e}")
            return {
                'min_leverage': 1,
                'max_leverage': 10,
                'leverage_available': True
            }

    async def fetch_liquidation_data(self, exchange, symbol: str) -> dict:
        try:
            if exchange.name == 'bybit':
                symbol_clean = symbol.replace('/', '')
                url = f"https://api.bybit.com/v2/public/liq-records?symbol={symbol_clean}&limit=50"
                async with self.session.get(url) as response:
                    data = await response.json()
                    if data.get('ret_code') == 0:
                        return data.get('result', {})
            elif exchange.name == 'okx':
                ccy = symbol.replace('/USDT', '')
                url = f"https://www.okx.com/api/v5/rubik/public/liquidation-orders?ccy={ccy}&limit=50"
                async with self.session.get(url) as response:
                    data = await response.json()
                    if data.get('code') == '0':
                        return data.get('data', {})
        except Exception as e:
            logger.error(f"Ошибка получения данных о ликвидациях для {symbol}: {e}")
        return {}

    async def fetch_oi_data(self, exchange, symbol: str) -> dict:
        try:
            if not exchange.has.get('fetchOpenInterest', False):
                return {}

            market = exchange.market(symbol)
            if not market.get('contract', False):
                return {}

            oi = exchange.fetch_open_interest(symbol)
            return oi
        except Exception as e:
            logger.error(f"Ошибка получения данных об открытом интересе для {symbol}: {e}")
            return {}

    async def fetch_whale_orders(self, exchange, symbol: str) -> dict:
        try:
            orderbook = exchange.fetch_order_book(symbol, limit=20)
            whale_buy = 0
            whale_sell = 0

            for bid in orderbook['bids']:
                if bid[0] * bid[1] > self.config['whale_alert_threshold']:
                    whale_buy += 1

            for ask in orderbook['asks']:
                if ask[0] * ask[1] > self.config['whale_alert_threshold']:
                    whale_sell += 1

            return {'buy': whale_buy, 'sell': whale_sell}
        except Exception as e:
            logger.error(f"Ошибка анализа китовых ордеров для {symbol}: {e}")
            return {'buy': 0, 'sell': 0}

    async def fetch_top_symbols(self) -> list:
        all_volume_map = {}
        exchange_weights = {
            'bybit': 1.2, 'okx': 1.1, 'mexc': 0.9, 'gateio': 0.9,
            'phemex': 0.8, 'coinex': 0.8, 'xt': 0.8, 'ascendex': 0.8,
            'bitrue': 0.8, 'blofin': 0.8
        }

        for exchange_name in self.config['priority_exchanges']:
            exchange = self.exchanges.get(exchange_name)
            if exchange is None:
                continue
            try:
                futures_symbols = self.get_futures_symbols(exchange, exchange_name)
                if not futures_symbols:
                    continue
                volume_map = await self.fetch_exchange_volume_data(exchange, exchange_name, futures_symbols)
                weight = exchange_weights.get(exchange_name, 1.0)
                for symbol, volume in volume_map.items():
                    weighted_volume = volume * weight
                    if symbol in all_volume_map:
                        all_volume_map[symbol] += weighted_volume
                    else:
                        all_volume_map[symbol] = weighted_volume
                logger.info(f"С {exchange_name} обработано {len(volume_map)} пар с объемом")
            except Exception as e:
                logger.error(f"Ошибка получения данных с {exchange_name}: {e}")
                continue

        sorted_symbols = sorted(all_volume_map.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, volume in sorted_symbols[:300]]
        self.symbol_24h_volume = dict(sorted_symbols[:300])
        logger.info(f"Отобрано топ {len(top_symbols)} пар для анализа")
        return top_symbols

    async def fetch_ohlcv_data(self, exchange_name: str, symbol: str, timeframe: str,
                               limit: int = None) -> pd.DataFrame:
        exchange = self.exchanges.get(exchange_name)
        if exchange is None:
            return None

        # Устанавливаем разный лимит для разных таймфреймов
        if limit is None:
            if timeframe == '1w':
                limit = 52  # 1 год данных
            elif timeframe == '1d':
                limit = 100  # 100 дней
            else:
                limit = 100  # для остальных таймфреймов

        try:
            if self.is_blacklisted(symbol):
                return None
            normalized_symbol = self.normalize_symbol_for_exchange(symbol, exchange_name)
            if not normalized_symbol:
                return None
            await asyncio.sleep(np.random.uniform(0.01, 0.05))
            ohlcv = exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:  # Минимальное количество свечей уменьшено для старших таймфреймов
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            if df.isnull().values.any():
                return None
            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных OHLCV для {symbol} на {timeframe}: {e}")
            return None

    def normalize_symbol_for_exchange(self, symbol: str, exchange_name: str) -> str:
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return None
        try:
            if symbol in exchange.symbols:
                return symbol
            variations = [
                symbol, symbol.replace('/', ''), symbol.replace('/', ':'),
                symbol.replace('/', '-'), symbol.replace('/USDT', 'USDT'),
                symbol.replace('/USDT', '-USDT'), symbol.replace('/USDT', ':USDT'),
            ]
            for variation in variations:
                if variation in exchange.symbols:
                    return variation
            return None
        except Exception:
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 20:  # Минимальное количество свечей уменьшено
            return df
        try:
            # Базовые индикаторы
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # Bollinger Bands (только если достаточно данных)
            if len(df) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
                df['bb_width'] = (bb_upper - bb_lower) / bb_middle
                df['bb_percent'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            else:
                df['bb_upper'] = df['close']
                df['bb_middle'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_width'] = 0
                df['bb_percent'] = 0.5

            # EMA (адаптивные периоды в зависимости от количества данных)
            if len(df) >= 8:
                df['ema_8'] = talib.EMA(df['close'], timeperiod=8)
            else:
                df['ema_8'] = df['close']

            if len(df) >= 21:
                df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
            else:
                df['ema_21'] = df['close']

            if len(df) >= 50:
                df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            else:
                df['ema_50'] = df['close']

            if len(df) >= 200:
                df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
            else:
                df['ema_200'] = df['close']

            # Volume
            if len(df) >= 20:
                df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
                df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
            else:
                df['volume_ma'] = df['volume']
                df['volume_ratio'] = 1

            # ATR
            if len(df) >= 14:
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            else:
                df['atr'] = (df['high'] - df['low']).rolling(window=min(5, len(df))).mean()

            # Stochastic
            if len(df) >= 14:
                slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                           fastk_period=14, slowk_period=3, slowd_period=3)
                df['stoch_k'] = slowk
                df['stoch_d'] = slowd
            else:
                df['stoch_k'] = 50
                df['stoch_d'] = 50

            # ADX
            if len(df) >= 14:
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            else:
                df['adx'] = 25

            # Momentum
            df['momentum'] = talib.MOM(df['close'], timeperiod=min(10, len(df) - 1))

            # OBV
            df['obv'] = talib.OBV(df['close'], df['volume'])

            # VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

            # Ichimoku Cloud (только если достаточно данных)
            if len(df) >= 52:
                tenkan_sen = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
                kijun_sen = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
                senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
                senkou_span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
                df['ichimoku_senkou_a'] = senkou_span_a
                df['ichimoku_senkou_b'] = senkou_span_b
                df['ichimoku_cloud_green'] = senkou_span_a > senkou_span_b
                df['ichimoku_cloud_red'] = senkou_span_a < senkou_span_b
            else:
                df['ichimoku_senkou_a'] = df['close']
                df['ichimoku_senkou_b'] = df['close']
                df['ichimoku_cloud_green'] = True
                df['ichimoku_cloud_red'] = False

            # Дополнительные индикаторы
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            df['uo'] = self.calculate_ultimate_oscillator(df)
            df['price_trend'] = self.calculate_price_trend(df)
            df['volume_trend'] = self.calculate_volume_trend(df)
            df['trix'] = talib.TRIX(df['close'], timeperiod=14)
            df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            df['chaikin'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
            df['linreg_slope'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)

            # Donchian Channel
            period = min(20, len(df))
            df['donchian_upper'] = df['high'].rolling(period).max()
            df['donchian_lower'] = df['low'].rolling(period).min()
            df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

            # Keltner Channel
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            keltner_middle = typical_price.rolling(period).mean()
            keltner_upper = keltner_middle + 2 * typical_price.rolling(period).std()
            keltner_lower = keltner_middle - 2 * typical_price.rolling(period).std()
            df['keltner_upper'] = keltner_upper
            df['keltner_lower'] = keltner_lower
            df['keltner_middle'] = keltner_middle
            df['keltner_position'] = (df['close'] - keltner_lower) / (keltner_upper - keltner_lower)

            # Heikin Ashi
            df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
            for i in range(1, len(df)):
                ha_open.append((ha_open[i - 1] + df['ha_close'].iloc[i - 1]) / 2)
            df['ha_open'] = ha_open
            df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
            df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
            df['ha_trend'] = np.where(df['ha_close'] > df['ha_open'], 1, -1)
            df['ha_trend_strength'] = abs(df['ha_close'] - df['ha_open']) / df['atr'].replace(0, 1)

            # Pin Bar detection
            df['pin_bar'] = self.detect_pin_bars(df)
            df['liquidation_heat'] = self.calculate_liquidation_heat(df)

        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
        return df

    def calculate_liquidation_heat(self, df: pd.DataFrame) -> pd.Series:
        try:
            atr_normalized = df['atr'] / df['close']
            volume_normalized = df['volume'] / df['volume_ma'].replace(0, 1)
            liquidation_heat = atr_normalized * volume_normalized * 100
            return liquidation_heat
        except Exception as e:
            logger.error(f"Ошибка расчета liquidation heat: {e}")
            return pd.Series(0, index=df.index)

    def detect_pin_bars(self, df: pd.DataFrame) -> pd.Series:
        try:
            pin_bars = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]

                if total_range == 0:
                    continue

                body_to_range_ratio = body_size / total_range

                if (body_to_range_ratio < self.config['pin_bar_threshold'] and
                        (df['close'].iloc[i] - df['low'].iloc[i]) / total_range > 0.6 and
                        df['close'].iloc[i] > df['open'].iloc[i] and
                        df['close'].iloc[i] > df['close'].iloc[i - 1]):
                    pin_bars.iloc[i] = 1

                elif (body_to_range_ratio < self.config['pin_bar_threshold'] and
                      (df['high'].iloc[i] - df['close'].iloc[i]) / total_range > 0.6 and
                      df['close'].iloc[i] < df['open'].iloc[i] and
                      df['close'].iloc[i] < df['close'].iloc[i - 1]):
                    pin_bars.iloc[i] = -1

            return pin_bars
        except Exception as e:
            logger.error(f"Ошибка обнаружения пин-баров: {e}")
            return pd.Series(0, index=df.index)

    def calculate_ultimate_oscillator(self, df, period1=7, period2=14, period3=28):
        try:
            bp = df['close'] - df[['low', 'close']].min(axis=1)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            avg7 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
            avg14 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
            avg28 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
            uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
            return uo
        except Exception:
            return pd.Series(np.nan, index=df.index)

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

    def calculate_volume_trend(self, df, period=20):
        try:
            x = np.arange(len(df))
            y = df['volume'].values
            if len(df) > period:
                x = x[-period:]
                y = y[-period:]
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * r_value ** 2
        except Exception:
            return 0

    def analyze_multiple_timeframes(self, dfs: dict) -> dict:
        timeframe_weights = self.config['timeframe_weights']
        analysis_results = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 10:  # Минимальное количество свечей уменьшено
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            prev2 = df.iloc[-3] if len(df) > 2 else prev

            tf_analysis = {
                'trend': 'neutral', 'momentum': 'neutral', 'volume': 'normal',
                'volatility': 'normal', 'signals': [], 'strength': 0,
                'price_action': 'neutral', 'market_condition': 'neutral'
            }

            # Анализ тренда по EMA (адаптивный в зависимости от таймфрейма)
            ema_trend_score = 0
            if 'ema_21' in df.columns and 'ema_50' in df.columns and 'ema_200' in df.columns:
                if last['ema_21'] > last['ema_50']:
                    ema_trend_score += 1
                if last['ema_50'] > last['ema_200']:
                    ema_trend_score += 1
                if last['close'] > last['ema_200']:
                    ema_trend_score += 1

            if ema_trend_score >= 2:
                tf_analysis['trend'] = 'bullish'
                tf_analysis['strength'] += ema_trend_score * 0.15
            elif ema_trend_score <= 1:
                tf_analysis['trend'] = 'bearish'
                tf_analysis['strength'] += (3 - ema_trend_score) * 0.15

            # Анализ momentum
            momentum_score = 0
            if last['rsi'] > 50: momentum_score += 1
            if last['macd'] > last['macd_signal']: momentum_score += 1
            if last['stoch_k'] > 50: momentum_score += 1
            if last['close'] > last['vwap']: momentum_score += 1
            if last['trix'] > 0: momentum_score += 1
            if last['roc'] > 0: momentum_score += 1

            if momentum_score >= 3:
                tf_analysis['momentum'] = 'bullish'
                tf_analysis['strength'] += momentum_score * 0.15
            elif momentum_score <= 2:
                tf_analysis['momentum'] = 'bearish'
                tf_analysis['strength'] += (6 - momentum_score) * 0.15

            # Анализ объема
            if last['volume_ratio'] > self.config['volume_spike_threshold']:
                tf_analysis['volume'] = 'high'
                tf_analysis['strength'] += 0.3
            elif last['volume_ratio'] < 0.5:
                tf_analysis['volume'] = 'low'
                tf_analysis['strength'] -= 0.15

            # Анализ волатильности
            if 'bb_width' in df.columns:
                bb_width_mean = df['bb_width'].mean()
                if bb_width_mean > 0:
                    if last['bb_width'] > bb_width_mean * 1.5:
                        tf_analysis['volatility'] = 'high'
                    elif last['bb_width'] < bb_width_mean * 0.5:
                        tf_analysis['volatility'] = 'low'

            # Price Action анализ
            price_action_score = 0
            is_bullish_candle = last['close'] > last['open']
            is_bearish_candle = last['close'] < last['open']

            if last['pin_bar'] == 1:
                price_action_score += 2
                tf_analysis['signals'].append(('bullish_pin_bar', 0.8))
            elif last['pin_bar'] == -1:
                price_action_score -= 2
                tf_analysis['signals'].append(('bearish_pin_bar', 0.8))

            if last['ha_trend'] > 0:
                price_action_score += 1
            else:
                price_action_score -= 1

            # Определение разворотных паттернов
            if is_bullish_candle and last['close'] > prev['high'] and last['open'] < prev['low']:
                price_action_score += 2
                tf_analysis['signals'].append(('bullish_engulfing', 0.8))
            elif is_bearish_candle and last['close'] < prev['low'] and last['open'] > prev['high']:
                price_action_score -= 2
                tf_analysis['signals'].append(('bearish_engulfing', 0.8))

            if price_action_score >= 1:
                tf_analysis['price_action'] = 'bullish'
            elif price_action_score <= -1:
                tf_analysis['price_action'] = 'bearish'

            # Сигналы перекупленности/перепроданности
            if last['rsi'] < 30:
                tf_analysis['signals'].append(('oversold', 0.5))
            elif last['rsi'] > 70:
                tf_analysis['signals'].append(('overbought', 0.5))

            if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bullish', 0.6))
            elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bearish', 0.6))

            # Для старших таймфреймов добавляем больше веса
            if tf in ['1w', '1d']:
                tf_analysis['strength'] *= 1.2  # Увеличиваем силу сигналов на старших ТФ

            analysis_results[tf] = tf_analysis

        return analysis_results

    def calculate_confidence_from_analysis(self, analysis_results: dict) -> float:
        total_confidence = 0
        total_weight = 0
        signals_count = 0
        trend_alignment = 0
        confirmed_timeframes = 0

        for tf, analysis in analysis_results.items():
            weight = self.config['timeframe_weights'].get(tf, 0.1)

            # Требуем подтверждения тренда на всех таймфреймах
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

            # Особый вес для старших таймфреймов
            if tf in ['1w', '1d']:
                if analysis['trend'] == 'bullish':
                    trend_alignment += weight * 1.5  # Больший вес для старших ТФ
                elif analysis['trend'] == 'bearish':
                    trend_alignment -= weight * 1.5

            total_weight += weight

        # Требуем подтверждения на минимум N таймфреймах
        if confirmed_timeframes < min(self.config['required_timeframes'], len(analysis_results)):
            return 0

        if total_weight > 0:
            confidence = total_confidence / total_weight
        else:
            confidence = 0

        # Увеличиваем уверенность при большом количестве сигналов
        if signals_count >= self.config['required_indicators'] and abs(confidence) > 0.4:
            confidence *= 1.5

        # Увеличиваем уверенность при согласованности тренда на старших таймфреймах
        if abs(trend_alignment) > 0.5:
            confidence *= 1.8  # Больший множитель для согласованности на старших ТФ

        return min(max(confidence, -1), 1)

    def calculate_liquidation_price(self, entry_price: float, signal_type: str, leverage: int = None) -> float:
        if leverage is None:
            leverage = self.config['default_leverage']

        try:
            if signal_type == 'LONG':
                liquidation_price = entry_price * (1 - 1 / leverage)
            else:
                liquidation_price = entry_price * (1 + 1 / leverage)
            return liquidation_price
        except Exception as e:
            logger.error(f"Ошибка расчета цены ликвидации: {e}")
            return None

    def calculate_position_size(self, entry_price: float, stop_loss: float, signal_type: str) -> float:
        try:
            if signal_type == 'LONG':
                risk_per_unit = entry_price - stop_loss
            else:
                risk_per_unit = stop_loss - entry_price

            risk_amount = self.config['virtual_balance'] * self.config['risk_per_trade']

            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit

                max_position_value = self.config['virtual_balance'] * self.config['max_position_size_percent']
                max_position_size = max_position_value / entry_price

                position_size = min(position_size, max_position_size)
                return round(position_size, 6)
            else:
                return 0
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            return 0

    def calculate_stop_loss_take_profit(self, df: pd.DataFrame, signal_type: str, current_price: float) -> tuple:
        try:
            atr = df['atr'].iloc[-1]
            min_sl_percent = 0.005
            max_sl_percent = 0.03

            if signal_type == 'LONG':
                base_sl = current_price - (atr * self.config['atr_multiplier_sl'])
                base_tp = current_price + (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                support_levels = self.find_support_levels(df)
                if support_levels:
                    closest_support = max([level for level in support_levels if level < current_price], default=None)
                    if closest_support and closest_support > base_sl:
                        base_sl = closest_support * 0.995

                min_sl_price = current_price * (1 - max_sl_percent)
                max_sl_price = current_price * (1 - min_sl_percent)
                base_sl = max(base_sl, min_sl_price)
                base_sl = min(base_sl, max_sl_price)

                risk = current_price - base_sl
                if risk > 0:
                    min_tp = current_price + risk * self.config['risk_reward_ratio']
                    base_tp = max(base_tp, min_tp)

                    resistance_levels = self.find_resistance_levels(df)
                    if resistance_levels:
                        closest_resistance = min([level for level in resistance_levels if level > current_price],
                                                 default=None)
                        if closest_resistance and closest_resistance < base_tp:
                            base_tp = closest_resistance * 0.995
            else:
                base_sl = current_price + (atr * self.config['atr_multiplier_sl'])
                base_tp = current_price - (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                resistance_levels = self.find_resistance_levels(df)
                if resistance_levels:
                    closest_resistance = min([level for level in resistance_levels if level > current_price],
                                             default=None)
                    if closest_resistance and closest_resistance < base_sl:
                        base_sl = closest_resistance * 1.005

                min_sl_price = current_price * (1 + min_sl_percent)
                max_sl_price = current_price * (1 + max_sl_percent)
                base_sl = min(base_sl, max_sl_price)
                base_sl = max(base_sl, min_sl_price)

                risk = base_sl - current_price
                if risk > 0:
                    min_tp = current_price - risk * self.config['risk_reward_ratio']
                    base_tp = min(base_tp, min_tp)

                    support_levels = self.find_support_levels(df)
                    if support_levels:
                        closest_support = max([level for level in support_levels if level < current_price],
                                              default=None)
                        if closest_support and closest_support > base_tp:
                            base_tp = closest_support * 1.005

            liquidation_price = self.calculate_liquidation_price(current_price, signal_type)

            if liquidation_price is None:
                return None, None, None

            if signal_type == 'LONG':
                if base_sl <= liquidation_price:
                    base_sl = liquidation_price * 1.01
            else:
                if base_sl >= liquidation_price:
                    base_sl = liquidation_price * 0.99

            return base_sl, base_tp, liquidation_price

        except Exception as e:
            logger.error(f"Ошибка расчета стоп-лосса и тейк-профита: {e}")
            return None, None, None

    def find_support_levels(self, df: pd.DataFrame, lookback_period: int = 50) -> list:
        try:
            support_levels = []
            period = min(lookback_period, len(df))

            for i in range(2, len(df) - 2):
                if (df['low'].iloc[i] < df['low'].iloc[i - 1] and
                        df['low'].iloc[i] < df['low'].iloc[i - 2] and
                        df['low'].iloc[i] < df['low'].iloc[i + 1] and
                        df['low'].iloc[i] < df['low'].iloc[i + 2]):
                    support_levels.append(df['low'].iloc[i])

            # Добавляем скользящие средние как динамические уровни поддержки
            if 'ema_21' in df.columns:
                support_levels.append(df['ema_21'].iloc[-1])
            if 'ema_50' in df.columns:
                support_levels.append(df['ema_50'].iloc[-1])
            if 'ema_200' in df.columns:
                support_levels.append(df['ema_200'].iloc[-1])
            if 'bb_lower' in df.columns:
                support_levels.append(df['bb_lower'].iloc[-1])

            return sorted(set([level for level in support_levels if not pd.isna(level)]), reverse=True)[:5]
        except Exception:
            return []

    def find_resistance_levels(self, df: pd.DataFrame, lookback_period: int = 50) -> list:
        try:
            resistance_levels = []
            period = min(lookback_period, len(df))

            for i in range(2, len(df) - 2):
                if (df['high'].iloc[i] > df['high'].iloc[i - 1] and
                        df['high'].iloc[i] > df['high'].iloc[i - 2] and
                        df['high'].iloc[i] > df['high'].iloc[i + 1] and
                        df['high'].iloc[i] > df['high'].iloc[i + 2]):
                    resistance_levels.append(df['high'].iloc[i])

            # Добавляем скользящие средние как динамические уровни сопротивления
            if 'ema_21' in df.columns:
                resistance_levels.append(df['ema_21'].iloc[-1])
            if 'ema_50' in df.columns:
                resistance_levels.append(df['ema_50'].iloc[-1])
            if 'ema_200' in df.columns:
                resistance_levels.append(df['ema_200'].iloc[-1])
            if 'bb_upper' in df.columns:
                resistance_levels.append(df['bb_upper'].iloc[-1])

            return sorted(set([level for level in resistance_levels if not pd.isna(level)]))[:5]
        except Exception:
            return []

    def determine_entry_point(self, df: pd.DataFrame, signal_type: str, current_price: float) -> float:
        try:
            if self.config['entry_point_strategy'] == 'aggressive':
                return current_price

            elif self.config['entry_point_strategy'] == 'conservative':
                if signal_type == 'LONG':
                    support_levels = self.find_support_levels(df)
                    if support_levels:
                        valid_supports = [level for level in support_levels if level < current_price]
                        if valid_supports:
                            closest_support = max(valid_supports)
                            return closest_support * 1.001
                else:
                    resistance_levels = self.find_resistance_levels(df)
                    if resistance_levels:
                        valid_resistances = [level for level in resistance_levels if level > current_price]
                        if valid_resistances:
                            closest_resistance = min(valid_resistances)
                            return closest_resistance * 0.999

            entry_price = current_price
            if signal_type == 'LONG':
                support_levels = self.find_support_levels(df)
                if support_levels:
                    valid_supports = [level for level in support_levels if level < current_price]
                    if valid_supports:
                        closest_support = max(valid_supports)
                        entry_price = (current_price + closest_support) / 2
            else:
                resistance_levels = self.find_resistance_levels(df)
                if resistance_levels:
                    valid_resistances = [level for level in resistance_levels if level > current_price]
                    if valid_resistances:
                        closest_resistance = min(valid_resistances)
                        entry_price = (current_price + closest_resistance) / 2

            return entry_price

        except Exception as e:
            logger.error(f"Ошибка определения точки входа: {e}")
            return current_price

    async def analyze_liquidation_levels(self, exchange, symbol: str, current_price: float) -> dict:
        try:
            liquidation_data = await self.fetch_liquidation_data(exchange, symbol)
            if not liquidation_data:
                return {'above': 0, 'below': 0}

            liquidations_above = 0
            liquidations_below = 0

            if 'buy' in liquidation_data and 'sell' in liquidation_data:
                for liq in liquidation_data.get('buy', []):
                    if liq['price'] > current_price:
                        liquidations_above += liq['qty'] * liq['price']
                    else:
                        liquidations_below += liq['qty'] * liq['price']

                for liq in liquidation_data.get('sell', []):
                    if liq['price'] > current_price:
                        liquidations_above += liq['qty'] * liq['price']
                    else:
                        liquidations_below += liq['qty'] * liq['price']

            return {'above': liquidations_above, 'below': liquidations_below}

        except Exception as e:
            logger.error(f"Ошибка анализа ликвидаций для {symbol}: {e}")
            return {'above': 0, 'below': 0}

    def calculate_profit_loss(self, entry_price: float, exit_price: float, position_size: float,
                              signal_type: str) -> float:
        try:
            if signal_type == 'LONG':
                return (exit_price - entry_price) * position_size
            else:
                return (entry_price - exit_price) * position_size
        except Exception as e:
            logger.error(f"Ошибка расчета P&L: {e}")
            return 0

    def generate_trading_signal(self, dfs: dict, symbol: str, exchange_name: str, leverage_info: dict,
                                analysis_start_time, liquidation_data: dict = None,
                                oi_data: dict = None, whale_orders: dict = None) -> dict:
        if not dfs:
            return None
        try:
            analysis_results = self.analyze_multiple_timeframes(dfs)
            if not analysis_results:
                return None

            confidence = self.calculate_confidence_from_analysis(analysis_results)

            # Используем 4h таймфрейм как основной для расчетов
            main_df = dfs.get('4h', dfs.get('1h', next(iter(dfs.values()))))
            last = main_df.iloc[-1]

            current_price = last['close']
            recommended_entry = self.determine_entry_point(main_df,
                                                           'LONG' if confidence > 0 else 'SHORT',
                                                           current_price)

            stop_loss, take_profit, liquidation_price = self.calculate_stop_loss_take_profit(
                main_df,
                'LONG' if confidence > 0 else 'SHORT',
                recommended_entry
            )

            if stop_loss is None or take_profit is None or liquidation_price is None:
                return None

            position_size = self.calculate_position_size(recommended_entry, stop_loss,
                                                         'LONG' if confidence > 0 else 'SHORT')

            if position_size <= 0:
                return None

            if confidence > 0:
                potential_profit = self.calculate_profit_loss(recommended_entry, take_profit, position_size, 'LONG')
                potential_loss = self.calculate_profit_loss(recommended_entry, stop_loss, position_size, 'LONG')
                signal_type = 'LONG'
            else:
                potential_profit = self.calculate_profit_loss(recommended_entry, take_profit, position_size, 'SHORT')
                potential_loss = self.calculate_profit_loss(recommended_entry, stop_loss, position_size, 'SHORT')
                signal_type = 'SHORT'

            if potential_loss <= 0 or potential_profit / potential_loss < self.config['risk_reward_ratio'] - 0.5:
                return None

            signal = {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': analysis_start_time,
                'current_price': current_price,
                'recommended_entry': recommended_entry,
                'signal': signal_type,
                'confidence': abs(confidence),
                'reasons': [],
                'recommended_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'liquidation_price': liquidation_price,
                'potential_profit_usd': potential_profit,
                'potential_loss_usd': abs(potential_loss),
                'timeframe_analysis': analysis_results,
                'signal_count_24h': self.get_signal_count_last_24h(symbol),
                'volume_24h': self.symbol_24h_volume.get(symbol, 0),
                'liquidation_zones': liquidation_data or {'above': 0, 'below': 0},
                'oi_data': oi_data or {},
                'whale_orders': whale_orders or {'buy': 0, 'sell': 0}
            }

            reasons = []
            for tf, analysis in analysis_results.items():
                if analysis['signals']:
                    reasons.append(f"{tf}: {', '.join([s[0] for s in analysis['signals']])}")
                if analysis['trend'] != 'neutral':
                    reasons.append(f"{tf} trend: {analysis['trend']}")
            signal['reasons'] = reasons

            if abs(confidence) < self.config['min_confidence']:
                return None

            price_change = abs((last['close'] - last['open']) / last['open'])
            if price_change < self.config['min_price_change']:
                return None

            # Особые требования для старших таймфреймов
            weekly_analysis = analysis_results.get('1w', {})
            daily_analysis = analysis_results.get('1d', {})

            # Требуем подтверждения тренда на недельном таймфрейме
            if weekly_analysis and weekly_analysis.get('trend') == 'neutral':
                return None

            # Требуем согласованности дневного и недельного трендов
            if (weekly_analysis and daily_analysis and
                    weekly_analysis.get('trend') != daily_analysis.get('trend')):
                return None

            self.update_signal_history(symbol, signal_type, signal['confidence'])
            return signal

        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str, analysis_start_time) -> dict:
        best_signal = None
        best_confidence = 0

        for exchange_name, exchange in self.exchanges.items():
            if exchange is None:
                continue
            try:
                if self.is_blacklisted(symbol):
                    continue

                normalized_symbol = self.normalize_symbol_for_exchange(symbol, exchange_name)
                if not normalized_symbol:
                    continue

                leverage_info = await self.fetch_leverage_info(exchange, normalized_symbol)
                if leverage_info['max_leverage'] < self.config['min_leverage']:
                    continue

                # Получаем данные для всех таймфреймов
                dfs = {}
                for timeframe in self.config['timeframes']:
                    try:
                        df = await self.fetch_ohlcv_data(exchange_name, symbol, timeframe)
                        if df is not None and len(df) >= 10:  # Минимум 10 свечей
                            df = self.calculate_technical_indicators(df)
                            dfs[timeframe] = df
                    except Exception as e:
                        logger.error(f"Ошибка получения данных {timeframe} для {symbol}: {e}")
                    await asyncio.sleep(0.02)

                if not dfs:
                    continue

                # Получаем текущую цену для анализа ликвидаций
                current_price = dfs[list(dfs.keys())[0]].iloc[-1]['close'] if dfs else None

                liquidation_data = None
                oi_data = None
                whale_orders = None

                if self.config['liquidation_analysis'] and current_price is not None:
                    liquidation_data = await self.analyze_liquidation_levels(exchange, normalized_symbol, current_price)

                if self.config['oi_analysis']:
                    oi_data = await self.fetch_oi_data(exchange, normalized_symbol)

                whale_orders = await self.fetch_whale_orders(exchange, normalized_symbol)

                signal = self.generate_trading_signal(dfs, symbol, exchange_name, leverage_info,
                                                      analysis_start_time, liquidation_data, oi_data, whale_orders)

                if signal and signal['confidence'] > best_confidence:
                    best_signal = signal
                    best_confidence = signal['confidence']

            except Exception as e:
                logger.error(f"Ошибка анализа {symbol} на {exchange_name}: {e}")
                continue

        return best_signal

    async def run_analysis(self):
        logger.info("Начало анализа торговых пар...")
        start_time = time.time()
        analysis_start_time = self.get_moscow_time()
        self.last_analysis_start_time = analysis_start_time

        self.top_symbols = await self.fetch_top_symbols()
        self.analysis_stats['total_analyzed'] = len(self.top_symbols)
        if not self.top_symbols:
            logger.warning("Не найдено символов для анализа")
            return []

        symbols_to_analyze = self.top_symbols[:200]  # Уменьшили количество для скорости
        tasks = []
        for symbol in symbols_to_analyze:
            task = asyncio.create_task(self.analyze_symbol(symbol, analysis_start_time))
            tasks.append(task)

        batch_size = 5  # Уменьшили размер батча из-за большего количества таймфреймов
        all_signals = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            try:
                results = await asyncio.gather(*batch, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    elif result is not None:
                        all_signals.append(result)
                processed = min(i + batch_size, len(tasks))
                logger.info(f"Обработано {processed}/{len(tasks)} символов")
            except Exception as e:
                logger.error(f"Ошибка обработки батча: {e}")
                continue
            await asyncio.sleep(1)

        self.signals = sorted(all_signals, key=lambda x: x['confidence'], reverse=True)
        self.analysis_stats['signals_found'] = len(self.signals)
        analysis_time = time.time() - start_time
        logger.info(f"Анализ завершен за {analysis_time:.1f} сек. Найдено {len(self.signals)} сигналов")
        await self.send_automatic_signals()
        return self.signals

    def print_signals(self, max_signals: int = 15):
        if not self.signals:
            print("🚫 Нет торговых сигналов")
            return

        print("\n" + "=" * 180)
        print("🎯 ТОРГОВЫЕ СИГНАЛЫ НА ФЬЮЧЕРСЫ (С ТАЙМФРЕЙМАМИ 1W/1D)")
        print(f"⏰ Время начала анализа: {self.format_moscow_time(self.last_analysis_start_time)}")
        print(f"💰 Виртуальный баланс: ${self.config['virtual_balance']}")
        print(f"⚖️ Плечо: {self.config['default_leverage']}x")
        print("=" * 180)
        print(
            f"{'Ранг':<4} {'Биржа':<8} {'Пара':<12} {'Сигнал':<8} {'Уверенность':<10} {'Тек.цена':<12} {'Вход':<12} {'Размер':<12} {'Прибыль':<10} {'Убыток':<10} {'R/R':<6} {'Вх.24ч':<6} {'Таймфреймы':<15} {'Причины'}")
        print("-" * 180)

        for i, signal in enumerate(self.signals[:max_signals]):
            rank = f"{i + 1}"
            exchange = self.format_exchange_name(signal['exchange'])[:8]
            symbol = signal['symbol'].replace('/USDT', '')[:12]
            signal_type = signal['signal'][:8]
            confidence = f"{signal['confidence'] * 100:.0f}%"
            current_price = f"{signal['current_price']:.6f}"
            entry = f"{signal['recommended_entry']:.6f}"
            size = f"{signal['recommended_size']:.4f}"
            profit = f"${signal['potential_profit_usd']:.2f}"
            loss = f"${signal['potential_loss_usd']:.2f}"
            rr_ratio = f"{signal['potential_profit_usd'] / signal['potential_loss_usd']:.1f}"
            signal_count = f"{signal['signal_count_24h']}"

            # Информация о таймфреймах
            timeframes = list(signal['timeframe_analysis'].keys())
            timeframe_str = ','.join(timeframes[:3])[:15]

            reasons = ', '.join(signal['reasons'][:2]) if signal['reasons'] else 'N/A'

            print(
                f"{rank:<4} {exchange:<8} {symbol:<12} {signal_type:<8} {confidence:<10} {current_price:<12} {entry:<12} {size:<12} {profit:<10} {loss:<10} {rr_ratio:<6} {signal_count:<6} {timeframe_str:<15} {reasons}")

        print("=" * 180)

        for i, signal in enumerate(self.signals[:3]):
            volume_24h = signal['volume_24h']
            volume_str = f"{volume_24h / 1e6:.2f}M" if volume_24h >= 1e6 else f"{volume_24h / 1e3:.1f}K"

            # Анализ таймфреймов
            timeframe_analysis = signal['timeframe_analysis']
            tf_info = []
            for tf in ['1w', '1d', '4h', '1h']:
                if tf in timeframe_analysis:
                    analysis = timeframe_analysis[tf]
                    tf_info.append(f"{tf}:{analysis['trend'][:1]}")

            print(
                f"\n🔥 ТОП-{i + 1}: {signal['symbol'].replace('/USDT', '')} на {self.format_exchange_name(signal['exchange'])}")
            print(f"📊 Сигнал: {signal['signal']} ({signal['confidence'] * 100:.0f}% уверенности)")
            print(f"📈 Таймфреймы: {', '.join(tf_info)}")
            print(f"💰 Текущая цена: {signal['current_price']:.8f}")
            print(f"🎯 Рекомендуемый вход: {signal['recommended_entry']:.8f}")
            print(f"⚖️ Размер позиции: {signal['recommended_size']:.6f}")
            print(f"📈 Объем 24ч: {volume_str}")

            if 'liquidation_zones' in signal:
                liq = signal['liquidation_zones']
                print(f"📊 Ликвидации: ↑{liq['above'] / 1e6:.2f}M / ↓{liq['below'] / 1e6:.2f}M")

            if 'whale_orders' in signal:
                whales = signal['whale_orders']
                print(f"🐋 Китовые ордера: 🟢{whales['buy']} | 🔴{whales['sell']}")

            print(f"🛑 Стоп-лосс: {signal['stop_loss']:.8f} (-${signal['potential_loss_usd']:.2f})")
            print(f"🎯 Тейк-профит: {signal['take_profit']:.8f} (+${signal['potential_profit_usd']:.2f})")
            print(
                f"💸 Цена ликвидации ({self.config['default_leverage']}X): {signal.get('liquidation_price', 'N/A'):.8f}")
            print(f"📈 R/R соотношение: 1:{signal['potential_profit_usd'] / signal['potential_loss_usd']:.1f}")
            print(f"🔢 Сигналов за 24ч: {signal['signal_count_24h']}")
            if signal['reasons']:
                print(f"🔍 Причины: {', '.join(signal['reasons'][:3])}")

    async def run_continuous(self):
        analysis_count = 0
        while True:
            try:
                analysis_count += 1
                current_time = self.format_moscow_time()
                print(f"\n{'=' * 80}")
                print(f"📊 АНАЛИЗ #{analysis_count} - {current_time} (МСК)")
                print(f"🕐 Таймфреймы: 1W, 1D, 4H, 1H, 15M, 5M")
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
                print("🔄 Немедленный запуск следующего анализа...")
            except KeyboardInterrupt:
                print("\n\n🛑 Бот остановлен пользователем")
                break
            except Exception as e:
                print(f"\n❌ Ошибка в основном цикле: {e}")
                print("🔄 Повторная попытка через 5 секунд...")
                await asyncio.sleep(5)


async def main():
    bot = FuturesTradingBot()
    try:
        await bot.initialize_session()
        current_time = bot.format_moscow_time()
        print("🚀 Запуск улучшенного торгового бота с поддержкой 14 бирж!")
        print(
            "📊 Поддерживаемые биржи: Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, HTX, BingX, Phemex, CoinEx, XT, AscendEX, Bitrue, Blofin")
        print("⚡ Мультитаймфрейм анализ (1W, 1D, 4h, 1h, 15m, 5m)")
        print(
            "📈 Технические индикаторы: RSI, MACD, Bollinger Bands, EMA, Volume, ATR, Stochastic, ADX, OBV, VWAP, Ichimoku")
        print("🐋 Анализ китовых ордеров и ликвидаций")
        print(
            f"⚙️ Настройки: мин. уверенность {bot.config['min_confidence'] * 100}%, R/R=1:{bot.config['risk_reward_ratio']}")
        print(f"💰 Виртуальный баланс: ${bot.config['virtual_balance']}")
        print(f"⚖️ Плечо по умолчанию: {bot.config['default_leverage']}x")
        print(f"🌍 Часовой пояс: Москва (UTC+3)")
        print(f"🕐 Текущее время: {current_time}")
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
        await bot.close_session()


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Программа завершена")
            break
        except Exception as e:
            print(f"🔄 Перезапуск после критической ошибки: {e}")
            time.sleep(10)
