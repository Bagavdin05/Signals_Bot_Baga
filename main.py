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

        self.config = {
            'timeframes': ['15m', '5m', '1h', '4h'],
            'min_volume_24h': 5000000,  # Увеличено до 5M
            'max_symbols_per_exchange': 20,
            'analysis_interval': 60,
            'risk_per_trade': 0.02,
            'virtual_balance': 10,
            'timeout': 10000,
            'min_confidence': 0.9,  # Увеличено до 90%
            'risk_reward_ratio': 1.5,  # Увеличено до 1:1.5
            'atr_multiplier_sl': 2.0,  # Увеличено для уменьшения ложных срабатываний
            'atr_multiplier_tp': 1.0,  # Увеличено для большего профита
            'blacklist': ['USDC/USDT', 'USDC/USD', 'USDCE/USDT', 'USDCB/USDT', 'BUSD/USDT'],
            'signal_validity_seconds': 300,
            'priority_exchanges': ['bybit', 'mexc', 'okx', 'gateio', 'bitget', 'kucoin', 'htx', 'bingx', 'phemex'],
            'required_indicators': 5,  # Увеличено до 5
            'min_price_change': 0.01,  # Увеличено до 1%
            'max_slippage_percent': 0.001,
            'volume_spike_threshold': 2.5,  # Увеличено для более сильного объема
            'trend_strength_threshold': 0.7,  # Увеличено для более сильного тренда
            'correction_filter': True,
            'multi_timeframe_confirmation': True,
            'market_trend_filter': True,
            'volume_confirmation': True,
            'volatility_filter': True,
            'price_action_filter': True,
            'min_leverage': 10,
            'required_timeframes': 3,  # Минимум 3 таймфрейма должны подтверждать сигнал
            'pin_bar_threshold': 0.7,  # Порог для пин-баров
            'trend_confirmation': True,  # Подтверждение тренда на старших таймфреймах
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
            # Увеличиваем таймауты для Telegram
            request = HTTPXRequest(
                connection_pool_size=10,
                connect_timeout=30.0,
                read_timeout=30.0,
                write_timeout=30.0
            )

            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()
            self.telegram_worker_task = asyncio.create_task(self.telegram_worker())
            logger.info("Telegram бот инициализирован")

            startup_message = (
                "🤖 <b>ТОРГОВЫЙ БОТ ЗАПУЩЕН!</b>\n\n"
                "📊 Бот начал анализ рынка и будет присылать ТОЧНЫЕ сигналы\n"
                f"⏰ Время запуска: {self.format_moscow_time()}\n"
                "🌍 Часовой пояс: Москва (UTC+3)\n\n"
                "⚡ <b>УЛУЧШЕННАЯ ВЕРСИЯ С ВЫСОКОЙ ТОЧНОСТЬЮ СИГНАЛОВ</b>"
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
                        # Добавляем повторные попытки отправки
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

        # Проверяем длину сообщения (ограничение Telegram - 4096 символов)
        if len(message) > 4096:
            message = message[:4090] + "..."

        for chat_id in chat_ids:
            await self.telegram_queue.put((chat_id, message))
            logger.info(f"Сообщение добавлено в очередь для chat_id: {chat_id}")

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        current_time = self.format_moscow_time()
        welcome_text = (
            "🚀 <b>ТОРГОВЫЙ БОТ ДЛЯ ФЬЮЧЕРСОВ</b>\n\n"
            "📊 <b>Поддерживаемые биржи:</b> Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, HTX, BingX, Phemex\n\n"
            "⚡ <b>Мультитаймфрейм анализ</b> (4h, 1h, 15m, 5m)\n"
            "📈 <b>Технические индикаторы:</b> RSI, MACD, Bollinger Bands, EMA, Volume, ATR, Stochastic, ADX, OBV, VWAP, Ichimoku\n\n"
            "🔮 <b>УЛУЧШЕННАЯ ВЕРСИЯ С ВЫСОКОЙ ТОЧНОСТЬЮ СИГНАЛОВ</b>\n\n"
            "⚙️ <b>Настройки:</b>\n"
            f"• Мин. уверенность: {self.config['min_confidence'] * 100}%\n"
            f"• Риск/вознаграждение: 1:{self.config['risk_reward_ratio']}\n"
            f"• Интервал анализа: {self.config['analysis_interval']} сек\n"
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

                message += (
                    f"{signal_emoji} <b>#{i + 1}: <a href='{exchange_url}'>{html.escape(formatted_exchange)}</a></b>\n"
                    f"<b>🪙 Монета:</b> <code>{html.escape(symbol_name)}</code>\n"
                    f"<b>📊 Сигнал:</b> <code>{html.escape(signal['signal'])}</code> <code>(Сила: {confidence_percent:.0f}%)</code>\n"
                    f"<b>⚖️ Размер:</b> <code>{signal['recommended_size']:.4f}</code>\n"
                    f"<b>📈 Объем 24ч:</b> <code>{volume_str}</code>\n"
                    f"<b>💰 Цена:</b> <code>{signal['price']:.6f}</code>\n"
                    f"<b>🎯 Тейк-профит:</b> <code>{signal['take_profit']:.6f}</code>\n"
                    f"<b>🛑 Стоп-лосс:</b> <code>{signal['stop_loss']:.6f}</code>\n"
                    f"<b>💸 Цена ликвидации (10X):</b> <code>{signal.get('liquidation_price', 'N/A'):.6f}</code>\n"
                    f"<b>🔢 Сигналов за 24ч:</b> <code>{signal_count}</code>\n\n"
                )

            message += f"<b>⏱️ Время начала анализа:</b> {html.escape(analysis_time_str)}\n"
            message += "<b>🌍 Часовой пояс:</b> Москва (UTC+3)\n"
            message += "<b>⚡ Автоматическое обновление</b>"

            logger.info(f"Отправка {len(self.signals)} сигналов в Telegram")
            await self.send_telegram_message(message)

        except Exception as e:
            logger.error(f"Ошибка при автоматической отправке сигналов: {e}")

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
            'phemex': f'https://phemex.com/contracts/{base_symbol}USDT'
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
            'phemex': 'Phemex'
        }
        return exchange_names.get(exchange_name, exchange_name.upper())

    def initialize_exchanges(self) -> dict:
        exchanges = {}
        exchange_configs = {
            'bybit': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'mexc': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'okx': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'gateio': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'bitget': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'kucoin': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'htx': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'bingx': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'phemex': {'options': {'defaultType': 'swap'}, 'timeout': 10000}
        }

        for exchange_name, config in exchange_configs.items():
            try:
                exchange_class = getattr(ccxt, exchange_name)
                exchange_instance = exchange_class({
                    'timeout': config['timeout'],
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
                'bitget': 50, 'kucoin': 50, 'htx': 50, 'bingx': 50, 'phemex': 50
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

    async def fetch_top_symbols(self) -> list:
        all_volume_map = {}
        exchange_weights = {'bybit': 1.2, 'okx': 1.1, 'mexc': 0.9, 'gateio': 0.9, 'phemex': 0.8}

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

    async def fetch_ohlcv_data(self, exchange_name: str, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        exchange = self.exchanges.get(exchange_name)
        if exchange is None:
            return None
        try:
            if self.is_blacklisted(symbol):
                return None
            normalized_symbol = self.normalize_symbol_for_exchange(symbol, exchange_name)
            if not normalized_symbol:
                return None
            await asyncio.sleep(np.random.uniform(0.01, 0.05))
            ohlcv = exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 50:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            if df.isnull().values.any():
                return None
            return df
        except Exception:
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
        if df is None or len(df) < 50:
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
            # Volume
            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
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
            # VWAP
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            # Ichimoku Cloud
            tenkan_sen = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
            kijun_sen = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
            df['ichimoku_senkou_a'] = senkou_span_a
            df['ichimoku_senkou_b'] = senkou_span_b
            df['ichimoku_cloud_green'] = senkou_span_a > senkou_span_b
            df['ichimoku_cloud_red'] = senkou_span_a < senkou_span_b
            # Price Rate of Change
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            # Commodity Channel Index
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            # MFI
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            # Ultimate Oscillator
            df['uo'] = self.calculate_ultimate_oscillator(df)
            # Price trends
            df['price_trend'] = self.calculate_price_trend(df)
            # Volume trends
            df['volume_trend'] = self.calculate_volume_trend(df)
            # TRIX
            df['trix'] = talib.TRIX(df['close'], timeperiod=14)
            # Parabolic SAR
            df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            # Chaikin Oscillator
            df['chaikin'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
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
            # Donchian Channel
            df['donchian_upper'] = df['high'].rolling(20).max()
            df['donchian_lower'] = df['low'].rolling(20).min()
            df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
            # Keltner Channel
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            keltner_middle = typical_price.rolling(20).mean()
            keltner_upper = keltner_middle + 2 * typical_price.rolling(20).std()
            keltner_lower = keltner_middle - 2 * typical_price.rolling(20).std()
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
            df['ha_trend_strength'] = abs(df['ha_close'] - df['ha_open']) / df['atr']
            # Pin Bar detection
            df['pin_bar'] = self.detect_pin_bars(df)
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
        return df

    def detect_pin_bars(self, df: pd.DataFrame) -> pd.Series:
        """Обнаружение пин-баров (модели разворота)"""
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
                    pin_bars.iloc[i] = 1  # Bullish pin bar

                # Bearish pin bar (shooting star)
                elif (body_to_range_ratio < self.config['pin_bar_threshold'] and
                      (df['high'].iloc[i] - df['close'].iloc[i]) / total_range > 0.6 and
                      df['close'].iloc[i] < df['open'].iloc[i] and
                      df['close'].iloc[i] < df['close'].iloc[i - 1]):
                    pin_bars.iloc[i] = -1  # Bearish pin bar

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
        timeframe_weights = {'4h': 0.40, '1h': 0.35, '15m': 0.15, '5m': 0.10}  # Увеличены веса старших ТФ
        analysis_results = {}
        for tf, df in dfs.items():
            if df is None or len(df) < 20:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]
            tf_analysis = {
                'trend': 'neutral', 'momentum': 'neutral', 'volume': 'normal',
                'volatility': 'normal', 'signals': [], 'strength': 0,
                'price_action': 'neutral', 'market_condition': 'neutral'
            }
            ema_trend_score = 0
            if last['ema_21'] > last['ema_50']: ema_trend_score += 1
            if last['ema_50'] > last['ema_200']: ema_trend_score += 1
            if last['close'] > last['ema_200']: ema_trend_score += 1
            if ema_trend_score >= 2:
                tf_analysis['trend'] = 'bullish'
                tf_analysis['strength'] += ema_trend_score * 0.15  # Увеличена сила тренда
            elif ema_trend_score <= 1:
                tf_analysis['trend'] = 'bearish'
                tf_analysis['strength'] += (3 - ema_trend_score) * 0.15  # Увеличена сила тренда

            momentum_score = 0
            if last['rsi'] > 55: momentum_score += 1  # Увеличен порог RSI
            if last['macd'] > last['macd_signal']: momentum_score += 1
            if last['stoch_k'] > 55: momentum_score += 1  # Увеличен порог Stochastic
            if last['close'] > last['vwap']: momentum_score += 1
            if last['trix'] > 0: momentum_score += 1
            if last['roc'] > 0: momentum_score += 1
            if momentum_score >= 4:
                tf_analysis['momentum'] = 'bullish'
                tf_analysis['strength'] += momentum_score * 0.15  # Увеличена сила момента
            elif momentum_score <= 2:
                tf_analysis['momentum'] = 'bearish'
                tf_analysis['strength'] += (6 - momentum_score) * 0.15  # Увеличена сила момента

            if last['volume_ratio'] > self.config['volume_spike_threshold']:
                tf_analysis['volume'] = 'high'
                tf_analysis['strength'] += 0.3  # Увеличена сила объема
            elif last['volume_ratio'] < 0.5:
                tf_analysis['volume'] = 'low'
                tf_analysis['strength'] -= 0.15  # Увеличена сила объема

            if last['bb_width'] > df['bb_width'].mean() * 1.5:
                tf_analysis['volatility'] = 'high'
            elif last['bb_width'] < df['bb_width'].mean() * 0.5:
                tf_analysis['volatility'] = 'low'

            price_action_score = 0
            is_bullish_candle = last['close'] > last['open']
            is_bearish_candle = last['close'] < last['open']

            # Улучшенное обнаружение пин-баров
            if last['pin_bar'] == 1:  # Bullish pin bar
                price_action_score += 2
                tf_analysis['signals'].append(('bullish_pin_bar', 0.8))
            elif last['pin_bar'] == -1:  # Bearish pin bar
                price_action_score -= 2
                tf_analysis['signals'].append(('bearish_pin_bar', 0.8))

            if last['ha_trend'] > 0:
                price_action_score += 1
            else:
                price_action_score -= 1

            if is_bullish_candle and last['close'] > prev['high'] and last['open'] < prev['low']:
                price_action_score += 2
                tf_analysis['signals'].append(('bullish_engulfing', 0.8))  # Увеличена уверенность
            elif is_bearish_candle and last['close'] < prev['low'] and last['open'] > prev['high']:
                price_action_score -= 2
                tf_analysis['signals'].append(('bearish_engulfing', 0.8))  # Увеличена уверенность

            if is_bullish_candle and (last['close'] - last['open']) / (last['high'] - last['low']) > 0.7:
                price_action_score += 1
                tf_analysis['signals'].append(('hammer', 0.6))  # Увеличена уверенность
            elif is_bearish_candle and (last['open'] - last['close']) / (last['high'] - last['low']) > 0.7:
                price_action_score -= 1
                tf_analysis['signals'].append(('shooting_star', 0.6))  # Увеличена уверенность

            if price_action_score >= 1:
                tf_analysis['price_action'] = 'bullish'
            elif price_action_score <= -1:
                tf_analysis['price_action'] = 'bearish'

            # Более строгие условия для перекупленности/перепроданности
            if last['rsi'] < 25 and last['close'] < last['bb_lower']:  # Уменьшен порог RSI
                tf_analysis['signals'].append(('oversold', 0.5))  # Увеличена уверенность
            elif last['rsi'] > 75 and last['close'] > last['bb_upper']:  # Увеличен порог RSI
                tf_analysis['signals'].append(('overbought', 0.5))  # Увеличена уверенность

            if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bullish', 0.6))  # Увеличена уверенность
            elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bearish', 0.6))  # Увеличена уверенность

            if last['stoch_k'] < 15 and last['stoch_d'] < 15:  # Уменьшены пороги
                tf_analysis['signals'].append(('stoch_oversold', 0.4))  # Увеличена уверенность
            elif last['stoch_k'] > 85 and last['stoch_d'] > 85:  # Увеличены пороги
                tf_analysis['signals'].append(('stoch_overbought', 0.4))  # Увеличена уверенность

            if last['adx'] > 30:  # Увеличен порог ADX
                tf_analysis['signals'].append(('strong_trend', 0.4))  # Увеличена уверенность

            if last['close'] > last['vwap'] and prev['close'] <= prev['vwap']:
                tf_analysis['signals'].append(('vwap_bullish', 0.5))  # Увеличена уверенность
            elif last['close'] < last['vwap'] and prev['close'] >= prev['vwap']:
                tf_analysis['signals'].append(('vwap_bearish', 0.5))  # Увеличена уверенность

            if not pd.isna(last['ichimoku_senkou_a']) and not pd.isna(last['ichimoku_senkou_b']):
                if last['ichimoku_cloud_green'] and last['close'] > last['ichimoku_senkou_a'] and last['close'] > last[
                    'ichimoku_senkou_b']:
                    tf_analysis['signals'].append(('ichimoku_bullish', 0.7))  # Увеличена уверенность
                elif last['ichimoku_cloud_red'] and last['close'] < last['ichimoku_senkou_a'] and last['close'] < last[
                    'ichimoku_senkou_b']:
                    tf_analysis['signals'].append(('ichimoku_bearish', 0.7))  # Увеличена уверенность

            if last['mfi'] < 15:  # Уменьшен порог MFI
                tf_analysis['signals'].append(('mfi_oversold', 0.4))  # Увеличена уверенность
            elif last['mfi'] > 85:  # Увеличен порог MFI
                tf_analysis['signals'].append(('mfi_overbought', 0.4))  # Увеличена уверенность

            if last['uo'] < 25:  # Уменьшен порог UO
                tf_analysis['signals'].append(('uo_oversold', 0.4))  # Увеличена уверенность
            elif last['uo'] > 75:  # Увеличен порог UO
                tf_analysis['signals'].append(('uo_overbought', 0.4))  # Увеличена уверенность

            if last['cci'] < -125:  # Уменьшен порог CCI
                tf_analysis['signals'].append(('cci_oversold', 0.4))  # Увеличена уверенность
            elif last['cci'] > 125:  # Увеличен порог CCI
                tf_analysis['signals'].append(('cci_overbought', 0.4))  # Увеличена уверенность

            if last['williams_r'] < -85:  # Уменьшен порог Williams %R
                tf_analysis['signals'].append(('williams_oversold', 0.4))  # Увеличена уверенность
            elif last['williams_r'] > -15:  # Увеличен порог Williams %R
                tf_analysis['signals'].append(('williams_overbought', 0.4))  # Увеличена уверенность

            if last['trix'] > 0 and prev['trix'] <= 0:
                tf_analysis['signals'].append(('trix_bullish', 0.6))  # Увеличена уверенность
            elif last['trix'] < 0 and prev['trix'] >= 0:
                tf_analysis['signals'].append(('trix_bearish', 0.6))  # Увеличена уверенность

            if last['close'] > last['sar']:
                tf_analysis['signals'].append(('sar_bullish', 0.5))  # Увеличена уверенность
            elif last['close'] < last['sar']:
                tf_analysis['signals'].append(('sar_bearish', 0.5))  # Увеличена уверенность

            if last['chaikin'] > 0:
                tf_analysis['signals'].append(('chaikin_bullish', 0.4))  # Увеличена уверенность
            elif last['chaikin'] < 0:
                tf_analysis['signals'].append(('chaikin_bearish', 0.4))  # Увеличена уверенность

            analysis_results[tf] = tf_analysis
        return analysis_results

    def calculate_confidence_from_analysis(self, analysis_results: dict) -> float:
        total_confidence = 0
        total_weight = 0
        signals_count = 0
        trend_alignment = 0
        confirmed_timeframes = 0

        for tf, analysis in analysis_results.items():
            weight = {'4h': 0.40, '1h': 0.35, '15m': 0.15, '5m': 0.10}.get(tf, 0.3)

            # Требуем подтверждения тренда на всех таймфреймах
            if analysis['trend'] != 'neutral':
                confirmed_timeframes += 1

            if analysis['trend'] == 'bullish':
                total_confidence += analysis['strength'] * weight
            elif analysis['trend'] == 'bearish':
                total_confidence -= analysis['strength'] * weight

            if analysis['volume'] == 'high':
                if analysis['trend'] == 'bullish':
                    total_confidence += 0.3 * weight  # Увеличено влияние объема
                elif analysis['trend'] == 'bearish':
                    total_confidence -= 0.3 * weight  # Увеличено влияние объема

            if analysis['price_action'] == 'bullish':
                total_confidence += 0.4 * weight  # Увеличено влияние price action
            elif analysis['price_action'] == 'bearish':
                total_confidence -= 0.4 * weight  # Увеличено влияние price action

            for signal_name, signal_strength in analysis['signals']:
                if 'bull' in signal_name or 'overbought' in signal_name:
                    total_confidence += signal_strength * weight
                elif 'bear' in signal_name or 'oversold' in signal_name:
                    total_confidence -= signal_strength * weight
                signals_count += 1

            if tf in ['4h', '1h']:
                if analysis['trend'] == 'bullish':
                    trend_alignment += weight
                elif analysis['trend'] == 'bearish':
                    trend_alignment -= weight

            total_weight += weight

        # Требуем подтверждения на минимум N таймфреймах
        if confirmed_timeframes < self.config['required_timeframes']:
            return 0

        if total_weight > 0:
            confidence = total_confidence / total_weight
        else:
            confidence = 0

        # Увеличиваем уверенность при большом количестве сигналов
        if signals_count >= self.config['required_indicators'] and abs(confidence) > 0.4:
            confidence *= 1.5  # Увеличено влияние количества сигналов

        # Увеличиваем уверенность при согласованности тренда
        if abs(trend_alignment) > 0.5:
            confidence *= 1.5  # Увеличено влияние согласованности тренда

        return min(max(confidence, -1), 1)

    def calculate_liquidation_price(self, entry_price: float, signal_type: str, leverage: int = 10) -> float:
        """Рассчитывает цену ликвидации для заданного плеча (10x по умолчанию)"""
        try:
            if signal_type == 'LONG':
                # Для LONG: liquidation_price = entry_price * (1 - 1/leverage)
                liquidation_price = entry_price * (1 - 1 / leverage)
            else:  # SHORT
                # Для SHORT: liquidation_price = entry_price * (1 + 1/leverage)
                liquidation_price = entry_price * (1 + 1 / leverage)
            return liquidation_price
        except Exception as e:
            logger.error(f"Ошибка расчета цены ликвидации: {e}")
            return None

    def calculate_stop_loss_take_profit(self, df: pd.DataFrame, signal_type: str, price: float) -> tuple:
        try:
            atr = df['atr'].iloc[-1]
            min_sl_percent = 0.01  # Увеличен минимальный стоп-лосс
            max_sl_percent = 0.04  # Увеличен максимальный стоп-лосс

            if signal_type == 'LONG':
                base_sl = price - (atr * self.config['atr_multiplier_sl'])
                base_tp = price + (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                # Ищем ближайшие уровни поддержки
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

                    # Ищем ближайшие уровни сопротивления для тейк-профита
                    resistance_levels = self.find_resistance_levels(df)
                    if resistance_levels:
                        closest_resistance = min([level for level in resistance_levels if level > price], default=None)
                        if closest_resistance and closest_resistance < base_tp:
                            base_tp = closest_resistance * 0.995
            else:
                base_sl = price + (atr * self.config['atr_multiplier_sl'])
                base_tp = price - (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                # Ищем ближайшие уровни сопротивления
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

                    # Ищем ближайшие уровни поддержки для тейк-профита
                    support_levels = self.find_support_levels(df)
                    if support_levels:
                        closest_support = max([level for level in support_levels if level < price], default=None)
                        if closest_support and closest_support > base_tp:
                            base_tp = closest_support * 1.005

            # Рассчитываем цену ликвидации для 10x плеча
            liquidation_price = self.calculate_liquidation_price(price, signal_type, 10)

            if liquidation_price is None:
                return None, None, None

            # Проверяем, чтобы стоп-лосс не был за ценой ликвидации
            if signal_type == 'LONG':
                if base_sl <= liquidation_price:
                    logger.info(f"Стоп-лосс {base_sl} ниже цены ликвидации {liquidation_price}")
                    return None, None, None
            else:  # SHORT
                if base_sl >= liquidation_price:
                    logger.info(f"Стоп-лосс {base_sl} выше цены ликвидации {liquidation_price}")
                    return None, None, None

            return base_sl, base_tp, liquidation_price

        except Exception as e:
            logger.error(f"Ошибка расчета стоп-лосса и тейк-профита: {e}")
            return None, None, None

    def find_support_levels(self, df: pd.DataFrame, lookback_period: int = 50) -> list:  # Увеличено окно поиска
        try:
            support_levels = []
            # Ищем минимумы
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

            # Добавляем скользящие средние как динамические уровни поддержки
            support_levels.append(df['ema_21'].iloc[-1])
            support_levels.append(df['ema_50'].iloc[-1])
            support_levels.append(df['ema_200'].iloc[-1])
            support_levels.append(df['bb_lower'].iloc[-1])

            return sorted(set(support_levels), reverse=True)[:5]  # Возвращаем топ-5 уровней
        except Exception:
            return []

    def find_resistance_levels(self, df: pd.DataFrame, lookback_period: int = 50) -> list:  # Увеличено окно поиска
        try:
            resistance_levels = []
            # Ищем максимумы
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

            # Добавляем скользящие средние как динамические уровни сопротивления
            resistance_levels.append(df['ema_21'].iloc[-1])
            resistance_levels.append(df['ema_50'].iloc[-1])
            resistance_levels.append(df['ema_200'].iloc[-1])
            resistance_levels.append(df['bb_upper'].iloc[-1])

            return sorted(set(resistance_levels))[:5]  # Возвращаем топ-5 уровней
        except Exception:
            return []

    def generate_trading_signal(self, dfs: dict, symbol: str, exchange_name: str, leverage_info: dict,
                                analysis_start_time) -> dict:
        if not dfs:
            return None
        try:
            analysis_results = self.analyze_multiple_timeframes(dfs)
            if not analysis_results:
                return None

            confidence = self.calculate_confidence_from_analysis(analysis_results)
            main_df = dfs.get('15m', next(iter(dfs.values())))
            last = main_df.iloc[-1]

            signal = {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': analysis_start_time,
                'price': last['close'],
                'signal': 'HOLD',
                'confidence': abs(confidence),
                'reasons': [],
                'recommended_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'liquidation_price': 0,
                'timeframe_analysis': analysis_results,
                'signal_count_24h': self.get_signal_count_last_24h(symbol),
                'volume_24h': self.symbol_24h_volume.get(symbol, 0)
            }

            reasons = []
            for tf, analysis in analysis_results.items():
                if analysis['signals']:
                    reasons.append(f"{tf}: {', '.join([s[0] for s in analysis['signals']])}")
                if analysis['trend'] != 'neutral':
                    reasons.append(f"{tf} trend: {analysis['trend']}")
            signal['reasons'] = reasons

            # Более строгие условия для фильтрации сигналов
            if abs(confidence) < self.config['min_confidence']:
                return None

            price_change = abs((last['close'] - last['open']) / last['open'])
            if price_change < self.config['min_price_change']:
                return None

            # Фильтр коррекции - не входим против тренда
            if self.config['correction_filter']:
                if confidence > 0:
                    recent_high = main_df['high'].rolling(20).max().iloc[-1]
                    if last['close'] > recent_high * 0.98:
                        return None
                else:
                    recent_low = main_df['low'].rolling(20).min().iloc[-1]
                    if last['close'] < recent_low * 1.02:
                        return None

            # Подтверждение объема - требуется высокий объем на всех таймфреймах
            if self.config['volume_confirmation']:
                volume_confirmations = 0
                for tf, analysis in analysis_results.items():
                    if analysis['volume'] == 'high':
                        volume_confirmations += 1
                if volume_confirmations < 2:  # Требуем подтверждения на минимум 2 таймфреймах
                    return None

            # Фильтр волатильности - избегаем низковолатильных рынков
            if self.config['volatility_filter']:
                if last['bb_width'] < main_df['bb_width'].mean() * 0.6:
                    return None

            # Подтверждение тренда на старших таймфреймах
            if self.config['trend_confirmation']:
                if confidence > 0:  # Потенциальный LONG
                    if not (analysis_results.get('4h', {}).get('trend') == 'bullish' and
                            analysis_results.get('1h', {}).get('trend') == 'bullish'):
                        return None
                else:  # Потенциальный SHORT
                    if not (analysis_results.get('4h', {}).get('trend') == 'bearish' and
                            analysis_results.get('1h', {}).get('trend') == 'bearish'):
                        return None

            if confidence > 0:
                signal['signal'] = 'LONG'
            else:
                signal['signal'] = 'SHORT'

            stop_loss, take_profit, liquidation_price = self.calculate_stop_loss_take_profit(
                main_df, signal['signal'], signal['price']
            )

            if stop_loss is None or take_profit is None or liquidation_price is None:
                return None

            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit
            signal['liquidation_price'] = liquidation_price

            risk_per_unit = abs(signal['price'] - signal['stop_loss'])
            if risk_per_unit > 0:
                risk_amount = self.config['virtual_balance'] * self.config['risk_per_trade']
                signal['recommended_size'] = round(risk_amount / risk_per_unit, 6)

            min_reward = risk_per_unit * self.config['risk_reward_ratio']
            actual_reward = abs(signal['take_profit'] - signal['price'])
            if actual_reward < min_reward:
                if signal['signal'] == 'LONG':
                    signal['take_profit'] = signal['price'] + min_reward
                else:
                    signal['take_profit'] = signal['price'] - min_reward

            self.update_signal_history(symbol, signal['signal'], signal['confidence'])
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
                    logger.info(
                        f"Символ {symbol} на {exchange_name} не поддерживает плечо {self.config['min_leverage']}x (макс: {leverage_info['max_leverage']}x)")
                    continue
                dfs = {}
                for timeframe in self.config['timeframes']:
                    df = await self.fetch_ohlcv_data(exchange_name, symbol, timeframe, limit=100)
                    if df is not None:
                        df = self.calculate_technical_indicators(df)
                        dfs[timeframe] = df
                    await asyncio.sleep(0.02)
                if not dfs:
                    continue
                signal = self.generate_trading_signal(dfs, symbol, exchange_name, leverage_info, analysis_start_time)
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

        symbols_to_analyze = self.top_symbols[:300]
        tasks = []
        for symbol in symbols_to_analyze:
            task = asyncio.create_task(self.analyze_symbol(symbol, analysis_start_time))
            tasks.append(task)

        batch_size = 8
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

        print("\n" + "=" * 160)
        print("🎯 ТОРГОВЫЕ СИГНАЛЫ НА ФЬЮЧЕРСЫ")
        print(f"⏰ Время начала анализа: {self.format_moscow_time(self.last_analysis_start_time)}")
        print("=" * 160)
        print(
            f"{'Ранг':<4} {'Биржа':<8} {'Пара':<12} {'Сигнал':<8} {'Уверенность':<10} {'Цена':<12} {'Объем 24ч':<12} {'R/R':<6} {'Вх.24ч':<6} {'Ликвидация (10X)':<15} {'Причины'}")
        print("-" * 160)

        for i, signal in enumerate(self.signals[:max_signals]):
            rank = f"{i + 1}"
            exchange = self.format_exchange_name(signal['exchange'])[:8]
            symbol = signal['symbol'].replace('/USDT', '')[:12]
            signal_type = signal['signal'][:8]
            confidence = f"{signal['confidence'] * 100:.0f}%"
            price = f"{signal['price']:.6f}"
            volume_24h = signal['volume_24h']
            volume_str = f"{volume_24h / 1e6:.2f}M" if volume_24h >= 1e6 else f"{volume_24h / 1e3:.1f}K"
            rr_ratio = f"{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}"
            signal_count = f"{signal['signal_count_24h']}"
            liquidation_price = f"{signal.get('liquidation_price', 0):.6f}"
            reasons = ', '.join(signal['reasons'][:2]) if signal['reasons'] else 'N/A'

            print(
                f"{rank:<4} {exchange:<8} {symbol:<12} {signal_type:<8} {confidence:<10} {price:<12} {volume_str:<12} {rr_ratio:<6} {signal_count:<6} {liquidation_price:<15} {reasons}")

        print("=" * 160)

        for i, signal in enumerate(self.signals[:3]):
            volume_24h = signal['volume_24h']
            volume_str = f"{volume_24h / 1e6:.2f}M" if volume_24h >= 1e6 else f"{volume_24h / 1e3:.1f}K"
            print(
                f"\n🔥 ТОП-{i + 1}: {signal['symbol'].replace('/USDT', '')} на {self.format_exchange_name(signal['exchange'])}")
            print(f"📊 Сигнал: {signal['signal']} ({signal['confidence'] * 100:.0f}% уверенности)")
            print(f"💰 Цена: {signal['price']:.8f}")
            print(f"📈 Объем 24ч: {volume_str}")
            print(f"🛑 Стоп-лосс: {signal['stop_loss']:.8f}")
            print(f"🎯 Тейк-профит: {signal['take_profit']:.8f}")
            print(f"💸 Цена ликвидации (10X): {signal.get('liquidation_price', 'N/A'):.8f}")
            print(f"⚖️ Размер позиции: {signal['recommended_size']:.6f}")
            print(
                f"📈 R/R соотношение: 1:{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}")
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
        print("🚀 Запуск улучшенного торгового бота с поддержкой 9 бирж!")
        print("📊 Поддерживаемые биржи: Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, HTX, BingX, Phemex")
        print("⚡ Мультитаймфрейм анализ (4h, 1h, 15m, 5m)")
        print(
            "📈 Технические индикаторы: RSI, MACD, Bollinger Bands, EMA, Volume, ATR, Stochastic, ADX, OBV, VWAP, Ichimoku")
        print(
            f"⚙️ Настройки: мин. уверенность {bot.config['min_confidence'] * 100}%, R/R=1:{bot.config['risk_reward_ratio']}")
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
