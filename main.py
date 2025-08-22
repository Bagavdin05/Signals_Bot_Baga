import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import asyncio
import talib
import warnings
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import html

warnings.filterwarnings('ignore')

# Настройка логирования только в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FuturesBot')

# Конфигурация Telegram бота
TELEGRAM_BOT_TOKEN = "8328135138:AAE5mLIWG59kM8STODbfPoLkd19iykbOmcM"
TELEGRAM_CHAT_IDS = ["1167694150", "7916502470", "5381553894"]


class FuturesTradingBot:
    def __init__(self):
        self.exchanges = self.initialize_exchanges()
        self.telegram_app = None
        self.last_analysis_time = None
        self.last_signals = []
        self.is_analyzing = False
        self.analysis_lock = asyncio.Lock()
        self.telegram_queue = asyncio.Queue()
        self.telegram_worker_task = None

        self.config = {
            'timeframes': ['15m', '5m', '1h'],
            'min_volume_24h': 1000000,
            'max_symbols_per_exchange': 30,
            'analysis_interval': 60,
            'risk_per_trade': 0.02,
            'virtual_balance': 1000,
            'timeout': 10000,
            'min_confidence': 0.80,
            'risk_reward_ratio': 2.0,
            'atr_multiplier_sl': 1.3,
            'atr_multiplier_tp': 1,
            'blacklist': ['USDC/USDT', 'USDC/USD', 'USDCE/USDT', 'USDCB/USDT', 'BUSD/USDT'],
            'signal_validity_seconds': 300,
            'priority_exchanges': ['bybit', 'mexc', 'okx', 'gateio', 'bitget', 'kucoin', 'htx', 'bingx', 'phemex'],
            'required_indicators': 3
        }

        self.top_symbols = []
        self.signals = []
        self.analysis_stats = {'total_analyzed': 0, 'signals_found': 0}

        logger.info("Торговый бот инициализирован")

    async def initialize_telegram(self):
        """Инициализация Telegram бота"""
        try:
            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

            # Добавляем только обработчик команды /start
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))

            # Запускаем бота в фоновом режиме
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()

            # Запускаем worker для обработки сообщений из очереди
            self.telegram_worker_task = asyncio.create_task(self.telegram_worker())

            logger.info("Telegram бот инициализирован")

        except Exception as e:
            logger.error(f"Ошибка инициализации Telegram бота: {e}")

    async def telegram_worker(self):
        """Работник для обработки сообщений из очереди"""
        while True:
            try:
                chat_id, message = await self.telegram_queue.get()

                if chat_id and message:
                    await self.telegram_app.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    )

                self.telegram_queue.task_done()
                await asyncio.sleep(0.1)  # Небольшая пауза между сообщениями

            except Exception as e:
                logger.error(f"Ошибка в telegram_worker: {e}")
                await asyncio.sleep(1)

    async def send_telegram_message(self, message: str, chat_ids: list = None):
        """Отправка сообщения в Telegram"""
        if chat_ids is None:
            chat_ids = TELEGRAM_CHAT_IDS

        for chat_id in chat_ids:
            await self.telegram_queue.put((chat_id, message))

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = (
            "🚀 Торговый бот для фьючерсов\n\n"
            "📊 Поддерживаемые биржи: Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, HTX, BingX, Phemex\n\n"
            "⚡ Мультитаймфрейм анализ (1h, 15m, 5m)\n"
            "📈 Технические индикаторы: RSI, MACD, Bollinger Bands, EMA, Volume\n\n"
            "Бот автоматически отправляет сигналы при их появлении!\n\n"
            "⚙️ Настройки:\n"
            f"• Мин. уверенность: {self.config['min_confidence'] * 100}%\n"
            f"• Риск/вознаграждение: 1:{self.config['risk_reward_ratio']}\n"
            f"• Интервал анализа: {self.config['analysis_interval']} сек\n\n"
            "📊 Сигналы отправляются автоматически после каждого анализа"
        )

        await self.send_telegram_message(welcome_text, [update.effective_chat.id])

    async def send_automatic_signals(self):
        """Автоматическая отправка сигналов после анализа"""
        if not self.signals:
            # Если сигналов нет, отправляем сообщение об этом
            no_signals_message = (
                "📊 <b>АНАЛИЗ ЗАВЕРШЕН</b>\n\n"
                "❌ Торговых сигналов не найдено\n\n"
                f"⏱️ Время анализа: {html.escape(self.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S'))}\n"
                "🔄 Следующий анализ через 60 секунд"
            )
            await self.send_telegram_message(no_signals_message)
            return

        try:
            message = "🚀 <b>НОВЫЕ ТОРГОВЫЕ СИГНАЛЫ</b>\n\n"
            message += "<i>Нажмите на данные, чтобы скопировать. Нажмите на биржу, чтобы перейти к торговле.</i>\n\n"

            for i, signal in enumerate(self.signals[:5]):  # Отправляем только топ-5 сигналов
                symbol_name = signal['symbol'].replace('/USDT', '')
                exchange_url = self.get_exchange_url(signal['exchange'], signal['symbol'])
                confidence_percent = signal['confidence'] * 100
                signal_emoji = "🟢" if signal['signal'] == 'LONG' else "🔴"
                formatted_exchange = self.format_exchange_name(signal['exchange'])

                message += (
                    f"{signal_emoji} <b>#{i + 1}: <a href='{exchange_url}'>{html.escape(formatted_exchange)}</a></b>\n"
                    f"<b>🪙 Монета:</b> <code>{html.escape(symbol_name)}</code>\n"
                    f"<b>📊 Сигнал:</b> <code>{html.escape(signal['signal'])}</code> <code>(Сила: {confidence_percent:.0f}%)</code>\n"
                    f"<b>💰 Цена:</b> <code>{signal['price']:.6f}</code>\n"
                    f"<b>🛑 Стоп-лосс:</b> <code>{signal['stop_loss']:.6f}</code>\n"
                    f"<b>🎯 Тейк-профит:</b> <code>{signal['take_profit']:.6f}</code>\n"
                    f"<b>⚖️ Размер:</b> <code>{signal['recommended_size']:.4f}</code>\n\n"
                )

            message += f"<b>⏱️ Время анализа:</b> {html.escape(self.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S'))}\n"
            message += "<b>⚡ Автоматическое обновление</b>"

            await self.send_telegram_message(message)

        except Exception as e:
            logger.error(f"Ошибка при автоматической отправке сигналов: {e}")

    def get_exchange_url(self, exchange_name: str, symbol: str) -> str:
        """Генерирует URL для торговой пары на бирже"""
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
        """Форматирует название биржи в правильный регистр"""
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
        """Проверяет, находится ли символ в черном списке"""
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
                    # Пропускаем пары из черного списка
                    if self.is_blacklisted(symbol):
                        continue

                    # Проверяем, что это фьючерсный рынок с USDT
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
                    # Пропускаем пары из черного списка
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

                # Применяем вес биржи к объемам
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
        top_symbols = [symbol for symbol, volume in sorted_symbols[:50]]

        logger.info(f"Отобрано топ {len(top_symbols)} пар для анализа")
        return top_symbols

    async def fetch_ohlcv_data(self, exchange_name: str, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        exchange = self.exchanges.get(exchange_name)
        if exchange is None:
            return None

        try:
            # Пропускаем пары из черного списка
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

            # Проверяем на наличие NaN значений
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

            # Пробуем различные варианты формата
            variations = [
                symbol,
                symbol.replace('/', ''),
                symbol.replace('/', ':'),
                symbol.replace('/', '-'),
                symbol.replace('/USDT', 'USDT'),
                symbol.replace('/USDT', '-USDT'),
                symbol.replace('/USDT', ':USDT'),
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

        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df

        return df

    def analyze_multiple_timeframes(self, dfs: dict) -> dict:
        """Анализ на нескольких таймфреймах"""
        timeframe_weights = {'1h': 0.4, '15m': 0.35, '5m': 0.25}
        analysis_results = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 20:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]

            tf_analysis = {
                'trend': 'neutral',
                'momentum': 'neutral',
                'volume': 'normal',
                'signals': []
            }

            # Анализ тренда
            if last['ema_21'] > last['ema_50'] > last['ema_200']:
                tf_analysis['trend'] = 'bullish'
            elif last['ema_21'] < last['ema_50'] < last['ema_200']:
                tf_analysis['trend'] = 'bearish'

            # Анализ импульса
            if last['rsi'] > 60 and last['macd'] > last['macd_signal']:
                tf_analysis['momentum'] = 'bullish'
            elif last['rsi'] < 40 and last['macd'] < last['macd_signal']:
                tf_analysis['momentum'] = 'bearish'

            # Анализ объема
            if last['volume_ratio'] > 1.8:
                tf_analysis['volume'] = 'high'
            elif last['volume_ratio'] < 0.5:
                tf_analysis['volume'] = 'low'

            # Сигналы
            if last['rsi'] < 35 and last['close'] < last['bb_lower']:
                tf_analysis['signals'].append(('oversold', 0.3))
            elif last['rsi'] > 65 and last['close'] > last['bb_upper']:
                tf_analysis['signals'].append(('overbought', 0.3))

            if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bullish', 0.4))
            elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                tf_analysis['signals'].append(('macd_bearish', 0.4))

            if last['stoch_k'] < 20 and last['stoch_d'] < 20:
                tf_analysis['signals'].append(('stoch_oversold', 0.2))
            elif last['stoch_k'] > 80 and last['stoch_d'] > 80:
                tf_analysis['signals'].append(('stoch_overbought', 0.2))

            analysis_results[tf] = tf_analysis

        return analysis_results

    def calculate_confidence_from_analysis(self, analysis_results: dict) -> float:
        """Расчет уверенности на основе анализа нескольких таймфреймов"""
        total_confidence = 0
        total_weight = 0
        signals_count = 0

        for tf, analysis in analysis_results.items():
            weight = {'1h': 0.4, '15m': 0.35, '5m': 0.25}.get(tf, 0.3)

            # Базовая уверенность от тренда
            if analysis['trend'] == 'bullish':
                total_confidence += 0.2 * weight
            elif analysis['trend'] == 'bearish':
                total_confidence -= 0.2 * weight

            # Уверенность от импульса
            if analysis['momentum'] == 'bullish':
                total_confidence += 0.3 * weight
            elif analysis['momentum'] == 'bearish':
                total_confidence -= 0.3 * weight

            # Сигналы
            for signal_name, signal_strength in analysis['signals']:
                if 'bull' in signal_name or 'overbought' in signal_name:
                    total_confidence += signal_strength * weight
                elif 'bear' in signal_name or 'oversold' in signal_name:
                    total_confidence -= signal_strength * weight
                signals_count += 1

            total_weight += weight

        # Нормализуем уверенность
        if total_weight > 0:
            confidence = total_confidence / total_weight
        else:
            confidence = 0

        # Увеличиваем уверенность при согласованных сигналах
        if signals_count >= 3 and abs(confidence) > 0.3:
            confidence *= 1.2

        return min(max(confidence, -1), 1)

    def calculate_stop_loss_take_profit(self, df: pd.DataFrame, signal_type: str, price: float) -> tuple:
        """Расчет стоп-лосса и тейк-профита на основе ATR и технических уровней"""
        try:
            # Используем ATR для расчета волатильности
            atr = df['atr'].iloc[-1]

            # Минимальный и максимальный стоп-лосс в процентах
            min_sl_percent = 0.005  # 0.5%
            max_sl_percent = 0.03  # 3%

            # Базовый стоп-лосс на основе ATR
            if signal_type == 'LONG':
                # Для лонга: стоп-лосс ниже текущей цены
                base_sl = price - (atr * self.config['atr_multiplier_sl'])
                base_tp = price + (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                # Проверяем поддержки как дополнительные уровни для стоп-лосса
                support_levels = self.find_support_levels(df)
                if support_levels:
                    # Берем ближайший уровень поддержки ниже текущей цены
                    closest_support = max([level for level in support_levels if level < price], default=None)
                    if closest_support and closest_support > base_sl:
                        base_sl = closest_support * 0.995  # Немного ниже уровня поддержки

                # Ограничиваем минимальный и максимальный стоп-лосс
                min_sl_price = price * (1 - max_sl_percent)
                max_sl_price = price * (1 - min_sl_percent)
                base_sl = max(base_sl, min_sl_price)
                base_sl = min(base_sl, max_sl_price)

            else:  # SHORT
                # Для шорта: стоп-лосс выше текущей цены
                base_sl = price + (atr * self.config['atr_multiplier_sl'])
                base_tp = price - (atr * self.config['atr_multiplier_tp'] * self.config['risk_reward_ratio'])

                # Проверяем сопротивления как дополнительные уровни для стоп-лосса
                resistance_levels = self.find_resistance_levels(df)
                if resistance_levels:
                    # Берем ближайший уровень сопротивления выше текущей цены
                    closest_resistance = min([level for level in resistance_levels if level > price], default=None)
                    if closest_resistance and closest_resistance < base_sl:
                        base_sl = closest_resistance * 1.005  # Немного выше уровня сопротивления

                # Ограничиваем минимальный и максимальный стоп-лосс
                min_sl_price = price * (1 + min_sl_percent)
                max_sl_price = price * (1 + max_sl_percent)
                base_sl = min(base_sl, max_sl_price)
                base_sl = max(base_sl, min_sl_price)

            return base_sl, base_tp

        except Exception as e:
            logger.error(f"Ошибка расчета стоп-лосса и тейк-профита: {e}")
            # Резервный расчет на основе процентов
            if signal_type == 'LONG':
                return price * 0.98, price * 1.04  # -2%, +4%
            else:
                return price * 1.02, price * 0.96  # +2%, -4%

    def find_support_levels(self, df: pd.DataFrame, lookback_period: int = 20) -> list:
        """Находит уровни поддержки на графике"""
        try:
            # Используем минимумы за последние периоды
            min_price = df['low'].rolling(window=lookback_period).min().iloc[-1]

            # Ищем кластеры цен вблизи минимумов
            support_levels = []
            for i in range(2, len(df) - 2):
                if (df['low'].iloc[i] < df['low'].iloc[i - 1] and
                        df['low'].iloc[i] < df['low'].iloc[i - 2] and
                        df['low'].iloc[i] < df['low'].iloc[i + 1] and
                        df['low'].iloc[i] < df['low'].iloc[i + 2]):
                    support_levels.append(df['low'].iloc[i])

            # Возвращаем уникальные уровни, отсортированные по убыванию
            return sorted(set(support_levels), reverse=True)[:3]  # Топ-3 уровня поддержки

        except Exception:
            return []

    def find_resistance_levels(self, df: pd.DataFrame, lookback_period: int = 20) -> list:
        """Находит уровни сопротивления на графике"""
        try:
            # Используем максимумы за последние периоды
            max_price = df['high'].rolling(window=lookback_period).max().iloc[-1]

            # Ищем кластеры цен вблизи максимумов
            resistance_levels = []
            for i in range(2, len(df) - 2):
                if (df['high'].iloc[i] > df['high'].iloc[i - 1] and
                        df['high'].iloc[i] > df['high'].iloc[i - 2] and
                        df['high'].iloc[i] > df['high'].iloc[i + 1] and
                        df['high'].iloc[i] > df['high'].iloc[i + 2]):
                    resistance_levels.append(df['high'].iloc[i])

            # Возвращаем уникальные уровни, отсортированные по возрастанию
            return sorted(set(resistance_levels))[:3]  # Топ-3 уровня сопротивления

        except Exception:
            return []

    def generate_trading_signal(self, dfs: dict, symbol: str, exchange_name: str) -> dict:
        if not dfs:
            return None

        try:
            # Анализ на нескольких таймфреймах
            analysis_results = self.analyze_multiple_timeframes(dfs)
            if not analysis_results:
                return None

            confidence = self.calculate_confidence_from_analysis(analysis_results)

            # Используем данные с основного таймфрейма (15m)
            main_df = dfs.get('15m', next(iter(dfs.values())))
            last = main_df.iloc[-1]

            signal = {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'price': last['close'],
                'signal': 'HOLD',
                'confidence': abs(confidence),
                'reasons': [],
                'recommended_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'timeframe_analysis': analysis_results
            }

            # Собираем причины
            reasons = []
            for tf, analysis in analysis_results.items():
                if analysis['signals']:
                    reasons.append(f"{tf}: {', '.join([s[0] for s in analysis['signals']])}")
                if analysis['trend'] != 'neutral':
                    reasons.append(f"{tf} trend: {analysis['trend']}")

            signal['reasons'] = reasons

            # Проверяем минимальную уверенность
            if abs(confidence) < self.config['min_confidence']:
                return None

            # Определяем направление сигнала
            if confidence > 0:
                signal['signal'] = 'LONG'
            else:
                signal['signal'] = 'SHORT'

            # Расчет стоп-лосса и тейк-профита
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                main_df, signal['signal'], signal['price']
            )

            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit

            # Расчет размера позиции
            risk_per_unit = abs(signal['price'] - signal['stop_loss'])
            if risk_per_unit > 0:
                risk_amount = self.config['virtual_balance'] * self.config['risk_per_trade']
                signal['recommended_size'] = round(risk_amount / risk_per_unit, 6)

            return signal

        except Exception as e:
            logger.error(f"Ошибка генерации сигнала для {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str) -> dict:
        best_signal = None
        best_confidence = 0

        for exchange_name, exchange in self.exchanges.items():
            if exchange is None:
                continue

            try:
                # Пропускаем пары из черного списка
                if self.is_blacklisted(symbol):
                    continue

                normalized_symbol = self.normalize_symbol_for_exchange(symbol, exchange_name)
                if not normalized_symbol:
                    continue

                # Загружаем данные для всех таймфреймов
                dfs = {}
                for timeframe in self.config['timeframes']:
                    df = await self.fetch_ohlcv_data(exchange_name, symbol, timeframe, limit=100)
                    if df is not None:
                        df = self.calculate_technical_indicators(df)
                        dfs[timeframe] = df
                    await asyncio.sleep(0.02)

                if not dfs:
                    continue

                signal = self.generate_trading_signal(dfs, symbol, exchange_name)

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

        self.top_symbols = await self.fetch_top_symbols()
        self.analysis_stats['total_analyzed'] = len(self.top_symbols)

        if not self.top_symbols:
            logger.warning("Не найдено символов для анализа")
            return []

        # Ограничиваем количество символов для анализа
        symbols_to_analyze = self.top_symbols[:40]

        tasks = []
        for symbol in symbols_to_analyze:
            task = asyncio.create_task(self.analyze_symbol(symbol))
            tasks.append(task)

        # Обрабатываем батчами по 8 символов
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

            # Небольшая пауза между батчами
            await asyncio.sleep(1)

        # Сортируем сигналы по уверенности
        self.signals = sorted(all_signals, key=lambda x: x['confidence'], reverse=True)
        self.analysis_stats['signals_found'] = len(self.signals)

        analysis_time = time.time() - start_time
        logger.info(f"Анализ завершен за {analysis_time:.1f} сек. Найдено {len(self.signals)} сигналов")

        # Автоматически отправляем сигналы, если они есть
        await self.send_automatic_signals()

        return self.signals

    def print_signals(self, max_signals: int = 15):
        if not self.signals:
            print("🚫 Нет торговых сигналов")
            return

        print("\n" + "=" * 140)
        print("🎯 ТОРГОВЫЕ СИГНАЛЫ НА ФЬЮЧЕРСЫ")
        print("=" * 140)
        print(
            f"{'Ранг':<4} {'Биржа':<8} {'Пара':<12} {'Сигнал':<8} {'Уверенность':<10} {'Цена':<12} {'R/R':<6} {'Причины'}")
        print("-" * 140)

        for i, signal in enumerate(self.signals[:max_signals]):
            rank = f"{i + 1}"
            exchange = self.format_exchange_name(signal['exchange'])[:8]
            symbol = signal['symbol'].replace('/USDT', '')[:12]
            signal_type = signal['signal'][:8]
            confidence = f"{signal['confidence'] * 100:.0f}%"
            price = f"{signal['price']:.6f}"
            rr_ratio = f"{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}"

            # Берем первые 2 причины
            reasons = ', '.join(signal['reasons'][:2]) if signal['reasons'] else 'N/A'

            print(
                f"{rank:<4} {exchange:<8} {symbol:<12} {signal_type:<8} {confidence:<10} {price:<12} {rr_ratio:<6} {reasons}")

        print("=" * 140)

        # Детали для топ-3 сигналов
        for i, signal in enumerate(self.signals[:3]):
            print(
                f"\n🔥 ТОП-{i + 1}: {signal['symbol'].replace('/USDT', '')} на {self.format_exchange_name(signal['exchange'])}")
            print(f"📊 Сигнал: {signal['signal']} ({signal['confidence'] * 100:.0f}% уверенности)")
            print(f"💰 Цена: {signal['price']:.8f}")
            print(f"🛑 Стоп-лосс: {signal['stop_loss']:.8f}")
            print(f"🎯 Тейк-профит: {signal['take_profit']:.8f}")
            print(f"⚖️ Размер позиции: {signal['recommended_size']:.6f}")
            print(
                f"📈 R/R соотношение: 1:{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}")
            if signal['reasons']:
                print(f"🔍 Причины: {', '.join(signal['reasons'][:3])}")

    async def run_continuous(self):
        """Бесконечный цикл анализа"""
        analysis_count = 0

        while True:
            try:
                analysis_count += 1
                print(f"\n{'=' * 80}")
                print(f"📊 АНАЛИЗ #{analysis_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'=' * 80}")

                start_time = time.time()
                await self.run_analysis()
                self.last_analysis_time = datetime.now()
                self.last_signals = self.signals.copy()

                if self.signals:
                    self.print_signals()
                else:
                    print("🚫 Сигналов не найдено")

                execution_time = time.time() - start_time
                print(f"\n⏱️ Время анализа: {execution_time:.1f} секунд")
                print(
                    f"📈 Статистика: {self.analysis_stats['total_analyzed']} пар, {self.analysis_stats['signals_found']} сигналов")

                # Расчет времени до следующего анализа
                wait_time = max(self.config['analysis_interval'] - execution_time, 30)
                next_analysis_time = datetime.now().timestamp() + wait_time
                next_time_str = datetime.fromtimestamp(next_analysis_time).strftime("%H:%M:%S")

                print(f"⏭️ Следующий анализ в {next_time_str} (через {wait_time:.0f} секунд)")
                print("📊 Ожидание следующего анализа..." + " " * 40, end='\r')

                # Ожидание до следующего анализа с прогресс-баром
                for sec in range(int(wait_time)):
                    try:
                        progress = (sec + 1) / wait_time * 50
                        bar = "█" * int(progress) + "░" * (50 - int(progress))
                        remaining = wait_time - sec - 1
                        print(f"⏳ Ожидание: [{bar}] {remaining}сек осталось", end='\r')
                        await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        await asyncio.sleep(1)

                print(" " * 80, end='\r')  # Очистка строки

            except KeyboardInterrupt:
                print("\n\n🛑 Бот остановлен пользователем")
                break
            except Exception as e:
                print(f"\n❌ Ошибка в основном цикле: {e}")
                print("🔄 Повторная попытка через 60 секунд...")
                await asyncio.sleep(60)


async def main():
    bot = FuturesTradingBot()

    try:
        print("🚀 Запуск улучшенного торгового бота с поддержкой 9 бирж!")
        print("📊 Поддерживаемые биржи: Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, HTX, BingX, Phemex")
        print("⚡ Мультитаймфрейм анализ (1h, 15m, 5m)")
        print("📈 Технические индикаторы: RSI, MACD, Bollinger Bands, EMA, Volume, Stochastic, ADX")
        print(
            f"⚙️ Настройки: мин. уверенность {bot.config['min_confidence'] * 100}%, R/R=1:{bot.config['risk_reward_ratio']}")
        print("⏸️ Для остановки нажмите Ctrl+C\n")

        # Инициализируем Telegram бота
        await bot.initialize_telegram()

        # Сразу выполняем первый анализ при запуске
        print("📈 Выполняю первоначальный анализ...")
        await bot.run_analysis()
        bot.last_analysis_time = datetime.now()
        bot.last_signals = bot.signals.copy()

        if bot.signals:
            bot.print_signals()
        else:
            print("📊 Сигналов не найдено")

        # Запускаем бесконечный цикл анализа
        await bot.run_continuous()

    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
    finally:
        # Останавливаем Telegram бота при завершении
        if bot.telegram_app:
            await bot.telegram_app.updater.stop()
            await bot.telegram_app.stop()
            await bot.telegram_app.shutdown()
        if bot.telegram_worker_task:
            bot.telegram_worker_task.cancel()


if __name__ == "__main__":
    # Бесконечный запуск с перезапуском при ошибках
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Программа завершена")
            break
        except Exception as e:
            print(f"🔄 Перезапуск после критической ошибки: {e}")
            time.sleep(10)
