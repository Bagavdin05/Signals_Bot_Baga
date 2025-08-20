import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
import asyncio
import talib
import warnings
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

warnings.filterwarnings('ignore')

# Настройка логирования только в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FuturesBot')

# Конфигурация Telegram бота
TELEGRAM_BOT_TOKEN = "8328135138:AAE5mLIWG59kM8STODbfPoLkd19iykbOmcM"  # Замените на ваш токен
TELEGRAM_CHAT_ID = "1167694150", "7916502470", "5381553894"  # Замените на ваш chat ID


class FuturesTradingBot:
    def __init__(self):
        self.exchanges = self.initialize_exchanges()
        self.telegram_app = None
        self.last_analysis_time = None
        self.last_signals = []

        self.config = {
            'timeframes': ['15m', '1h'],  # анализируемые таймфреймы
            'min_volume_24h': 1000000,  # минимальный объем для анализа
            'max_symbols_per_exchange': 50,  # максимальное количество пар с биржи
            'analysis_interval': 150,  # интервал анализа в секундах
            'risk_per_trade': 0.02,  # риск на сделку (2% от депозита)
            'virtual_balance': 10000,  # виртуальный баланс для расчета позиции
            'timeout': 10000,
            'min_confidence': 0.50,  # минимальная уверенность для сигнала
            'risk_reward_ratio': 2.0,  # соотношение риск/вознаграждение
            'atr_multiplier_sl': 1.5,  # множитель ATR для стоп-лосса
            'atr_multiplier_tp': 2.0,  # множитель ATR для тейк-профита
            'blacklist': ['USDC/USDT', 'USDC/USD', 'USDCE/USDT', 'USDCB/USDT'],  # черный список пар
            'signal_validity_seconds': 30  # время валидности сигналов
        }

        self.top_symbols = []
        self.signals = []

        logger.info("Торговый бот инициализирован")

    async def initialize_telegram(self):
        """Инициализация Telegram бота"""
        try:
            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

            # Добавляем обработчики
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
            self.telegram_app.add_handler(CommandHandler("signals", self.telegram_signals))
            self.telegram_app.add_handler(MessageHandler(filters.Text(["🔄 Обновить сигналы"]), self.telegram_signals))

            # Запускаем бота в фоновом режиме
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()

            logger.info("Telegram бот инициализирован")

        except Exception as e:
            logger.error(f"Ошибка инициализации Telegram бота: {e}")

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        keyboard = [["🔄 Обновить сигналы"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

        welcome_text = (
            "🚀 Торговый бот для фьючерсов\n\n"
            "Поддерживаемые биржи: Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, Huobi, Phemex, BingX\n\n"
            "Нажмите кнопку ниже для получения актуальных сигналов:"
        )

        await update.message.reply_text(welcome_text, reply_markup=reply_markup)

    def escape_markdown(self, text: str) -> str:
        """Экранирует специальные символы Markdown"""
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return ''.join(['\\' + char if char in escape_chars else char for char in text])

    async def telegram_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Отправка сигналов в Telegram"""
        try:
            # Проверяем актуальность сигналов
            current_time = datetime.now()
            if (self.last_analysis_time is None or
                    (current_time - self.last_analysis_time).total_seconds() > self.config['signal_validity_seconds']):
                await update.message.reply_text("⏳ Анализирую рынок...")
                await self.run_analysis()
                self.last_analysis_time = current_time
                self.last_signals = self.signals.copy()

            # Отправляем сигналы
            if not self.last_signals:
                await update.message.reply_text("📊 Сигналов не найдено")
                return

            message = "📈 *АКТУАЛЬНЫЕ ТОРГОВЫЕ СИГНАЛЫ*\n\n"

            for i, signal in enumerate(self.last_signals[:10]):  # Ограничиваем 10 сигналами
                escaped_symbol = self.escape_markdown(signal['symbol'])
                escaped_exchange = self.escape_markdown(signal['exchange'])
                escaped_reasons = self.escape_markdown(', '.join(signal['reasons']))

                message += (
                    f"*Сигнал #{i + 1}:* {escaped_symbol} на {escaped_exchange}\n"
                    f"*Сигнал:* {self.escape_markdown(signal['signal'])} (уверенность: {signal['confidence']:.2f})\n"
                    f"*Цена:* {signal['price']:.8f}\n"
                    f"*Стоп-лосс:* {signal['stop_loss']:.8f}\n"
                    f"*Тейк-профит:* {signal['take_profit']:.8f}\n"
                    f"*Размер позиции:* {signal['recommended_size']:.6f}\n"
                    f"*Причины:* {escaped_reasons}\n\n"
                )

            # Добавляем время анализа
            message += f"*Время анализа:* {self.escape_markdown(self.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S'))}\n"
            message += "*Используйте кнопку ниже для обновления*"

            # Отправляем сообщение
            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            error_msg = f"❌ Ошибка при отправке сигналов: {e}"
            await update.message.reply_text(error_msg)
            logger.error(error_msg)

    def initialize_exchanges(self) -> dict:
        exchanges = {}

        exchange_configs = {
            'bybit': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'mexc': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'okx': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'gateio': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'bitget': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'kucoin': {'options': {'defaultType': 'future'}, 'timeout': 10000},
            'huobi': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
            'phemex': {'options': {'defaultType': 'swap'}, 'timeout': 10000},
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

        # Добавляем BingX через специальную обработку
        exchanges['bingx'] = self.initialize_bingx()

        return exchanges

    def initialize_bingx(self):
        """Инициализация BingX с использованием обходных методов"""
        try:
            # Создаем обычный экземпляр
            bingx = ccxt.bingx({
                'timeout': 10000,
                'enableRateLimit': True,
            })

            # Пробуем загрузить рынки
            bingx.load_markets()
            logger.info("Успешное подключение к BingX")
            return bingx

        except Exception as e:
            logger.error(f"Ошибка подключения к BingX: {e}")

            # Пробуем альтернативный подход для BingX
            try:
                # Создаем экземпляр с другими настройками
                bingx = ccxt.bingx({
                    'timeout': 15000,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True,
                    }
                })

                # Пробуем получить только основные пары
                markets = bingx.load_markets()
                logger.info(f"BingX: загружено {len(markets)} рынков")
                return bingx

            except Exception as e2:
                logger.error(f"Вторая попытка подключения к BingX также failed: {e2}")
                return None

    def is_blacklisted(self, symbol: str) -> bool:
        """Проверяет, находится ли символ в черном списке"""
        for blacklisted_symbol in self.config['blacklist']:
            if blacklisted_symbol in symbol:
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

                    # Общая логика для большинства бирж
                    if (market.get('swap', False) or market.get('future', False) or
                            'swap' in symbol.lower() or 'future' in symbol.lower() or
                            '/USDT:' in symbol or symbol.endswith('USDT')):

                        if 'USDT' in symbol and not symbol.startswith('.'):
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
            if exchange_name == 'bybit':
                for symbol in symbols:
                    try:
                        # Пропускаем пары из черного списка
                        if self.is_blacklisted(symbol):
                            continue

                        ticker = exchange.fetch_ticker(symbol)
                        volume = ticker.get('quoteVolume', 0)
                        if volume and volume > self.config['min_volume_24h']:
                            normalized_symbol = symbol.replace(':', '/').replace('-', '/')
                            volume_map[normalized_symbol] = volume
                        await asyncio.sleep(0.02)
                    except Exception:
                        continue
            elif exchange_name == 'bingx':
                # Специальная обработка для BingX
                for symbol in symbols:
                    try:
                        # Пропускаем пары из черного списка
                        if self.is_blacklisted(symbol):
                            continue

                        ticker = exchange.fetch_ticker(symbol)
                        volume = ticker.get('quoteVolume', ticker.get('baseVolume', 0))
                        if volume and volume > self.config['min_volume_24h']:
                            volume_map[symbol] = volume
                        await asyncio.sleep(0.03)
                    except Exception:
                        continue
            else:
                try:
                    tickers = exchange.fetch_tickers(symbols)
                    for symbol, ticker in tickers.items():
                        # Пропускаем пары из черного списка
                        if self.is_blacklisted(symbol):
                            continue

                        volume = ticker.get('quoteVolume', 0)
                        if volume and volume > self.config['min_volume_24h']:
                            normalized_symbol = symbol.replace(':', '/').replace('-', '/')
                            volume_map[normalized_symbol] = volume
                except Exception:
                    for symbol in symbols:
                        try:
                            # Пропускаем пары из черного списка
                            if self.is_blacklisted(symbol):
                                continue

                            ticker = exchange.fetch_ticker(symbol)
                            volume = ticker.get('quoteVolume', 0)
                            if volume and volume > self.config['min_volume_24h']:
                                normalized_symbol = symbol.replace(':', '/').replace('-', '/')
                                volume_map[normalized_symbol] = volume
                            await asyncio.sleep(0.02)
                        except Exception:
                            continue

        except Exception as e:
            logger.error(f"Ошибка получения данных объема с {exchange_name}: {e}")

        return volume_map

    async def fetch_top_symbols(self) -> list:
        all_volume_map = {}

        for exchange_name, exchange in self.exchanges.items():
            if exchange is None:
                continue

            try:
                futures_symbols = self.get_futures_symbols(exchange, exchange_name)

                if not futures_symbols:
                    continue

                volume_map = await self.fetch_exchange_volume_data(exchange, exchange_name, futures_symbols)
                all_volume_map.update(volume_map)

                logger.info(f"С {exchange_name} обработано {len(volume_map)} пар с объемом")

            except Exception as e:
                logger.error(f"Ошибка получения данных с {exchange_name}: {e}")
                continue

        sorted_symbols = sorted(all_volume_map.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, volume in sorted_symbols[:70]]

        logger.info(f"Отобрано топ {len(top_symbols)} пар для анализа")
        return top_symbols

    async def fetch_ohlcv_data(self, exchange_name: str, symbol: str, timeframe: str, limit: int = 50) -> pd.DataFrame:
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

            await asyncio.sleep(np.random.uniform(0.01, 0.03))

            ohlcv = exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) < 20:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
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
                symbol,
                symbol.replace('/', ''),
                symbol.replace('/', ':'),
                symbol.replace('/', '-'),
                symbol + 'USDT',
                symbol.replace('USDT', ''),
                symbol.replace('/', '') + 'T',
            ]

            for variation in variations:
                if variation in exchange.symbols:
                    return variation

            # Для BingX пробуем найти похожие символы
            if exchange_name == 'bingx':
                for exchange_symbol in exchange.symbols:
                    if symbol.replace('/', '').replace('-', '').replace(':', '') in exchange_symbol:
                        return exchange_symbol

            return None

        except Exception:
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 20:
            return df

        try:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)

            macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal

            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower

            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)

            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)

            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        except Exception:
            return df

        return df

    def generate_trading_signal(self, df: pd.DataFrame, symbol: str, exchange_name: str) -> dict:
        if df is None or len(df) < 20:
            return None

        try:
            last = df.iloc[-1]

            signal = {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'price': last['close'],
                'signal': 'HOLD',
                'confidence': 0,
                'reasons': [],
                'recommended_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
            }

            reasons = []
            confidence = 0

            # RSI анализ
            if not pd.isna(last['rsi']):
                if last['rsi'] < 30:
                    confidence += 0.3
                    reasons.append("RSI < 30")
                elif last['rsi'] > 70:
                    confidence -= 0.3
                    reasons.append("RSI > 70")

            # MACD анализ
            if not pd.isna(last['macd']) and not pd.isna(last['macd_signal']):
                if last['macd'] > last['macd_signal']:
                    confidence += 0.2
                    reasons.append("MACD bullish")
                else:
                    confidence -= 0.2
                    reasons.append("MACD bearish")

            # Bollinger Bands анализ
            if (not pd.isna(last['bb_upper']) and not pd.isna(last['bb_lower'])):
                if last['close'] < last['bb_lower']:
                    confidence += 0.2
                    reasons.append("Below BB")
                elif last['close'] > last['bb_upper']:
                    confidence -= 0.2
                    reasons.append("Above BB")

            # EMA анализ
            if not pd.isna(last['ema_20']) and not pd.isna(last['ema_50']):
                if last['close'] > last['ema_20'] > last['ema_50']:
                    confidence += 0.2
                    reasons.append("EMA uptrend")
                elif last['close'] < last['ema_20'] < last['ema_50']:
                    confidence -= 0.2
                    reasons.append("EMA downtrend")

            # Volume анализ
            if not pd.isna(last['volume_ratio']):
                if last['volume_ratio'] > 1.5:
                    confidence += 0.1 if last['close'] > df['close'].iloc[-2] else -0.1
                    reasons.append(f"Volume x{last['volume_ratio']:.1f}")

            signal['reasons'] = reasons
            signal['confidence'] = abs(confidence)  # Убираем минус для отображения

            # Проверяем минимальную уверенность
            if abs(confidence) < self.config['min_confidence']:
                return None

            # Определяем сигнал
            if confidence >= 0.6:
                signal['signal'] = 'LONG'
            elif confidence >= self.config['min_confidence']:
                signal['signal'] = 'WEAK_LONG'
            elif confidence <= -0.6:
                signal['signal'] = 'SHORT'
            elif confidence <= -self.config['min_confidence']:
                signal['signal'] = 'WEAK_SHORT'

            # Расчет параметров сделки
            if signal['signal'] != 'HOLD':
                atr = last['atr'] if not pd.isna(last['atr']) else last['close'] * 0.02
                price = signal['price']

                if 'LONG' in signal['signal']:
                    signal['stop_loss'] = price - (atr * self.config['atr_multiplier_sl'])
                    signal['take_profit'] = price + (atr * self.config['atr_multiplier_tp'])
                else:
                    signal['stop_loss'] = price + (atr * self.config['atr_multiplier_sl'])
                    signal['take_profit'] = price - (atr * self.config['atr_multiplier_tp'])

                risk_per_unit = abs(price - signal['stop_loss'])
                if risk_per_unit > 0:
                    risk_amount = self.config['virtual_balance'] * self.config['risk_per_trade']
                    signal['recommended_size'] = round(risk_amount / risk_per_unit, 4)

            return signal if signal['signal'] != 'HOLD' else None

        except Exception:
            return None

    async def analyze_symbol(self, symbol: str) -> dict:
        best_signal = None

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

                df = await self.fetch_ohlcv_data(exchange_name, symbol, self.config['timeframes'][0])
                if df is None:
                    continue

                df = self.calculate_technical_indicators(df)
                signal = self.generate_trading_signal(df, symbol, exchange_name)

                if signal and (best_signal is None or signal['confidence'] > best_signal['confidence']):
                    best_signal = signal

            except Exception:
                continue

        return best_signal

    async def run_analysis(self):
        logger.info("Начало анализа торговых пар...")

        self.top_symbols = await self.fetch_top_symbols()

        if not self.top_symbols:
            logger.warning("Не найдено символов для анализа")
            return []

        tasks = []
        for symbol in self.top_symbols:
            task = asyncio.create_task(self.analyze_symbol(symbol))
            tasks.append(task)

        batch_size = 15
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

                logger.info(f"Обработано {min(i + batch_size, len(tasks))}/{len(tasks)} символов")

            except Exception:
                continue

        self.signals = sorted(all_signals, key=lambda x: x['confidence'], reverse=True)

        logger.info(f"Анализ завершен. Найдено {len(self.signals)} сигналов")
        return self.signals

    def print_signals(self, max_signals: int = 10):
        if not self.signals:
            print("Нет торговых сигналов")
            return

        print("\n" + "=" * 120)
        print("ТОРГОВЫЕ СИГНАЛЫ НА ФЬЮЧЕРСЫ")
        print("=" * 120)
        print(f"{'Время':<15} {'Биржа':<10} {'Пара':<15} {'Сигнал':<12} {'Цена':<12} {'Conf':<6} {'R/R':<5}")
        print("-" * 120)

        for signal in self.signals[:max_signals]:
            time_str = signal['timestamp'].strftime("%H:%M:%S")
            exchange = signal['exchange'][:10]
            symbol = signal['symbol'].replace('/USDT', '')[:12]
            signal_type = signal['signal'][:12]
            price = f"{signal['price']:.6f}"
            confidence = f"{signal['confidence']:.2f}"
            rr_ratio = f"{abs(signal['take_profit'] - signal['price']) / abs(signal['price'] - signal['stop_loss']):.1f}"

            print(
                f"{time_str:<15} {exchange:<10} {symbol:<15} {signal_type:<12} {price:<12} {confidence:<6} {rr_ratio:<5}")

        print("=" * 120)

        for i, signal in enumerate(self.signals[:10]):
            print(f"\nСигнал #{i + 1}: {signal['symbol']} на {signal['exchange']}")
            print(f"Сигнал: {signal['signal']} (уверенность: {signal['confidence']:.2f})")
            print(f"Цена: {signal['price']:.8f}")
            print(f"Стоп-лосс: {signal['stop_loss']:.8f}")
            print(f"Тейк-профит: {signal['take_profit']:.8f}")
            print(f"Размер позиции: {signal['recommended_size']:.6f}")
            print("Причины: " + ", ".join(signal['reasons']))

    async def run_continuous(self):
        """Бесконечный цикл анализа"""
        analysis_count = 0

        while True:
            try:
                analysis_count += 1
                print(f"\n{'=' * 60}")
                print(f"АНАЛИЗ #{analysis_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'=' * 60}")

                start_time = time.time()
                await self.run_analysis()

                if self.signals:
                    self.print_signals()
                else:
                    print("🚫 Сигналов не найдено")

                execution_time = time.time() - start_time
                print(f"\n⏱️ Время анализа: {execution_time:.1f} секунд")

                # Расчет времени до следующего анализа
                wait_time = max(self.config['analysis_interval'] - execution_time, 30)
                next_analysis_time = datetime.now().timestamp() + wait_time
                next_time_str = datetime.fromtimestamp(next_analysis_time).strftime("%H:%M:%S")

                print(f"⏭️ Следующий анализ в {next_time_str} (через {wait_time:.0f} секунд)")
                print("📊 Ожидание следующего анализа..." + " " * 30, end='\r')

                # Ожидание до следующего анализа с прогресс-баром
                for sec in range(int(wait_time)):
                    try:
                        progress = (sec + 1) / wait_time * 50
                        bar = "█" * int(progress) + "░" * (50 - int(progress))
                        print(f"⏳ Ожидание: [{bar}] {sec + 1}/{int(wait_time)}сек", end='\r')
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
        print("🚀 Запуск торгового бота с поддержкой 9 бирж!")
        print("📊 Поддерживаемые биржи: Bybit, MEXC, OKX, Gate.io, Bitget, KuCoin, Huobi, Phemex, BingX")
        print(
            f"⚙️ Настройки: мин. уверенность {bot.config['min_confidence']}, SL={bot.config['atr_multiplier_sl']}ATR, TP={bot.config['atr_multiplier_tp']}ATR")
        print("⏸️ Для остановки нажмите Ctrl+C\n")

        # Инициализируем Telegram бота
        await bot.initialize_telegram()

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
