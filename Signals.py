import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import talib
from threading import Thread
import logging
from telegram import Bot
from telegram.error import TelegramError
import asyncio
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CryptoScannerBot:
    # ============= КОНФИГУРАЦИЯ БОТА =============
    EXCHANGES = ['bybit', 'mexc', 'okx', 'gate', 'bitget', 'kucoin', 'htx', 'bingx', 'phemex']
    TIMEFRAMES = ['15min', '1h', '4h']
    MIN_VOLUME = 1000000  # Минимальный объем торгов в USDT
    SYMBOLS_PER_EXCHANGE = 500  # Количество топовых пар (по объему) для анализа на каждой бирже.
    SCAN_INTERVAL = 120  # Увеличил интервал сканирования до 120 секунд
    SIGNAL_STRENGTH_THRESHOLD = 2  # Минимальная "сила" сигнала для фильтрации

    # Параметры индикаторов
    RSI_OVERBOUGHT = 70  # Порог перекупленности для RSI
    RSI_OVERSOLD = 30  # Порог перепроданности для RSI
    ADX_STRONG_TREND = 25  # Минимальное значение ADX для сильного тренда
    BBANDS_PERIOD = 20  # Длина периода для расчета линий Боллинджера.

    # Настройки риска
    RISK_PER_TRADE = 0.02  # 2% от депозита на сделку
    REWARD_RATIO = 2.5  # Соотношение прибыли к риску 2.5:1
    MIN_PROFIT_PERCENT = 0.5  # Минимальная прибыль в процентах от входа
    MAX_POSITION_SIZE = 0.1  # 10% от депозита на одну позицию

    # Настройки Telegram
    TELEGRAM_TOKEN = "8328135138:AAE5mLIWG59kM8STODbfPoLkd19iykbOmcM"
    TELEGRAM_CHAT_ID = "1167694150, 7916502470, 5381553894"

    # Настройки депозита (обновленные значения)
    MIN_DEPOSIT = 5       # Минимальный депозит $
    MAX_DEPOSIT = 1000   # Максимальный депозит $

    def __init__(self, deposit=100):
        if not self.MIN_DEPOSIT <= deposit <= self.MAX_DEPOSIT:
            raise ValueError(f"Депозит должен быть между {self.MIN_DEPOSIT} и {self.MAX_DEPOSIT} USDT")

        self.signals = []
        self.running = True
        self.telegram_bot = Bot(token=self.TELEGRAM_TOKEN)
        self.loop = asyncio.new_event_loop()
        self.deposit = deposit  # Стартовый депозит в USDT
        logger.info(f"Бот инициализирован с депозитом: {self.deposit} USDT")

    def get_fallback_symbols(self):
        """Резервный список символов при ошибке"""
        common_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
                          'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT',
                          'AVAX/USDT', 'LINK/USDT', 'ATOM/USDT', 'UNI/USDT', 'XLM/USDT',
                          'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'FIL/USDT', 'TRX/USDT',
                          'TON/USDT', 'ARB/USDT', 'OP/USDT', 'APT/USDT', 'SUI/USDT']
        return common_symbols[:self.SYMBOLS_PER_EXCHANGE]

    def set_deposit(self, new_deposit):
        """Установка нового значения депозита с проверкой диапазона"""
        if not self.MIN_DEPOSIT <= new_deposit <= self.MAX_DEPOSIT:
            logger.error(
                f"Ошибка: Депозит {new_deposit} вне допустимого диапазона ({self.MIN_DEPOSIT}-{self.MAX_DEPOSIT})")
            return False

        self.deposit = new_deposit
        logger.info(f"Депозит успешно изменен на {self.deposit} USDT")
        return True

    def calculate_position_size(self, entry_price, stop_loss):
        """Рассчитываем размер позиции с учетом риска"""
        risk_amount = self.deposit * self.RISK_PER_TRADE
        risk_per_unit = abs(entry_price - stop_loss)
        position_size = risk_amount / risk_per_unit

        # Проверяем максимальный размер позиции
        max_size = (self.deposit * self.MAX_POSITION_SIZE) / entry_price
        return min(position_size, max_size)

    def calculate_tp_sl(self, entry_price, stop_loss, direction):
        """Рассчитываем тейк-профит и стоп-лосс с учетом соотношения прибыли к риску"""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.REWARD_RATIO

        if direction == 'ЛОНГ':
            take_profit = entry_price + reward
        else:  # ШОРТ
            take_profit = entry_price - reward

        return take_profit, stop_loss

    def normalize_symbol(self, symbol, exchange):
        """Приводим символы к единому формату: BTC/USDT"""
        # Специальная обработка для Phemex
        if exchange == 'phemex':
            if symbol.endswith('USDT'):
                return f"{symbol[:-4]}/USDT"
            return symbol

        # Общий случай для других бирж
        clean_symbol = re.sub(r'[^a-zA-Z]', '', symbol)

        # Обработка для BingX
        if exchange == 'bingx' and 'USDT' in clean_symbol:
            base = clean_symbol.split('USDT')[0]
            return f"{base}/USDT"

        if clean_symbol.endswith('USDT'):
            base = clean_symbol[:-4]
            return f"{base}/USDT"
        return symbol

    def get_top_symbols(self, exchange):
        """Получаем топ символов по объему с реальных бирж"""
        try:
            symbols = []
            if exchange == 'bybit':
                url = "https://api.bybit.com/v5/market/tickers?category=spot"
                response = requests.get(url, timeout=15)
                data = response.json()
                if data['retCode'] == 0:
                    tickers = [t for t in data['result']['list'] if 'USDT' in t['symbol']]
                    tickers.sort(key=lambda x: float(x['turnover24h']), reverse=True)
                    symbols = [self.normalize_symbol(t['symbol'], exchange) for t in
                               tickers[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'mexc':
                url = "https://api.mexc.com/api/v3/ticker/24hr"
                response = requests.get(url, timeout=15)
                data = response.json()
                usdt_pairs = [t for t in data if t['symbol'].endswith('USDT')]
                usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                symbols = [self.normalize_symbol(t['symbol'], exchange) for t in usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'okx':
                url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
                response = requests.get(url, timeout=15)
                data = response.json()
                if data['code'] == '0':
                    usdt_pairs = [t for t in data['data'] if t['instId'].endswith('USDT')]
                    usdt_pairs.sort(key=lambda x: float(x['volCcy24h']), reverse=True)
                    symbols = [self.normalize_symbol(t['instId'], exchange) for t in
                               usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'gate':
                url = "https://api.gateio.ws/api/v4/spot/tickers"
                response = requests.get(url, timeout=15)
                data = response.json()
                usdt_pairs = [t for t in data if t['currency_pair'].endswith('_USDT')]
                usdt_pairs.sort(key=lambda x: float(x['quote_volume']), reverse=True)
                symbols = [self.normalize_symbol(t['currency_pair'].replace('_', ''), exchange) for t in
                           usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'bitget':
                url = "https://api.bitget.com/api/spot/v1/market/tickers"
                response = requests.get(url, timeout=15)
                data = response.json()['data']
                usdt_pairs = [t for t in data if t['symbol'].endswith('USDT')]
                usdt_pairs.sort(key=lambda x: float(x['usdtVol']), reverse=True)
                symbols = [self.normalize_symbol(t['symbol'], exchange) for t in usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'kucoin':
                url = "https://api.kucoin.com/api/v1/market/allTickers"
                response = requests.get(url, timeout=15)
                data = response.json()['data']['ticker']
                usdt_pairs = [t for t in data if t['symbol'].endswith('USDT')]
                usdt_pairs.sort(key=lambda x: float(x['volValue']), reverse=True)
                symbols = [self.normalize_symbol(t['symbol'], exchange) for t in usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'htx':
                url = "https://api.huobi.pro/market/tickers"
                response = requests.get(url, timeout=15)
                data = response.json()['data']
                usdt_pairs = [t for t in data if t['symbol'].endswith('usdt')]
                usdt_pairs.sort(key=lambda x: float(x['vol']), reverse=True)
                symbols = [self.normalize_symbol(t['symbol'], exchange).upper() for t in
                           usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]]

            elif exchange == 'bingx':
                url = "https://api.bingx.com/openApi/spot/v1/ticker/24hr"
                response = requests.get(url, timeout=15)
                data = response.json()

                # Проверка кода ответа
                if data.get('code') != 0 or 'data' not in data:
                    logger.error(f"BingX API error: {data.get('msg', 'Unknown error')}")
                    return self.get_fallback_symbols()

                # Фильтрация и обработка пар
                usdt_pairs = [t for t in data['data'] if t['symbol'].endswith('-USDT')]
                usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                symbols = [
                    self.normalize_symbol(t['symbol'].replace('-', '/'), exchange)
                    for t in usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]
                ]

            elif exchange == 'phemex':
                # Используем обновленный URL согласно документации
                url = "https://api.phemex.com/md/spot/ticker/24hr/all"
                response = requests.get(url, timeout=15)

                if response.status_code != 200:
                    logger.error(f"Phemex API status code: {response.status_code}")
                    return self.get_fallback_symbols()

                data = response.json()

                # Проверка структуры ответа
                if 'result' not in data or 'data' not in data['result']:
                    logger.error(f"Phemex API error: Unexpected response format")
                    return self.get_fallback_symbols()

                # Фильтрация USDT пар и обработка объема
                usdt_pairs = []
                for t in data['result']['data']:
                    if 'symbol' in t and t['symbol'].endswith('USDT'):
                        try:
                            # Конвертация turnoverEv в объем USDT
                            turnover_ev = int(t.get('turnoverEv', 0))
                            volume = turnover_ev / 1e6  # Предполагаем, что 1 USDT = 10^6
                            usdt_pairs.append({
                                'symbol': t['symbol'],
                                'volume': volume
                            })
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Phemex volume conversion error: {e}")

                usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
                symbols = [
                    self.normalize_symbol(t['symbol'], exchange)
                    for t in usdt_pairs[:self.SYMBOLS_PER_EXCHANGE]
                ]

            logger.info(f"Успешно получено {len(symbols)} пар с {exchange}")
            return symbols

        except Exception as e:
            logger.error(f"Ошибка получения топовых пар с {exchange}: {str(e)[:200]}")
            return self.get_fallback_symbols()

    def get_ohlcv_data(self, exchange, symbol, timeframe='1H', limit=100):
        """Получаем OHLCV данные с биржи (реальные или имитация)"""
        try:
            # Для простоты оставим генерацию случайных данных
            # В реальном боте здесь будет запрос к API биржи
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)

            base = np.random.uniform(100, 500)
            noise = np.random.normal(0, 5, limit)
            close_prices = base + np.cumsum(noise)
            close_prices = np.abs(close_prices)

            open_prices = close_prices - np.random.uniform(0.1, 1, limit)
            high_prices = close_prices + np.random.uniform(0.5, 3, limit)
            low_prices = close_prices - np.random.uniform(0.5, 3, limit)
            volume = np.random.lognormal(10, 2, limit)

            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volume
            })

            # Добавляем ATR для расчетов
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных с {exchange} для {symbol}: {e}")
            return None

    def analyze_symbol(self, exchange, symbol):
        """Анализируем символ на разных таймфреймах"""
        try:
            for tf in self.TIMEFRAMES:
                df = self.get_ohlcv_data(exchange, symbol, tf)
                if df is None or len(df) < 50:
                    continue

                # Рассчитываем индикаторы
                df['rsi'] = talib.RSI(df['close'], timeperiod=14)
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
                df['ema20'] = talib.EMA(df['close'], timeperiod=20)
                df['ema50'] = talib.EMA(df['close'], timeperiod=50)
                df['ema200'] = talib.EMA(df['close'], timeperiod=200)
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'],
                                                                               timeperiod=self.BBANDS_PERIOD)

                # Последние значения
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]

                # Проверяем условия для сигналов
                long_signal = False
                short_signal = False
                signal_strength = 0

                # Условия для лонга
                ema_condition = (last_row['ema20'] > last_row['ema50'] > last_row['ema200'])
                macd_condition = (last_row['macd_hist'] > 0 and last_row['macd'] > last_row['macd_signal'])
                rsi_condition = (self.RSI_OVERSOLD < last_row['rsi'] < self.RSI_OVERBOUGHT)
                volume_condition = (last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5)

                if ema_condition and macd_condition and rsi_condition and volume_condition:
                    long_signal = True
                    signal_strength += 1
                    if last_row['close'] > last_row['bb_upper']:
                        signal_strength += 1

                # Условия для шорта
                ema_condition_short = (last_row['ema20'] < last_row['ema50'] < last_row['ema200'])
                macd_condition_short = (last_row['macd_hist'] < 0 and last_row['macd'] < last_row['macd_signal'])
                rsi_condition_short = (self.RSI_OVERSOLD < last_row['rsi'] < self.RSI_OVERBOUGHT)

                if ema_condition_short and macd_condition_short and rsi_condition_short and volume_condition:
                    short_signal = True
                    signal_strength += 1
                    if last_row['close'] < last_row['bb_lower']:
                        signal_strength += 1

                # Дополнительные условия
                if last_row['adx'] > self.ADX_STRONG_TREND:
                    signal_strength += 1

                if (last_row['close'] < last_row['bb_lower']) and (prev_row['close'] > prev_row['bb_lower']):
                    signal_strength += 1

                if (last_row['close'] > last_row['bb_upper']) and (prev_row['close'] < prev_row['bb_upper']):
                    signal_strength += 1

                if long_signal or short_signal:
                    direction = 'ЛОНГ' if long_signal else 'ШОРТ'
                    entry_price = last_row['close']

                    # Расчет стоп-лосса
                    atr = last_row['atr']
                    if direction == 'ЛОНГ':
                        stop_loss = df['low'].iloc[-1] - atr * 1.5
                    else:
                        stop_loss = df['high'].iloc[-1] + atr * 1.5

                    # Расчет тейк-профита
                    take_profit, stop_loss = self.calculate_tp_sl(entry_price, stop_loss, direction)

                    # Расчет потенциальной прибыли в процентах
                    if direction == 'ЛОНГ':
                        profit_percent = (take_profit - entry_price) / entry_price * 100
                    else:
                        profit_percent = (entry_price - take_profit) / entry_price * 100

                    # Пропускаем сигналы с малой прибылью
                    if profit_percent < self.MIN_PROFIT_PERCENT:
                        continue

                    position_size = self.calculate_position_size(entry_price, stop_loss)

                    signal = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': tf,
                        'direction': direction,
                        'entry_price': round(entry_price, 4),
                        'take_profit': round(take_profit, 4),
                        'stop_loss': round(stop_loss, 4),
                        'position_size': round(position_size, 4),
                        'profit_percent': round(profit_percent, 2),
                        'risk_reward': f"1:{self.REWARD_RATIO}",
                        'volume': last_row['volume'],
                        'strength': signal_strength,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'indicators': {
                            'rsi': round(last_row['rsi'], 2),
                            'macd_hist': round(last_row['macd_hist'], 4),
                            'ema_cross': f"20:{round(last_row['ema20'], 2)} > 50:{round(last_row['ema50'], 2)} > 200:{round(last_row['ema200'], 2)}" if long_signal else f"20:{round(last_row['ema20'], 2)} < 50:{round(last_row['ema50'], 2)} < 200:{round(last_row['ema200'], 2)}",
                            'adx': round(last_row['adx'], 2),
                            'atr': round(last_row['atr'], 2),
                            'bb_percent': round((last_row['close'] - last_row['bb_lower']) / (
                                    last_row['bb_upper'] - last_row['bb_lower']) * 100, 2)
                        }
                    }
                    self.signals.append(signal)
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol} на {exchange}: {e}")

    def scan_exchange(self, exchange):
        """Сканируем биржу на наличие сигналов"""
        logger.info(f"Сканируем {exchange}...")
        symbols = self.get_top_symbols(exchange)
        logger.info(f"Топ {len(symbols)} пар на {exchange}: {', '.join(symbols[:5])}...")

        for symbol in symbols:
            self.analyze_symbol(exchange, symbol)
            time.sleep(0.1)

    def start_scanning(self):
        """Запускаем сканирование всех бирж"""
        threads = []
        for exchange in self.EXCHANGES:
            thread = Thread(target=self.scan_exchange, args=(exchange,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def filter_signals(self):
        """Фильтруем сигналы по силе, объему и минимальной прибыли"""
        if not self.signals:
            return []

        df = pd.DataFrame(self.signals)
        df = df[df['strength'] >= self.SIGNAL_STRENGTH_THRESHOLD]
        df = df[df['volume'] >= self.MIN_VOLUME]
        df = df[df['profit_percent'] >= self.MIN_PROFIT_PERCENT]

        df = df.sort_values(['strength', 'volume'], ascending=[False, False])
        df = df.drop_duplicates(['exchange', 'symbol'], keep='first')

        return df.to_dict('records')

    def generate_report(self, signals):
        """Генерируем отчет по сигналам"""
        if not signals:
            return None

        report = "🚀 **СИГНАЛЫ ДЛЯ ТОРГОВЛИ КРИПТОВАЛЮТАМИ** 🚀\n\n"
        report += f"📅 Время генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"💵 Текущий депозит: ${self.deposit:,.2f}\n"
        report += f"📊 Риск на сделку: {self.RISK_PER_TRADE * 100}% от депозита\n"
        report += f"🎯 Минимальная прибыль: {self.MIN_PROFIT_PERCENT}%\n\n"

        for signal in signals:
            emoji = "📈" if signal['direction'] == 'ЛОНГ' else "📉"
            report += (
                f"{emoji} **СИГНАЛ НА {signal['direction']}** {emoji}\n"
                f"🏦 Биржа: {signal['exchange'].upper()}\n"
                f"💰 Пара: {signal['symbol']}\n"
                f"⏰ Таймфрейм: {signal['timeframe']}\n"
                f"🔢 Размер позиции: {signal['position_size']} {signal['symbol'].split('/')[0]}\n"
                f"💵 Цена входа: {signal['entry_price']:.4f}\n"
                f"🎯 Тейк-профит: {signal['take_profit']:.4f} (+{signal['profit_percent']:.2f}%)\n"
                f"🛑 Стоп-лосс: {signal['stop_loss']:.4f}\n"
                f"📈 Риск/Прибыль: {signal['risk_reward']}\n"
                f"📊 Объем (24ч): {signal['volume']:,.0f}\n"
                f"💪 Сила сигнала: {'⭐' * signal['strength']}\n\n"
                f"📊 **Технические индикаторы**\n"
                f"  • RSI: {signal['indicators']['rsi']} ({'Нейтрально' if self.RSI_OVERSOLD < signal['indicators']['rsi'] < self.RSI_OVERBOUGHT else 'Перекупленность' if signal['indicators']['rsi'] >= self.RSI_OVERBOUGHT else 'Перепроданность'})\n"
                f"  • MACD Hist: {signal['indicators']['macd_hist']:.4f} ({'Бычий' if signal['indicators']['macd_hist'] > 0 else 'Медвежий'})\n"
                f"  • EMA Cross: {signal['indicators']['ema_cross']}\n"
                f"  • ADX: {signal['indicators']['adx']} ({'Сильный тренд' if signal['indicators']['adx'] > self.ADX_STRONG_TREND else 'Слабый тренд'})\n"
                f"  • ATR: {signal['indicators']['atr']:.2f} (Волатильность)\n"
                f"  • BB %: {signal['indicators']['bb_percent']}% ({'Верхняя граница' if signal['indicators']['bb_percent'] > 80 else 'Нижняя граница' if signal['indicators']['bb_percent'] < 20 else 'Средний диапазон'})\n"
                f"\n{'=' * 50}\n\n"
            )

        report += "\n⚠️ **Управление капиталом**:\n"
        report += f"• Риск на сделку: {self.RISK_PER_TRADE * 100}% от депозита\n"
        report += f"• Соотношение прибыли к риску: {self.REWARD_RATIO}:1\n"
        report += f"• Минимальная прибыль: {self.MIN_PROFIT_PERCENT}%\n"
        report += f"• Максимальный размер позиции: {self.MAX_POSITION_SIZE * 100}% от депозита\n\n"
        report += "⚠️ **Предупреждение**: Это не финансовая рекомендация. Всегда проводите собственное исследование перед торговлей."
        return report

    async def send_to_telegram_async(self, message):
        """Асинхронная отправка сообщения"""
        try:
            max_length = 4096
            if len(message) > max_length:
                parts = [message[i:i + max_length] for i in range(0, len(message), max_length)]
                for part in parts:
                    await self.telegram_bot.send_message(chat_id=self.TELEGRAM_CHAT_ID, text=part,
                                                         parse_mode='Markdown')
                await asyncio.sleep(1)
            else:
                await self.telegram_bot.send_message(chat_id=self.TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")

    def send_to_telegram(self, message):
        """Синхронная обертка для отправки"""
        if message is None:
            logger.info("Сигналы не найдены, пропускаем отправку в Telegram")
            return

        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.send_to_telegram_async(message))
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")

    def run(self):
        """Основной цикл работы бота"""
        logger.info("Запуск Crypto Scanner Bot...")
        logger.info(f"Анализируемые биржи: {', '.join(self.EXCHANGES)}")
        logger.info(f"Таймфреймы: {', '.join(self.TIMEFRAMES)}")
        logger.info(f"Минимальный объем: ${self.MIN_VOLUME:,}")
        logger.info(
            f"Критерии сигналов: RSI ({self.RSI_OVERSOLD}-{self.RSI_OVERBOUGHT}), ADX > {self.ADX_STRONG_TREND}")

        while self.running:
            try:
                start_time = time.time()
                self.signals = []

                self.start_scanning()
                strong_signals = self.filter_signals()
                report = self.generate_report(strong_signals)

                self.send_to_telegram(report)

                scan_time = time.time() - start_time
                logger.info(
                    f"Сканирование завершено за {scan_time:.2f} секунд. Найдено сигналов: {len(strong_signals)}")

                sleep_time = max(0, self.SCAN_INTERVAL - scan_time)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.running = False
                logger.info("Бот остановлен пользователем")
            except Exception as e:
                logger.error(f"Ошибка в основном цикле: {e}")
                time.sleep(60)


if __name__ == "__main__":
    try:
        # Инициализация и запуск бота с дефолтным депозитом
        bot = CryptoScannerBot()
        bot.run()
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")