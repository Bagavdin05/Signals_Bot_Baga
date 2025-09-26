import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import logging
import asyncio
import warnings
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest
import html
from collections import defaultdict
import pytz
import aiohttp
import math
import talib

warnings.filterwarnings('ignore')

# Настройка часового пояса Москвы (UTC+3)
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# Настройка логирования только в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('MEXC_5min_Bot')

# Конфигурация Telegram бота
TELEGRAM_BOT_TOKEN = "7952768185:AAGuhybXaGPJqtlGPd1-O4nc6_FpUL2rOgw"
TELEGRAM_CHAT_IDS = ["1167694150", "7916502470", "1111230981"]

class AdvancedTechnicalAnalyzer:
    """Продвинутый технический анализатор с использованием TA-Lib"""
    
    @staticmethod
    def ema(data, period):
        """Экспоненциальная скользящая средняя с TA-Lib"""
        return talib.EMA(data, timeperiod=period)
    
    @staticmethod
    def sma(data, period):
        """Простая скользящая средняя с TA-Lib"""
        return talib.SMA(data, timeperiod=period)
    
    @staticmethod
    def rsi(data, period=14):
        """Индекс относительной силы с TA-Lib"""
        return talib.RSI(data, timeperiod=period)
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD с TA-Lib"""
        macd, macd_signal, macd_hist = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def bollinger_bands(data, period=20, std=2):
        """Полосы Боллинджера с TA-Lib"""
        upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std, nbdevdn=std)
        return upper, middle, lower
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Стохастический осциллятор с TA-Lib"""
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=3, slowd_period=3)
        return slowk, slowd
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Индекс Williams %R с TA-Lib"""
        return talib.WILLR(high, low, close, timeperiod=period)
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index с TA-Lib"""
        return talib.ADX(high, low, close, timeperiod=period)
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range с TA-Lib"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    @staticmethod
    def cci(high, low, close, period=14):
        """Commodity Channel Index с TA-Lib"""
        return talib.CCI(high, low, close, timeperiod=period)
    
    @staticmethod
    def obv(close, volume):
        """On Balance Volume с TA-Lib"""
        return talib.OBV(close, volume)
    
    @staticmethod
    def volume_profile(volume, period=20):
        """Анализ объемов"""
        volume_sma = talib.SMA(volume, timeperiod=period)
        return volume / volume_sma
    
    @staticmethod
    def mfi(high, low, close, volume, period=14):
        """Money Flow Index с TA-Lib"""
        return talib.MFI(high, low, close, volume, timeperiod=period)
    
    @staticmethod
    def awesome_oscillator(high, low, fast_period=5, slow_period=34):
        """Awesome Oscillator"""
        fast_sma = talib.SMA((high + low) / 2, timeperiod=fast_period)
        slow_sma = talib.SMA((high + low) / 2, timeperiod=slow_period)
        return fast_sma - slow_sma
    
    @staticmethod
    def parabolic_sar(high, low, acceleration=0.02, maximum=0.2):
        """Parabolic SAR"""
        return talib.SAR(high, low, acceleration=acceleration, maximum=maximum)

class StrictSignalGenerator:
    """Строгий генератор сигналов с минимальной уверенностью 70%"""
    
    def __init__(self):
        self.ta = AdvancedTechnicalAnalyzer()
        self.signal_history = defaultdict(list)
        self.max_history_size = 5
    
    def analyze_multiple_timeframes(self, df_1m, df_5m, df_15m, df_1h):
        """Анализ всех таймфреймов для строгих сигналов"""
        signals = {}
        
        timeframes = [
            ('1m', df_1m), ('5m', df_5m), 
            ('15m', df_15m), ('1h', df_1h)
        ]
        
        for timeframe, df in timeframes:
            if df is not None and len(df) > 20:
                signal = self.analyze_single_timeframe(df, timeframe)
                signals[timeframe] = signal
        
        return signals
    
    def analyze_single_timeframe(self, df, timeframe):
        """Строгий анализ одного таймфрейма с минимальной уверенностью 70%"""
        if len(df) < 20:
            return {'signal': 'NEUTRAL', 'confidence': 0}
        
        try:
            # Расчет всех индикаторов
            df = self.calculate_indicators(df)
            current = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else current
            
            # Строгая система баллов
            bullish_score = 0
            bearish_score = 0
            max_score = 0
            
            # 1. EMA направление (высокий вес)
            if not self.is_nan(current['ema_9']) and not self.is_nan(current['ema_21']):
                max_score += 4
                if current['ema_9'] > current['ema_21'] and current['ema_21'] > current['ema_50']:
                    bullish_score += 4
                elif current['ema_9'] < current['ema_21'] and current['ema_21'] < current['ema_50']:
                    bearish_score += 4
            
            # 2. RSI строгая проверка
            if not self.is_nan(current['rsi']):
                max_score += 3
                if current['rsi'] < 30:  # Строгие границы
                    bullish_score += 3
                elif current['rsi'] > 70:
                    bearish_score += 3
            
            # 3. MACD строгая проверка
            if not self.is_nan(current['macd']) and not self.is_nan(current['macd_signal']):
                max_score += 3
                if current['macd'] > current['macd_signal'] and current['macd'] > 0:
                    bullish_score += 3
                elif current['macd'] < current['macd_signal'] and current['macd'] < 0:
                    bearish_score += 3
            
            # 4. Стохастик строгая проверка
            if not self.is_nan(current['stoch_k']) and not self.is_nan(current['stoch_d']):
                max_score += 2
                if current['stoch_k'] < 20 and current['stoch_k'] > current['stoch_d']:
                    bullish_score += 2
                elif current['stoch_k'] > 80 and current['stoch_k'] < current['stoch_d']:
                    bearish_score += 2
            
            # 5. Полосы Боллинджера
            if not self.is_nan(current['bb_upper']) and not self.is_nan(current['bb_lower']):
                max_score += 2
                if current['close'] <= current['bb_lower']:
                    bullish_score += 2
                elif current['close'] >= current['bb_upper']:
                    bearish_score += 2
            
            # 6. Объемная активность
            if not self.is_nan(current['volume_ratio']):
                max_score += 1
                if current['volume_ratio'] > 2.0:  # Высокий объем
                    if current['close'] > current['open']:
                        bullish_score += 1
                    else:
                        bearish_score += 1
            
            # Расчет уверенности
            total_score = max(1, max_score)
            confidence = (abs(bullish_score - bearish_score) / total_score) * 100
            
            # Минимальная уверенность 70%
            if confidence < 70:
                return {'signal': 'NEUTRAL', 'confidence': confidence}
            
            if bullish_score > bearish_score:
                signal_type = 'LONG'
            elif bearish_score > bullish_score:
                signal_type = 'SHORT'
            else:
                return {'signal': 'NEUTRAL', 'confidence': confidence}
            
            return {'signal': signal_type, 'confidence': confidence}
            
        except Exception as e:
            logger.error(f"Ошибка анализа таймфрейма {timeframe}: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}
    
    def calculate_indicators(self, df):
        """Расчет индикаторов для строгого анализа"""
        try:
            # EMA
            df['ema_9'] = self.ta.ema(df['close'], 9)
            df['ema_21'] = self.ta.ema(df['close'], 21)
            df['ema_50'] = self.ta.ema(df['close'], 50)
            
            # Моментум
            df['rsi'] = self.ta.rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = self.ta.macd(df['close'])
            df['stoch_k'], df['stoch_d'] = self.ta.stochastic(df['high'], df['low'], df['close'])
            
            # Волатильность
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.ta.bollinger_bands(df['close'])
            
            # Объем
            df['volume_ratio'] = self.ta.volume_profile(df['volume'], 20)
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
        
        return df
    
    def is_nan(self, value):
        """Проверка на NaN"""
        if value is None:
            return True
        return pd.isna(value) or np.isnan(value)
    
    def generate_final_signal(self, timeframe_signals, symbol):
        """Генерация окончательного сигнала с минимальной уверенностью 70%"""
        if not timeframe_signals:
            return None
        
        # Веса для разных таймфреймов
        weights = {'1m': 1, '5m': 2, '15m': 3, '1h': 4}  # Старшие ТФ имеют больший вес
        
        long_score = 0
        short_score = 0
        total_confidence = 0
        total_weight = 0
        
        for timeframe, signal_info in timeframe_signals.items():
            weight = weights.get(timeframe, 1)
            confidence = signal_info.get('confidence', 0)
            
            # Учитываем только сигналы с уверенностью >= 70%
            if confidence >= 70:
                if signal_info['signal'] == 'LONG':
                    long_score += weight * (confidence / 100)
                elif signal_info['signal'] == 'SHORT':
                    short_score += weight * (confidence / 100)
                
                total_confidence += confidence
                total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Средняя уверенность должна быть >= 70%
        avg_confidence = total_confidence / len([s for s in timeframe_signals.values() if s.get('confidence', 0) >= 70])
        if avg_confidence < 70:
            return None
        
        # Требуем явного превосходства
        if long_score > short_score * 1.5:  # LONG должен быть значительно сильнее
            final_signal = 'LONG'
        elif short_score > long_score * 1.5:  # SHORT должен быть значительно сильнее
            final_signal = 'SHORT'
        else:
            return None
        
        # Проверяем историю сигналов
        current_time = time.time()
        recent_signals = [s for s in self.signal_history[symbol] 
                         if current_time - s['timestamp'] < 300]  # 5 минут
        
        if recent_signals:
            return None
        
        # Сохраняем сигнал в историю
        self.signal_history[symbol].append({
            'signal': final_signal,
            'timestamp': current_time
        })
        
        # Ограничиваем размер истории
        if len(self.signal_history[symbol]) > self.max_history_size:
            self.signal_history[symbol] = self.signal_history[symbol][-self.max_history_size:]
        
        return final_signal

class MEXCStrictOptionsBot:
    def __init__(self):
        self.exchange = self.initialize_exchange()
        self.telegram_app = None
        self.last_analysis_time = None
        self.signals = []
        self.is_analyzing = False
        self.analysis_lock = asyncio.Lock()
        self.telegram_queue = asyncio.Queue()
        self.telegram_worker_task = None
        self.session = None

        # Только BTC, ETH, SOL на MEXC
        self.target_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Настройки для строгих 5-минутных опционов
        self.config = {
            'timeframes': ['1m', '5m', '15m', '1h'],
            'analysis_interval': 60,  # Анализ каждые 60 секунд
            'min_candles': 20,
            'max_analysis_time': 30
        }

        # Инициализация строгого анализатора
        self.signal_generator = StrictSignalGenerator()
        
        # Клавиатура с кнопками
        self.reply_keyboard = ReplyKeyboardMarkup(
            [
                ["📊 Активные сигналы", "🔄 Строгий анализ"],
                ["⏰ Время торговли", "⚡ Торговля"],
                ["📈 Статистика", "ℹ️ Инструкция"]
            ],
            resize_keyboard=True,
            input_field_placeholder="Выберите действие..."
        )

        # Статистика
        self.statistics = {
            'total_analyses': 0,
            'signals_generated': 0,
            'last_signal_time': None,
            'symbol_stats': defaultdict(lambda: {'long': 0, 'short': 0, 'neutral': 0})
        }

        logger.info("Строгий MEXC Бот для 5-минутных опционов инициализирован")

    async def initialize_session(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))

    async def close_session(self):
        if self.session:
            await self.session.close()

    def get_moscow_time(self, dt=None):
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(MOSCOW_TZ)

    def format_moscow_time(self, dt=None, format_str='%H:%M:%S'):
        moscow_time = self.get_moscow_time(dt)
        return moscow_time.strftime(format_str)

    async def initialize_telegram(self):
        try:
            request = HTTPXRequest(connection_pool_size=3, read_timeout=10, write_timeout=10, connect_timeout=10)
            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
            
            self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
            self.telegram_app.add_handler(CommandHandler("stats", self.show_stats))
            self.telegram_app.add_handler(CommandHandler("analyze", self.quick_analysis))
            
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()
            self.telegram_worker_task = asyncio.create_task(self.telegram_worker())
            logger.info("Telegram бот инициализирован")

            startup_message = (
                "🎯 <b>СТРОГИЙ MEXC БОТ ДЛЯ 5-МИНУТНЫХ ОПЦИОНОВ</b>\n\n"
                "🎯 <b>Монеты:</b> BTC, ETH, SOL\n"
                "⏰ <b>Таймфреймы:</b> 1M, 5M, 15M, 1H\n"
                "📊 <b>Сигналы:</b> ТОЧНЫЕ LONG / SHORT\n"
                "🏢 <b>Биржа:</b> MEXC\n\n"
                "✅ <b>Особенности:</b>\n"
                "• Минимальная уверенность: 70%\n"
                "• Строгая фильтрация сигналов\n"
                "• Только качественные setup\n"
                "• Анализ каждые 60 секунд\n\n"
                f"🕐 <b>Запуск:</b> {self.format_moscow_time()}"
            )
            await self.send_telegram_message(startup_message)

        except Exception as e:
            logger.error(f"Ошибка инициализации Telegram бота: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        text = update.message.text
        
        if text == "📊 Активные сигналы":
            await self.handle_signals(update, context)
        elif text == "🔄 Строгий анализ":
            await self.handle_analysis(update, context)
        elif text == "⏰ Время торговли":
            await self.handle_trading_time(update, context)
        elif text == "⚡ Торговля":
            await self.handle_trading(update, context)
        elif text == "📈 Статистика":
            await self.show_stats(update, context)
        elif text == "ℹ️ Инструкция":
            await self.handle_help(update, context)
        else:
            await self.handle_unknown(update, context)

    async def handle_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Активные сигналы'"""
        if not self.signals:
            await update.message.reply_text(
                "📊 <b>Активных сигналов нет</b>\n\n"
                "Сигналы появляются только при уверенности ≥70%\n"
                "Нажмите '🔄 Строгий анализ' для сканирования",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            return

        try:
            message = "🎯 <b>СТРОГИЕ ТОРГОВЫЕ СИГНАЛЫ</b>\n\n"
            message += "⏰ <b>5-минутные опционы • MEXC</b>\n"
            message += "✅ <b>Минимальная уверенность: 70%</b>\n\n"

            for i, signal in enumerate(self.signals[:5]):
                symbol_name = signal['symbol'].replace('/USDT', '')
                
                if signal['signal'] == 'LONG':
                    signal_emoji = "🟢"
                    action_text = "LONG"
                else:
                    signal_emoji = "🔴" 
                    action_text = "SHORT"
                
                expiration_str = signal['expiration_time'].strftime('%H:%M:%S')
                entry_str = signal['entry_time'].strftime('%H:%M:%S')
                
                message += (
                    f"{signal_emoji} <b>{symbol_name}</b>\n"
                    f"📈 Направление: <b>{action_text}</b>\n"
                    f"⏰ Вход: <b>{entry_str}</b>\n"
                    f"📅 Экспирация: <b>{expiration_str}</b>\n"
                )
                
                # Время до экспирации
                current_time = self.get_moscow_time()
                time_to_expiry = signal['expiration_time'] - current_time
                minutes_to_expiry = max(0, int(time_to_expiry.total_seconds() / 60))
                seconds_to_expiry = max(0, int(time_to_expiry.total_seconds() % 60))
                
                if minutes_to_expiry <= 1:
                    message += f"⚡ <b>Экспирация через: {seconds_to_expiry} сек!</b>\n"
                else:
                    message += f"⏱️ <b>До экспирации: {minutes_to_expiry} мин</b>\n"
                
                message += "━━━━━━━━━━━━━━━━━━━\n\n"

            message += f"🕐 <b>Обновлено:</b> {self.format_moscow_time(self.last_analysis_time)}\n"
            message += "🔁 <b>Автообновление каждые 60 сек.</b>"

            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

        except Exception as e:
            logger.error(f"Ошибка отправки сигналов: {e}")
            await update.message.reply_text(
                "❌ <b>Ошибка формирования сигналов</b>",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

    async def handle_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Торговля'"""
        try:
            current_time = self.get_moscow_time()
            
            # Рассчитываем ближайшие 5-минутные интервалы
            current_second = current_time.second
            current_minute = current_time.minute
            
            # Следующий 5-минутный интервал
            minutes_to_next = (5 - (current_minute % 5)) % 5
            seconds_to_next = (60 - current_second) if minutes_to_next == 0 else (minutes_to_next * 60 - current_second)
            
            next_entry = current_time + timedelta(seconds=seconds_to_next)
            next_expiry = next_entry + timedelta(minutes=5)
            
            time_message = (
                "⚡ <b>ТОРГОВЛЯ 5-МИНУТНЫМИ ОПЦИОНАМИ</b>\n\n"
                f"🕐 <b>Текущее время:</b> {current_time.strftime('%H:%M:%S')}\n"
                f"🎯 <b>Следующий вход:</b> {next_entry.strftime('%H:%M:%S')}\n"
                f"📅 <b>Экспирация:</b> {next_expiry.strftime('%H:%M:%S')}\n"
                f"⏱️ <b>До входа:</b> {seconds_to_next} сек\n\n"
            )
            
            # Рекомендации для торговли
            time_message += "<b>💡 Стратегия торговли:</b>\n"
            time_message += "• Входите за 15-30 секунд до начала интервала\n"
            time_message += "• Используйте только сигналы с уверенностью ≥70%\n"
            time_message += "• Следите за объемом и волатильностью\n"
            time_message += "• Ставьте стоп-лосс 10-15%\n\n"
            
            time_message += "<b>📊 Ближайшие интервалы:</b>\n"
            for i in range(4):
                entry_time = next_entry + timedelta(minutes=5*i)
                expiry_time = entry_time + timedelta(minutes=5)
                time_message += f"• {entry_time.strftime('%H:%M')} - {expiry_time.strftime('%H:%M')}\n"
            
            await update.message.reply_text(
                time_message,
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета времени: {e}")
            await update.message.reply_text(
                "❌ <b>Ошибка расчета времени</b>",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать статистику бота"""
        try:
            stats = self.statistics
            total_signals = stats['signals_generated']
            total_analyses = stats['total_analyses']
            
            if total_analyses > 0:
                signal_rate = (total_signals / total_analyses) * 100
            else:
                signal_rate = 0
            
            message = (
                "📈 <b>СТАТИСТИКА СТРОГОГО БОТА</b>\n\n"
                f"📊 Всего анализов: <b>{total_analyses}</b>\n"
                f"🎯 Сгенерировано сигналов: <b>{total_signals}</b>\n"
                f"📈 Процент сигналов: <b>{signal_rate:.1f}%</b>\n"
            )
            
            if stats['last_signal_time']:
                last_signal_str = self.format_moscow_time(stats['last_signal_time'])
                message += f"⏰ Последний сигнал: <b>{last_signal_str}</b>\n\n"
            else:
                message += "⏰ Последний сигнал: <b>нет данных</b>\n\n"
            
            # Статистика по символам
            message += "<b>🎯 Статистика по монетам:</b>\n"
            for symbol in self.target_symbols:
                symbol_name = symbol.replace('/USDT', '')
                symbol_stats = stats['symbol_stats'][symbol_name]
                total = symbol_stats['long'] + symbol_stats['short'] + symbol_stats['neutral']
                
                if total > 0:
                    long_percent = (symbol_stats['long'] / total) * 100
                    short_percent = (symbol_stats['short'] / total) * 100
                    message += (
                        f"• {symbol_name}: LONG {long_percent:.1f}% | "
                        f"SHORT {short_percent:.1f}%\n"
                    )
            
            message += "\n✅ <b>Минимальная уверенность: 70%</b>"
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            
        except Exception as e:
            logger.error(f"Ошибка показа статистики: {e}")
            await update.message.reply_text(
                "❌ <b>Ошибка загрузки статистики</b>",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

    async def handle_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Строгий анализ'"""
        try:
            await update.message.reply_text(
                "🎯 <b>Запускаю строгий анализ...</b>\n"
                "Сканирую BTC, ETH, SOL...\n"
                "Минимальная уверенность: 70%",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            
            asyncio.create_task(self.run_single_analysis(update))
            
        except Exception as e:
            logger.error(f"Ошибка запуска анализа: {e}")
            await update.message.reply_text(
                "❌ <b>Ошибка запуска анализа</b>",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

    async def quick_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Анализ через команду /analyze"""
        await self.handle_analysis(update, context)

    async def handle_trading_time(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Время торговли'"""
        try:
            current_time = self.get_moscow_time()
            
            # Рассчитываем ближайшие 5-минутные интервалы
            current_minute = current_time.minute
            minutes_to_next = (5 - (current_minute % 5)) % 5
            seconds_to_next = (60 - current_time.second) if minutes_to_next == 0 else (minutes_to_next * 60 - current_time.second)
            
            if minutes_to_next == 0 and seconds_to_next <= 30:
                next_entry = current_time
                next_expiry = current_time + timedelta(minutes=5)
                status = "⚡ СЕЙЧАС"
            else:
                next_entry = current_time + timedelta(seconds=seconds_to_next)
                next_expiry = next_entry + timedelta(minutes=5)
                status = f"⏱️ Через {minutes_to_next} мин {seconds_to_next} сек"
            
            time_message = (
                "⏰ <b>РАСПИСАНИЕ ТОРГОВЛИ</b>\n\n"
                f"🕐 <b>Текущее время:</b> {current_time.strftime('%H:%M:%S')}\n"
                f"🎯 <b>Следующий вход:</b> {next_entry.strftime('%H:%M:%S')}\n"
                f"📅 <b>Экспирация:</b> {next_expiry.strftime('%H:%M:%S')}\n"
                f"📊 <b>Статус:</b> {status}\n\n"
            )
            
            # Расписание на ближайшие 20 минут
            time_message += "<b>Ближайшие интервалы:</b>\n"
            for i in range(4):
                entry_time = next_entry + timedelta(minutes=5*i)
                expiry_time = entry_time + timedelta(minutes=5)
                time_message += f"• {entry_time.strftime('%H:%M')} - {expiry_time.strftime('%H:%M')}\n"
            
            time_message += "\n💡 <b>Совет для торговли:</b>\n"
            time_message += "• Входите за 15-30 секунд до начала интервала\n"
            time_message += "• Используйте только сигналы с уверенностью ≥70%\n"
            time_message += "• Следите за трендом на старших таймфреймах"
            
            await update.message.reply_text(
                time_message,
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            
        except Exception as e:
            logger.error(f"Ошибка расчета времени: {e}")
            await update.message.reply_text(
                "❌ <b>Ошибка расчета времени</b>",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Инструкция'"""
        help_message = (
            "🤖 <b>ИНСТРУКЦИЯ ПО СТРОГОЙ ТОРГОВЛЕ</b>\n\n"
            "🎯 <b>Стратегия для 5-минутных опционов:</b>\n"
            "• LONG - покупаем при сильном восходящем тренде\n"
            "• SHORT - продаем при сильном нисходящем тренде\n"
            "• Экспирация через 5 минут\n"
            "• Вход за 15-30 секунд до начала интервала\n\n"
            "📊 <b>Как использовать бота:</b>\n"
            "1. Нажмите '🔄 Строгий анализ' для сканирования\n"
            "2. Смотрите сигналы в '📊 Активные сигналы'\n"
            "3. Используйте '⚡ Торговля' для timing\n"
            "4. Входите в сделку за 15-30 секунд до интервала\n\n"
            "⏰ <b>Временные интервалы:</b>\n"
            "• Интервалы начинаются каждые 5 минут\n"
            "• Пример: 14:00, 14:05, 14:10 и т.д.\n"
            "• Экспирация через 5 минут после входа\n\n"
            "⚡ <b>Пример торговли:</b>\n"
            "• Текущее время: 14:04:45\n"
            "• Следующий вход: 14:05:00\n"
            "• Входить в: 14:04:45-14:04:50\n"
            "• Экспирация: 14:10:00\n\n"
            "✅ <b>Преимущества строгого бота:</b>\n"
            "• Минимальная уверенность 70%\n"
            "• Строгая фильтрация сигналов\n"
            "• Только качественные setup\n"
            "• Меньше ложных срабатываний"
        )
        await update.message.reply_text(
            help_message,
            parse_mode='HTML',
            reply_markup=self.reply_keyboard
        )

    async def handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "❓ <b>Используйте кнопки для управления ботом</b>",
            parse_mode='HTML',
            reply_markup=self.reply_keyboard
        )

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = (
            "🎯 <b>СТРОГИЙ MEXC БОТ ДЛЯ 5-МИНУТНЫХ ОПЦИОНОВ</b>\n\n"
            "🎯 <b>Специализация:</b> Качественные 5-минутные опционы LONG/SHORT\n"
            "💰 <b>Монеты:</b> Bitcoin (BTC), Ethereum (ETH), Solana (SOL)\n"
            "⏰ <b>Экспирация:</b> 5 минут\n"
            "🏢 <b>Биржа:</b> MEXC\n\n"
            "✅ <b>Ключевые особенности:</b>\n"
            "• Минимальная уверенность сигналов: 70%\n"
            "• Строгая фильтрация ложных срабатываний\n"
            "• Анализ по 4 таймфреймам: 1M, 5M, 15M, 1H\n"
            "• Только сильные торговые setup\n\n"
            "📱 <b>Используйте кнопки ниже для управления:</b>"
        )
        await update.message.reply_text(
            welcome_text,
            parse_mode='HTML',
            reply_markup=self.reply_keyboard
        )

    async def telegram_worker(self):
        logger.info("Telegram worker запущен")
        while True:
            try:
                chat_id, message = await self.telegram_queue.get()
                if chat_id and message:
                    await self.telegram_app.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=self.reply_keyboard
                    )
                self.telegram_queue.task_done()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Ошибка в telegram_worker: {e}")
                await asyncio.sleep(1)

    async def send_telegram_message(self, message: str, chat_ids: list = None):
        if chat_ids is None:
            chat_ids = TELEGRAM_CHAT_IDS
        for chat_id in chat_ids:
            await self.telegram_queue.put((chat_id, message))

    async def run_single_analysis(self, update=None):
        try:
            async with self.analysis_lock:
                if self.is_analyzing:
                    if update:
                        await update.message.reply_text(
                            "🎯 <b>Анализ уже выполняется...</b>",
                            parse_mode='HTML',
                            reply_markup=self.reply_keyboard
                        )
                    return

                self.is_analyzing = True
                
                if update:
                    await update.message.reply_text(
                        "🎯 <b>Запускаю строгий анализ...</b>\n"
                        "BTC → ETH → SOL\n"
                        "1M → 5M → 15M → 1H\n"
                        "Минимальная уверенность: 70%\n"
                        "Ожидайте качественные сигналы...",
                        parse_mode='HTML',
                        reply_markup=self.reply_keyboard
                    )

                signals = await self.analyze_market()

                if update:
                    if signals:
                        await update.message.reply_text(
                            f"✅ <b>Строгий анализ завершен</b>\n\n"
                            f"📊 Найдено сигналов: <b>{len(signals)}</b>\n"
                            f"🎯 Минимальная уверенность: <b>70%</b>\n\n"
                            f"Нажмите '📊 Активные сигналы' для просмотра",
                            parse_mode='HTML',
                            reply_markup=self.reply_keyboard
                        )
                    else:
                        await update.message.reply_text(
                            "ℹ️ <b>Анализ завершен</b>\n"
                            "Сигналов с уверенностью ≥70% не найдено\n"
                            "Попробуйте через 60 секунд",
                            parse_mode='HTML',
                            reply_markup=self.reply_keyboard
                        )

        except Exception as e:
            logger.error(f"Ошибка в run_single_analysis: {e}")
            if update:
                await update.message.reply_text(
                    "❌ <b>Ошибка анализа</b>",
                    parse_mode='HTML',
                    reply_markup=self.reply_keyboard
                )
        finally:
            self.is_analyzing = False

    def initialize_exchange(self):
        try:
            exchange = ccxt.mexc({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
                'timeout': 15000,
            })
            exchange.load_markets()
            
            timeframes = exchange.timeframes
            logger.info(f"Доступные таймфреймы MEXC: {timeframes}")
            
            return exchange
        except Exception as e:
            logger.error(f"Ошибка подключения к MEXC: {e}")
            return None

    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 25):
        if self.exchange is None:
            return None

        try:
            if timeframe not in self.exchange.timeframes:
                logger.error(f"Таймфрейм {timeframe} не поддерживается MEXC")
                return None
            
            normalized_symbol = symbol.replace('/', '')
            await asyncio.sleep(0.05)
            
            ohlcv = self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 20:
                logger.warning(f"Недостаточно данных для {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Ошибка получения данных {symbol} {timeframe}: {e}")
            return None

    def calculate_optimal_times(self):
        """Расчет времени входа и экспирации"""
        current_time = self.get_moscow_time()
        current_minute = current_time.minute
        current_second = current_time.second
        
        # Расчет следующего 5-минутного интервала
        minutes_to_next = (5 - (current_minute % 5)) % 5
        seconds_to_next = (60 - current_second) if minutes_to_next == 0 else (minutes_to_next * 60 - current_second)
        
        # Вход за 15 секунд до начала интервала
        entry_offset = 15
        if seconds_to_next > entry_offset:
            entry_time = current_time + timedelta(seconds=seconds_to_next - entry_offset)
        else:
            entry_time = current_time + timedelta(seconds=1)
        
        # Экспирация через 5 минут
        expiration_time = entry_time + timedelta(minutes=5)
        
        return entry_time, expiration_time, seconds_to_next

    async def analyze_symbol(self, symbol: str):
        """Строгий анализ одного символа с минимальной уверенностью 70%"""
        try:
            # Получаем данные для всех таймфреймов
            df_1m = await self.fetch_ohlcv_data(symbol, '1m', 25)
            df_5m = await self.fetch_ohlcv_data(symbol, '5m', 25)
            df_15m = await self.fetch_ohlcv_data(symbol, '15m', 25)
            df_1h = await self.fetch_ohlcv_data(symbol, '1h', 25)
            
            # Проверяем минимальное количество данных
            timeframes_data = {
                '1m': df_1m, '5m': df_5m, 
                '15m': df_15m, '1h': df_1h
            }
            
            valid_timeframes = {}
            for tf, df in timeframes_data.items():
                if df is not None and len(df) >= 20:
                    valid_timeframes[tf] = df
            
            if len(valid_timeframes) < 3:  # Минимум 3 таймфрейма
                return None
            
            # Анализируем multiple timeframe
            timeframe_signals = self.signal_generator.analyze_multiple_timeframes(
                df_1m, df_5m, df_15m, df_1h
            )
            
            # Генерируем окончательный сигнал (только с уверенностью >= 70%)
            final_signal = self.signal_generator.generate_final_signal(timeframe_signals, symbol)
            
            if final_signal is None:
                symbol_name = symbol.replace('/USDT', '')
                self.statistics['symbol_stats'][symbol_name]['neutral'] += 1
                return None
            
            # Рассчитываем время входа
            entry_time, expiration_time, seconds_to_next = self.calculate_optimal_times()
            
            # Получаем текущую цену
            try:
                ticker = self.exchange.fetch_ticker(symbol.replace('/', ''))
                current_price = ticker['last']
            except:
                current_price = None
            
            signal_data = {
                'symbol': symbol,
                'signal': final_signal,
                'entry_time': entry_time,
                'expiration_time': expiration_time,
                'current_price': current_price,
                'analysis_time': self.get_moscow_time(),
                'time_to_entry': seconds_to_next
            }
            
            # Обновляем статистику
            symbol_name = symbol.replace('/USDT', '')
            if final_signal == 'LONG':
                self.statistics['symbol_stats'][symbol_name]['long'] += 1
            else:
                self.statistics['symbol_stats'][symbol_name]['short'] += 1
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

    async def analyze_market(self):
        """Строгий анализ всего рынка с минимальной уверенностью 70%"""
        logger.info("Запуск строгого анализа рынка...")
        start_time = time.time()
        self.last_analysis_time = self.get_moscow_time()
        self.statistics['total_analyses'] += 1
        
        signals = []
        
        # Анализируем каждый символ
        for symbol in self.target_symbols:
            try:
                if time.time() - start_time > self.config['max_analysis_time']:
                    break
                
                signal = await self.analyze_symbol(symbol)
                if signal:  # Сигналы уже отфильтрованы по уверенности 70%
                    signals.append(signal)
                    self.statistics['signals_generated'] += 1
                    self.statistics['last_signal_time'] = self.get_moscow_time()
                    logger.info(f"Найден строгий сигнал для {symbol}: {signal['signal']}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")
                continue
        
        # Сортируем сигналы по времени до входа
        signals.sort(key=lambda x: x['time_to_entry'])
        self.signals = signals
        
        analysis_time = time.time() - start_time
        logger.info(f"Строгий анализ завершен за {analysis_time:.1f}с. Найдено {len(signals)} сигналов")
        
        # Отправляем уведомление о сильных сигналах
        if signals:
            await self.send_telegram_message(
                f"🎯 <b>Обнаружены строгие сигналы!</b>\n"
                f"📊 Количество: {len(signals)}\n"
                f"✅ Уверенность: ≥70%\n"
                f"⏰ Время: {self.format_moscow_time()}"
            )
        
        return signals

    def print_signals(self):
        """Вывод сигналов в консоль"""
        if not self.signals:
            print("🚫 Нет торговых сигналов с уверенностью ≥70%")
            return
            
        print("\n" + "="*80)
        print("🎯 СТРОГИЕ СИГНАЛЫ ДЛЯ 5-МИНУТНЫХ ОПЦИОНОВ")
        print("✅ Минимальная уверенность: 70%")
        print(f"⏰ Время анализа: {self.format_moscow_time(self.last_analysis_time)}")
        print("="*80)
        
        for i, signal in enumerate(self.signals):
            symbol_name = signal['symbol'].replace('/USDT', '')
            time_to_entry = signal['entry_time'] - self.get_moscow_time()
            seconds_to_entry = max(0, int(time_to_entry.total_seconds()))
            
            print(f"\n{i+1}. {symbol_name} | {signal['signal']}")
            print(f"   ⏰ Вход: {signal['entry_time'].strftime('%H:%M:%S')} (через {seconds_to_entry} сек)")
            print(f"   📅 Экспирация: {signal['expiration_time'].strftime('%H:%M:%S')}")
            
        print("="*80)

    async def run_continuous(self):
        """Непрерывный режим работы"""
        analysis_count = 0
        
        while True:
            try:
                analysis_count += 1
                current_time = self.format_moscow_time()
                
                print(f"\n{'='*50}")
                print(f"🎯 СТРОГИЙ АНАЛИЗ #{analysis_count} - {current_time}")
                print(f"💰 Монеты: BTC, ETH, SOL")
                print(f"✅ Минимальная уверенность: 70%")
                print(f"⏰ Таймфреймы: 1M, 5M, 15M, 1H")
                print(f"{'='*50}")
                
                await self.analyze_market()
                self.print_signals()
                
                print(f"⏳ Следующий анализ через {self.config['analysis_interval']}с...")
                await asyncio.sleep(self.config['analysis_interval'])
                
            except KeyboardInterrupt:
                print("\n\n🛑 Бот остановлен пользователем")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                await asyncio.sleep(10)

    async def cleanup(self):
        """Корректное завершение работы"""
        if self.telegram_app:
            try:
                await self.telegram_app.updater.stop()
                await self.telegram_app.stop()
                await self.telegram_app.shutdown()
            except:
                pass
        if self.telegram_worker_task:
            self.telegram_worker_task.cancel()
        await self.close_session()

async def main():
    bot = MEXCStrictOptionsBot()
    
    try:
        await bot.initialize_session()
        
        print("🎯 Запуск строгого MEXC бота для 5-минутных опционов...")
        print("💰 Монеты: BTC, ETH, SOL")
        print("⏰ Тип сделок: 5-минутные опционы LONG/SHORT")
        print("🏢 Биржа: MEXC")
        print(f"🌍 Часовой пояс: Москва (UTC+3)")
        print(f"🕐 Текущее время: {bot.format_moscow_time()}")
        print("📱 Управление через кнопки в Telegram")
        print("✅ Минимальная уверенность сигналов: 70%")
        print("⏰ Таймфреймы: 1M, 5M, 15M, 1H")
        print("🔁 Анализ каждые 60 секунд")
        print("⏸️ Для остановки нажмите Ctrl+C\n")
        
        await bot.initialize_telegram()
        
        print("🎯 Выполняю первоначальный строгий анализ...")
        await bot.analyze_market()
        bot.print_signals()
        
        await bot.run_continuous()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Остановка бота...")
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
    finally:
        await bot.cleanup()
        print("👋 Бот завершил работу")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Программа завершена")
    except Exception as e:
        print(f"🔄 Перезапуск после ошибки: {e}")
        time.sleep(10)