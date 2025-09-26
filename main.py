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
        """Экспоненциальная скользящая средная с TA-Lib"""
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
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
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
    def ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26, senkou_period=52):
        """Ишимоку Кинко Хайо (Облако Ишимоку)"""
        tenkan_sen = (talib.MAX(high, tenkan_period) + talib.MIN(low, tenkan_period)) / 2
        kijun_sen = (talib.MAX(high, kijun_period) + talib.MIN(low, kijun_period)) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        senkou_span_b = ((talib.MAX(high, senkou_period) + talib.MIN(low, senkou_period)) / 2).shift(kijun_period)
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

class EnhancedSignalGenerator:
    """Улучшенный генератор торговых сигналов с фильтрацией ложных срабатываний"""
    
    def __init__(self):
        self.ta = AdvancedTechnicalAnalyzer()
        self.last_signals = defaultdict(lambda: {'signal': 'NEUTRAL', 'timestamp': 0})
    
    def analyze_multiple_timeframes(self, df_5m, df_15m, df_1h):
        """Анализ нескольких таймфреймов для подтверждения сигнала"""
        signals = {}
        
        for timeframe, df in [('5m', df_5m), ('15m', df_15m), ('1h', df_1h)]:
            if df is not None and len(df) > 50:  # Увеличили минимальное количество свечей
                signals[timeframe] = self.analyze_single_timeframe(df, timeframe)
        
        return signals
    
    def analyze_single_timeframe(self, df, timeframe):
        """Анализ одного таймфрейма с улучшенной логикой"""
        if len(df) < 50:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': 0, 'indicators': {}}
        
        # Расчет всех индикаторов
        df = self.calculate_all_indicators(df)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Проверка качества данных
        if self.has_invalid_data(current):
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': 0, 'indicators': {}}
        
        # Система взвешенных баллов
        bullish_score = 0
        bearish_score = 0
        max_score = 0
        
        indicators_status = {}
        
        # 1. Трендовые индикаторы (высокий вес)
        # EMA направление
        if not self.is_nan(current['ema_9']) and not self.is_nan(current['ema_21']):
            max_score += 4  # Увеличили вес
            if current['ema_9'] > current['ema_21'] and current['ema_9'] > current['ema_50']:
                bullish_score += 4
                indicators_status['ema'] = 'BULLISH'
            elif current['ema_9'] < current['ema_21'] and current['ema_9'] < current['ema_50']:
                bearish_score += 4
                indicators_status['ema'] = 'BEARISH'
            else:
                indicators_status['ema'] = 'NEUTRAL'
        
        # 2. MACD анализ (средний вес)
        if not self.is_nan(current['macd']) and not self.is_nan(current['macd_signal']):
            max_score += 3
            macd_above_signal = current['macd'] > current['macd_signal']
            macd_positive = current['macd'] > 0
            
            if macd_above_signal and macd_positive and current['macd_hist'] > prev['macd_hist']:
                bullish_score += 3
                indicators_status['macd'] = 'BULLISH'
            elif not macd_above_signal and not macd_positive and current['macd_hist'] < prev['macd_hist']:
                bearish_score += 3
                indicators_status['macd'] = 'BEARISH'
            else:
                indicators_status['macd'] = 'NEUTRAL'
        
        # 3. RSI с дивергенцией (высокий вес)
        if not self.is_nan(current['rsi']):
            max_score += 3
            rsi_signal = self.analyze_rsi(df['rsi'].tail(10), df['close'].tail(10))
            if rsi_signal == 'BULLISH':
                bullish_score += 3
                indicators_status['rsi'] = 'BULLISH'
            elif rsi_signal == 'BEARISH':
                bearish_score += 3
                indicators_status['rsi'] = 'BEARISH'
            else:
                indicators_status['rsi'] = 'NEUTRAL'
        
        # 4. Стохастик (средний вес)
        if not self.is_nan(current['stoch_k']) and not self.is_nan(current['stoch_d']):
            max_score += 2
            if current['stoch_k'] < 20 and current['stoch_k'] > current['stoch_d']:
                bullish_score += 2
                indicators_status['stoch'] = 'BULLISH'
            elif current['stoch_k'] > 80 and current['stoch_k'] < current['stoch_d']:
                bearish_score += 2
                indicators_status['stoch'] = 'BEARISH'
            else:
                indicators_status['stoch'] = 'NEUTRAL'
        
        # 5. Полосы Боллинджера (высокий вес)
        bb_signal = self.analyze_bollinger_bands(current, df)
        max_score += 3
        if bb_signal == 'LONG':
            bullish_score += 3
            indicators_status['bb'] = 'BULLISH'
        elif bb_signal == 'SHORT':
            bearish_score += 3
            indicators_status['bb'] = 'BEARISH'
        else:
            indicators_status['bb'] = 'NEUTRAL'
        
        # 6. Объемы и MFI (средний вес)
        if not self.is_nan(current['volume_ratio']) and not self.is_nan(current['mfi']):
            max_score += 2
            volume_condition = current['volume_ratio'] > 1.5
            mfi_condition = current['mfi'] < 20 or current['mfi'] > 80
            
            if volume_condition and current['close'] > current['open'] and current['mfi'] < 20:
                bullish_score += 2
                indicators_status['volume'] = 'BULLISH'
            elif volume_condition and current['close'] < current['open'] and current['mfi'] > 80:
                bearish_score += 2
                indicators_status['volume'] = 'BEARISH'
            else:
                indicators_status['volume'] = 'NEUTRAL'
        
        # 7. ADX для силы тренда (низкий вес)
        if not self.is_nan(current['adx']):
            max_score += 1
            if current['adx'] > 25:  # Сильный тренд
                if bullish_score > bearish_score:
                    bullish_score += 1
                else:
                    bearish_score += 1
                indicators_status['adx'] = 'STRONG_TREND'
            else:
                indicators_status['adx'] = 'WEAK_TREND'
        
        # 8. Ишимоку Кинко Хайо (высокий вес)
        ichimoku_signal = self.analyze_ichimoku(current, df)
        max_score += 4
        if ichimoku_signal == 'BULLISH':
            bullish_score += 4
            indicators_status['ichimoku'] = 'BULLISH'
        elif ichimoku_signal == 'BEARISH':
            bearish_score += 4
            indicators_status['ichimoku'] = 'BEARISH'
        else:
            indicators_status['ichimoku'] = 'NEUTRAL'
        
        # 9. Awesome Oscillator (средний вес)
        ao_signal = self.analyze_awesome_oscillator(df['ao'].tail(5))
        max_score += 2
        if ao_signal == 'BULLISH':
            bullish_score += 2
            indicators_status['ao'] = 'BULLISH'
        elif ao_signal == 'BEARISH':
            bearish_score += 2
            indicators_status['ao'] = 'BEARISH'
        else:
            indicators_status['ao'] = 'NEUTRAL'
        
        # Расчет уверенности
        total_score = max(1, max_score)
        confidence = (abs(bullish_score - bearish_score) / total_score) * 100
        
        # Увеличиваем порог уверенности для уменьшения ложных сигналов
        min_confidence = 65 if timeframe == '1h' else 55
        
        # Определение сигнала с порогом уверенности
        if confidence < min_confidence or abs(bullish_score - bearish_score) < 5:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': confidence, 'indicators': indicators_status}
        
        if bullish_score > bearish_score:
            return {'signal': 'LONG', 'strength': bullish_score, 'confidence': confidence, 'indicators': indicators_status}
        elif bearish_score > bullish_score:
            return {'signal': 'SHORT', 'strength': bearish_score, 'confidence': confidence, 'indicators': indicators_status}
        else:
            return {'signal': 'NEUTRAL', 'strength': 0, 'confidence': confidence, 'indicators': indicators_status}
    
    def calculate_all_indicators(self, df):
        """Расчет всех технических индикаторов"""
        try:
            # Трендовые
            df['ema_9'] = self.ta.ema(df['close'], 9)
            df['ema_21'] = self.ta.ema(df['close'], 21)
            df['ema_50'] = self.ta.ema(df['close'], 50)
            df['sma_100'] = self.ta.sma(df['close'], 100)
            
            # Моментум
            df['rsi'] = self.ta.rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = self.ta.macd(df['close'])
            df['stoch_k'], df['stoch_d'] = self.ta.stochastic(df['high'], df['low'], df['close'])
            df['williams_r'] = self.ta.williams_r(df['high'], df['low'], df['close'])
            df['cci'] = self.ta.cci(df['high'], df['low'], df['close'])
            
            # Волатильность
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.ta.bollinger_bands(df['close'])
            df['atr'] = self.ta.atr(df['high'], df['low'], df['close'])
            
            # Тренд сила
            df['adx'] = self.ta.adx(df['high'], df['low'], df['close'])
            
            # Объем
            df['obv'] = self.ta.obv(df['close'], df['volume'])
            df['volume_ratio'] = self.ta.volume_profile(df['volume'], 20)
            df['mfi'] = self.ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            
            # Дополнительные индикаторы
            df['ao'] = self.ta.awesome_oscillator(df['high'], df['low'])
            
            # Ишимоку (упрощенная версия)
            tenkan, kijun, senkou_a, senkou_b = self.ta.ichimoku_cloud(df['high'], df['low'], df['close'])
            df['tenkan_sen'] = tenkan
            df['kijun_sen'] = kijun
            
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
        
        return df
    
    def analyze_rsi(self, rsi_data, price_data):
        """Анализ RSI с проверкой дивергенции"""
        if len(rsi_data) < 5:
            return 'NEUTRAL'
        
        current_rsi = rsi_data.iloc[-1]
        prev_rsi = rsi_data.iloc[-2]
        current_price = price_data.iloc[-1]
        prev_price = price_data.iloc[-2]
        
        # Бычья дивергенция: цена делает нижний минимум, RSI - более высокий
        if (current_price < prev_price and current_rsi > prev_rsi and 
            current_rsi < 35 and current_rsi > prev_rsi):
            return 'BULLISH'
        
        # Медвежья дивергенция: цена делает верхний максимум, RSI - более низкий
        if (current_price > prev_price and current_rsi < prev_rsi and 
            current_rsi > 65 and current_rsi < prev_rsi):
            return 'BEARISH'
        
        # Стандартный анализ RSI
        if current_rsi < 30:
            return 'BULLISH'
        elif current_rsi > 70:
            return 'BEARISH'
        
        return 'NEUTRAL'
    
    def analyze_bollinger_bands(self, current, df):
        """Анализ положения цены относительно полос Боллинджера"""
        if (self.is_nan(current['bb_upper']) or self.is_nan(current['bb_lower']) or 
            self.is_nan(current['bb_middle'])):
            return 'NEUTRAL'
        
        price = current['close']
        bb_width = (current['bb_upper'] - current['bb_lower']) / current['bb_middle']
        
        if bb_width < 0.015:  # Увеличили порог для узких полос
            return 'NEUTRAL'
        
        # Проверяем несколько последних свечей для подтверждения
        recent_prices = df['close'].tail(3)
        recent_lowers = df['bb_lower'].tail(3)
        recent_uppers = df['bb_upper'].tail(3)
        
        # Бычий сигнал: цена коснулась нижней полосы и начинает рост
        if (price <= current['bb_lower'] * 1.005 and 
            all(recent_prices.iloc[i] <= recent_lowers.iloc[i] * 1.01 for i in range(2))):
            return 'LONG'
        
        # Медвежий сигнал: цена коснулась верхней полосы и начинает падение
        if (price >= current['bb_upper'] * 0.995 and 
            all(recent_prices.iloc[i] >= recent_uppers.iloc[i] * 0.99 for i in range(2))):
            return 'SHORT'
        
        return 'NEUTRAL'
    
    def analyze_ichimoku(self, current, df):
        """Анализ облака Ишимоку"""
        if (self.is_nan(current['tenkan_sen']) or self.is_nan(current['kijun_sen'])):
            return 'NEUTRAL'
        
        price = current['close']
        tenkan = current['tenkan_sen']
        kijun = current['kijun_sen']
        
        # Бычий сигнал: цена выше тэнкан-сен и киджун-сен, тэнкан-сен выше киджун-сен
        if (price > tenkan and price > kijun and tenkan > kijun and
            df['close'].iloc[-2] > df['tenkan_sen'].iloc[-2] and
            df['close'].iloc[-2] > df['kijun_sen'].iloc[-2]):
            return 'BULLISH'
        
        # Медвежий сигнал: цена ниже тэнкан-сен и киджун-сен, тэнкан-сен ниже киджун-сен
        if (price < tenkan and price < kijun and tenkan < kijun and
            df['close'].iloc[-2] < df['tenkan_sen'].iloc[-2] and
            df['close'].iloc[-2] < df['kijun_sen'].iloc[-2]):
            return 'BEARISH'
        
        return 'NEUTRAL'
    
    def analyze_awesome_oscillator(self, ao_data):
        """Анализ Awesome Oscillator"""
        if len(ao_data) < 3:
            return 'NEUTRAL'
        
        current_ao = ao_data.iloc[-1]
        prev_ao = ao_data.iloc[-2]
        prev2_ao = ao_data.iloc[-3]
        
        # Бычий сигнал: AO перешел из отрицательной в положительную зону
        if current_ao > 0 and prev_ao <= 0 and prev2_ao <= 0:
            return 'BULLISH'
        
        # Медвежий сигнал: AO перешел из положительной в отрицательную зону
        if current_ao < 0 and prev_ao >= 0 and prev2_ao >= 0:
            return 'BEARISH'
        
        # Сигнал пинцета: два последовательных столбца одного цвета
        if current_ao > prev_ao and prev_ao > prev2_ao and current_ao > 0:
            return 'BULLISH'
        elif current_ao < prev_ao and prev_ao < prev2_ao and current_ao < 0:
            return 'BEARISH'
        
        return 'NEUTRAL'
    
    def is_nan(self, value):
        """Проверка на NaN"""
        return pd.isna(value) or np.isnan(value)
    
    def has_invalid_data(self, current):
        """Проверка на валидность данных"""
        required_indicators = ['ema_9', 'ema_21', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        for indicator in required_indicators:
            if self.is_nan(current[indicator]):
                return True
        return False
    
    def generate_final_signal(self, timeframe_signals, symbol):
        """Генерация окончательного сигнала с улучшенной логикой"""
        if not timeframe_signals:
            return None
        
        # Взвешивание по таймфреймам (старшие ТФ имеют больший вес)
        weights = {'1h': 5, '15m': 3, '5m': 2}
        
        long_score = 0
        short_score = 0
        total_weight = 0
        confidences = []
        all_indicators = {}
        
        for timeframe, signal_info in timeframe_signals.items():
            weight = weights.get(timeframe, 1)
            confidence = signal_info.get('confidence', 0)
            
            # Требуем минимальной уверенности для каждого таймфрейма
            min_tf_confidence = 50 if timeframe == '5m' else 60
            if confidence < min_tf_confidence:
                continue
            
            if signal_info['signal'] == 'LONG':
                long_score += weight * (confidence / 100)
                confidences.append(confidence)
            elif signal_info['signal'] == 'SHORT':
                short_score += weight * (confidence / 100)
                confidences.append(confidence)
            
            total_weight += weight
            all_indicators[timeframe] = signal_info.get('indicators', {})
        
        if total_weight == 0 or not confidences:
            return None
        
        # Средняя уверенность (взвешенная)
        avg_confidence = np.average(confidences, weights=[weights.get(tf, 1) for tf in timeframe_signals.keys()])
        
        # Повышенный порог для финального сигнала
        if avg_confidence < 70:  # Увеличили порог уверенности
            return None
        
        # Требуем явного превосходства одного направления
        min_ratio = 1.8  # Увеличили соотношение для уменьшения ложных сигналов
        
        if long_score > short_score * min_ratio:
            # Проверяем согласованность индикаторов
            if self.check_indicators_consistency(all_indicators, 'LONG'):
                return 'LONG'
        elif short_score > long_score * min_ratio:
            if self.check_indicators_consistency(all_indicators, 'SHORT'):
                return 'SHORT'
        
        return None
    
    def check_indicators_consistency(self, all_indicators, direction):
        """Проверка согласованности индикаторов across timeframes"""
        if not all_indicators:
            return False
        
        required_consistent = ['ema', 'macd', 'bb']
        consistent_count = 0
        
        for indicator in required_consistent:
            tf_agreement = 0
            total_tf = 0
            
            for tf_indicators in all_indicators.values():
                if indicator in tf_indicators:
                    total_tf += 1
                    indicator_status = tf_indicators[indicator]
                    
                    if (direction == 'LONG' and indicator_status == 'BULLISH') or \
                       (direction == 'SHORT' and indicator_status == 'BEARISH'):
                        tf_agreement += 1
            
            # Требуем согласия в большинстве таймфреймов
            if total_tf > 0 and tf_agreement / total_tf >= 0.6:
                consistent_count += 1
        
        # Требуем согласованности как минимум 2 из 3 ключевых индикаторов
        return consistent_count >= 2

class MEXC5MinuteBot:
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
        
        # Настройки для 5-минутных ордеров
        self.config = {
            'timeframes': ['5m', '15m', '1h'],
            'analysis_interval': 300,  # 5 минут
            'min_candles': 50,  # Увеличили для более точного анализа
            'max_analysis_time': 30
        }

        # Инициализация улучшенного анализатора
        self.signal_generator = EnhancedSignalGenerator()
        
        # Клавиатура с кнопками
        self.reply_keyboard = ReplyKeyboardMarkup(
            [
                ["📊 Получить сигналы", "🔄 Новый анализ"],
                ["⏰ Время торговли", "ℹ️ Инструкция"],
                ["📈 Статистика", "⚙️ Настройки"]
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

        logger.info("Улучшенный MEXC Бот для 5-минутных ордеров инициализирован")

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

    def format_moscow_time(self, dt=None, format_str='%H:%M'):
        moscow_time = self.get_moscow_time(dt)
        return moscow_time.strftime(format_str)

    async def initialize_telegram(self):
        try:
            request = HTTPXRequest(connection_pool_size=3, read_timeout=10, write_timeout=10, connect_timeout=10)
            self.telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(request).build()
            
            self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
            self.telegram_app.add_handler(CommandHandler("stats", self.show_stats))
            self.telegram_app.add_handler(CommandHandler("settings", self.show_settings))
            
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()
            self.telegram_worker_task = asyncio.create_task(self.telegram_worker())
            logger.info("Telegram бот инициализирован")

            startup_message = (
                "🤖 <b>УЛУЧШЕННЫЙ MEXC БОТ ДЛЯ 5-МИНУТНЫХ ОРДЕРОВ</b>\n\n"
                "🎯 <b>Монеты:</b> BTC, ETH, SOL\n"
                "⏰ <b>Таймфреймы:</b> 5M, 15M, 1H\n"
                "📊 <b>Сигналы:</b> ТОЧНЫЕ LONG / SHORT\n"
                "🏢 <b>Биржа:</b> MEXC\n\n"
                "✅ <b>Улучшенный анализ:</b>\n"
                "• Мультитаймфреймная проверка\n"
                "• Фильтрация ложных сигналов\n"
                "• TA-Lib индикаторы\n"
                "• Проверка дивергенций\n\n"
                f"🕐 <b>Запуск:</b> {self.format_moscow_time()}"
            )
            await self.send_telegram_message(startup_message)

        except Exception as e:
            logger.error(f"Ошибка инициализации Telegram бота: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        text = update.message.text
        
        if text == "📊 Получить сигналы":
            await self.handle_signals(update, context)
        elif text == "🔄 Новый анализ":
            await self.handle_analysis(update, context)
        elif text == "⏰ Время торговли":
            await self.handle_trading_time(update, context)
        elif text == "ℹ️ Инструкция":
            await self.handle_help(update, context)
        elif text == "📈 Статистика":
            await self.show_stats(update, context)
        elif text == "⚙️ Настройки":
            await self.show_settings(update, context)
        else:
            await self.handle_unknown(update, context)

    async def handle_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Получить сигналы'"""
        if not self.signals:
            await update.message.reply_text(
                "📊 <b>Активных сигналов нет</b>\n\n"
                "Нажмите '🔄 Новый анализ' для сканирования рынка",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            return

        try:
            message = "🎯 <b>ТОЧНЫЕ ТОРГОВЫЕ СИГНАЛЫ</b>\n\n"
            message += "⏰ <b>5-минутные ордера • MEXC</b>\n\n"

            for i, signal in enumerate(self.signals[:5]):
                symbol_name = signal['symbol'].replace('/USDT', '')
                
                # Определяем эмодзи для сигнала
                if signal['signal'] == 'LONG':
                    signal_emoji = "🟢"
                    action_text = "LONG"
                    confidence_color = "🟢"
                else:
                    signal_emoji = "🔴" 
                    action_text = "SHORT"
                    confidence_color = "🔴"
                
                # Форматируем время
                expiration_str = signal['expiration_time'].strftime('%H:%M')
                entry_str = signal['entry_time'].strftime('%H:%M')
                
                message += (
                    f"{signal_emoji} <b>{symbol_name}</b>\n"
                    f"📈 Направление: <b>{action_text}</b>\n"
                    f"⏰ Вход: <b>{entry_str}</b>\n"
                    f"📅 Экспирация: <b>{expiration_str}</b>\n"
                    f"🎯 Уверенность: <b>{signal['confidence']:.1f}%</b> {confidence_color}\n"
                )
                
                # Добавляем рекомендацию по времени
                current_time = self.get_moscow_time()
                time_to_entry = signal['entry_time'] - current_time
                minutes_to_entry = max(0, int(time_to_entry.total_seconds() / 60))
                
                if minutes_to_entry <= 1:
                    message += "⚡ <b>ВХОД СЕЙЧАС!</b>\n"
                elif minutes_to_entry <= 3:
                    message += f"⏱️ <b>Готовьтесь: через {minutes_to_entry} мин</b>\n"
                else:
                    message += f"🕐 <b>До входа: {minutes_to_entry} мин</b>\n"
                
                message += "━━━━━━━━━━━━━━━━━━━\n\n"

            message += f"🕐 <b>Обновлено:</b> {self.format_moscow_time(self.last_analysis_time)}\n"
            message += "🔁 <b>Автообновление каждые 5 мин.</b>"

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
                "📈 <b>СТАТИСТИКА БОТА</b>\n\n"
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
            message += "<b>Статистика по монетам:</b>\n"
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

    async def show_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать настройки бота"""
        try:
            message = (
                "⚙️ <b>НАСТРОЙКИ БОТА</b>\n\n"
                f"🎯 Монеты: <b>{', '.join([s.replace('/USDT', '') for s in self.target_symbols])}</b>\n"
                f"⏰ Таймфреймы: <b>{', '.join(self.config['timeframes'])}</b>\n"
                f"📊 Интервал анализа: <b>{self.config['analysis_interval']} сек</b>\n"
                f"🕐 Часовой пояс: <b>Москва (UTC+3)</b>\n\n"
                "<b>Параметры сигналов:</b>\n"
                "• Минимальная уверенность: 70%\n"
                "• Мультитаймфреймная проверка\n"
                "• Фильтрация ложных сигналов\n"
                "• Проверка дивергенций\n"
            )
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )
            
        except Exception as e:
            logger.error(f"Ошибка показа настроек: {e}")
            await update.message.reply_text(
                "❌ <b>Ошибка загрузки настроек</b>",
                parse_mode='HTML',
                reply_markup=self.reply_keyboard
            )

    async def handle_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Новый анализ'"""
        try:
            await update.message.reply_text(
                "🔍 <b>Запускаю улучшенный анализ...</b>\n"
                "Сканирую BTC, ETH, SOL...",
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

    async def handle_trading_time(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик кнопки 'Время торговли'"""
        try:
            current_time = self.get_moscow_time()
            
            # Рассчитываем ближайшие 5-минутные интервалы
            current_minute = current_time.minute
            minutes_to_next = (5 - (current_minute % 5)) % 5
            
            if minutes_to_next == 0:
                next_entry = current_time
                next_expiry = current_time + timedelta(minutes=5)
                status = "⚡ СЕЙЧАС"
            else:
                next_entry = current_time + timedelta(minutes=minutes_to_next)
                next_expiry = next_entry + timedelta(minutes=5)
                status = f"⏱️ Через {minutes_to_next} мин"
            
            time_message = (
                "⏰ <b>РАСПИСАНИЕ ТОРГОВЛИ</b>\n\n"
                f"🕐 <b>Текущее время:</b> {current_time.strftime('%H:%M')}\n"
                f"🎯 <b>Следующий вход:</b> {next_entry.strftime('%H:%M')}\n"
                f"📅 <b>Экспирация:</b> {next_expiry.strftime('%H:%M')}\n"
                f"📊 <b>Статус:</b> {status}\n\n"
            )
            
            # Расписание на ближайшие 30 минут
            time_message += "<b>Ближайшие интервалы:</b>\n"
            for i in range(6):
                entry_time = next_entry + timedelta(minutes=5*i)
                expiry_time = entry_time + timedelta(minutes=5)
                time_message += f"• {entry_time.strftime('%H:%M')} - {expiry_time.strftime('%H:%M')}\n"
            
            time_message += "\n💡 <b>Совет:</b> Входите за 1-2 минуты до начала интервала"
            
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
            "🤖 <b>ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ</b>\n\n"
            "🎯 <b>Улучшенная стратегия торговли:</b>\n"
            "• LONG - покупаем при подтвержденном восходящем тренде\n"
            "• SHORT - продаем при подтвержденном нисходящем тренде\n"
            "• Экспирация через 5 минут\n"
            "• Вход за 1-2 минуты до начала интервала\n\n"
            "📊 <b>Как использовать бота:</b>\n"
            "1. Нажмите '🔄 Новый анализ' для сканирования\n"
            "2. Посмотрите сигналы в '📊 Получить сигналы'\n"
            "3. Следите за временем в '⏰ Время торговли'\n"
            "4. Входите в сделку за 1-2 минуты до начала интервала\n\n"
            "⏰ <b>Временные интервалы:</b>\n"
            "• Интервалы начинаются каждые 5 минут\n"
            "• Пример: 14:00, 14:05, 14:10 и т.д.\n"
            "• Экспирация через 5 минут после входа\n\n"
            "⚡ <b>Пример работы:</b>\n"
            "• Текущее время: 14:03\n"
            "• Следующий вход: 14:05\n"
            "• Входить в: 14:04\n"
            "• Экспирация: 14:10\n\n"
            "✅ <b>Улучшения против ложных сигналов:</b>\n"
            "• Мультитаймфреймная проверка (5M, 15M, 1H)\n"
            "• Минимальная уверенность 70%\n"
            "• Проверка дивергенций RSI\n"
            "• Согласованность индикаторов\n"
            "• Фильтрация по объему и волатильности"
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
            "🚀 <b>УЛУЧШЕННЫЙ MEXC БОТ ДЛЯ 5-МИНУТНЫХ ОРДЕРОВ</b>\n\n"
            "🎯 <b>Специализация:</b> Точные 5-минутные ордера LONG/SHORT\n"
            "💰 <b>Монеты:</b> Bitcoin (BTC), Ethereum (ETH), Solana (SOL)\n"
            "⏰ <b>Экспирация:</b> 5 минут\n"
            "🏢 <b>Биржа:</b> MEXC\n\n"
            "✅ <b>Улучшенные преимущества:</b>\n"
            "• TA-Lib индикаторы для точности\n"
            "• Мультитаймфреймный анализ (5M, 15M, 1H)\n"
            "• Фильтрация ложных сигналов\n"
            "• Проверка дивергенций RSI\n"
            "• Минимальная уверенность 70%\n\n"
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
                            "⏳ <b>Анализ уже выполняется...</b>",
                            parse_mode='HTML',
                            reply_markup=self.reply_keyboard
                        )
                    return

                self.is_analyzing = True
                
                if update:
                    await update.message.reply_text(
                        "🔍 <b>Запускаю улучшенный анализ...</b>\n"
                        "BTC → ETH → SOL\n"
                        "5M → 15M → 1H таймфреймы\n"
                        "Проверка дивергенций...",
                        parse_mode='HTML',
                        reply_markup=self.reply_keyboard
                    )

                signals = await self.analyze_market()

                if update:
                    if signals:
                        await update.message.reply_text(
                            f"✅ <b>Анализ завершен</b>\n\n"
                            f"📊 Найдено сигналов: <b>{len(signals)}</b>\n"
                            f"🎯 Уверенность: <b>70%+</b>\n\n"
                            f"Нажмите '📊 Получить сигналы' для просмотра",
                            parse_mode='HTML',
                            reply_markup=self.reply_keyboard
                        )
                    else:
                        await update.message.reply_text(
                            "ℹ️ <b>Анализ завершен</b>\n"
                            "Сильных сигналов не найдено\n"
                            "Попробуйте позже",
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
                'timeout': 15000,  # Увеличили таймаут
            })
            exchange.load_markets()
            
            # Проверяем доступные таймфреймы
            timeframes = exchange.timeframes
            logger.info(f"Доступные таймфреймы MEXC: {timeframes}")
            
            return exchange
        except Exception as e:
            logger.error(f"Ошибка подключения к MEXC: {e}")
            return None

    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100):
        if self.exchange is None:
            return None

        try:
            # Проверяем доступность таймфрейма
            if timeframe not in self.exchange.timeframes:
                logger.error(f"Таймфрейм {timeframe} не поддерживается MEXC")
                return None
            
            normalized_symbol = symbol.replace('/', '')
            await asyncio.sleep(0.05)  # Увеличили задержку для избежания лимитов
            
            ohlcv = self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 50:  # Увеличили минимальное количество свечей
                logger.warning(f"Недостаточно данных для {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Проверяем качество данных
            if df['close'].isna().any() or df['volume'].isna().any():
                logger.warning(f"Обнаружены NaN значения в данных {symbol}")
                return None
                
            return df

        except Exception as e:
            logger.error(f"Ошибка получения данных {symbol} {timeframe}: {e}")
            return None

    def calculate_optimal_times(self):
        """Расчет оптимального времени входа и экспирации"""
        current_time = self.get_moscow_time()
        current_minute = current_time.minute
        
        # Расчет следующего 5-минутного интервала
        minutes_to_next = (5 - (current_minute % 5)) % 5
        
        if minutes_to_next == 0:
            # Мы в начале интервала
            entry_time = current_time
        else:
            # Ждем следующего интервала
            entry_time = current_time + timedelta(minutes=minutes_to_next)
        
        # Экспирация через 5 минут
        expiration_time = entry_time + timedelta(minutes=5)
        
        return entry_time, expiration_time, minutes_to_next

    async def analyze_symbol(self, symbol: str):
        """Анализ одного символа с улучшенной логикой"""
        try:
            # Получаем данные для всех таймфреймов с увеличенным лимитом
            df_5m = await self.fetch_ohlcv_data(symbol, '5m', 100)
            df_15m = await self.fetch_ohlcv_data(symbol, '15m', 100) 
            df_1h = await self.fetch_ohlcv_data(symbol, '1h', 100)
            
            if df_5m is None or len(df_5m) < 50:
                logger.warning(f"Недостаточно данных 5m для {symbol}")
                return None
            if df_15m is None or len(df_15m) < 50:
                logger.warning(f"Недостаточно данных 15m для {symbol}")
                return None
            if df_1h is None or len(df_1h) < 50:
                logger.warning(f"Недостаточно данных 1h для {symbol}")
                return None
            
            # Анализируем multiple timeframe
            timeframe_signals = self.signal_generator.analyze_multiple_timeframes(df_5m, df_15m, df_1h)
            
            # Генерируем окончательный сигнал
            final_signal = self.signal_generator.generate_final_signal(timeframe_signals, symbol)
            
            if final_signal is None:
                # Обновляем статистику
                symbol_name = symbol.replace('/USDT', '')
                self.statistics['symbol_stats'][symbol_name]['neutral'] += 1
                return None
            
            # Рассчитываем время входа
            entry_time, expiration_time, minutes_to_next = self.calculate_optimal_times()
            
            # Получаем текущую цену
            try:
                ticker = self.exchange.fetch_ticker(symbol.replace('/', ''))
                current_price = ticker['last']
            except:
                current_price = None
            
            # Рассчитываем общую уверенность
            confidences = [s.get('confidence', 0) for s in timeframe_signals.values()]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            signal_data = {
                'symbol': symbol,
                'signal': final_signal,
                'entry_time': entry_time,
                'expiration_time': expiration_time,
                'current_price': current_price,
                'confidence': avg_confidence,
                'analysis_time': self.get_moscow_time(),
                'timeframe_signals': timeframe_signals
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
        """Анализ всего рынка"""
        logger.info("Запуск улучшенного анализа рынка...")
        start_time = time.time()
        self.last_analysis_time = self.get_moscow_time()
        self.statistics['total_analyses'] += 1
        
        signals = []
        
        # Анализируем каждый символ
        for symbol in self.target_symbols:
            try:
                if time.time() - start_time > self.config['max_analysis_time']:
                    logger.warning(f"Превышено время анализа для {symbol}")
                    break
                
                signal = await self.analyze_symbol(symbol)
                if signal and signal['confidence'] >= 70:  # Фильтр по уверенности
                    signals.append(signal)
                    self.statistics['signals_generated'] += 1
                    self.statistics['last_signal_time'] = self.get_moscow_time()
                
                await asyncio.sleep(0.1)  # Увеличили задержку
                
            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")
                continue
        
        # Сортируем сигналы по уверенности
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        self.signals = signals
        
        analysis_time = time.time() - start_time
        logger.info(f"Анализ завершен за {analysis_time:.1f}с. Найдено {len(signals)} сигналов")
        
        # Отправляем уведомление в Telegram если есть сигналы
        if signals:
            strong_signals = [s for s in signals if s['confidence'] >= 80]
            if strong_signals:
                await self.send_telegram_message(
                    f"🚨 <b>Обнаружены сильные сигналы!</b>\n"
                    f"📊 Количество: {len(strong_signals)}\n"
                    f"🎯 Уверенность: 80%+\n"
                    f"⏰ Время: {self.format_moscow_time()}"
                )
        
        return signals

    def print_signals(self):
        """Вывод сигналов в консоль"""
        if not self.signals:
            print("🚫 Нет торговых сигналов")
            return
            
        print("\n" + "="*80)
        print("🎯 УЛУЧШЕННЫЕ СИГНАЛЫ ДЛЯ 5-МИНУТНЫХ ОРДЕРОВ")
        print(f"⏰ Время анализа: {self.format_moscow_time(self.last_analysis_time)}")
        print("="*80)
        
        for i, signal in enumerate(self.signals):
            symbol_name = signal['symbol'].replace('/USDT', '')
            time_to_entry = signal['entry_time'] - self.get_moscow_time()
            minutes_to_entry = max(0, int(time_to_entry.total_seconds() / 60))
            
            print(f"\n{i+1}. {symbol_name} | {signal['signal']} | Уверенность: {signal['confidence']:.1f}%")
            print(f"   ⏰ Вход: {signal['entry_time'].strftime('%H:%M')} (через {minutes_to_entry} мин)")
            print(f"   📅 Экспирация: {signal['expiration_time'].strftime('%H:%M')}")
            
        print("="*80)

    async def run_continuous(self):
        """Непрерывный режим работы"""
        analysis_count = 0
        
        while True:
            try:
                analysis_count += 1
                current_time = self.format_moscow_time()
                
                print(f"\n{'='*50}")
                print(f"📊 АНАЛИЗ #{analysis_count} - {current_time}")
                print(f"🎯 5-минутные ордера: BTC, ETH, SOL")
                print(f"🎯 Минимальная уверенность: 70%")
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
                await asyncio.sleep(10)  # Увеличили задержку при ошибке

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
    bot = MEXC5MinuteBot()
    
    try:
        await bot.initialize_session()
        
        print("🚀 Запуск улучшенного MEXC бота для 5-минутных ордеров...")
        print("🎯 Монеты: BTC, ETH, SOL")
        print("⏰ Тип сделок: 5-минутные ордера LONG/SHORT")
        print("🏢 Биржа: MEXC")
        print(f"🌍 Часовой пояс: Москва (UTC+3)")
        print(f"🕐 Текущее время: {bot.format_moscow_time()}")
        print("📱 Управление через кнопки в Telegram")
        print("🎯 Минимальная уверенность сигналов: 70%")
        print("⏸️ Для остановки нажмите Ctrl+C\n")
        
        await bot.initialize_telegram()
        
        print("📊 Выполняю первоначальный анализ...")
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
