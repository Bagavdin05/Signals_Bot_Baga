import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
import asyncio
import talib
from typing import Dict, List, Optional
import requests
import json
import warnings

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MEXCBot')

# Конфигурация Telegram
TELEGRAM_BOT_TOKEN = "7952768185:AAGuhybXaGPJqtlGPd1-O4nc6_FpUL2rOgw"
TELEGRAM_CHAT_IDS = ["1167694150", "7916502470", "1111230981"]

class TelegramBot:
    def __init__(self, token: str, chat_ids: List[str]):
        self.token = token
        self.chat_ids = chat_ids
        self.base_url = f"https://api.telegram.org/bot{token}/"

    async def send_message(self, message: str, parse_mode: str = "HTML"):
        """Отправка сообщения во все чаты"""
        for chat_id in self.chat_ids:
            try:
                url = f"{self.base_url}sendMessage"
                payload = {
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': parse_mode
                }
                response = requests.post(url, data=payload, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Ошибка отправки в Telegram: {response.text}")
            except Exception as e:
                logger.error(f"Ошибка отправки в Telegram: {e}")

class TradingStatistics:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_streak = 0
        self.last_trade_profit = 0.0

    def add_trade(self, profit: float):
        """Добавление сделки в статистику"""
        self.total_trades += 1
        self.total_profit += profit
        self.last_trade_profit = profit

        if profit > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.current_streak = self.consecutive_wins
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.current_streak = -self.consecutive_losses

        if profit > self.max_profit:
            self.max_profit = profit
        if profit < self.max_loss:
            self.max_loss = profit

    def get_statistics(self) -> Dict:
        """Получение статистики"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_profit = (self.total_profit / self.total_trades) if self.total_trades > 0 else 0

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit': avg_profit,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'current_streak': self.current_streak,
            'last_trade_profit': self.last_trade_profit
        }

class MEXCTradingBot:
    def __init__(self):
        self.exchange = self.initialize_exchange()
        self.telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS)
        self.stats = TradingStatistics()

        # Обновленная конфигурация с таймфреймами 1м, 5м, 15м
        self.config = {
            'timeframes': ['1m', '5m', '15m'],  # Обновленные таймфреймы
            'symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'],
            'analysis_interval': 4,  # Интервал анализа
            'min_confidence': 0.78,  # Минимальная уверенность
            'position_hold_minutes': 5,  # время удержания
            'rsi_period': 9,  # Более чувствительный RSI
            'volume_ma_period': 10,
            'atr_period': 7,
            'min_volume_ratio': 1.3,  # Минимальное соотношение объема
            'max_volatility': 2.0,  # Максимальная волатильность (ATR %)
            'trend_strength_min': 0.6  # Минимальная сила тренда
        }

        self.active_positions = {}
        self.signal_history = []
        self.analysis_count = 0
        self.last_signals = {}

        logger.info("Улучшенный MEXC Trading Bot инициализирован с таймфреймами 1м, 5м, 15м")

    def initialize_exchange(self):
        """Инициализация подключения к MEXC"""
        try:
            exchange = ccxt.mexc({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap'
                },
                'timeout': 30000,
            })
            exchange.load_markets()
            logger.info("Успешное подключение к MEXC")
            return exchange
        except Exception as e:
            logger.error(f"Ошибка подключения к MEXC: {e}")
            return None

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Получение OHLCV данных с улучшенной обработкой"""
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Проверка качества данных
            if len(df) < 20:
                return None
                
            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol} на {timeframe}: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет улучшенных технических индикаторов"""
        try:
            # Быстрый RSI
            df['rsi_fast'] = talib.RSI(df['close'], timeperiod=9)
            df['rsi_slow'] = talib.RSI(df['close'], timeperiod=14)
            
            # Улучшенный MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=8, slowperiod=21, signalperiod=5)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Множественные EMA для определения тренда
            df['ema_5'] = talib.EMA(df['close'], timeperiod=5)
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema_13'] = talib.EMA(df['close'], timeperiod=13)
            df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
            
            # Bollinger Bands с разными параметрами
            bb_upper1, bb_middle1, bb_lower1 = talib.BBANDS(df['close'], timeperiod=13, nbdevup=1.5, nbdevdn=1.5)
            df['bb_upper'] = bb_upper1
            df['bb_lower'] = bb_lower1
            df['bb_position'] = (df['close'] - bb_lower1) / (bb_upper1 - bb_lower1)
            
            # Volume analysis
            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=self.config['volume_ma_period'])
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # ATR для волатильности
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.config['atr_period'])
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # Stochastic RSI
            stoch_k, stoch_d = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=3, fastd_period=3)
            df['stoch_rsi_k'] = stoch_k
            df['stoch_rsi_d'] = stoch_d
            
            # Momentum indicators
            df['momentum'] = talib.MOM(df['close'], timeperiod=6)
            df['rate_of_change'] = talib.ROC(df['close'], timeperiod=8)
            
            # Trend strength
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=9)
            
            # Support/Resistance levels
            df['resistance'] = df['high'].rolling(window=10).max()
            df['support'] = df['low'].rolling(window=10).min()
            
            # Price position relative to support/resistance
            df['price_vs_resistance'] = (df['close'] - df['resistance']) / df['resistance'] * 100
            df['price_vs_support'] = (df['close'] - df['support']) / df['support'] * 100
            
            return df
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df

    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Анализ рыночных условий"""
        if df is None or len(df) < 20:
            return {'trend': 'neutral', 'volatility': 'low', 'volume': 'low'}
            
        last = df.iloc[-1]
        
        # Определение тренда
        trend_score = 0
        if last['close'] > last['ema_13'] > last['ema_21']:
            trend_score += 2
        if last['ema_5'] > last['ema_13']:
            trend_score += 1
            
        if trend_score >= 2:
            trend = 'bullish'
        elif trend_score <= 0:
            trend = 'bearish'
        else:
            trend = 'neutral'
            
        # Анализ волатильности
        atr_percent = last['atr_percent']
        if atr_percent > 1.5:
            volatility = 'high'
        elif atr_percent > 0.8:
            volatility = 'medium'
        else:
            volatility = 'low'
            
        # Анализ объема
        volume_ratio = last['volume_ratio']
        if volume_ratio > 1.5:
            volume_status = 'high'
        elif volume_ratio > 1.0:
            volume_status = 'medium'
        else:
            volume_status = 'low'
            
        return {
            'trend': trend,
            'volatility': volatility,
            'volume': volume_status,
            'trend_strength': abs(last['adx']) if not np.isnan(last['adx']) else 0,
            'atr_percent': atr_percent,
            'volume_ratio': volume_ratio
        }

    def analyze_timeframe_advanced(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Улучшенный анализ одного таймфрейма"""
        if df is None or len(df) < 20:
            return {'signal': 'neutral', 'strength': 0, 'indicators': {}, 'market_conditions': {}}

        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]

            market_conditions = self.analyze_market_conditions(df)
            
            # Балльная система для сигналов
            bullish_points = 0
            bearish_points = 0
            max_points = 0
            
            indicators_analysis = {}

            # 1. RSI анализ (макс 3 балла)
            max_points += 3
            if last['rsi_fast'] < 35 and last['rsi_fast'] > prev['rsi_fast']:
                bullish_points += 3
                indicators_analysis['rsi'] = 'oversold_bullish'
            elif last['rsi_fast'] > 65 and last['rsi_fast'] < prev['rsi_fast']:
                bearish_points += 3
                indicators_analysis['rsi'] = 'overbought_bearish'
            elif 40 < last['rsi_fast'] < 60 and last['rsi_fast'] > prev['rsi_fast']:
                bullish_points += 1
                indicators_analysis['rsi'] = 'neutral_bullish'
            elif 40 < last['rsi_fast'] < 60 and last['rsi_fast'] < prev['rsi_fast']:
                bearish_points += 1
                indicators_analysis['rsi'] = 'neutral_bearish'

            # 2. MACD анализ (макс 3 балла)
            max_points += 3
            if last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
                bullish_points += 3
                indicators_analysis['macd'] = 'bullish_strong'
            elif last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']:
                bearish_points += 3
                indicators_analysis['macd'] = 'bearish_strong'
            elif last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                bullish_points += 2
                indicators_analysis['macd'] = 'bullish_cross'
            elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                bearish_points += 2
                indicators_analysis['macd'] = 'bearish_cross'

            # 3. EMA анализ (макс 3 балла)
            max_points += 3
            if last['close'] > last['ema_5'] > last['ema_13'] > last['ema_21']:
                bullish_points += 3
                indicators_analysis['ema'] = 'strong_bullish'
            elif last['close'] < last['ema_5'] < last['ema_13'] < last['ema_21']:
                bearish_points += 3
                indicators_analysis['ema'] = 'strong_bearish'
            elif last['close'] > last['ema_13'] and last['ema_5'] > last['ema_13']:
                bullish_points += 2
                indicators_analysis['ema'] = 'bullish'
            elif last['close'] < last['ema_13'] and last['ema_5'] < last['ema_13']:
                bearish_points += 2
                indicators_analysis['ema'] = 'bearish'

            # 4. Bollinger Bands (макс 2 балла)
            max_points += 2
            if last['bb_position'] < 0.1:
                bullish_points += 2
                indicators_analysis['bb'] = 'oversold'
            elif last['bb_position'] > 0.9:
                bearish_points += 2
                indicators_analysis['bb'] = 'overbought'
            elif last['bb_position'] < 0.3 and last['close'] > prev['close']:
                bullish_points += 1
                indicators_analysis['bb'] = 'lower_bullish'
            elif last['bb_position'] > 0.7 and last['close'] < prev['close']:
                bearish_points += 1
                indicators_analysis['bb'] = 'upper_bearish'

            # 5. Volume анализ (макс 2 балла)
            max_points += 2
            volume_ratio = last['volume_ratio']
            if volume_ratio > 1.5:
                if last['close'] > last['open']:
                    bullish_points += 2
                    indicators_analysis['volume'] = 'high_volume_bullish'
                else:
                    bearish_points += 2
                    indicators_analysis['volume'] = 'high_volume_bearish'
            elif volume_ratio > 1.0:
                if last['close'] > last['open']:
                    bullish_points += 1
                    indicators_analysis['volume'] = 'medium_volume_bullish'
                else:
                    bearish_points += 1
                    indicators_analysis['volume'] = 'medium_volume_bearish'

            # 6. Stochastic RSI (макс 2 балла)
            max_points += 2
            if last['stoch_rsi_k'] < 20 and last['stoch_rsi_k'] > last['stoch_rsi_d']:
                bullish_points += 2
                indicators_analysis['stoch_rsi'] = 'oversold_bullish'
            elif last['stoch_rsi_k'] > 80 and last['stoch_rsi_k'] < last['stoch_rsi_d']:
                bearish_points += 2
                indicators_analysis['stoch_rsi'] = 'overbought_bearish'

            # 7. Momentum (макс 2 балла)
            max_points += 2
            if last['momentum'] > 0 and last['rate_of_change'] > 0:
                bullish_points += 2
                indicators_analysis['momentum'] = 'bullish'
            elif last['momentum'] < 0 and last['rate_of_change'] < 0:
                bearish_points += 2
                indicators_analysis['momentum'] = 'bearish'

            # 8. Support/Resistance (макс 2 балла)
            max_points += 2
            if abs(last['price_vs_resistance']) < 0.5:  # Near resistance
                bearish_points += 2
                indicators_analysis['sr'] = 'near_resistance'
            elif abs(last['price_vs_support']) < 0.5:  # Near support
                bullish_points += 2
                indicators_analysis['sr'] = 'near_support'

            # Расчет общей силы сигнала
            if max_points == 0:
                return {'signal': 'neutral', 'strength': 0, 'indicators': indicators_analysis, 'market_conditions': market_conditions}

            total_bullish_ratio = bullish_points / max_points
            total_bearish_ratio = bearish_points / max_points

            # Определение доминирующего сигнала
            if total_bullish_ratio > total_bearish_ratio and total_bullish_ratio > 0.4:
                signal = 'bullish'
                strength = total_bullish_ratio
            elif total_bearish_ratio > total_bullish_ratio and total_bearish_ratio > 0.4:
                signal = 'bearish'
                strength = total_bearish_ratio
            else:
                signal = 'neutral'
                strength = 0

            return {
                'signal': signal,
                'strength': strength,
                'indicators': indicators_analysis,
                'market_conditions': market_conditions,
                'bullish_points': bullish_points,
                'bearish_points': bearish_points,
                'max_points': max_points
            }

        except Exception as e:
            logger.error(f"Ошибка анализа таймфрейма {timeframe}: {e}")
            return {'signal': 'neutral', 'strength': 0, 'indicators': {}, 'market_conditions': {}}

    def analyze_multiple_timeframes_advanced(self, symbol: str, timeframe_data: Dict) -> Dict:
        """Улучшенный анализ всех таймфреймов для символа"""
        # Веса в зависимости от таймфрейма (5m самый важный)
        timeframe_weights = {
            '1m': 0.25,  # 25% вес
            '5m': 0.40,  # 40% вес (основной)
            '15m': 0.35  # 35% вес
        }

        analysis_results = {}
        total_weight = 0
        weighted_signal = 0
        market_conditions = {}

        for timeframe, df in timeframe_data.items():
            if df is not None:
                result = self.analyze_timeframe_advanced(df, timeframe)
                analysis_results[timeframe] = result
                market_conditions[timeframe] = result['market_conditions']

                weight = timeframe_weights.get(timeframe, 0.3)

                if result['signal'] == 'bullish':
                    weighted_signal += result['strength'] * weight
                elif result['signal'] == 'bearish':
                    weighted_signal -= result['strength'] * weight

                total_weight += weight

        if total_weight == 0:
            return {
                'symbol': symbol,
                'signal': 'HOLD', 
                'confidence': 0,
                'timeframe_analysis': analysis_results,
                'market_conditions': market_conditions,
                'current_price': None
            }

        confidence = abs(weighted_signal) / total_weight

        # Определение финального сигнала с учетом уверенности
        if weighted_signal > 0.1 and confidence > self.config['min_confidence']:
            final_signal = 'LONG'
        elif weighted_signal < -0.1 and confidence > self.config['min_confidence']:
            final_signal = 'SHORT'
        else:
            final_signal = 'HOLD'

        return {
            'symbol': symbol,
            'signal': final_signal,
            'confidence': confidence,
            'weighted_signal': weighted_signal,
            'timeframe_analysis': analysis_results,
            'market_conditions': market_conditions,
            'current_price': timeframe_data['5m'].iloc[-1]['close'] if '5m' in timeframe_data else None
        }

    def should_open_position_advanced(self, analysis: Dict) -> bool:
        """Улучшенная проверка условий для открытия позиции"""
        if analysis['signal'] == 'HOLD':
            return False

        if analysis['confidence'] < self.config['min_confidence']:
            return False

        # Проверяем, нет ли уже активной позиции по этому символу
        if analysis['symbol'] in self.active_positions:
            return False

        # Анализ рыночных условий на 5m таймфрейме
        five_min_conditions = analysis['market_conditions'].get('5m', {})
        
        # Проверка объема
        if five_min_conditions.get('volume_ratio', 0) < self.config['min_volume_ratio']:
            return False

        # Проверка волатильности
        if five_min_conditions.get('atr_percent', 0) > self.config['max_volatility']:
            return False

        # Проверка силы тренда
        if five_min_conditions.get('trend_strength', 0) < self.config['trend_strength_min']:
            return False

        # Проверка согласованности таймфреймов
        bullish_count = 0
        bearish_count = 0
        
        for tf, tf_analysis in analysis['timeframe_analysis'].items():
            if tf_analysis['signal'] == 'bullish':
                bullish_count += 1
            elif tf_analysis['signal'] == 'bearish':
                bearish_count += 1

        # Требуем согласованности как минимум на 2 таймфреймах
        if analysis['signal'] == 'LONG' and bullish_count < 2:
            return False
        if analysis['signal'] == 'SHORT' and bearish_count < 2:
            return False

        # Проверка на частые сигналы (избегаем чрезмерной торговли)
        current_time = datetime.now()
        if analysis['symbol'] in self.last_signals:
            time_since_last = (current_time - self.last_signals[analysis['symbol']]).total_seconds()
            if time_since_last < 300:  # 5 минут между сигналами на один символ
                return False

        return True

    def manage_positions_advanced(self):
        """Улучшенное управление открытыми позициями"""
        current_time = datetime.now()
        positions_to_close = []

        for symbol, position in self.active_positions.items():
            try:
                # Закрытие по времени
                position_age = current_time - position['open_time']
                if position_age.total_seconds() >= self.config['position_hold_minutes'] * 60:
                    positions_to_close.append(symbol)
                    continue

            except Exception as e:
                logger.error(f"Ошибка управления позицией {symbol}: {e}")

        for symbol in positions_to_close:
            self.close_position_advanced(symbol)

    def close_position_advanced(self, symbol: str):
        """Улучшенное закрытие позиции"""
        if symbol in self.active_positions:
            position = self.active_positions.pop(symbol)
            close_time = datetime.now()

            # Расчет P&L
            open_price = position['open_price']
            close_price = position.get('current_price', open_price)

            if position['signal'] == 'LONG':
                profit_pct = (close_price - open_price) / open_price * 100
                result = "ВЫИГРЫШ" if close_price > open_price else "ПРОИГРЫШ"
            else:  # SHORT
                profit_pct = (open_price - close_price) / open_price * 100
                result = "ВЫИГРЫШ" if close_price < open_price else "ПРОИГРЫШ"

            # Добавление в статистику
            self.stats.add_trade(profit_pct)

            # Логирование закрытия позиции
            logger.info(f"🔒 ЗАКРЫТИЕ ПОЗИЦИИ: {symbol} | "
                        f"Сигнал: {position['signal']} | "
                        f"Результат: {result} | "
                        f"P&L: {profit_pct:+.2f}% | "
                        f"Время: {position['hold_time']:.1f} мин")

            # Отправка в Telegram
            result_emoji = "🟢" if result == "ВЫИГРЫШ" else "🔴"
            hold_time = (close_time - position['open_time']).total_seconds() / 60
            
            telegram_message = (
                f"🔒 <b>РЕЗУЛЬТАТ СДЕЛКИ</b>\n"
                f"🎯 <b>{symbol.replace('/USDT:USDT', '')}</b>\n"
                f"📊 Сигнал: {position['signal']}\n"
                f"💰 Цена открытия: {open_price:.6f}\n"
                f"💰 Цена закрытия: {close_price:.6f}\n"
                f"💵 P&L: <b>{profit_pct:+.2f}%</b>\n"
                f"📈 Результат: {result_emoji} <b>{result}</b>\n"
                f"⏱️ Время удержания: {hold_time:.1f} мин\n"
                f"🎯 Уверенность входа: {position['confidence']:.2%}"
            )
            asyncio.create_task(self.telegram_bot.send_message(telegram_message))

            # Добавляем в историю
            self.signal_history.append({
                'symbol': symbol,
                'signal': position['signal'],
                'open_price': position['open_price'],
                'close_price': close_price,
                'open_time': position['open_time'],
                'close_time': close_time,
                'profit_pct': profit_pct,
                'result': result,
                'confidence': position['confidence'],
                'hold_time': hold_time
            })

    def open_position_advanced(self, analysis: Dict):
        """Улучшенное открытие новой позиции"""
        symbol = analysis['symbol']
        
        # Обновляем время последнего сигнала
        self.last_signals[symbol] = datetime.now()

        position = {
            'symbol': symbol,
            'signal': analysis['signal'],
            'open_price': analysis['current_price'],
            'open_time': datetime.now(),
            'confidence': analysis['confidence'],
            'timeframe_analysis': analysis['timeframe_analysis'],
            'market_conditions': analysis['market_conditions'],
            'current_price': analysis['current_price'],
            'hold_time': 0
        }

        self.active_positions[symbol] = position

        # Логирование открытия позиции
        logger.info(f"🎯 ОТКРЫТИЕ ПОЗИЦИИ: {symbol} | "
                    f"Сигнал: {analysis['signal']} | "
                    f"Уверенность: {analysis['confidence']:.2%} | "
                    f"Цена: {analysis['current_price']:.6f}")

        # Отправка в Telegram
        confidence_emoji = "🟢" if analysis['confidence'] > 0.85 else "🟡" if analysis['confidence'] > 0.75 else "🔴"
        signal_emoji = "📈" if analysis['signal'] == 'LONG' else "📉"

        # Анализ индикаторов для 5m
        five_min_analysis = analysis['timeframe_analysis'].get('5m', {})
        indicators_text = ""
        if 'indicators' in five_min_analysis:
            for ind, status in five_min_analysis['indicators'].items():
                if len(indicators_text) < 100:  # Ограничиваем длину
                    indicators_text += f"{ind}: {status}, "

        telegram_message = (
            f"🎯 <b>НОВАЯ СДЕЛКА</b>\n"
            f"{signal_emoji} <b>{symbol.replace('/USDT:USDT', '')}</b>\n"
            f"📊 Направление: <b>{analysis['signal']}</b>\n"
            f"💰 Цена входа: <b>{analysis['current_price']:.6f}</b>\n"
            f"📈 Уверенность: {confidence_emoji} <b>{analysis['confidence']:.2%}</b>\n"
            f"⏰ Макс. время: {self.config['position_hold_minutes']} мин\n"
            f"📊 Индикаторы: {indicators_text}\n"
            f"🔥 Условия: {analysis['market_conditions'].get('5m', {}).get('trend', 'N/A')} тренд, "
            f"{analysis['market_conditions'].get('5m', {}).get('volume', 'N/A')} объем"
        )
        asyncio.create_task(self.telegram_bot.send_message(telegram_message))

    async def analyze_symbol_advanced(self, symbol: str) -> Optional[Dict]:
        """Улучшенный полный анализ одного символа"""
        try:
            timeframe_data = {}

            # Получаем данные для всех таймфреймов
            for timeframe in self.config['timeframes']:
                df = await self.fetch_ohlcv(symbol, timeframe, 50)  # Уменьшили лимит для скорости
                if df is not None and len(df) > 20:
                    df = self.calculate_advanced_indicators(df)
                    timeframe_data[timeframe] = df
                await asyncio.sleep(0.05)  # Уменьшенная задержка

            if not timeframe_data:
                return None

            # Анализируем все таймфреймы
            analysis = self.analyze_multiple_timeframes_advanced(symbol, timeframe_data)
            return analysis

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

    async def send_statistics_report(self):
        """Отправка отчета статистики в Telegram"""
        stats = self.stats.get_statistics()

        if stats['total_trades'] == 0:
            return

        # Анализ последних 10 сделок
        recent_trades = self.signal_history[-10:] if len(self.signal_history) > 10 else self.signal_history
        recent_wins = sum(1 for trade in recent_trades if trade['result'] == 'ВЫИГРЫШ')
        recent_win_rate = (recent_wins / len(recent_trades) * 100) if recent_trades else 0

        telegram_message = (
            f"📊 <b>СТАТИСТИКА ТОРГОВЛИ</b>\n"
            f"📈 Всего сделок: <b>{stats['total_trades']}</b>\n"
            f"🟢 Выигрышных: <b>{stats['winning_trades']}</b>\n"
            f"🔴 Проигрышных: <b>{stats['losing_trades']}</b>\n"
            f"🎯 Общий винрейт: <b>{stats['win_rate']:.1f}%</b>\n"
            f"📊 Винрейт (10 посл.): <b>{recent_win_rate:.1f}%</b>\n"
            f"💰 Общий P&L: <b>{stats['total_profit']:+.2f}%</b>\n"
            f"📊 Средний P&L: <b>{stats['avg_profit']:+.2f}%</b>\n"
            f"🚀 Макс. профит: <b>{stats['max_profit']:+.2f}%</b>\n"
            f"🛑 Макс. убыток: <b>{stats['max_loss']:+.2f}%</b>\n"
            f"🔥 Текущая серия: <b>{stats['current_streak']}</b>\n"
            f"⏰ Активных позиций: <b>{len(self.active_positions)}</b>"
        )

        await self.telegram_bot.send_message(telegram_message)

    async def run_advanced_analysis(self):
        """Запуск улучшенного анализа всех символов"""
        self.analysis_count += 1
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"\n{'=' * 70}")
        logger.info(f"📊 УЛУЧШЕННЫЙ АНАЛИЗ #{self.analysis_count} | {current_time}")
        logger.info(f"{'=' * 70}")

        # Обновляем цены и время удержания для открытых позиций
        for symbol in list(self.active_positions.keys()):
            try:
                df = await self.fetch_ohlcv(symbol, '1m', 2)
                if df is not None and len(df) > 0:
                    self.active_positions[symbol]['current_price'] = df.iloc[-1]['close']
                    # Обновляем время удержания
                    hold_time = (datetime.now() - self.active_positions[symbol]['open_time']).total_seconds() / 60
                    self.active_positions[symbol]['hold_time'] = hold_time
            except Exception as e:
                logger.error(f"Ошибка обновления цены для {symbol}: {e}")

        # Управляем существующими позициями
        self.manage_positions_advanced()

        # Анализируем все символы
        tasks = [self.analyze_symbol_advanced(symbol) for symbol in self.config['symbols']]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_signals = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if result and result['signal'] != 'HOLD':
                valid_signals.append(result)

        # Сортируем сигналы по уверенности
        valid_signals.sort(key=lambda x: x['confidence'], reverse=True)

        # Обрабатываем сигналы для открытия позиций
        for signal in valid_signals:
            if self.should_open_position_advanced(signal):
                self.open_position_advanced(signal)
                break  # Открываем только одну лучшую позицию за анализ

        # Выводим результаты
        self.print_advanced_analysis_results(valid_signals)

        # Отправляем статистику каждые 5 анализов
        if self.analysis_count % 5 == 0 and self.stats.total_trades > 0:
            await self.send_statistics_report()

    def print_advanced_analysis_results(self, signals: List[Dict]):
        """Вывод улучшенных результатов анализа"""
        if not signals:
            logger.info("📭 Нет торговых сигналов, удовлетворяющих критериям")
            return

        logger.info("🎯 УЛУЧШЕННЫЕ ТОРГОВЫЕ СИГНАЛЫ:")
        for signal in signals:
            symbol = signal['symbol'].replace('/USDT:USDT', '')
            confidence = signal['confidence']
            
            if confidence > 0.85:
                confidence_color = "🟢"
            elif confidence > 0.75:
                confidence_color = "🟡"
            else:
                confidence_color = "🔴"

            # Анализ условий рынка
            market_cond = signal['market_conditions'].get('5m', {})
            trend = market_cond.get('trend', 'N/A')
            volume = market_cond.get('volume', 'N/A')
            volatility = market_cond.get('volatility', 'N/A')

            logger.info(f"{confidence_color} {symbol:<6} | "
                       f"{signal['signal']:<6} | "
                       f"Уверенность: {confidence:.2%} | "
                       f"Цена: {signal['current_price']:.6f}")
            logger.info(f"   📊 Условия: {trend} тренд, {volume} объем, {volatility} волатильность")

            # Детали по индикаторам для 5m
            five_min_analysis = signal['timeframe_analysis'].get('5m', {})
            if 'indicators' in five_min_analysis:
                for ind, status in list(five_min_analysis['indicators'].items())[:3]:  # Показываем первые 3
                    logger.info(f"   📈 {ind}: {status}")

    def print_advanced_positions_status(self):
        """Вывод улучшенного статуса открытых позиций"""
        if not self.active_positions:
            logger.info("📭 Нет открытых позиций")
            return

        logger.info("📈 ОТКРЫТЫЕ ПОЗИЦИИ:")
        for symbol, position in self.active_positions.items():
            symbol_clean = symbol.replace('/USDT:USDT', '')
            
            # Расчет текущего P&L
            current_price = position.get('current_price', position['open_price'])
            if position['signal'] == 'LONG':
                profit_pct = (current_price - position['open_price']) / position['open_price'] * 100
                result = "ВЫИГРЫШ" if current_price > position['open_price'] else "ПРОИГРЫШ"
            else:
                profit_pct = (position['open_price'] - current_price) / position['open_price'] * 100
                result = "ВЫИГРЫШ" if current_price < position['open_price'] else "ПРОИГРЫШ"

            result_emoji = "🟢" if result == "ВЫИГРЫШ" else "🔴"
            hold_time = position.get('hold_time', 0)
            minutes_remaining = max(0, self.config['position_hold_minutes'] - hold_time)

            logger.info(f"🔷 {symbol_clean:<6} | "
                       f"{position['signal']:<6} | "
                       f"Открыта: {hold_time:.1f} мин | "
                       f"Закрытие через: {minutes_remaining:.1f} мин | "
                       f"P&L: {result_emoji} {profit_pct:+.2f}%")

    async def run_continuous_advanced(self):
        """Запуск непрерывного улучшенного анализа"""
        # Отправка сообщения о запуске
        start_message = (
            f"🚀 <b>УЛУЧШЕННЫЙ MEXC BOT ЗАПУЩЕН</b>\n"
            f"🎯 <b>РЕЖИМ: ТАЙМФРЕЙМЫ 1М, 5М, 15М</b>\n"
            f"📊 Символы: {', '.join([s.replace('/USDT:USDT', '') for s in self.config['symbols']])}\n"
            f"⏱️ Таймфреймы: {', '.join(self.config['timeframes'])}\n"
            f"🔄 Интервал анализа: {self.config['analysis_interval']} сек\n"
            f"⏳ Время экспирации: {self.config['position_hold_minutes']} мин\n"
            f"📈 Мин. уверенность: {self.config['min_confidence']:.0%}"
        )
        await self.telegram_bot.send_message(start_message)

        logger.info("🚀 ЗАПУСК УЛУЧШЕННОГО НЕПРЕРЫВНОГО АНАЛИЗА")
        logger.info(f"🎯 РЕЖИМ: ТАЙМФРЕЙМЫ 1М, 5М, 15М")
        logger.info(f"📊 Символы: {', '.join([s.replace('/USDT:USDT', '') for s in self.config['symbols']])}")
        logger.info(f"⏱️ Таймфреймы: {', '.join(self.config['timeframes'])}")
        logger.info(f"🔄 Интервал анализа: {self.config['analysis_interval']} сек")
        logger.info(f"⏳ Время экспирации: {self.config['position_hold_minutes']} мин")

        while True:
            try:
                await self.run_advanced_analysis()
                self.print_advanced_positions_status()

                logger.info(f"⏳ Следующий анализ через {self.config['analysis_interval']} сек...")
                await asyncio.sleep(self.config['analysis_interval'])

            except KeyboardInterrupt:
                logger.info("🛑 Остановка бота...")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в основном цикле: {e}")
                await asyncio.sleep(5)

async def main():
    """Основная функция"""
    bot = MEXCTradingBot()

    if bot.exchange is None:
        logger.error("Не удалось подключиться к MEXC. Проверьте интернет соединение.")
        return

    try:
        await bot.run_continuous_advanced()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

        # Отправка сообщения об ошибке
        error_message = f"❌ <b>КРИТИЧЕСКАЯ ОШИБКА БОТА</b>\n{str(e)}"
        await bot.telegram_bot.send_message(error_message)
    finally:
        # Отправка финальной статистики
        if bot.stats.total_trades > 0:
            await bot.send_statistics_report()

        stop_message = "🛑 <b>УЛУЧШЕННЫЙ MEXC BOT ОСТАНОВЛЕН</b>"
        await bot.telegram_bot.send_message(stop_message)

        logger.info("👋 Работа бота завершена")

if __name__ == "__main__":
    # Запуск улучшенного бота
    asyncio.run(main())
