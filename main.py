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

        self.config = {
            'timeframes': ['1m', '5m', '15m'],
            'symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'],
            'analysis_interval': 4,  # секунды между анализами
            'min_confidence': 0.65,
            'position_hold_minutes': 5,
            'max_open_positions': 3,
            'rsi_period': 14,
            'volume_ma_period': 20,
            'atr_period': 14
        }

        self.active_positions = {}
        self.signal_history = []
        self.analysis_count = 0

        logger.info("MEXC Trading Bot инициализирован")

    def initialize_exchange(self):
        """Инициализация подключения к MEXC"""
        try:
            exchange = ccxt.mexc({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap'
                }
            })
            exchange.load_markets()
            logger.info("Успешное подключение к MEXC")
            return exchange
        except Exception as e:
            logger.error(f"Ошибка подключения к MEXC: {e}")
            return None

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Получение OHLCV данных"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol} на {timeframe}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов"""
        try:
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.config['rsi_period'])

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # EMA
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

            # Volume
            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=self.config['volume_ma_period'])
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.config['atr_period'])

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'],
                                           fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            # Momentum
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)

            return df
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df

    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Анализ одного таймфрейма"""
        if df is None or len(df) < 50:
            return {'signal': 'neutral', 'strength': 0, 'indicators': {}}

        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]

            indicators = {
                'rsi': last['rsi'],
                'macd_hist': last['macd_hist'],
                'price_vs_ema9': (last['close'] - last['ema_9']) / last['ema_9'],
                'price_vs_ema21': (last['close'] - last['ema_21']) / last['ema_21'],
                'bb_position': last['bb_position'],
                'volume_ratio': last['volume_ratio'],
                'stoch_k': last['stoch_k'],
                'momentum': last['momentum']
            }

            # Подсчет сигналов
            bullish_signals = 0
            bearish_signals = 0

            # RSI анализ
            if last['rsi'] < 30:
                bullish_signals += 1.5
            elif last['rsi'] > 70:
                bearish_signals += 1.5
            elif 30 <= last['rsi'] <= 50:
                bullish_signals += 0.5
            elif 50 <= last['rsi'] <= 70:
                bearish_signals += 0.5

            # MACD анализ
            if last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
                bullish_signals += 1
            elif last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']:
                bearish_signals += 1

            # EMA анализ
            if last['close'] > last['ema_9'] > last['ema_21']:
                bullish_signals += 1
            elif last['close'] < last['ema_9'] < last['ema_21']:
                bearish_signals += 1

            # Bollinger Bands
            if last['bb_position'] < 0.2:
                bullish_signals += 1
            elif last['bb_position'] > 0.8:
                bearish_signals += 1

            # Volume анализ
            if last['volume_ratio'] > 1.5:
                if last['close'] > last['open']:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # Stochastic
            if last['stoch_k'] < 20:
                bullish_signals += 0.5
            elif last['stoch_k'] > 80:
                bearish_signals += 0.5

            # Momentum
            if last['momentum'] > 0:
                bullish_signals += 0.5
            else:
                bearish_signals += 0.5

            # Определение сигнала и силы
            total_signals = bullish_signals + bearish_signals
            if total_signals == 0:
                return {'signal': 'neutral', 'strength': 0, 'indicators': indicators}

            if bullish_signals > bearish_signals:
                strength = bullish_signals / total_signals
                return {'signal': 'bullish', 'strength': strength, 'indicators': indicators}
            else:
                strength = bearish_signals / total_signals
                return {'signal': 'bearish', 'strength': strength, 'indicators': indicators}

        except Exception as e:
            logger.error(f"Ошибка анализа таймфрейма {timeframe}: {e}")
            return {'signal': 'neutral', 'strength': 0, 'indicators': {}}

    def analyze_multiple_timeframes(self, symbol: str, timeframe_data: Dict) -> Dict:
        """Анализ всех таймфреймов для символа"""
        timeframe_weights = {
            '1m': 0.2,  # 20% вес
            '5m': 0.5,  # 50% вес (основной)
            '15m': 0.3  # 30% вес
        }

        analysis_results = {}
        total_weight = 0
        weighted_signal = 0

        for timeframe, df in timeframe_data.items():
            if df is not None:
                result = self.analyze_timeframe(df, timeframe)
                analysis_results[timeframe] = result

                weight = timeframe_weights.get(timeframe, 0.2)

                if result['signal'] == 'bullish':
                    weighted_signal += result['strength'] * weight
                elif result['signal'] == 'bearish':
                    weighted_signal -= result['strength'] * weight

                total_weight += weight

        if total_weight == 0:
            return {'signal': 'neutral', 'confidence': 0, 'timeframe_analysis': analysis_results}

        confidence = abs(weighted_signal) / total_weight

        if weighted_signal > 0:
            final_signal = 'LONG'
        elif weighted_signal < 0:
            final_signal = 'SHORT'
        else:
            final_signal = 'HOLD'

        return {
            'symbol': symbol,
            'signal': final_signal,
            'confidence': confidence,
            'timeframe_analysis': analysis_results,
            'current_price': timeframe_data['5m'].iloc[-1]['close'] if '5m' in timeframe_data else None
        }

    def should_open_position(self, analysis: Dict) -> bool:
        """Проверка условий для открытия позиции"""
        if analysis['signal'] == 'HOLD':
            return False

        if analysis['confidence'] < self.config['min_confidence']:
            return False

        # Проверяем, нет ли уже активной позиции по этому символу
        if analysis['symbol'] in self.active_positions:
            return False

        # Проверяем лимит открытых позиций
        if len(self.active_positions) >= self.config['max_open_positions']:
            return False

        # Проверяем 5-минутный таймфрейм (основной)
        five_min_analysis = analysis['timeframe_analysis'].get('5m', {})
        if five_min_analysis.get('signal') == 'neutral':
            return False

        return True

    def manage_positions(self):
        """Управление открытыми позициями - закрытие по истечении времени"""
        current_time = datetime.now()
        positions_to_close = []

        for symbol, position in self.active_positions.items():
            position_age = current_time - position['open_time']
            if position_age.total_seconds() >= self.config['position_hold_minutes'] * 60:
                positions_to_close.append(symbol)

        for symbol in positions_to_close:
            self.close_position(symbol)

    def close_position(self, symbol: str):
        """Закрытие позиции"""
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
                        f"P&L: {profit_pct:+.2f}%")

            # Отправка в Telegram
            result_emoji = "🟢" if result == "ВЫИГРЫШ" else "🔴"
            telegram_message = (
                f"🔒 <b>РЕЗУЛЬТАТ СДЕЛКИ</b>\n"
                f"🎯 <b>{symbol.replace('/USDT:USDT', '')}</b>\n"
                f"📊 Сигнал: {position['signal']}\n"
                f"📈 Цена открытия: {open_price:.6f}\n"
                f"📉 Цена закрытия: {close_price:.6f}\n"
                f"💰 Результат: {result_emoji} <b>{result}</b>\n"
                f"💵 P&L: <b>{profit_pct:+.2f}%</b>\n"
                f"⏱️ Время удержания: {self.config['position_hold_minutes']} мин"
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
                'result': result
            })

    def open_position(self, analysis: Dict):
        """Открытие новой позиции"""
        symbol = analysis['symbol']

        position = {
            'symbol': symbol,
            'signal': analysis['signal'],
            'open_price': analysis['current_price'],
            'open_time': datetime.now(),
            'confidence': analysis['confidence'],
            'timeframe_analysis': analysis['timeframe_analysis'],
            'current_price': analysis['current_price']
        }

        self.active_positions[symbol] = position

        # Логирование открытия позиции
        logger.info(f"🎯 ОТКРЫТИЕ ПОЗИЦИИ: {symbol} | "
                    f"Сигнал: {analysis['signal']} | "
                    f"Уверенность: {analysis['confidence']:.2%} | "
                    f"Цена: {analysis['current_price']:.6f}")

        # Отправка в Telegram
        confidence_emoji = "🟢" if analysis['confidence'] > 0.8 else "🟡" if analysis['confidence'] > 0.7 else "🔴"
        signal_emoji = "📈" if analysis['signal'] == 'LONG' else "📉"

        telegram_message = (
            f"🎯 <b>НОВАЯ СДЕЛКА</b>\n"
            f"{signal_emoji} <b>{symbol.replace('/USDT:USDT', '')}</b>\n"
            f"📊 Направление: <b>{analysis['signal']}</b>\n"
            f"💰 Цена входа: <b>{analysis['current_price']:.6f}</b>\n"
            f"📈 Уверенность: {confidence_emoji} <b>{analysis['confidence']:.2%}</b>\n"
            f"⏰ Экспирация: через {self.config['position_hold_minutes']} мин\n"
            f"🎯 Прогноз: цена {'вырастет' if analysis['signal'] == 'LONG' else 'упадет'}"
        )
        asyncio.create_task(self.telegram_bot.send_message(telegram_message))

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Полный анализ одного символа"""
        try:
            timeframe_data = {}

            # Получаем данные для всех таймфреймов
            for timeframe in self.config['timeframes']:
                df = await self.fetch_ohlcv(symbol, timeframe, 100)
                if df is not None and len(df) > 50:
                    df = self.calculate_indicators(df)
                    timeframe_data[timeframe] = df
                await asyncio.sleep(0.1)  # Задержка между запросами

            if not timeframe_data:
                return None

            # Анализируем все таймфреймы
            analysis = self.analyze_multiple_timeframes(symbol, timeframe_data)
            return analysis

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

    async def send_statistics_report(self):
        """Отправка отчета статистики в Telegram"""
        stats = self.stats.get_statistics()

        if stats['total_trades'] == 0:
            return

        telegram_message = (
            f"📊 <b>СТАТИСТИКА ТОРГОВЛИ</b>\n"
            f"📈 Всего сделок: <b>{stats['total_trades']}</b>\n"
            f"🟢 Выигрышных: <b>{stats['winning_trades']}</b>\n"
            f"🔴 Проигрышных: <b>{stats['losing_trades']}</b>\n"
            f"🎯 Винрейт: <b>{stats['win_rate']:.1f}%</b>\n"
            f"💰 Общий P&L: <b>{stats['total_profit']:+.2f}%</b>\n"
            f"📊 Средний P&L: <b>{stats['avg_profit']:+.2f}%</b>\n"
            f"🚀 Макс. профит: <b>{stats['max_profit']:+.2f}%</b>\n"
            f"🛑 Макс. убыток: <b>{stats['max_loss']:+.2f}%</b>\n"
            f"🔥 Текущая серия: <b>{stats['current_streak']}</b>\n"
            f"🎯 Последняя сделка: <b>{stats['last_trade_profit']:+.2f}%</b>"
        )

        await self.telegram_bot.send_message(telegram_message)

    async def run_analysis(self):
        """Запуск анализа всех символов"""
        self.analysis_count += 1
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 АНАЛИЗ #{self.analysis_count} | {current_time}")
        logger.info(f"{'=' * 60}")

        # Обновляем цены для открытых позиций
        for symbol in list(self.active_positions.keys()):
            try:
                df = await self.fetch_ohlcv(symbol, '1m', 2)
                if df is not None and len(df) > 0:
                    self.active_positions[symbol]['current_price'] = df.iloc[-1]['close']
            except Exception as e:
                logger.error(f"Ошибка обновления цены для {symbol}: {e}")

        # Управляем существующими позициями
        self.manage_positions()

        # Анализируем все символы
        tasks = [self.analyze_symbol(symbol) for symbol in self.config['symbols']]
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
            if self.should_open_position(signal):
                self.open_position(signal)

        # Выводим результаты
        self.print_analysis_results(valid_signals)

        # Отправляем статистику каждые 10 анализов
        if self.analysis_count % 10 == 0 and self.stats.total_trades > 0:
            await self.send_statistics_report()

    def print_analysis_results(self, signals: List[Dict]):
        """Вывод результатов анализа"""
        if not signals:
            logger.info("📭 Нет торговых сигналов")
            return

        logger.info("🎯 ТОРГОВЫЕ СИГНАЛЫ:")
        for signal in signals:
            symbol = signal['symbol'].replace('/USDT:USDT', '')
            confidence_color = "🟢" if signal['confidence'] > 0.8 else "🟡" if signal['confidence'] > 0.7 else "🔴"

            logger.info(f"{confidence_color} {symbol:<6} | "
                        f"{signal['signal']:<6} | "
                        f"Уверенность: {signal['confidence']:.2%} | "
                        f"Цена: {signal['current_price']:.6f}")

            # Детали по таймфреймам
            for tf, analysis in signal['timeframe_analysis'].items():
                if analysis['signal'] != 'neutral':
                    logger.info(f"   {tf}: {analysis['signal']} (сила: {analysis['strength']:.2f})")

    def print_positions_status(self):
        """Вывод статуса открытых позиций"""
        if not self.active_positions:
            logger.info("📭 Нет открытых позиций")
            return

        logger.info("📈 ОТКРЫТЫЕ ПОЗИЦИИ:")
        for symbol, position in self.active_positions.items():
            symbol_clean = symbol.replace('/USDT:USDT', '')
            position_age = datetime.now() - position['open_time']
            minutes_open = int(position_age.total_seconds() / 60)
            minutes_remaining = max(0, self.config['position_hold_minutes'] - minutes_open)

            # Расчет текущего P&L
            current_price = position.get('current_price', position['open_price'])
            if position['signal'] == 'LONG':
                profit_pct = (current_price - position['open_price']) / position['open_price'] * 100
                result = "ВЫИГРЫШ" if current_price > position['open_price'] else "ПРОИГРЫШ"
            else:
                profit_pct = (position['open_price'] - current_price) / position['open_price'] * 100
                result = "ВЫИГРЫШ" if current_price < position['open_price'] else "ПРОИГРЫШ"

            result_emoji = "🟢" if result == "ВЫИГРЫШ" else "🔴"

            logger.info(f"🔷 {symbol_clean:<6} | "
                        f"{position['signal']:<6} | "
                        f"Открыта: {minutes_open} мин назад | "
                        f"Закрытие через: {minutes_remaining} мин | "
                        f"Текущий результат: {result_emoji} {result} | "
                        f"P&L: {profit_pct:+.2f}%")

    async def run_continuous(self):
        """Запуск непрерывного анализа"""
        # Отправка сообщения о запуске
        start_message = (
            f"🚀 <b>MEXC TRADING BOT ЗАПУЩЕН</b>\n"
            f"🎯 <b>ФЬЮЧЕРСНЫЕ ПРОГНОЗЫ (ОПЦИОНЫ)</b>\n"
            f"📊 Символы: {', '.join([s.replace('/USDT:USDT', '') for s in self.config['symbols']])}\n"
            f"⏱️ Таймфреймы: {', '.join(self.config['timeframes'])}\n"
            f"🔄 Интервал анализа: {self.config['analysis_interval']} сек\n"
            f"⏳ Время экспирации: {self.config['position_hold_minutes']} мин\n"
            f"📈 Мин. уверенность: {self.config['min_confidence']:.0%}\n"
            f"💰 Тип торговли: Прогноз направления цены"
        )
        await self.telegram_bot.send_message(start_message)

        logger.info("🚀 ЗАПУСК НЕПРЕРЫВНОГО АНАЛИЗА")
        logger.info(f"🎯 РЕЖИМ: ФЬЮЧЕРСНЫЕ ПРОГНОЗЫ (ОПЦИОНЫ)")
        logger.info(f"📊 Символы: {', '.join([s.replace('/USDT:USDT', '') for s in self.config['symbols']])}")
        logger.info(f"⏱️ Таймфреймы: {', '.join(self.config['timeframes'])}")
        logger.info(f"🔄 Интервал анализа: {self.config['analysis_interval']} сек")
        logger.info(f"⏳ Время экспирации: {self.config['position_hold_minutes']} мин")

        while True:
            try:
                await self.run_analysis()
                self.print_positions_status()

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
        await bot.run_continuous()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

        # Отправка сообщения об ошибке
        error_message = f"❌ <b>КРИТИЧЕСКАЯ ОШИБКА БОТА</b>\n{str(e)}"
        await bot.telegram_bot.send_message(error_message)
    finally:
        # Отправка финальной статистики
        if bot.stats.total_trades > 0:
            await bot.send_statistics_report()

        stop_message = "🛑 <b>MEXC TRADING BOT ОСТАНОВЛЕН</b>"
        await bot.telegram_bot.send_message(stop_message)

        logger.info("👋 Работа бота завершена")


if __name__ == "__main__":
    # Запуск бота
    asyncio.run(main())
