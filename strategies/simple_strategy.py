import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Strategy
from backtesting.lib import FractionalBacktest as Backtest
from backtesting.lib import crossover
import pandas as pd
from data.indicators import load_with_indicators


class RSIStrategy(Strategy):
    rsi_low = 35    # RSI 이 아래면 롱
    rsi_high = 65   # RSI 이 위면 숏

    def init(self):
        self.rsi = self.I(lambda: self.data.rsi, name='RSI')

    def next(self):
        if self.rsi[-1] < self.rsi_low:
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy(size=0.99)

        elif self.rsi[-1] > self.rsi_high:
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell(size=0.99)


if __name__ == "__main__":
    df = load_with_indicators()

    # backtesting.py는 컬럼명 대문자 필요
    df.columns = [c.capitalize() for c in df.columns]
    df = df.rename(columns={
        'Ema_20': 'ema_20', 'Ema_50': 'ema_50', 'Macd': 'macd',
        'Macd_signal': 'macd_signal', 'Rsi': 'rsi',
        'Bb_upper': 'bb_upper', 'Bb_lower': 'bb_lower',
        'Bb_mid': 'bb_mid', 'Volume_ema': 'volume_ema'
    })

    bt = Backtest(
        df,
        RSIStrategy,
        cash=10_000,       # 시작 자금 (USDT)
        commission=0.0004, # 바이낸스 선물 수수료
        exclusive_orders=True
    )

    result = bt.run()
    print(result)
    bt.plot()
