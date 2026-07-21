import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Strategy
from backtesting.lib import FractionalBacktest as Backtest
from data.indicators import load_with_indicators


class TestStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if len(self.data) == 500:
            print(f"Open:  {self.data.Open[-1]:.2f}")
            print(f"High:  {self.data.High[-1]:.2f}")
            print(f"Low:   {self.data.Low[-1]:.2f}")
            print(f"Close: {self.data.Close[-1]:.2f}")
            print(f"ema_20: {self.data.ema_20[-1]:.2f}")
            print(f"atr:    {self.data.atr[-1]:.2f}")

        if not self.position and len(self.data) == 500:
            print("매수 시도...")
            self.buy(size=0.99)


if __name__ == "__main__":
    df = load_with_indicators()
    df.columns = [c.capitalize() for c in df.columns]
    df = df.rename(columns={
        'Ema_20': 'ema_20', 'Ema_50': 'ema_50', 'Ema_200': 'ema_200',
        'Macd': 'macd', 'Macd_signal': 'macd_signal', 'Rsi': 'rsi',
        'Bb_upper': 'bb_upper', 'Bb_lower': 'bb_lower',
        'Bb_mid': 'bb_mid', 'Volume_ema': 'volume_ema', 'Atr': 'atr'
    })

    bt = Backtest(df, TestStrategy, cash=10_000, commission=0.0004, exclusive_orders=True)
    result = bt.run()
    print(f"거래 횟수: {result['# Trades']}")
