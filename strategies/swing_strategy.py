import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Backtest, Strategy
from data.indicators import load_with_indicators


def prepare_df():
    df = load_with_indicators()
    df.columns = [c.capitalize() for c in df.columns]
    df = df.rename(columns={
        'Ema_20': 'ema_20', 'Ema_50': 'ema_50', 'Ema_200': 'ema_200',
        'Macd': 'macd', 'Macd_signal': 'macd_signal', 'Rsi': 'rsi',
        'Bb_upper': 'bb_upper', 'Bb_lower': 'bb_lower',
        'Bb_mid': 'bb_mid', 'Volume_ema': 'volume_ema', 'Atr': 'atr'
    })
    return df


class SwingStrategy(Strategy):
    trail_pct    = 0.06
    position_pct = 0.50  # 자본의 몇 % 투입할지

    def init(self):
        self.peak   = 0.0
        self.trough = 0.0

    def next(self):
        price  = self.data.Close[-1]
        ema20  = self.data.ema_20[-1]
        ema50  = self.data.ema_50[-1]
        ema200 = self.data.ema_200[-1]

        if price <= 0 or np.isnan(ema200):
            return

        # 트리플 EMA 완전 정렬 확인
        full_bull = ema20 > ema50 > ema200
        full_bear = ema20 < ema50 < ema200

        prev_full_bull = self.data.ema_20[-2] > self.data.ema_50[-2] > self.data.ema_200[-2]
        prev_full_bear = self.data.ema_20[-2] < self.data.ema_50[-2] < self.data.ema_200[-2]

        bull_entry = full_bull and not prev_full_bull
        bear_entry = full_bear and not prev_full_bear

        if not self.position:
            units = max(1, int(self.equity * self.position_pct / price))

            if bull_entry:
                self.buy(size=units)
                self.peak = price

            elif bear_entry:
                self.sell(size=units)
                self.trough = price

        else:
            if self.position.is_long:
                if price > self.peak:
                    self.peak = price
                trailing_sl = self.peak * (1 - self.trail_pct)

                # 트레일링 스탑 또는 정렬 붕괴 시 청산
                if price < trailing_sl or not full_bull:
                    self.position.close()
                    self.peak = 0.0

            elif self.position.is_short:
                if price < self.trough:
                    self.trough = price
                trailing_sl = self.trough * (1 + self.trail_pct)

                if price > trailing_sl or not full_bear:
                    self.position.close()
                    self.trough = 0.0


if __name__ == "__main__":
    df = prepare_df()
    bt = Backtest(df, SwingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)

    print("최적화 실행 중...")
    stats, _ = bt.optimize(
        trail_pct=[i/100 for i in range(5, 30, 1)],
        maximize='Return [%]',
        return_heatmap=True
    )

    print(f"\n최적 trail_pct: {stats._strategy.trail_pct * 100:.0f}%")
    print(stats[['Return [%]', 'Buy & Hold Return [%]', 'Win Rate [%]',
                  'Max. Drawdown [%]', '# Trades', 'Profit Factor', 'Sharpe Ratio']])
    bt.plot()
