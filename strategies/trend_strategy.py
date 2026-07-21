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


class TrendStrategy(Strategy):
    atr_sl_mult = 2.0
    atr_tp_mult = 4.0
    risk_pct    = 0.02  # 거래당 자본의 2% 리스크

    def init(self):
        pass

    def next(self):
        price = self.data.Close[-1]
        atr   = self.data.atr[-1]

        if price <= 0 or np.isnan(atr):
            return

        bull_market = price > self.data.ema_200[-1]
        bear_market = price < self.data.ema_200[-1]
        uptrend     = self.data.ema_20[-1] > self.data.ema_50[-1]
        downtrend   = self.data.ema_20[-1] < self.data.ema_50[-1]

        macd_cross_up   = (self.data.macd[-1] > self.data.macd_signal[-1]) and \
                          (self.data.macd[-2] <= self.data.macd_signal[-2])
        macd_cross_down = (self.data.macd[-1] < self.data.macd_signal[-1]) and \
                          (self.data.macd[-2] >= self.data.macd_signal[-2])

        rsi_ok = 30 < self.data.rsi[-1] < 70

        # EMA200 위에서만 롱, 아래에서만 숏 — 큰 추세에 역행 금지
        long_signal  = bull_market and uptrend   and macd_cross_up   and rsi_ok
        short_signal = bear_market and downtrend and macd_cross_down and rsi_ok

        if self.position:
            return

        # 최대 자본의 10% 노출, 리스크 2% 고정
        max_units = max(1, int(self.equity * 0.10 / price))

        if long_signal:
            sl = price - atr * self.atr_sl_mult
            tp = price + atr * self.atr_tp_mult
            if sl <= 0 or np.isnan(sl) or np.isnan(tp):
                return
            sl_pct = (price - sl) / price
            if sl_pct <= 0:
                return
            units = min(max(1, int(self.equity * self.risk_pct / (price * sl_pct))), max_units)
            self.buy(size=units, sl=sl, tp=tp)

        elif short_signal:
            sl = price + atr * self.atr_sl_mult
            tp = price - atr * self.atr_tp_mult
            if tp <= 0 or np.isnan(sl) or np.isnan(tp):
                return
            sl_pct = (sl - price) / price
            if sl_pct <= 0:
                return
            units = min(max(1, int(self.equity * self.risk_pct / (price * sl_pct))), max_units)
            self.sell(size=units, sl=sl, tp=tp)


if __name__ == "__main__":
    df = prepare_df()

    bt = Backtest(df, TrendStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    result = bt.run()

    print(result[['Return [%]', 'Buy & Hold Return [%]', 'Win Rate [%]',
                   'Max. Drawdown [%]', '# Trades', 'Profit Factor', 'Sharpe Ratio']])
    bt.plot()
