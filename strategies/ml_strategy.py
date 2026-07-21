import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ta
import pandas as pd
from backtesting import Backtest, Strategy
from config import DATA_DIR
from ml.predictor import MLPredictor


def prepare_df(timeframe='4h', ema_fast=20, ema_slow=50, ema_trend=200):
    path = os.path.join(DATA_DIR, f"ohlcv_{timeframe}_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=ema_fast).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=ema_slow).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=ema_trend).ema_indicator()
    df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_fast']  = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    df['macd_hist'] = ta.trend.MACD(df['close']).macd_diff()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_pct']   = df['atr'] / df['close']

    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_pct']    = bb.bollinger_pband()
    df['bb_width']  = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    df['volume_ratio']  = df['volume'] / df['volume'].rolling(20).mean()
    df['ema_gap_fs']    = (df['ema_fast'] - df['ema_slow']) / df['close']
    df['ema_gap_st']    = (df['ema_slow'] - df['ema_trend']) / df['close']
    df['price_vs_200']  = (df['close'] - df['ema_trend']) / df['close']
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_5']  = df['close'].pct_change().rolling(5).std()
    df['vol_regime']    = df['volatility_5'] / df['volatility_20']
    df['trend_duration'] = (
        (df['ema_fast'] > df['ema_slow']).astype(int)
        .groupby((df['ema_fast'] > df['ema_slow']).ne(
            (df['ema_fast'] > df['ema_slow']).shift()).cumsum()).cumsum()
    )
    for lag in [1, 3, 5, 10]:
        df[f'ret_{lag}']      = df['close'].pct_change(lag)
        df[f'rsi_lag_{lag}']  = df['rsi'].shift(lag)
        df[f'atr_lag_{lag}']  = df['atr_pct'].shift(lag)
        df[f'vol_lag_{lag}']  = df['volume_ratio'].shift(lag)
        df[f'macd_lag_{lag}'] = df['macd_hist'].shift(lag)

    df.dropna(inplace=True)
    df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                  for c in df.columns]
    return df


class MLSwingStrategy(Strategy):
    trail_pct    = 0.07
    position_pct = 0.30
    ml_threshold = 0.50  # ML 확률 임계값

    def init(self):
        self.predictor = MLPredictor(timeframe='4h')
        self.peak   = 0.0
        self.trough = 0.0

    def next(self):
        price = self.data.close[-1] if hasattr(self.data, 'close') else self.data.Close[-1]
        if price <= 0:
            return

        ema_f  = self.data.ema_fast[-1]
        ema_s  = self.data.ema_slow[-1]
        ema_t  = self.data.ema_trend[-1]

        if np.isnan(ema_t):
            return

        full_bull = ema_f > ema_s > ema_t
        full_bear = ema_f < ema_s < ema_t
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]

        bull_entry = full_bull and not prev_bull
        bear_entry = full_bear and not prev_bear

        if not self.position:
            if bull_entry or bear_entry:
                # ML 피처 수집
                features = {col: getattr(self.data, col)[-1]
                            for col in self.predictor.feature_cols
                            if hasattr(self.data, col)}

                direction = 'long' if bull_entry else 'short'
                prob = self.predictor.predict(features, direction)

                units = max(1, int(self.equity * self.position_pct / price))

                if bull_entry and prob >= self.ml_threshold:
                    self.buy(size=units)
                    self.peak = price

                elif bear_entry and prob >= self.ml_threshold:
                    self.sell(size=units)
                    self.trough = price

        else:
            if self.position.is_long:
                if price > self.peak:
                    self.peak = price
                if price < self.peak * (1 - self.trail_pct) or not full_bull:
                    self.position.close()
                    self.peak = 0.0

            elif self.position.is_short:
                if price < self.trough:
                    self.trough = price
                if price > self.trough * (1 + self.trail_pct) or not full_bear:
                    self.position.close()
                    self.trough = 0.0


if __name__ == "__main__":
    from results.tracker import save_result

    df = prepare_df()

    print("ML 없는 기준 전략:")
    from strategies.swing_strategy import SwingStrategy, prepare_df as prep2
    df2 = prep2()
    bt_base = Backtest(df2, SwingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    base = bt_base.run(trail_pct=0.07, position_pct=0.30)
    print(f"  Return={base['Return [%]']:.1f}% MDD={base['Max. Drawdown [%]']:.1f}% Sharpe={base['Sharpe Ratio']:.3f}")

    print("\nML 필터 전략 (임계값 탐색 중):")
    best_sharpe = -999
    best_stats  = None
    best_thresh = 0.5

    for thresh in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]:
        bt = Backtest(df, MLSwingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        stats = bt.run(trail_pct=0.07, position_pct=0.30, ml_threshold=thresh)

        if stats['# Trades'] < 5:
            continue

        print(f"  threshold={thresh} → Return={stats['Return [%]']:.1f}% "
              f"MDD={stats['Max. Drawdown [%]']:.1f}% "
              f"Sharpe={stats['Sharpe Ratio']:.3f} "
              f"Trades={stats['# Trades']}")

        if stats['Sharpe Ratio'] > best_sharpe:
            best_sharpe = stats['Sharpe Ratio']
            best_stats  = stats
            best_thresh = thresh

    print(f"\n최적 임계값: {best_thresh}")
    print(f"ML 전략 최종: Return={best_stats['Return [%]']:.1f}% "
          f"MDD={best_stats['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={best_stats['Sharpe Ratio']:.3f}")

    save_result(best_stats, 'ML_SwingStrategy', '4h', leverage=1,
                trail_pct=0.07, position_pct=0.30,
                ema_fast=20, ema_slow=50, ema_trend=200,
                note=f"ML_thresh={best_thresh}")
