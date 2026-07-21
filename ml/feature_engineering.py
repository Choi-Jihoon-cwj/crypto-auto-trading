import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ta
from config import DATA_DIR


def build_features(timeframe='4h', ema_fast=20, ema_slow=50, ema_trend=200,
                   trail_pct=0.05, forward_bars=40):
    path = os.path.join(DATA_DIR, f"ohlcv_{timeframe}_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    # 기본 지표
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

    # 모멘텀 피처 (과거 흐름)
    for lag in [1, 3, 5, 10]:
        df[f'ret_{lag}']       = df['close'].pct_change(lag)
        df[f'rsi_lag_{lag}']   = df['rsi'].shift(lag)
        df[f'atr_lag_{lag}']   = df['atr_pct'].shift(lag)
        df[f'vol_lag_{lag}']   = df['volume_ratio'].shift(lag)
        df[f'macd_lag_{lag}']  = df['macd_hist'].shift(lag)

    # 변동성 레짐
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_5']  = df['close'].pct_change().rolling(5).std()
    df['vol_regime']    = df['volatility_5'] / df['volatility_20']  # 단기/장기 변동성 비율

    # 추세 지속 기간
    df['trend_duration'] = (
        (df['ema_fast'] > df['ema_slow']).astype(int)
        .groupby((df['ema_fast'] > df['ema_slow']).ne(
            (df['ema_fast'] > df['ema_slow']).shift()).cumsum()).cumsum()
    )

    df.dropna(inplace=True)

    # Triple EMA 신호 감지
    full_bull = (df['ema_fast'] > df['ema_slow']) & (df['ema_slow'] > df['ema_trend'])
    full_bear = (df['ema_fast'] < df['ema_slow']) & (df['ema_slow'] < df['ema_trend'])
    bull_entry = full_bull & (~full_bull.shift(1).fillna(False))
    bear_entry = full_bear & (~full_bear.shift(1).fillna(False))
    signal_dir = pd.Series(0, index=df.index)
    signal_dir[bull_entry] = 1
    signal_dir[bear_entry] = -1

    feature_cols = (
        ['rsi', 'rsi_fast', 'macd_hist', 'atr_pct', 'bb_pct', 'bb_width',
         'volume_ratio', 'ema_gap_fs', 'ema_gap_st', 'price_vs_200',
         'volatility_20', 'volatility_5', 'vol_regime', 'trend_duration'] +
        [f'ret_{l}'      for l in [1,3,5,10]] +
        [f'rsi_lag_{l}'  for l in [1,3,5,10]] +
        [f'atr_lag_{l}'  for l in [1,3,5,10]] +
        [f'vol_lag_{l}'  for l in [1,3,5,10]] +
        [f'macd_lag_{l}' for l in [1,3,5,10]]
    )

    rows = []
    for idx in df.index[bull_entry | bear_entry]:
        loc = df.index.get_loc(idx)
        if loc + forward_bars >= len(df):
            continue

        direction   = signal_dir[idx]
        entry_price = df['close'].iloc[loc]
        future      = df['close'].iloc[loc+1 : loc+forward_bars+1]

        if direction == 1:
            max_gain = (future.max() - entry_price) / entry_price
        else:
            max_gain = (entry_price - future.min()) / entry_price

        label = 1 if max_gain >= trail_pct else 0

        row = {'timestamp': idx, 'direction': direction, 'label': label}
        for col in feature_cols:
            row[col] = df[col].iloc[loc]
        rows.append(row)

    result = pd.DataFrame(rows).set_index('timestamp')
    print(f"신호 {len(result)}개 | 수익 {result['label'].sum()}개 ({result['label'].mean()*100:.1f}%)")
    return result, feature_cols


if __name__ == "__main__":
    df, cols = build_features()
    print(f"피처 수: {len(cols)}")
    print(df.head())
