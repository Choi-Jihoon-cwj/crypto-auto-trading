import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from backtesting import Backtest, Strategy
from config import DATA_DIR
from results.tracker import save_result


# ── 피처 컬럼 정의 ───────────────────────────────────────────
FEATURE_COLS = (
    ['rsi', 'rsi_fast', 'macd_hist', 'atr_pct', 'bb_pct', 'bb_width',
     'volume_ratio', 'ema_gap_fs', 'ema_gap_st', 'price_vs_200',
     'volatility_20', 'volatility_5', 'vol_regime', 'trend_duration'] +
    [f'ret_{l}'      for l in [1,3,5,10]] +
    [f'rsi_lag_{l}'  for l in [1,3,5,10]] +
    [f'atr_lag_{l}'  for l in [1,3,5,10]] +
    [f'vol_lag_{l}'  for l in [1,3,5,10]] +
    [f'macd_lag_{l}' for l in [1,3,5,10]]
)


def build_full_df(timeframe='4h', ef=20, es=50, et=200):
    path = os.path.join(DATA_DIR, f"ohlcv_{timeframe}_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=ef).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=es).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=et).ema_indicator()
    df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_fast']  = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    df['macd_hist'] = ta.trend.MACD(df['close']).macd_diff()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close'],14).average_true_range()
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
    return df


def build_signal_df(df, trail_pct=0.05, forward_bars=40):
    full_bull = (df['ema_fast'] > df['ema_slow']) & (df['ema_slow'] > df['ema_trend'])
    full_bear = (df['ema_fast'] < df['ema_slow']) & (df['ema_slow'] < df['ema_trend'])
    bull_entry = full_bull & (~full_bull.shift(1).fillna(False))
    bear_entry = full_bear & (~full_bear.shift(1).fillna(False))
    signal_dir = pd.Series(0, index=df.index)
    signal_dir[bull_entry] = 1
    signal_dir[bear_entry] = -1

    rows = []
    for idx in df.index[bull_entry | bear_entry]:
        loc = df.index.get_loc(idx)
        if loc + forward_bars >= len(df):
            continue
        direction = signal_dir[idx]
        entry = df['close'].iloc[loc]
        future = df['close'].iloc[loc+1:loc+forward_bars+1]
        gain = (future.max()-entry)/entry if direction==1 else (entry-future.min())/entry
        label = 1 if gain >= trail_pct else 0
        row = {'timestamp': idx, 'direction': direction, 'label': label}
        for col in FEATURE_COLS:
            row[col] = df[col].iloc[loc]
        rows.append(row)
    return pd.DataFrame(rows).set_index('timestamp')


def train_model(signal_df, direction):
    subset = signal_df[signal_df['direction'] == direction]
    if len(subset) < 20:
        return None, None
    X = subset[FEATURE_COLS].values
    y = subset['label'].values
    scale = (len(y) - y.sum()) / max(y.sum(), 1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    xgb  = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=scale, eval_metric='logloss',
                          random_state=42, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=scale, random_state=42, verbose=-1)

    # 간단한 hold-out으로 모델 선택
    split = int(len(X_s) * 0.8)
    xgb.fit(X_s[:split], y[:split])
    lgbm.fit(pd.DataFrame(X_s[:split], columns=FEATURE_COLS), y[:split])

    xgb_auc  = roc_auc_score(y[split:], xgb.predict_proba(X_s[split:])[:,1]) if len(set(y[split:])) > 1 else 0.5
    lgbm_auc = roc_auc_score(y[split:], lgbm.predict_proba(pd.DataFrame(X_s[split:], columns=FEATURE_COLS))[:,1]) if len(set(y[split:])) > 1 else 0.5

    best = xgb if xgb_auc >= lgbm_auc else lgbm
    best.fit(X_s if isinstance(best, XGBClassifier) else pd.DataFrame(X_s, columns=FEATURE_COLS), y)
    return best, scaler


class WalkForwardStrategy(Strategy):
    trail_pct    = 0.07
    position_pct = 0.70
    ml_threshold = 0.55
    model_long   = None
    model_short  = None
    scaler_long  = None
    scaler_short = None

    def init(self):
        self.peak   = 0.0
        self.trough = 0.0

    def predict(self, direction):
        model  = self.model_long  if direction == 'long' else self.model_short
        scaler = self.scaler_long if direction == 'long' else self.scaler_short
        if model is None or scaler is None:
            return 0.5
        X = np.array([[getattr(self.data, col)[-1] if hasattr(self.data, col) else 0
                       for col in FEATURE_COLS]])
        X_s = scaler.transform(X)
        if isinstance(model, LGBMClassifier):
            X_s = pd.DataFrame(X_s, columns=FEATURE_COLS)
        return model.predict_proba(X_s)[0][1]

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef = self.data.ema_fast[-1]
        es = self.data.ema_slow[-1]
        et = self.data.ema_trend[-1]
        if np.isnan(et): return

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull
        bear_entry = full_bear and not prev_bear

        if not self.position:
            if bull_entry and self.predict('long') >= self.ml_threshold:
                units = max(1, int(self.equity * self.position_pct / price))
                self.buy(size=units)
                self.peak = price
            elif bear_entry and self.predict('short') >= self.ml_threshold:
                units = max(1, int(self.equity * self.position_pct / price))
                self.sell(size=units)
                self.trough = price
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price
                if price < self.peak*(1-self.trail_pct) or not full_bull:
                    self.position.close(); self.peak = 0.0
            elif self.position.is_short:
                if price < self.trough: self.trough = price
                if price > self.trough*(1+self.trail_pct) or not full_bear:
                    self.position.close(); self.trough = 0.0


if __name__ == "__main__":
    print("워크포워드 검증 시작\n" + "="*60)
    print("학습: 2019-09 ~ 2023-12 | 테스트: 2024-01 ~ 2026-06\n")

    df = build_full_df()

    train_df = df[df.index < '2024-01-01']
    test_df  = df[df.index >= '2024-01-01']

    print(f"학습 기간: {train_df.index[0].date()} ~ {train_df.index[-1].date()} ({len(train_df)}개)")
    print(f"테스트 기간: {test_df.index[0].date()} ~ {test_df.index[-1].date()} ({len(test_df)}개)\n")

    # 학습 데이터로 신호 생성 및 모델 학습
    print("모델 학습 중...")
    signal_df = build_signal_df(train_df)

    model_long,  scaler_long  = train_model(signal_df, 1)
    model_short, scaler_short = train_model(signal_df, -1)
    print("학습 완료\n")

    # 테스트 데이터로 백테스트 (모델이 한 번도 못 본 데이터)
    test_bt_df = test_df.copy()
    test_bt_df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                           for c in test_bt_df.columns]

    print("테스트 기간 백테스트 중...")
    bt = Backtest(test_bt_df, WalkForwardStrategy, cash=500_000,
                  commission=0.0004, exclusive_orders=True)

    stats = bt.run(
        trail_pct=0.07, position_pct=0.70, ml_threshold=0.55,
        model_long=model_long, model_short=model_short,
        scaler_long=scaler_long, scaler_short=scaler_short
    )

    print(f"\n{'='*60}")
    print(f"워크포워드 검증 결과 (미래 데이터)")
    print(f"{'='*60}")
    print(stats[['Return [%]', 'Buy & Hold Return [%]', 'Win Rate [%]',
                  'Max. Drawdown [%]', '# Trades', 'Profit Factor', 'Sharpe Ratio']])

    # 전체 기간 결과와 비교
    print(f"\n{'='*60}")
    print("전체 기간(7년) vs 미래 검증 비교")
    print(f"{'='*60}")
    print(f"{'항목':<20} {'전체(7년)':<15} {'미래 검증'}")
    print(f"{'수익률':<20} {'252.5%':<15} {stats['Return [%]']:.1f}%")
    print(f"{'MDD':<20} {'-7.2%':<15} {stats['Max. Drawdown [%]']:.1f}%")
    print(f"{'Sharpe':<20} {'1.738':<15} {stats['Sharpe Ratio']:.3f}")
    print(f"{'거래수':<20} {'61':<15} {stats['# Trades']}")

    save_result(stats, 'WalkForward_ML', '4h', leverage=1,
                trail_pct=0.07, position_pct=0.70,
                ema_fast=20, ema_slow=50, ema_trend=200,
                note='walk_forward_2024_2026')

    bt.plot()
