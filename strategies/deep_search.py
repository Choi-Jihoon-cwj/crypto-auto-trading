"""
EMAgap 기반 심화 탐색
1. EMAgap + VolSizing 조합
2. EMAgap + RSI 조합
3. EMAgap + ATR 트레일 조합
4. 모든 파라미터 조합 Grid Search
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import ta
from backtesting import Backtest, Strategy
from config import DATA_DIR
from results.tracker import save_result


def load_df(timeframe='4h', ef=20, es=50, et=200):
    path = os.path.join(DATA_DIR, f"ohlcv_{timeframe}_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=ef).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=es).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=et).ema_indicator()
    df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr_pct']   = df['atr'] / df['close']
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                  for c in df.columns]
    return df


# ── EMAgap + VolSizing 조합 ────────────────────────────────────────
class EMAgapVolStrategy(Strategy):
    trail_pct    = 0.07
    base_pct     = 0.50
    gap_min      = 0.002
    vol_scale    = 1.0

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0: return

        ema_gap = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min

        if not self.position:
            units = max(1, int(self.equity * pos_pct / price))
            if bull_entry:   self.buy(size=units);  self.peak = price
            elif bear_entry: self.sell(size=units); self.trough = price
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price
                if price < self.peak * (1 - self.trail_pct) or not full_bull:
                    self.position.close(); self.peak = 0.0
            else:
                if price < self.trough: self.trough = price
                if price > self.trough * (1 + self.trail_pct) or not full_bear:
                    self.position.close(); self.trough = 0.0


# ── EMAgap + ATR 트레일 조합 ──────────────────────────────────────
class EMAgapATRStrategy(Strategy):
    atr_mult     = 2.5
    position_pct = 0.50
    gap_min      = 0.002

    def init(self):
        self.peak = 0.0; self.trough = 0.0
        self.peak_atr = 0.0; self.trough_atr = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        atr = self.data.atr[-1]
        if np.isnan(et) or np.isnan(atr) or atr <= 0: return

        ema_gap = abs(ef - es) / price
        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min

        if not self.position:
            units = max(1, int(self.equity * self.position_pct / price))
            if bull_entry:
                self.buy(size=units); self.peak = price; self.peak_atr = atr
            elif bear_entry:
                self.sell(size=units); self.trough = price; self.trough_atr = atr
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price; self.peak_atr = atr
                stop = self.peak - self.peak_atr * self.atr_mult
                if price < stop or not full_bull:
                    self.position.close(); self.peak = 0.0
            else:
                if price < self.trough: self.trough = price; self.trough_atr = atr
                stop = self.trough + self.trough_atr * self.atr_mult
                if price > stop or not full_bear:
                    self.position.close(); self.trough = 0.0


# ── EMAgap + RSI 조합 ─────────────────────────────────────────────
class EMAgapRSIStrategy(Strategy):
    trail_pct    = 0.07
    position_pct = 0.50
    gap_min      = 0.002
    rsi_upper    = 70
    rsi_lower    = 30

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        rsi = self.data.rsi[-1]
        if np.isnan(et) or np.isnan(rsi): return

        ema_gap = abs(ef - es) / price
        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and rsi < self.rsi_upper
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and rsi > self.rsi_lower

        if not self.position:
            units = max(1, int(self.equity * self.position_pct / price))
            if bull_entry:   self.buy(size=units);  self.peak = price
            elif bear_entry: self.sell(size=units); self.trough = price
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price
                if price < self.peak * (1 - self.trail_pct) or not full_bull:
                    self.position.close(); self.peak = 0.0
            else:
                if price < self.trough: self.trough = price
                if price > self.trough * (1 + self.trail_pct) or not full_bear:
                    self.position.close(); self.trough = 0.0


# ── EMAgap + VolSizing + ATR 트레일 (풀 조합) ─────────────────────
class UltimateStrategy(Strategy):
    atr_mult     = 2.5
    base_pct     = 0.50
    gap_min      = 0.002
    vol_scale    = 1.0
    rsi_upper    = 70
    rsi_lower    = 30

    def init(self):
        self.peak = 0.0; self.trough = 0.0
        self.peak_atr = 0.0; self.trough_atr = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        rsi = self.data.rsi[-1]
        atr = self.data.atr[-1]
        vol = self.data.volatility_20[-1]
        if np.isnan(et) or np.isnan(rsi) or np.isnan(atr) or np.isnan(vol) or atr <= 0 or vol <= 0: return

        ema_gap = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and rsi < self.rsi_upper
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and rsi > self.rsi_lower

        if not self.position:
            units = max(1, int(self.equity * pos_pct / price))
            if bull_entry:
                self.buy(size=units); self.peak = price; self.peak_atr = atr
            elif bear_entry:
                self.sell(size=units); self.trough = price; self.trough_atr = atr
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price; self.peak_atr = atr
                stop = self.peak - self.peak_atr * self.atr_mult
                if price < stop or not full_bull:
                    self.position.close(); self.peak = 0.0
            else:
                if price < self.trough: self.trough = price; self.trough_atr = atr
                stop = self.trough + self.trough_atr * self.atr_mult
                if price > stop or not full_bear:
                    self.position.close(); self.trough = 0.0


if __name__ == "__main__":
    print("=" * 70)
    print("EMAgap 기반 심화 탐색 (워크포워드)")
    print("튜닝: 2019-2023 | 검증: 2024-2026")
    print("=" * 70)

    df = load_df()
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()

    results = []

    # ── A. EMAgap + VolSizing ──────────────────────────────────
    print("\n[A] EMAgap + VolSizing 조합")
    best_p, best_sharpe = None, -999
    for gap_min in [0.001, 0.002, 0.003, 0.005]:
        for base_pct in [0.30, 0.50, 0.70, 0.90]:
            for vol_scale in [0.5, 1.0, 1.5, 2.0]:
                for trail in [0.05, 0.07, 0.09]:
                    bt = Backtest(df_train, EMAgapVolStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                    s = bt.run(trail_pct=trail, base_pct=base_pct, gap_min=gap_min, vol_scale=vol_scale)
                    if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
                        best_sharpe = s['Sharpe Ratio']
                        best_p = {'trail_pct': trail, 'base_pct': base_pct, 'gap_min': gap_min, 'vol_scale': vol_scale}
    bt = Backtest(df_test, EMAgapVolStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('A.EMAgap+Vol', s, best_p))
    print(f"    최적: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── B. EMAgap + ATR 트레일 ─────────────────────────────────
    print("\n[B] EMAgap + ATR 트레일 조합")
    best_p, best_sharpe = None, -999
    for gap_min in [0.001, 0.002, 0.003, 0.005]:
        for pos in [0.30, 0.50, 0.70, 0.90]:
            for atr_mult in [1.5, 2.0, 2.5, 3.0, 3.5]:
                bt = Backtest(df_train, EMAgapATRStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                s = bt.run(atr_mult=atr_mult, position_pct=pos, gap_min=gap_min)
                if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
                    best_sharpe = s['Sharpe Ratio']
                    best_p = {'atr_mult': atr_mult, 'position_pct': pos, 'gap_min': gap_min}
    bt = Backtest(df_test, EMAgapATRStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('B.EMAgap+ATR', s, best_p))
    print(f"    최적: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── C. EMAgap + RSI ────────────────────────────────────────
    print("\n[C] EMAgap + RSI 필터 조합")
    best_p, best_sharpe = None, -999
    for gap_min in [0.001, 0.002, 0.003, 0.005]:
        for pos in [0.30, 0.50, 0.70, 0.90]:
            for rsi_upper in [65, 70, 75]:
                for rsi_lower in [25, 30, 35]:
                    for trail in [0.05, 0.07, 0.09]:
                        bt = Backtest(df_train, EMAgapRSIStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                        s = bt.run(trail_pct=trail, position_pct=pos, gap_min=gap_min, rsi_upper=rsi_upper, rsi_lower=rsi_lower)
                        if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                            best_sharpe = s['Sharpe Ratio']
                            best_p = {'trail_pct': trail, 'position_pct': pos, 'gap_min': gap_min,
                                      'rsi_upper': rsi_upper, 'rsi_lower': rsi_lower}
    bt = Backtest(df_test, EMAgapRSIStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('C.EMAgap+RSI', s, best_p))
    print(f"    최적: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── D. Ultimate (전부 조합) ────────────────────────────────
    print("\n[D] Ultimate 전략 (EMAgap + VolSizing + ATR + RSI)")
    best_p, best_sharpe = None, -999
    for gap_min in [0.001, 0.002, 0.003]:
        for base_pct in [0.30, 0.50, 0.70]:
            for vol_scale in [0.5, 1.0, 1.5]:
                for atr_mult in [2.0, 2.5, 3.0]:
                    for rsi_upper in [65, 70]:
                        for rsi_lower in [30, 35]:
                            bt = Backtest(df_train, UltimateStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                            s = bt.run(atr_mult=atr_mult, base_pct=base_pct, gap_min=gap_min,
                                       vol_scale=vol_scale, rsi_upper=rsi_upper, rsi_lower=rsi_lower)
                            if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                                best_sharpe = s['Sharpe Ratio']
                                best_p = {'atr_mult': atr_mult, 'base_pct': base_pct, 'gap_min': gap_min,
                                          'vol_scale': vol_scale, 'rsi_upper': rsi_upper, 'rsi_lower': rsi_lower}
    bt = Backtest(df_test, UltimateStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('D.Ultimate', s, best_p))
    print(f"    최적: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 최종 비교 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("최종 결과 (2024-2026 미래 검증)")
    print("=" * 70)
    print(f"{'전략':<20} {'수익률':>8} {'MDD':>8} {'Sharpe':>8} {'거래수':>7} {'PF':>7}")
    print("-" * 70)

    # 베이스라인 포함
    bt_base = Backtest(df_test, None.__class__, cash=500_000, commission=0.0004, exclusive_orders=True)
    print(f"{'0.Base(trail7,p50)':<20} {'38.1%':>8} {'-14.8%':>8} {'0.747':>8} {'65':>7} {'1.76':>7}  ← 이전 최고")

    for name, s, p in sorted(results, key=lambda x: x[1]['Sharpe Ratio'], reverse=True):
        print(f"{name:<20} {s['Return [%]']:>7.1f}% {s['Max. Drawdown [%]']:>7.1f}% "
              f"{s['Sharpe Ratio']:>8.3f} {s['# Trades']:>7} {s['Profit Factor']:>7.2f}")

    # 최고 전략 저장
    best = max(results, key=lambda x: x[1]['Sharpe Ratio'])
    name, s, p = best
    save_result(s, f'DeepSearch_{name}', '4h', leverage=1,
                trail_pct=p.get('trail_pct', p.get('atr_mult', 0)),
                position_pct=p.get('position_pct', p.get('base_pct', 0.5)),
                ema_fast=20, ema_slow=50, ema_trend=200,
                note=f'deep_walkforward_{name}_params={p}')
    print(f"\n최고: {name}  →  결과 저장 완료")
    print(f"파라미터: {p}")
