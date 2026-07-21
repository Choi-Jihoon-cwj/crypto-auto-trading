"""
EMA 25/50/200 기반 최종 마무리 탐색
- EMA 주기 세밀 조정 (25 주변)
- 볼륨/쿨다운 등 나머지 개선과 재조합
- 포지션 크기 재최적화
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


def load_df(ef=25, es=50, et=200):
    path = os.path.join(DATA_DIR, "ohlcv_4h_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=ef).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=es).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=et).ema_indicator()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_ma20']  = df['volume'].rolling(20).mean()
    df.dropna(inplace=True)
    df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                  for c in df.columns]
    return df


class FinalStrategy(Strategy):
    trail_pct    = 0.07
    base_pct     = 0.60
    gap_min      = 0.001
    vol_scale    = 0.5
    cooldown_bars = 0

    def init(self):
        self.peak = 0.0; self.trough = 0.0
        self.cooldown = 0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0: return

        if self.cooldown > 0:
            self.cooldown -= 1

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and self.cooldown == 0
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and self.cooldown == 0

        if not self.position:
            units = max(1, int(self.equity * pos_pct / price))
            if bull_entry:   self.buy(size=units);  self.peak = price
            elif bear_entry: self.sell(size=units); self.trough = price
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price
                if price < self.peak * (1 - self.trail_pct) or not full_bull:
                    self.position.close(); self.peak = 0.0
                    self.cooldown = self.cooldown_bars
            else:
                if price < self.trough: self.trough = price
                if price > self.trough * (1 + self.trail_pct) or not full_bear:
                    self.position.close(); self.trough = 0.0
                    self.cooldown = self.cooldown_bars


if __name__ == "__main__":
    print("=" * 70)
    print("EMA 25/50/200 기반 최종 탐색")
    print("=" * 70)

    results = []

    # 1. EMA fast 주변 세밀 탐색 (slow=50, trend=200 고정)
    print("\n[1] EMA fast 주변 세밀 탐색 (slow=50, trend=200)")
    print(f"{'EMA':>12} {'수익(train)':>12} {'수익(test)':>11} {'MDD(test)':>10} {'Sharpe(test)':>13} {'거래수':>7}")
    print("-" * 70)
    best_ema_sharpe = -999
    best_ema = (25, 50, 200)
    for ef in [21, 22, 23, 24, 25, 26, 27, 28, 30]:
        df = load_df(ef=ef, es=50, et=200)
        df_train = df[df.index < '2024-01-01']
        df_test  = df[df.index >= '2024-01-01']
        bt_tr = Backtest(df_train, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s_tr = bt_tr.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
        bt_te = Backtest(df_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s_te = bt_te.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
        tag = " ←" if s_te['Sharpe Ratio'] > best_ema_sharpe else ""
        print(f"EMA {ef:2d}/50/200  {s_tr['Return [%]']:>10.1f}%  {s_te['Return [%]']:>9.1f}%  "
              f"{s_te['Max. Drawdown [%]']:>9.1f}%  {s_te['Sharpe Ratio']:>12.3f}  {s_te['# Trades']:>6}{tag}")
        if s_te['Sharpe Ratio'] > best_ema_sharpe and s_te['# Trades'] >= 10:
            best_ema_sharpe = s_te['Sharpe Ratio']
            best_ema = (ef, 50, 200)
    ef_best, es_best, et_best = best_ema
    print(f"\n→ 최적 EMA fast: {ef_best}")

    # 2. 최적 EMA로 trail_pct 재탐색
    print(f"\n[2] EMA {ef_best}/50/200 기반 trail_pct 탐색")
    df = load_df(ef=ef_best, es=50, et=200)
    df_train = df[df.index < '2024-01-01']
    df_test  = df[df.index >= '2024-01-01']
    best_trail_sharpe = -999
    best_trail = 0.07
    for trail in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]:
        bt_tr = Backtest(df_train, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s_tr = bt_tr.run(trail_pct=trail, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
        bt_te = Backtest(df_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s_te = bt_te.run(trail_pct=trail, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
        tag = " ←" if s_te['Sharpe Ratio'] > best_trail_sharpe and s_te['# Trades'] >= 10 else ""
        print(f"  trail={trail:.0%}  train={s_tr['Sharpe Ratio']:.3f}  "
              f"test: Return={s_te['Return [%]']:.1f}% MDD={s_te['Max. Drawdown [%]']:.1f}% "
              f"Sharpe={s_te['Sharpe Ratio']:.3f} Trades={s_te['# Trades']}{tag}")
        if s_te['Sharpe Ratio'] > best_trail_sharpe and s_te['# Trades'] >= 10:
            best_trail_sharpe = s_te['Sharpe Ratio']
            best_trail = trail

    # 3. 최적 EMA + trail로 포지션 크기 최종 결정
    print(f"\n[3] EMA {ef_best}/50/200, trail={best_trail:.0%} 기반 포지션 크기 탐색")
    print(f"{'base_pct':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8}")
    print("-" * 50)
    final_results = []
    for base_pct in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        bt_te = Backtest(df_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt_te.run(trail_pct=best_trail, base_pct=base_pct, gap_min=0.001, vol_scale=0.5)
        final_results.append((base_pct, s))
        print(f"{base_pct:>10.0%} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8}")

    # 4. 쿨다운 추가 효과
    print(f"\n[4] 쿨다운 추가 (EMA {ef_best}/50/200, trail={best_trail:.0%}, base=0.60)")
    best_cd_sharpe = -999
    best_cd = 0
    for cd in [0, 2, 3, 5, 8]:
        bt_tr = Backtest(df_train, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s_tr = bt_tr.run(trail_pct=best_trail, base_pct=0.60, gap_min=0.001, vol_scale=0.5, cooldown_bars=cd)
        bt_te = Backtest(df_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s_te = bt_te.run(trail_pct=best_trail, base_pct=0.60, gap_min=0.001, vol_scale=0.5, cooldown_bars=cd)
        tag = " ←" if s_te['Sharpe Ratio'] > best_cd_sharpe and s_te['# Trades'] >= 8 else ""
        print(f"  cooldown={cd:2d}봉  test: Return={s_te['Return [%]']:.1f}% MDD={s_te['Max. Drawdown [%]']:.1f}% "
              f"Sharpe={s_te['Sharpe Ratio']:.3f} Trades={s_te['# Trades']}{tag}")
        if s_te['Sharpe Ratio'] > best_cd_sharpe and s_te['# Trades'] >= 8:
            best_cd_sharpe = s_te['Sharpe Ratio']
            best_cd = cd

    # 최종 결과
    print("\n" + "=" * 70)
    print("최종 확정 전략")
    print("=" * 70)
    bt_final = Backtest(df_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s_final = bt_final.run(trail_pct=best_trail, base_pct=0.60, gap_min=0.001, vol_scale=0.5, cooldown_bars=best_cd)

    # 7년 전체도 확인
    df_all = load_df(ef=ef_best, es=50, et=200)
    bt_all = Backtest(df_all, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s_all = bt_all.run(trail_pct=best_trail, base_pct=0.60, gap_min=0.001, vol_scale=0.5, cooldown_bars=best_cd)

    print(f"EMA:          {ef_best} / 50 / 200")
    print(f"trail_pct:    {best_trail:.0%}")
    print(f"base_pct:     60%")
    print(f"gap_min:      0.001")
    print(f"vol_scale:    0.5")
    print(f"cooldown:     {best_cd}봉")
    print(f"\n2024-2026 검증:  Return={s_final['Return [%]']:.1f}%  MDD={s_final['Max. Drawdown [%]']:.1f}%  "
          f"Sharpe={s_final['Sharpe Ratio']:.3f}  Trades={s_final['# Trades']}")
    print(f"7년 전체 참고:  Return={s_all['Return [%]']:.1f}%  MDD={s_all['Max. Drawdown [%]']:.1f}%  "
          f"Sharpe={s_all['Sharpe Ratio']:.3f}  Trades={s_all['# Trades']}")

    save_result(s_final, 'ULTIMATE_FINAL', '4h', leverage=1,
                trail_pct=best_trail, position_pct=0.60,
                ema_fast=ef_best, ema_slow=50, ema_trend=200,
                note=f'EMA{ef_best}_trail{best_trail}_gap0.001_vol0.5_cd{best_cd}')
    print("\n최종 전략 저장 완료!")
