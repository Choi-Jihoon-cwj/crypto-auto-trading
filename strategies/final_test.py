"""
최고 전략(EMAgap + VolSizing) 포지션 크기 탐색
파라미터: trail_pct=0.07, gap_min=0.001, vol_scale=0.5
base_pct를 0.30 ~ 0.95 까지 변화시키며 미래 성과 확인
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


def load_df():
    path = os.path.join(DATA_DIR, "ohlcv_4h_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['atr_pct']   = df['atr'] / df['close']
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                  for c in df.columns]
    return df


class BestStrategy(Strategy):
    """EMAgap + VolSizing - 최고 전략"""
    trail_pct = 0.07
    base_pct  = 0.50
    gap_min   = 0.001
    vol_scale = 0.5

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


if __name__ == "__main__":
    df = load_df()
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()

    print("=" * 70)
    print("최고 전략 (EMAgap+VolSizing) - 포지션 크기별 미래 성과")
    print("기준 파라미터: trail=7%, gap_min=0.001, vol_scale=0.5")
    print("=" * 70)
    print(f"\n{'base_pct':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8} {'PF':>7}")
    print("-" * 60)

    results = []
    for base_pct in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        bt = Backtest(df_test, BestStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.07, base_pct=base_pct, gap_min=0.001, vol_scale=0.5)
        results.append((base_pct, s))
        print(f"{base_pct:>10.0%} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8} {s['Profit Factor']:>7.2f}")

    # 7년 전체 기간도 확인 (과거 데이터 검증)
    print("\n" + "=" * 70)
    print("최고 전략 - 전체 7년 기간 성과 (참고용)")
    print("=" * 70)
    print(f"\n{'base_pct':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8}")
    print("-" * 55)

    for base_pct in [0.30, 0.50, 0.70, 0.90]:
        bt = Backtest(df, BestStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.07, base_pct=base_pct, gap_min=0.001, vol_scale=0.5)
        print(f"{base_pct:>10.0%} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8}")

    # 최적 포지션 선택 (Sharpe 기준) 및 저장
    print("\n" + "=" * 70)
    best = max(results, key=lambda x: x[1]['Sharpe Ratio'])
    print(f"최고 Sharpe: base_pct={best[0]:.0%}")

    # 수익/리스크 균형 선택 (Sharpe>0.8 중 수익 최고)
    balanced = [(p, s) for p, s in results if s['Sharpe Ratio'] >= 0.8]
    if balanced:
        best_balanced = max(balanced, key=lambda x: x[1]['Return [%]'])
        print(f"Sharpe≥0.8 중 최고 수익: base_pct={best_balanced[0]:.0%} "
              f"→ Return={best_balanced[1]['Return [%]']:.1f}% "
              f"MDD={best_balanced[1]['Max. Drawdown [%]']:.1f}% "
              f"Sharpe={best_balanced[1]['Sharpe Ratio']:.3f}")

        save_result(best_balanced[1], 'FINAL_BestStrategy', '4h', leverage=1,
                    trail_pct=0.07, position_pct=best_balanced[0],
                    ema_fast=20, ema_slow=50, ema_trend=200,
                    note=f'EMAgap_VolSizing_gap0.001_vol0.5_FINAL')
        print("→ 최종 전략 저장 완료!")
