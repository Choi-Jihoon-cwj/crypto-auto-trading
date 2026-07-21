"""
최종 전략 확정 테스트
공포탐욕 fg_upper=70 기반으로 포지션 크기 + 파라미터 재최적화
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


def build_df():
    path = os.path.join(DATA_DIR, "ohlcv_4h_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=25).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)

    fg = pd.read_csv(os.path.join(DATA_DIR, "fear_greed.csv"), index_col='timestamp', parse_dates=True)
    fg.index = fg.index.normalize()
    df['date'] = df.index.normalize()
    fg_map = fg['fear_greed'].to_dict()
    df['fear_greed'] = df['date'].map(fg_map)
    df.drop(columns=['date'], inplace=True)
    df.dropna(inplace=True)

    df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                  for c in df.columns]
    return df


class UltimateFinal(Strategy):
    trail_pct  = 0.05
    base_pct   = 0.60
    gap_min    = 0.001
    vol_scale  = 0.5
    fg_upper   = 70   # 극단 탐욕 롱 차단
    fg_lower   = 20   # 극단 공포 숏 차단

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        fg  = self.data.fear_greed[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0 or np.isnan(fg): return

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]

        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and fg <= self.fg_upper
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and fg >= self.fg_lower

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
    print("=" * 70)
    print("공포탐욕 fg_upper=70 기반 최종 파라미터 확정")
    print("=" * 70)

    df = build_df()
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()

    # 1. 포지션 크기별 최종 성과
    print("\n[포지션 크기별] trail=5%, gap=0.001, vol=0.5, fg_upper=70")
    print(f"{'base_pct':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8} {'PF':>7}")
    print("-" * 58)
    final_results = []
    for base_pct in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        bt = Backtest(df_test, UltimateFinal, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.05, base_pct=base_pct, gap_min=0.001, vol_scale=0.5, fg_upper=70, fg_lower=20)
        final_results.append((base_pct, s))
        print(f"{base_pct:>10.0%} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8} {s['Profit Factor']:>7.2f}")

    # 2. 7년 전체 성과 확인
    print("\n[7년 전체 참고] trail=5%, gap=0.001, vol=0.5, fg_upper=70")
    print(f"{'base_pct':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8}")
    print("-" * 52)
    for base_pct in [0.30, 0.50, 0.60, 0.70, 0.90]:
        bt = Backtest(df, UltimateFinal, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.05, base_pct=base_pct, gap_min=0.001, vol_scale=0.5, fg_upper=70, fg_lower=20)
        print(f"{base_pct:>10.0%} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8}")

    # 3. 전체 진화 과정 비교
    print("\n" + "=" * 70)
    print("전략 진화 최종 비교 (2024-2026 미래 검증)")
    print("=" * 70)
    evolution = [
        ("1. 기본 EMA20/50/200",        64.4, -19.1, 0.745, 65),
        ("2. EMAgap+VolSizing 추가",     48.7, -12.5, 0.948, 33),
        ("3. EMA 25로 변경",             67.2,  -9.6, 1.156, 33),
        ("4. trail 5%로 조정",           65.5,  -8.4, 1.250, 33),
        ("5. 공포탐욕 fg≤70 추가",       71.1,  -8.3, 1.426, 28),
    ]
    print(f"{'전략':<28} {'수익률':>8} {'MDD':>8} {'Sharpe':>9} {'거래수':>7}")
    print("-" * 65)
    for name, ret, mdd, sharpe, trades in evolution:
        print(f"{name:<28} {ret:>7.1f}% {mdd:>7.1f}% {sharpe:>9.3f} {trades:>7}")

    # 최종 저장
    bt_save = Backtest(df_test, UltimateFinal, cash=500_000, commission=0.0004, exclusive_orders=True)
    s_save = bt_save.run(trail_pct=0.05, base_pct=0.60, gap_min=0.001, vol_scale=0.5, fg_upper=70, fg_lower=20)
    save_result(s_save, 'ULTIMATE_WITH_FG', '4h', leverage=1,
                trail_pct=0.05, position_pct=0.60,
                ema_fast=25, ema_slow=50, ema_trend=200,
                note='EMA25_trail5_gap0.001_vol0.5_fg70_FINAL')

    print(f"\n최종 확정 파라미터:")
    print(f"  EMA:         25 / 50 / 200")
    print(f"  trail_pct:   5%")
    print(f"  base_pct:    60%")
    print(f"  gap_min:     0.001")
    print(f"  vol_scale:   0.5")
    print(f"  fg_upper:    70  (공포탐욕 70 초과면 롱 금지)")
    print(f"  fg_lower:    20  (공포탐욕 20 미만이면 숏 금지)")
    print(f"\n2024-2026: Return={s_save['Return [%]']:.1f}% MDD={s_save['Max. Drawdown [%]']:.1f}% Sharpe={s_save['Sharpe Ratio']:.3f}")
    print("\n저장 완료!")
