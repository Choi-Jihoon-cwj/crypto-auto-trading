"""
추가 개선 실험 - 모두 워크포워드 검증
베이스: EMAgap+VolSizing (trail=7%, gap_min=0.001, vol_scale=0.5, base_pct=0.60)

테스트:
  E. 볼륨 필터 (거래량 > N배 평균)
  F. EMA 주기 재탐색 (현 전략에 최적 EMA 찾기)
  G. 분할 청산 (반 청산 후 나머지 트레일)
  H. 재진입 쿨다운 (손절 후 N봉 대기)
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


def load_df(ef=20, es=50, et=200):
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


# ── 현재 최고 전략 (베이스라인) ───────────────────────────────────
class BestBase(Strategy):
    trail_pct = 0.07
    base_pct  = 0.60
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


# ── E. 볼륨 필터 ─────────────────────────────────────────────────
class VolumeFilterStrategy(Strategy):
    trail_pct  = 0.07
    base_pct   = 0.60
    gap_min    = 0.001
    vol_scale  = 0.5
    vol_thresh = 1.2   # 거래량 > 평균 * vol_thresh 일 때만 진입

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        volatility = self.data.volatility_20[-1]
        volume     = self.data.Volume[-1]
        vol_ma     = self.data.vol_ma20[-1]
        if np.isnan(et) or np.isnan(volatility) or volatility <= 0 or np.isnan(vol_ma) or vol_ma <= 0: return

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / volatility, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)
        vol_ok     = volume >= vol_ma * self.vol_thresh

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and vol_ok
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and vol_ok

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


# ── G. 분할 청산 ─────────────────────────────────────────────────
# 목표 수익(half_exit_pct) 달성 시 절반 청산, 나머지는 트레일링
class HalfExitStrategy(Strategy):
    trail_pct     = 0.07
    base_pct      = 0.60
    gap_min       = 0.001
    vol_scale     = 0.5
    half_exit_pct = 0.05   # 5% 수익 시 절반 청산

    def init(self):
        self.peak = 0.0; self.trough = 0.0
        self.entry_price = 0.0
        self.half_exited = False

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0: return

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min

        if not self.position:
            units = max(1, int(self.equity * pos_pct / price))
            if bull_entry:
                self.buy(size=units); self.peak = price
                self.entry_price = price; self.half_exited = False
            elif bear_entry:
                self.sell(size=units); self.trough = price
                self.entry_price = price; self.half_exited = False
        else:
            if self.position.is_long:
                if price > self.peak: self.peak = price
                # 목표 수익 달성 → 절반 청산
                if not self.half_exited and price >= self.entry_price * (1 + self.half_exit_pct):
                    half = max(1, self.position.size // 2)
                    self.sell(size=half)
                    self.half_exited = True
                # 트레일링 스탑 or EMA 붕괴
                elif price < self.peak * (1 - self.trail_pct) or not full_bull:
                    self.position.close(); self.peak = 0.0; self.half_exited = False
            elif self.position.is_short:
                if price < self.trough: self.trough = price
                if not self.half_exited and price <= self.entry_price * (1 - self.half_exit_pct):
                    half = max(1, self.position.size // 2)
                    self.buy(size=half)
                    self.half_exited = True
                elif price > self.trough * (1 + self.trail_pct) or not full_bear:
                    self.position.close(); self.trough = 0.0; self.half_exited = False


# ── H. 재진입 쿨다운 ─────────────────────────────────────────────
class CooldownStrategy(Strategy):
    trail_pct    = 0.07
    base_pct     = 0.60
    gap_min      = 0.001
    vol_scale    = 0.5
    cooldown_bars = 5   # 손절 후 N봉 대기

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
    print("추가 개선 실험 (워크포워드)")
    print("베이스: EMAgap+VolSizing | 튜닝:2019-23 | 검증:2024-26")
    print("=" * 70)

    df = load_df()
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()

    results = []

    # 베이스라인
    print("\n[현재 최고] EMAgap+VolSizing base")
    bt = Backtest(df_test, BestBase, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
    results.append(('현재최고(base)', s, {}))
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # E. 볼륨 필터
    print("\n[E] 볼륨 필터 탐색")
    best_p, best_sharpe = None, -999
    for vol_thresh in [1.0, 1.1, 1.2, 1.5, 2.0]:
        bt = Backtest(df_train, VolumeFilterStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5, vol_thresh=vol_thresh)
        if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
            best_sharpe = s['Sharpe Ratio']
            best_p = {'trail_pct': 0.07, 'base_pct': 0.60, 'gap_min': 0.001, 'vol_scale': 0.5, 'vol_thresh': vol_thresh}
    bt = Backtest(df_test, VolumeFilterStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('E.VolumeFilter', s, best_p))
    print(f"    최적: vol_thresh={best_p['vol_thresh']}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # F. EMA 주기 재탐색
    print("\n[F] EMA 주기 재탐색 (현 전략 기반)")
    best_p, best_sharpe = None, -999
    ema_combos = [
        (10, 30, 100), (10, 50, 200), (15, 40, 150),
        (20, 50, 200), (20, 60, 200), (25, 50, 200),
        (12, 26, 100), (8, 21, 89),
    ]
    for ef, es, et in ema_combos:
        df_tmp = load_df(ef=ef, es=es, et=et)
        df_tr  = df_tmp[df_tmp.index < '2024-01-01'].copy()
        bt = Backtest(df_tr, BestBase, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
        if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
            best_sharpe = s['Sharpe Ratio']
            best_p = {'ef': ef, 'es': es, 'et': et}
    ef, es, et = best_p['ef'], best_p['es'], best_p['et']
    df_best = load_df(ef=ef, es=es, et=et)
    df_te = df_best[df_best.index >= '2024-01-01'].copy()
    bt = Backtest(df_te, BestBase, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
    results.append((f'F.EMA({ef}/{es}/{et})', s, best_p))
    print(f"    최적: EMA {ef}/{es}/{et}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # G. 분할 청산
    print("\n[G] 분할 청산 탐색")
    best_p, best_sharpe = None, -999
    for half_pct in [0.03, 0.05, 0.07, 0.10, 0.15]:
        bt = Backtest(df_train, HalfExitStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5, half_exit_pct=half_pct)
        if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
            best_sharpe = s['Sharpe Ratio']
            best_p = {'trail_pct': 0.07, 'base_pct': 0.60, 'gap_min': 0.001, 'vol_scale': 0.5, 'half_exit_pct': half_pct}
    bt = Backtest(df_test, HalfExitStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('G.HalfExit', s, best_p))
    print(f"    최적: half_exit_pct={best_p['half_exit_pct']:.0%}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # H. 쿨다운
    print("\n[H] 재진입 쿨다운 탐색")
    best_p, best_sharpe = None, -999
    for cooldown in [2, 3, 5, 8, 10]:
        bt = Backtest(df_train, CooldownStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.07, base_pct=0.60, gap_min=0.001, vol_scale=0.5, cooldown_bars=cooldown)
        if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
            best_sharpe = s['Sharpe Ratio']
            best_p = {'trail_pct': 0.07, 'base_pct': 0.60, 'gap_min': 0.001, 'vol_scale': 0.5, 'cooldown_bars': cooldown}
    bt = Backtest(df_test, CooldownStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('H.Cooldown', s, best_p))
    print(f"    최적: cooldown_bars={best_p['cooldown_bars']}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # 최종 비교
    print("\n" + "=" * 70)
    print("전체 결과 비교 (2024-2026 미래 검증)")
    print("=" * 70)
    print(f"{'전략':<22} {'수익률':>8} {'MDD':>8} {'Sharpe':>8} {'거래수':>7} {'PF':>7}")
    print("-" * 65)
    for name, s, p in sorted(results, key=lambda x: x[1]['Sharpe Ratio'], reverse=True):
        tag = " ← 신기록" if s['Sharpe Ratio'] > 0.979 else ""
        print(f"{name:<22} {s['Return [%]']:>7.1f}% {s['Max. Drawdown [%]']:>7.1f}% "
              f"{s['Sharpe Ratio']:>8.3f} {s['# Trades']:>7} {s['Profit Factor']:>7.2f}{tag}")

    # 신기록 저장
    best = max(results, key=lambda x: x[1]['Sharpe Ratio'])
    name, s, p = best
    if s['Sharpe Ratio'] > 0.979:
        save_result(s, f'Improvement2_{name}', '4h', leverage=1,
                    trail_pct=p.get('trail_pct', 0.07),
                    position_pct=p.get('base_pct', p.get('position_pct', 0.60)),
                    ema_fast=20, ema_slow=50, ema_trend=200,
                    note=f'{name}_params={p}')
        print(f"\n신기록! {name} → 저장 완료")
    else:
        print(f"\n기존 최고(Sharpe 0.979)를 넘지 못함 → 현재 전략 유지")
