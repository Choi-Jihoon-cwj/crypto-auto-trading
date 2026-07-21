"""
Enhanced Strategy Experiments
모든 개선안을 워크포워드 방식으로 검증:
  - 튜닝: 2019-2023 (학습/최적화)
  - 검증: 2024-2026 (실제 미래 성과)
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


# ── 데이터 로더 ────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
# 전략 1: 기본 전략 (베이스라인)
# ══════════════════════════════════════════════════════════════
class BaseStrategy(Strategy):
    trail_pct    = 0.07
    position_pct = 0.50

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        if np.isnan(et): return

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull
        bear_entry = full_bear and not prev_bear

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


# ══════════════════════════════════════════════════════════════
# 전략 2: ATR 기반 트레일링 스탑
# 변동성이 클 때 더 넓은 스탑, 작을 때 더 좁은 스탑
# ══════════════════════════════════════════════════════════════
class ATRTrailStrategy(Strategy):
    atr_mult     = 2.5   # ATR 배수 (2.0 ~ 3.5)
    position_pct = 0.50

    def init(self):
        self.peak = 0.0; self.trough = 0.0; self.peak_atr = 0.0; self.trough_atr = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        atr = self.data.atr[-1]
        if np.isnan(et) or np.isnan(atr) or atr <= 0: return

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull
        bear_entry = full_bear and not prev_bear

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


# ══════════════════════════════════════════════════════════════
# 전략 3: RSI 진입 필터
# 롱: RSI < 70 (과매수 제외)
# 숏: RSI > 30 (과매도 제외)
# ══════════════════════════════════════════════════════════════
class RSIFilterStrategy(Strategy):
    trail_pct    = 0.07
    position_pct = 0.50
    rsi_upper    = 70   # 롱 진입 RSI 상한
    rsi_lower    = 30   # 숏 진입 RSI 하한

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        rsi = self.data.rsi[-1]
        if np.isnan(et) or np.isnan(rsi): return

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and rsi < self.rsi_upper
        bear_entry = full_bear and not prev_bear and rsi > self.rsi_lower

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


# ══════════════════════════════════════════════════════════════
# 전략 4: EMA 간격 강도 필터
# EMA 간격이 일정 이상일 때만 진입 (추세 강도 확인)
# ══════════════════════════════════════════════════════════════
class EMAgapStrategy(Strategy):
    trail_pct    = 0.07
    position_pct = 0.50
    gap_min      = 0.005  # EMA fast-slow 최소 간격 (가격 대비 %)

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        if np.isnan(et): return

        ema_gap = abs(ef - es) / price

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min

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


# ══════════════════════════════════════════════════════════════
# 전략 5: 변동성 기반 포지션 사이징
# 변동성 낮을 때 크게, 높을 때 작게 (역변동성 사이징)
# ══════════════════════════════════════════════════════════════
class VolSizingStrategy(Strategy):
    trail_pct    = 0.07
    base_pct     = 0.50   # 기본 포지션 비율
    vol_scale    = 1.5    # 변동성 조정 강도

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0: return

        # 변동성이 낮을수록 더 큰 포지션 (역변동성)
        # 기준 변동성: 20일 변동성의 역수 정규화
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)  # 0.3x ~ 2.0x 범위 제한
        pos_pct = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]
        bull_entry = full_bull and not prev_bull
        bear_entry = full_bear and not prev_bear

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


# ══════════════════════════════════════════════════════════════
# 전략 6: 종합 (ATR 트레일 + RSI 필터 + EMA 강도 필터)
# ══════════════════════════════════════════════════════════════
class CombinedStrategy(Strategy):
    atr_mult     = 2.5
    position_pct = 0.50
    rsi_upper    = 70
    rsi_lower    = 30
    gap_min      = 0.003

    def init(self):
        self.peak = 0.0; self.trough = 0.0; self.peak_atr = 0.0; self.trough_atr = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        rsi = self.data.rsi[-1]
        atr = self.data.atr[-1]
        if np.isnan(et) or np.isnan(rsi) or np.isnan(atr) or atr <= 0: return

        ema_gap = abs(ef - es) / price
        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]

        bull_entry = full_bull and not prev_bull and rsi < self.rsi_upper and ema_gap >= self.gap_min
        bear_entry = full_bear and not prev_bear and rsi > self.rsi_lower and ema_gap >= self.gap_min

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


# ══════════════════════════════════════════════════════════════
# 실험 실행
# ══════════════════════════════════════════════════════════════
def run_experiment(df_train, df_test, strategy_cls, params, name):
    bt_train = Backtest(df_train, strategy_cls, cash=500_000,
                        commission=0.0004, exclusive_orders=True)

    # 튜닝 (학습 기간)
    try:
        opt = bt_train.optimize(
            **params,
            maximize='Sharpe Ratio',
            constraint=lambda p: p.get('trail_pct', 0.07) > 0,
            return_heatmap=False
        )
    except Exception as e:
        print(f"  [{name}] 튜닝 실패: {e}")
        return None

    best_params = {k: getattr(opt._strategy, k) for k in params}

    # 검증 (미래 데이터)
    bt_test = Backtest(df_test, strategy_cls, cash=500_000,
                       commission=0.0004, exclusive_orders=True)
    stats = bt_test.run(**best_params)
    return stats, best_params


if __name__ == "__main__":
    print("=" * 70)
    print("향상된 전략 실험 (워크포워드 검증)")
    print("튜닝: 2019-2023 | 검증: 2024-2026 (미래 데이터)")
    print("=" * 70)

    df = load_df()
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()
    print(f"학습 {len(df_train)}개 | 검증 {len(df_test)}개\n")

    results = []

    # ── 0. 베이스라인 ──────────────────────────────────────────
    print("[0] 베이스라인 (기본 전략)")
    bt = Backtest(df_test, BaseStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(trail_pct=0.07, position_pct=0.50)
    results.append(('0.BaseStrategy', 0.50, 0.07, s))
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 1. ATR 트레일링 스탑 ───────────────────────────────────
    print("\n[1] ATR 트레일링 스탑 (atr_mult 최적화)")
    best_s, best_p, best_sharpe = None, None, -999
    for atr_mult in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for pos in [0.30, 0.50, 0.70]:
            bt = Backtest(df_train, ATRTrailStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
            s = bt.run(atr_mult=atr_mult, position_pct=pos)
            if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
                best_sharpe = s['Sharpe Ratio']
                best_p = {'atr_mult': atr_mult, 'position_pct': pos}
    bt = Backtest(df_test, ATRTrailStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('1.ATRTrail', best_p['position_pct'], best_p['atr_mult'], s))
    print(f"    최적파라미터: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 2. RSI 진입 필터 ──────────────────────────────────────
    print("\n[2] RSI 진입 필터 (rsi_upper/lower 최적화)")
    best_s, best_p, best_sharpe = None, None, -999
    for rsi_upper in [65, 70, 75]:
        for rsi_lower in [25, 30, 35]:
            for pos in [0.30, 0.50, 0.70]:
                bt = Backtest(df_train, RSIFilterStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                s = bt.run(trail_pct=0.07, position_pct=pos, rsi_upper=rsi_upper, rsi_lower=rsi_lower)
                if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
                    best_sharpe = s['Sharpe Ratio']
                    best_p = {'trail_pct': 0.07, 'position_pct': pos, 'rsi_upper': rsi_upper, 'rsi_lower': rsi_lower}
    bt = Backtest(df_test, RSIFilterStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('2.RSIFilter', best_p['position_pct'], best_p['rsi_upper'], s))
    print(f"    최적파라미터: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 3. EMA 강도 필터 ──────────────────────────────────────
    print("\n[3] EMA 간격 강도 필터 (gap_min 최적화)")
    best_p, best_sharpe = None, -999
    for gap_min in [0.002, 0.003, 0.005, 0.008, 0.01]:
        for pos in [0.30, 0.50, 0.70]:
            bt = Backtest(df_train, EMAgapStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
            s = bt.run(trail_pct=0.07, position_pct=pos, gap_min=gap_min)
            if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
                best_sharpe = s['Sharpe Ratio']
                best_p = {'trail_pct': 0.07, 'position_pct': pos, 'gap_min': gap_min}
    bt = Backtest(df_test, EMAgapStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('3.EMAgap', best_p['position_pct'], best_p['gap_min'], s))
    print(f"    최적파라미터: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 4. 변동성 포지션 사이징 ────────────────────────────────
    print("\n[4] 변동성 기반 포지션 사이징 (vol_scale 최적화)")
    best_p, best_sharpe = None, -999
    for vol_scale in [0.5, 1.0, 1.5, 2.0]:
        for base_pct in [0.30, 0.50, 0.70]:
            bt = Backtest(df_train, VolSizingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
            s = bt.run(trail_pct=0.07, base_pct=base_pct, vol_scale=vol_scale)
            if s['# Trades'] >= 10 and s['Sharpe Ratio'] > best_sharpe:
                best_sharpe = s['Sharpe Ratio']
                best_p = {'trail_pct': 0.07, 'base_pct': base_pct, 'vol_scale': vol_scale}
    bt = Backtest(df_test, VolSizingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('4.VolSizing', best_p['base_pct'], best_p['vol_scale'], s))
    print(f"    최적파라미터: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 5. 종합 전략 ──────────────────────────────────────────
    print("\n[5] 종합 전략 (ATR + RSI + EMA강도 조합)")
    best_p, best_sharpe = None, -999
    for atr_mult in [2.0, 2.5, 3.0]:
        for rsi_upper in [65, 70]:
            for rsi_lower in [30, 35]:
                for gap_min in [0.003, 0.005]:
                    for pos in [0.30, 0.50, 0.70]:
                        bt = Backtest(df_train, CombinedStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                        s = bt.run(atr_mult=atr_mult, position_pct=pos,
                                   rsi_upper=rsi_upper, rsi_lower=rsi_lower, gap_min=gap_min)
                        if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                            best_sharpe = s['Sharpe Ratio']
                            best_p = {'atr_mult': atr_mult, 'position_pct': pos,
                                      'rsi_upper': rsi_upper, 'rsi_lower': rsi_lower, 'gap_min': gap_min}
    bt = Backtest(df_test, CombinedStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('5.Combined', best_p['position_pct'], str(best_p), s))
    print(f"    최적파라미터: {best_p}")
    print(f"    Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% "
          f"Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 최종 비교표 ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("최종 결과 비교 (2024-2026 미래 검증)")
    print("=" * 70)
    print(f"{'전략':<20} {'수익률':>8} {'MDD':>8} {'Sharpe':>8} {'거래수':>7} {'PF':>7}")
    print("-" * 70)
    for name, pos, param, s in sorted(results, key=lambda x: x[3]['Sharpe Ratio'], reverse=True):
        print(f"{name:<20} {s['Return [%]']:>7.1f}% {s['Max. Drawdown [%]']:>7.1f}% "
              f"{s['Sharpe Ratio']:>8.3f} {s['# Trades']:>7} {s['Profit Factor']:>7.2f}")

    # 최고 결과 저장
    best = max(results, key=lambda x: x[3]['Sharpe Ratio'])
    save_result(best[3], f'Enhanced_{best[0]}', '4h', leverage=1,
                trail_pct=0.07, position_pct=best[1],
                ema_fast=20, ema_slow=50, ema_trend=200,
                note=f'walkforward_2024_best')
    print(f"\n최고 전략: {best[0]} → 결과 저장 완료")
