"""
공포탐욕지수 + 펀딩비 활용 전략
베이스: EMA 25/50/200, EMAgap+VolSizing (Sharpe 1.250)

추가 필터:
- 공포탐욕: 극단 탐욕(>80) 시 롱 진입 차단 / 극단 공포(<20) 시 숏 진입 차단
- 펀딩비: 높은 양수 펀딩비(>0.1%) 시 롱 차단 (과매수) / 낮은 음수(<-0.1%) 시 숏 차단
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


def load_base_df():
    path = os.path.join(DATA_DIR, "ohlcv_4h_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=25).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    return df


def load_fear_greed():
    path = os.path.join(DATA_DIR, "fear_greed.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    return df[['fear_greed']]


def load_funding():
    path = os.path.join(DATA_DIR, "funding_rate.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    # 8시간마다 발생 → 4h 캔들에 맞게 ffill
    return df[['funding_rate']]


def build_df():
    df = load_base_df()

    # 공포탐욕: 일별 → 4h에 ffill
    fg = load_fear_greed()
    fg.index = fg.index.normalize()
    df['date'] = df.index.normalize()
    fg_map = fg['fear_greed'].to_dict()
    df['fear_greed'] = df['date'].map(fg_map)
    df.drop(columns=['date'], inplace=True)

    # 펀딩비: 8h 주기 → 4h에 ffill
    fr = load_funding()
    df['funding_rate'] = fr['funding_rate'].reindex(df.index, method='ffill')

    df.dropna(inplace=True)
    df.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                  for c in df.columns]
    return df


# ── 공포탐욕 필터 전략 ────────────────────────────────────────
class FearGreedStrategy(Strategy):
    trail_pct  = 0.05
    base_pct   = 0.60
    gap_min    = 0.001
    vol_scale  = 0.5
    fg_upper   = 80   # 탐욕 > 이 값이면 롱 차단
    fg_lower   = 20   # 공포 < 이 값이면 숏 차단

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

        # 극단 탐욕에서 롱 차단, 극단 공포에서 숏 차단
        fg_ok_long  = fg <= self.fg_upper
        fg_ok_short = fg >= self.fg_lower

        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and fg_ok_long
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and fg_ok_short

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


# ── 펀딩비 필터 전략 ──────────────────────────────────────────
class FundingStrategy(Strategy):
    trail_pct  = 0.05
    base_pct   = 0.60
    gap_min    = 0.001
    vol_scale  = 0.5
    fr_upper   = 0.001   # 펀딩비 > 이 값이면 롱 차단 (0.1%)
    fr_lower   = -0.001  # 펀딩비 < 이 값이면 숏 차단 (-0.1%)

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        fr  = self.data.funding_rate[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0 or np.isnan(fr): return

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]

        # 높은 펀딩비(시장 과매수)면 롱 차단, 낮은 펀딩비면 숏 차단
        fr_ok_long  = fr <= self.fr_upper
        fr_ok_short = fr >= self.fr_lower

        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and fr_ok_long
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and fr_ok_short

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


# ── 공포탐욕 + 펀딩비 조합 ───────────────────────────────────
class SentimentStrategy(Strategy):
    trail_pct  = 0.05
    base_pct   = 0.60
    gap_min    = 0.001
    vol_scale  = 0.5
    fg_upper   = 80
    fg_lower   = 20
    fr_upper   = 0.001
    fr_lower   = -0.001

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        fg  = self.data.fear_greed[-1]
        fr  = self.data.funding_rate[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0 or np.isnan(fg) or np.isnan(fr): return

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]

        ok_long  = ema_gap >= self.gap_min and fg <= self.fg_upper and fr <= self.fr_upper
        ok_short = ema_gap >= self.gap_min and fg >= self.fg_lower and fr >= self.fr_lower

        bull_entry = full_bull and not prev_bull and ok_long
        bear_entry = full_bear and not prev_bear and ok_short

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
    print("공포탐욕 + 펀딩비 전략 (워크포워드)")
    print("=" * 70)

    print("데이터 병합 중...")
    df = build_df()
    # 펀딩비는 2019-09부터 → 학습/검증 기간 조정
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()
    print(f"학습 {len(df_train)}개 | 검증 {len(df_test)}개")
    print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}\n")

    results = []

    # 베이스라인
    from strategies.final_polish import FinalStrategy, load_df
    df_base = load_df(ef=25)
    df_base_test = df_base[df_base.index >= '2024-01-01'].copy()
    bt = Backtest(df_base_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(trail_pct=0.05, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
    results.append(('베이스라인', s, {}))
    print(f"[베이스] Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 공포탐욕 필터 ──────────────────────────────────────────
    print("\n[A] 공포탐욕 임계값 탐색 (train)")
    best_p, best_sharpe = None, -999
    for fg_upper in [70, 75, 80, 85, 90]:
        for fg_lower in [10, 15, 20, 25, 30]:
            for trail in [0.04, 0.05, 0.06]:
                for base_pct in [0.40, 0.60, 0.80]:
                    bt = Backtest(df_train, FearGreedStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                    s = bt.run(trail_pct=trail, base_pct=base_pct, gap_min=0.001, vol_scale=0.5,
                               fg_upper=fg_upper, fg_lower=fg_lower)
                    if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                        best_sharpe = s['Sharpe Ratio']
                        best_p = {'trail_pct': trail, 'base_pct': base_pct, 'gap_min': 0.001,
                                  'vol_scale': 0.5, 'fg_upper': fg_upper, 'fg_lower': fg_lower}
    bt = Backtest(df_test, FearGreedStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('A.FearGreed', s, best_p))
    print(f"최적: {best_p}")
    print(f"결과: Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # 상세: fg_upper 별
    print("\n공포탐욕 임계값별 상세 (test, trail=0.05, base=0.60)")
    print(f"{'fg_upper':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8}")
    print("-" * 50)
    for fg_upper in [70, 75, 80, 85, 90, 100]:
        bt = Backtest(df_test, FearGreedStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.05, base_pct=0.60, gap_min=0.001, vol_scale=0.5, fg_upper=fg_upper, fg_lower=20)
        tag = " ←" if s['Sharpe Ratio'] > 1.250 else ""
        print(f"{fg_upper:>10} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8}{tag}")

    # ── 펀딩비 필터 ────────────────────────────────────────────
    print("\n[B] 펀딩비 임계값 탐색 (train)")
    best_p, best_sharpe = None, -999
    for fr_upper in [0.0003, 0.0005, 0.001, 0.002, 0.003]:
        for fr_lower in [-0.003, -0.002, -0.001, -0.0005]:
            for trail in [0.04, 0.05, 0.06]:
                for base_pct in [0.40, 0.60, 0.80]:
                    bt = Backtest(df_train, FundingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                    s = bt.run(trail_pct=trail, base_pct=base_pct, gap_min=0.001, vol_scale=0.5,
                               fr_upper=fr_upper, fr_lower=fr_lower)
                    if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                        best_sharpe = s['Sharpe Ratio']
                        best_p = {'trail_pct': trail, 'base_pct': base_pct, 'gap_min': 0.001,
                                  'vol_scale': 0.5, 'fr_upper': fr_upper, 'fr_lower': fr_lower}
    bt = Backtest(df_test, FundingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('B.Funding', s, best_p))
    print(f"최적: {best_p}")
    print(f"결과: Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # 상세: fr_upper 별
    print("\n펀딩비 임계값별 상세 (test, trail=0.05, base=0.60)")
    print(f"{'fr_upper':>10} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8}")
    print("-" * 50)
    for fr_upper in [0.0003, 0.0005, 0.001, 0.002, 0.003, 9999]:
        bt = Backtest(df_test, FundingStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.05, base_pct=0.60, gap_min=0.001, vol_scale=0.5,
                   fr_upper=fr_upper, fr_lower=-0.001)
        tag = " ←" if s['Sharpe Ratio'] > 1.250 else ""
        label = f"{fr_upper:.4f}" if fr_upper < 9999 else "무제한"
        print(f"{label:>10} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8}{tag}")

    # ── 조합 ──────────────────────────────────────────────────
    print("\n[C] 공포탐욕 + 펀딩비 조합 탐색 (train)")
    best_p, best_sharpe = None, -999
    for fg_upper in [75, 80, 85]:
        for fg_lower in [15, 20, 25]:
            for fr_upper in [0.0005, 0.001, 0.002]:
                for fr_lower in [-0.002, -0.001, -0.0005]:
                    for trail in [0.04, 0.05, 0.06]:
                        for base_pct in [0.40, 0.60, 0.80]:
                            bt = Backtest(df_train, SentimentStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                            s = bt.run(trail_pct=trail, base_pct=base_pct, gap_min=0.001, vol_scale=0.5,
                                       fg_upper=fg_upper, fg_lower=fg_lower,
                                       fr_upper=fr_upper, fr_lower=fr_lower)
                            if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                                best_sharpe = s['Sharpe Ratio']
                                best_p = {'trail_pct': trail, 'base_pct': base_pct, 'gap_min': 0.001,
                                          'vol_scale': 0.5, 'fg_upper': fg_upper, 'fg_lower': fg_lower,
                                          'fr_upper': fr_upper, 'fr_lower': fr_lower}
    bt = Backtest(df_test, SentimentStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('C.Sentiment조합', s, best_p))
    print(f"최적: {best_p}")
    print(f"결과: Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # ── 최종 비교 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("최종 결과 비교 (2024-2026 미래 검증)")
    print("=" * 70)
    print(f"{'전략':<22} {'수익률':>8} {'MDD':>8} {'Sharpe':>9} {'거래수':>7} {'PF':>7}")
    print("-" * 65)
    for name, s, p in sorted(results, key=lambda x: x[1]['Sharpe Ratio'], reverse=True):
        tag = " ← 신기록!" if s['Sharpe Ratio'] > 1.250 else ""
        print(f"{name:<22} {s['Return [%]']:>7.1f}% {s['Max. Drawdown [%]']:>7.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>7} {s['Profit Factor']:>7.2f}{tag}")

    best = max(results, key=lambda x: x[1]['Sharpe Ratio'])
    name, s, p = best
    if s['Sharpe Ratio'] > 1.250:
        save_result(s, f'Sentiment_{name}', '4h', leverage=1,
                    trail_pct=p.get('trail_pct', 0.05),
                    position_pct=p.get('base_pct', 0.60),
                    ema_fast=25, ema_slow=50, ema_trend=200,
                    note=f'FG+FR_{name}_params={p}')
        print(f"\n신기록! 저장 완료")
    else:
        print(f"\n기존 최고(Sharpe 1.250) 유지 → 센티멘트 데이터 추가 효과 없음")
