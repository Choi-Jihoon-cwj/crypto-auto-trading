"""
멀티타임프레임 전략
4h 신호 + 1h 추세 방향 확인
- 4h EMA 25/50/200 정렬 신호 발생 시
- 1h EMA 방향이 같은 방향일 때만 진입
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


def load_4h():
    path = os.path.join(DATA_DIR, "ohlcv_4h_full.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=25).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    return df


def load_1h_trend():
    """1h 데이터에서 추세 방향 추출 → 4h 타임스탬프에 맞게 리샘플"""
    path = os.path.join(DATA_DIR, "ohlcv_1h_full.csv")
    df1h = pd.read_csv(path, index_col='timestamp', parse_dates=True)

    # 1h EMA로 추세 방향 판단
    df1h['ema_fast'] = ta.trend.EMAIndicator(df1h['close'], window=25).ema_indicator()
    df1h['ema_slow'] = ta.trend.EMAIndicator(df1h['close'], window=50).ema_indicator()
    df1h['h1_bull']  = (df1h['ema_fast'] > df1h['ema_slow']).astype(int)  # 1=상승추세, 0=하락추세
    df1h['h1_trend'] = df1h['h1_bull'].rolling(3).mean()  # 최근 3봉 평균 (확신도)

    df1h.dropna(inplace=True)
    return df1h[['h1_bull', 'h1_trend']]


def build_mtf_df():
    """4h 데이터에 1h 추세 정보 병합"""
    df4h = load_4h()
    df1h_trend = load_1h_trend()

    # 1h → 4h 타임스탬프에 맞게 forward fill
    df_merged = df4h.copy()
    df_merged['h1_bull']  = df1h_trend['h1_bull'].reindex(df4h.index, method='ffill')
    df_merged['h1_trend'] = df1h_trend['h1_trend'].reindex(df4h.index, method='ffill')
    df_merged.dropna(inplace=True)

    df_merged.columns = [c.capitalize() if c in ['open','high','low','close','volume'] else c
                         for c in df_merged.columns]
    return df_merged


class MTFStrategy(Strategy):
    """4h 신호 + 1h 추세 확인"""
    trail_pct     = 0.05
    base_pct      = 0.60
    gap_min       = 0.001
    vol_scale     = 0.5
    h1_min_trend  = 0.60  # 1h 최근 3봉 중 N% 이상이 같은 방향이어야 진입

    def init(self):
        self.peak = 0.0; self.trough = 0.0

    def next(self):
        price = self.data.Close[-1]
        if price <= 0: return
        ef, es, et = self.data.ema_fast[-1], self.data.ema_slow[-1], self.data.ema_trend[-1]
        vol = self.data.volatility_20[-1]
        h1_trend = self.data.h1_trend[-1]
        if np.isnan(et) or np.isnan(vol) or vol <= 0 or np.isnan(h1_trend): return

        ema_gap    = abs(ef - es) / price
        vol_factor = min(max(0.02 / vol, 0.3), 2.0)
        pos_pct    = min(self.base_pct * vol_factor * self.vol_scale, 0.95)

        full_bull = ef > es > et
        full_bear = ef < es < et
        prev_bull = self.data.ema_fast[-2] > self.data.ema_slow[-2] > self.data.ema_trend[-2]
        prev_bear = self.data.ema_fast[-2] < self.data.ema_slow[-2] < self.data.ema_trend[-2]

        # 1h 추세 확인: 롱은 1h도 상승추세, 숏은 1h도 하락추세
        h1_ok_long  = h1_trend >= self.h1_min_trend
        h1_ok_short = h1_trend <= (1 - self.h1_min_trend)

        bull_entry = full_bull and not prev_bull and ema_gap >= self.gap_min and h1_ok_long
        bear_entry = full_bear and not prev_bear and ema_gap >= self.gap_min and h1_ok_short

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
    print("멀티타임프레임 전략 (4h 신호 + 1h 확인)")
    print("=" * 70)

    print("데이터 병합 중...")
    df = build_mtf_df()
    df_train = df[df.index < '2024-01-01'].copy()
    df_test  = df[df.index >= '2024-01-01'].copy()
    print(f"학습 {len(df_train)}개 | 검증 {len(df_test)}개\n")

    results = []

    # 베이스라인 (MTF 없이)
    from strategies.final_polish import FinalStrategy, load_df
    df_base = load_df(ef=25)
    df_base_test = df_base[df_base.index >= '2024-01-01'].copy()
    bt = Backtest(df_base_test, FinalStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(trail_pct=0.05, base_pct=0.60, gap_min=0.001, vol_scale=0.5)
    results.append(('베이스라인(MTF없음)', s, {}))
    print(f"[베이스] Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # 1h 확인 임계값 탐색
    print("\n[MTF] 1h 확인 임계값 탐색 (train)")
    best_p, best_sharpe = None, -999
    for h1_min in [0.50, 0.60, 0.67, 0.75, 1.00]:
        for trail in [0.04, 0.05, 0.06, 0.07]:
            for base_pct in [0.40, 0.60, 0.80]:
                bt = Backtest(df_train, MTFStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
                s = bt.run(trail_pct=trail, base_pct=base_pct, gap_min=0.001, vol_scale=0.5, h1_min_trend=h1_min)
                if s['# Trades'] >= 8 and s['Sharpe Ratio'] > best_sharpe:
                    best_sharpe = s['Sharpe Ratio']
                    best_p = {'trail_pct': trail, 'base_pct': base_pct, 'gap_min': 0.001,
                              'vol_scale': 0.5, 'h1_min_trend': h1_min}

    bt = Backtest(df_test, MTFStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
    s = bt.run(**best_p)
    results.append(('MTF(4h+1h)', s, best_p))
    print(f"최적파라미터: {best_p}")
    print(f"결과: Return={s['Return [%]']:.1f}% MDD={s['Max. Drawdown [%]']:.1f}% Sharpe={s['Sharpe Ratio']:.3f} Trades={s['# Trades']}")

    # h1_min 별 상세 보기
    print("\n[MTF] 1h 임계값별 상세 결과 (test, trail=0.05, base=0.60)")
    print(f"{'h1_min':>8} {'수익률':>9} {'MDD':>9} {'Sharpe':>9} {'거래수':>8}")
    print("-" * 50)
    for h1_min in [0.50, 0.60, 0.67, 0.75, 1.00]:
        bt = Backtest(df_test, MTFStrategy, cash=500_000, commission=0.0004, exclusive_orders=True)
        s = bt.run(trail_pct=0.05, base_pct=0.60, gap_min=0.001, vol_scale=0.5, h1_min_trend=h1_min)
        tag = " ←" if s['Sharpe Ratio'] > 1.250 else ""
        print(f"{h1_min:>8.0%} {s['Return [%]']:>8.1f}% {s['Max. Drawdown [%]']:>8.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>8}{tag}")

    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    print(f"{'전략':<22} {'수익률':>8} {'MDD':>8} {'Sharpe':>9} {'거래수':>7}")
    print("-" * 58)
    for name, s, p in sorted(results, key=lambda x: x[1]['Sharpe Ratio'], reverse=True):
        tag = " ← 신기록!" if s['Sharpe Ratio'] > 1.250 else ""
        print(f"{name:<22} {s['Return [%]']:>7.1f}% {s['Max. Drawdown [%]']:>7.1f}% "
              f"{s['Sharpe Ratio']:>9.3f} {s['# Trades']:>7}{tag}")

    best = max(results, key=lambda x: x[1]['Sharpe Ratio'])
    if best[1]['Sharpe Ratio'] > 1.250:
        save_result(best[1], 'MTF_Strategy', '4h', leverage=1,
                    trail_pct=best[2].get('trail_pct', 0.05),
                    position_pct=best[2].get('base_pct', 0.60),
                    ema_fast=25, ema_slow=50, ema_trend=200,
                    note=f'MTF_4h1h_{best[2]}')
        print(f"\n신기록! 저장 완료")
    else:
        print(f"\n기존 최고(Sharpe 1.250) 유지")
