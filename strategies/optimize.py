import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import Backtest
from strategies.swing_strategy import SwingStrategy, prepare_df

if __name__ == "__main__":
    df = prepare_df()

    print("레버리지별 결과 비교\n" + "=" * 60)

    results = []
    for leverage in [1, 2, 3, 5]:
        bt = Backtest(
            df, SwingStrategy,
            cash=500_000,
            commission=0.0004,
            margin=1/leverage,       # 레버리지 설정
            exclusive_orders=True
        )
        stats = bt.run(trail_pct=0.07, position_pct=0.30)
        results.append({
            'leverage': leverage,
            'return':   stats['Return [%]'],
            'mdd':      stats['Max. Drawdown [%]'],
            'sharpe':   stats['Sharpe Ratio'],
            'pf':       stats['Profit Factor'],
            'trades':   stats['# Trades'],
        })

    print(f"{'레버리지':<10} {'수익률':<12} {'MDD':<12} {'Sharpe':<10} {'PF':<8} {'거래수'}")
    print("-" * 60)
    for r in results:
        print(f"{r['leverage']}x{'':<8} {r['return']:<12.1f} {r['mdd']:<12.1f} "
              f"{r['sharpe']:<10.3f} {r['pf']:<8.3f} {r['trades']}")

    # 최적 레버리지로 재최적화
    print("\n" + "=" * 60)
    print("최적 레버리지(3x) 파라미터 세밀 최적화")
    print("=" * 60)

    bt3 = Backtest(
        df, SwingStrategy,
        cash=500_000,
        commission=0.0004,
        margin=1/3,
        exclusive_orders=True
    )
    stats3, _ = bt3.optimize(
        trail_pct    = [0.05, 0.06, 0.07, 0.08, 0.10],
        position_pct = [0.20, 0.25, 0.30, 0.35, 0.40],
        maximize='Sharpe Ratio',
        return_heatmap=True
    )

    print(f"최적 trail_pct:    {stats3._strategy.trail_pct * 100:.0f}%")
    print(f"최적 position_pct: {stats3._strategy.position_pct * 100:.0f}%")
    print(stats3[['Return [%]', 'Buy & Hold Return [%]', 'Win Rate [%]',
                   'Max. Drawdown [%]', '# Trades', 'Profit Factor', 'Sharpe Ratio']])

    bt3.plot()
