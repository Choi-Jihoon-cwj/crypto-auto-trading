import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indicators import load_with_indicators
import pandas as pd

df = load_with_indicators()

# MACD 크로스 감지
macd_cross_up   = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

uptrend   = df['ema_20'] > df['ema_50']
downtrend = df['ema_20'] < df['ema_50']
rsi_ok    = (df['rsi'] > 30) & (df['rsi'] < 70)

long_signal  = uptrend & macd_cross_up   & rsi_ok
short_signal = downtrend & macd_cross_down & rsi_ok

print(f"전체 데이터: {len(df)}개")
print(f"MACD 골든크로스: {macd_cross_up.sum()}번")
print(f"MACD 데드크로스: {macd_cross_down.sum()}번")
print(f"상승추세 구간: {uptrend.sum()}개 캔들")
print(f"하락추세 구간: {downtrend.sum()}개 캔들")
print(f"롱 신호: {long_signal.sum()}번")
print(f"숏 신호: {short_signal.sum()}번")
print(f"\nATR 최솟값: {df['atr'].min():.2f}")
print(f"ATR 최댓값: {df['atr'].max():.2f}")
print(f"가격 최솟값: {df['close'].min():.2f}")
print(f"\n롱 신호 샘플:")
print(df[long_signal][['close', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'rsi', 'atr']].head())
