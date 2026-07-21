"""
신호 계산 엔진
실시간 캔들 데이터 → 매매 신호 계산
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
import ta
from config import (EMA_FAST, EMA_SLOW, EMA_TREND,
                    GAP_MIN, VOL_SCALE, FG_UPPER, FG_LOWER)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast']  = ta.trend.EMAIndicator(df['close'], window=EMA_FAST).ema_indicator()
    df['ema_slow']  = ta.trend.EMAIndicator(df['close'], window=EMA_SLOW).ema_indicator()
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=EMA_TREND).ema_indicator()
    df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    return df


def get_fear_greed() -> int:
    """공포탐욕지수 현재값 조회"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        return int(r.json()['data'][0]['value'])
    except Exception:
        return 50  # 조회 실패 시 중립값


def compute_signal(df: pd.DataFrame, fear_greed: int) -> dict:
    """
    최신 완성 캔들 기준 신호 계산
    반환: {
        'signal': 'long' | 'short' | 'none',
        'pos_pct': float,   # 진입 포지션 비율
        'ema_fast': float, 'ema_slow': float, 'ema_trend': float,
        'fear_greed': int,
        'reason': str       # 신호 or 미발생 이유
    }
    """
    df = add_indicators(df)
    # 마지막 봉은 현재 진행 중 → 완성된 봉 기준 ([-2])
    cur  = df.iloc[-2]
    prev = df.iloc[-3]

    ef,  es,  et  = cur['ema_fast'],  cur['ema_slow'],  cur['ema_trend']
    pef, pes, pet = prev['ema_fast'], prev['ema_slow'], prev['ema_trend']
    vol = cur['volatility_20']

    if any(pd.isna([ef, es, et, vol])) or vol <= 0:
        return {'signal': 'none', 'reason': '지표 계산 중'}

    ema_gap    = abs(ef - es) / cur['close']
    vol_factor = min(max(0.02 / vol, 0.3), 2.0)

    from config import BASE_PCT
    pos_pct = min(BASE_PCT * vol_factor * VOL_SCALE, 0.95)

    full_bull = ef > es > et
    full_bear = ef < es < et
    prev_bull = pef > pes > pet
    prev_bear = pef < pes < pet

    # 기본 진입 조건
    bull_signal = full_bull and not prev_bull
    bear_signal = full_bear and not prev_bear

    result = {
        'ema_fast':    ef,
        'ema_slow':    es,
        'ema_trend':   et,
        'ema_gap_pct': ema_gap * 100,
        'volatility':  vol,
        'fear_greed':  fear_greed,
        'pos_pct':     pos_pct,
        'full_bull':   full_bull,
        'full_bear':   full_bear,
    }

    if bull_signal:
        if ema_gap < GAP_MIN:
            result.update({'signal': 'none', 'reason': f'롱 신호 but EMA 간격 부족 ({ema_gap*100:.3f}% < {GAP_MIN*100:.3f}%)'})
        elif fear_greed > FG_UPPER:
            result.update({'signal': 'none', 'reason': f'롱 신호 but 공포탐욕 과열 ({fear_greed} > {FG_UPPER})'})
        else:
            result.update({'signal': 'long', 'reason': f'Triple EMA 정배열 전환 | FG={fear_greed} | 포지션={pos_pct:.0%}'})

    elif bear_signal:
        if ema_gap < GAP_MIN:
            result.update({'signal': 'none', 'reason': f'숏 신호 but EMA 간격 부족 ({ema_gap*100:.3f}%)'})
        elif fear_greed < FG_LOWER:
            result.update({'signal': 'none', 'reason': f'숏 신호 but 공포탐욕 과매도 ({fear_greed} < {FG_LOWER})'})
        else:
            result.update({'signal': 'short', 'reason': f'Triple EMA 역배열 전환 | FG={fear_greed} | 포지션={pos_pct:.0%}'})

    else:
        trend = '정배열(보유)' if full_bull else ('역배열(보유)' if full_bear else '중립')
        result.update({'signal': 'none', 'reason': f'신호 없음 | 현재 추세: {trend}'})

    return result


def check_exit(df: pd.DataFrame, position: dict) -> dict:
    """
    포지션 청산 조건 확인
    position: {'side': 'long'|'short', 'entry': float, 'peak': float, 'trough': float}
    반환: {'exit': bool, 'reason': str}
    """
    df = add_indicators(df)
    cur = df.iloc[-2]  # 완성된 봉

    from config import TRAIL_PCT
    ef, es, et = cur['ema_fast'], cur['ema_slow'], cur['ema_trend']
    price = cur['close']

    full_bull = ef > es > et
    full_bear = ef < es < et

    if position['side'] == 'long':
        trail_stop = position['peak'] * (1 - TRAIL_PCT)
        if price < trail_stop:
            return {'exit': True, 'reason': f'트레일링 스탑 ({price:.0f} < {trail_stop:.0f})'}
        if not full_bull:
            return {'exit': True, 'reason': 'EMA 정배열 붕괴'}
        return {'exit': False, 'reason': f'보유 중 | peak={position["peak"]:.0f} | stop={trail_stop:.0f}'}

    else:  # short
        trail_stop = position['trough'] * (1 + TRAIL_PCT)
        if price > trail_stop:
            return {'exit': True, 'reason': f'트레일링 스탑 ({price:.0f} > {trail_stop:.0f})'}
        if not full_bear:
            return {'exit': True, 'reason': 'EMA 역배열 붕괴'}
        return {'exit': False, 'reason': f'보유 중 | trough={position["trough"]:.0f} | stop={trail_stop:.0f}'}
