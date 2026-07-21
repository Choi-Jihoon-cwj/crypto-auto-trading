"""
5m 진입 타이밍 정밀화
4h 신호 발생 후 5m 봉으로 더 좋은 가격에 진입

로직:
  롱 신호: 5m RSI < 45 (눌림목) 또는 4h 이후 1봉 이상 지남 → 즉시 진입
  숏 신호: 5m RSI > 55 (되돌림) 또는 4h 이후 1봉 이상 지남 → 즉시 진입
  타임아웃: 4h 봉 1개(4시간) 내 조건 미충족 시 시장가 진입
"""
import sys, os, time, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import ta
from trading.binance_client import get_klines, get_price

log = logging.getLogger(__name__)

RSI_LONG_ENTRY  = 45   # 롱: 5m RSI 이 값 이하일 때 진입 (눌림목)
RSI_SHORT_ENTRY = 55   # 숏: 5m RSI 이 값 이상일 때 진입 (되돌림)
TIMEOUT_SECONDS = 4 * 3600  # 4시간 타임아웃 (다음 4h 봉 전에는 반드시 진입)
CHECK_INTERVAL  = 60        # 5m 신호 체크 주기 (60초)


def get_5m_rsi() -> float:
    """5m 봉 최신 RSI 조회"""
    try:
        df = get_klines(interval='5m', limit=50)
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        return float(rsi.iloc[-2])  # 완성된 봉 기준
    except Exception as e:
        log.warning(f"5m RSI 조회 실패: {e}")
        return 50.0  # 실패 시 중립값


def wait_for_entry(signal: str, timeout_sec: int = TIMEOUT_SECONDS) -> dict:
    """
    5m 봉 기준으로 최적 진입 타이밍 대기
    반환: {'price': float, 'rsi': float, 'reason': str, 'waited_sec': int}
    """
    start = time.time()
    best_price = get_price()
    checks = 0

    log.info(f"[5m 타이밍] {signal.upper()} 진입 대기 시작 | 타임아웃 {timeout_sec//3600}h")

    while True:
        elapsed = int(time.time() - start)
        rsi = get_5m_rsi()
        price = get_price()
        checks += 1

        log.info(f"[5m #{checks}] 가격=${price:,.0f} | RSI={rsi:.1f} | 경과={elapsed//60}분")

        # 롱: RSI 눌림목 확인
        if signal == 'long' and rsi <= RSI_LONG_ENTRY:
            return {
                'price': price, 'rsi': rsi,
                'reason': f'5m RSI 눌림목 ({rsi:.1f} ≤ {RSI_LONG_ENTRY})',
                'waited_sec': elapsed
            }

        # 숏: RSI 되돌림 확인
        if signal == 'short' and rsi >= RSI_SHORT_ENTRY:
            return {
                'price': price, 'rsi': rsi,
                'reason': f'5m RSI 되돌림 ({rsi:.1f} ≥ {RSI_SHORT_ENTRY})',
                'waited_sec': elapsed
            }

        # 타임아웃: 4h 봉 마감 전까지 조건 미충족 → 시장가 즉시 진입
        if elapsed >= timeout_sec:
            return {
                'price': price, 'rsi': rsi,
                'reason': f'타임아웃 ({elapsed//3600}h 경과) → 시장가 진입',
                'waited_sec': elapsed
            }

        time.sleep(CHECK_INTERVAL)
