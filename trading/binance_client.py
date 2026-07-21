"""
바이낸스 선물 API 클라이언트 (테스트넷/실거래 공통)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import hmac
import hashlib
from urllib.parse import urlencode
import pandas as pd
from config import API_KEY, API_SECRET, TESTNET, SYMBOL, LEVERAGE


BASE_URL = (
    "https://testnet.binancefuture.com"
    if TESTNET else
    "https://fapi.binance.com"
)


def _sign(params: dict) -> str:
    return hmac.new(
        API_SECRET.encode(), urlencode(params).encode(), hashlib.sha256
    ).hexdigest()


def _get(endpoint, params=None, signed=False):
    params = params or {}
    if signed:
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = _sign(params)
    headers = {'X-MBX-APIKEY': API_KEY}
    r = requests.get(BASE_URL + endpoint, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def _post(endpoint, params=None):
    params = params or {}
    params['timestamp'] = int(time.time() * 1000)
    params['signature'] = _sign(params)
    headers = {'X-MBX-APIKEY': API_KEY}
    r = requests.post(BASE_URL + endpoint, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


# ── 계좌 정보 ─────────────────────────────────────────────────
def get_account():
    return _get("/fapi/v2/account", signed=True)


def get_balance():
    acc = get_account()
    for asset in acc['assets']:
        if asset['asset'] == 'USDT':
            return float(asset['availableBalance'])
    return 0.0


def get_position():
    """현재 포지션 반환 (없으면 None)"""
    acc = get_account()
    for p in acc['positions']:
        if p['symbol'] == SYMBOL:
            amt = float(p['positionAmt'])
            if amt != 0:
                return {
                    'side':   'long' if amt > 0 else 'short',
                    'size':   abs(amt),
                    'entry':  float(p['entryPrice']),
                    'pnl':    float(p['unrealizedProfit']),
                }
    return None


# ── 가격 조회 ─────────────────────────────────────────────────
def get_price():
    data = _get("/fapi/v1/ticker/price", {'symbol': SYMBOL})
    return float(data['price'])


def get_klines(interval='4h', limit=300):
    """최근 캔들 데이터 DataFrame 반환"""
    data = _get("/fapi/v1/klines", {
        'symbol': SYMBOL, 'interval': interval, 'limit': limit
    })
    df = pd.DataFrame(data, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','qav','num_trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df[['open','high','low','close','volume']]


# ── 레버리지 설정 ─────────────────────────────────────────────
def set_leverage(leverage=LEVERAGE):
    return _post("/fapi/v1/leverage", {
        'symbol': SYMBOL, 'leverage': leverage
    })


# ── 수량 계산 ─────────────────────────────────────────────────
def calc_quantity(usdt_amount: float, price: float) -> float:
    """USDT 금액 → BTC 수량 (소수점 3자리)"""
    qty = usdt_amount / price
    return round(qty, 3)


# ── 주문 실행 ─────────────────────────────────────────────────
def place_order(side: str, quantity: float, order_type='MARKET'):
    """
    side: 'BUY' or 'SELL'
    quantity: BTC 수량
    """
    return _post("/fapi/v1/order", {
        'symbol':   SYMBOL,
        'side':     side,
        'type':     order_type,
        'quantity': quantity,
    })


def close_position():
    """현재 포지션 전량 청산"""
    pos = get_position()
    if pos is None:
        return None
    side = 'SELL' if pos['side'] == 'long' else 'BUY'
    return _post("/fapi/v1/order", {
        'symbol':           SYMBOL,
        'side':             side,
        'type':             'MARKET',
        'quantity':         pos['size'],
        'reduceOnly':       'true',
    })


# ── 연결 테스트 ───────────────────────────────────────────────
def ping():
    try:
        _get("/fapi/v1/ping")
        return True
    except Exception as e:
        print(f"ping 실패: {e}")
        return False


if __name__ == "__main__":
    print(f"{'테스트넷' if TESTNET else '실거래'} 연결 테스트")
    if ping():
        print("✓ 연결 성공")
        bal = get_balance()
        print(f"✓ 잔고: {bal:,.2f} USDT")
        price = get_price()
        print(f"✓ BTC 현재가: ${price:,.2f}")
        pos = get_position()
        if pos:
            print(f"✓ 현재 포지션: {pos}")
        else:
            print("✓ 현재 포지션 없음")
    else:
        print("✗ 연결 실패 - API 키를 확인하세요")
