"""
라이브 트레이딩 봇 v2 (4h 신호 + 5m 진입 정밀화)

흐름:
  1. 4h 봉 마감마다 신호 확인
  2. 신호 발생 시 → 5m RSI로 더 좋은 진입 타이밍 대기
  3. 4h 타임아웃 내 조건 미충족 → 시장가 즉시 진입
  4. 포지션 보유 중 → 4h 기준 트레일링 스탑 / EMA 붕괴 청산
"""
import sys, os, time, json, logging
from datetime import datetime, timezone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (SYMBOL, TIMEFRAME, TESTNET, LEVERAGE,
                    TRAIL_PCT, BASE_PCT, EMA_FAST, EMA_SLOW, EMA_TREND)
from trading.binance_client import (
    ping, get_balance, get_price, get_klines,
    get_position, set_leverage, place_order, close_position, calc_quantity
)
from trading.signal_engine import compute_signal, check_exit, get_fear_greed
from trading.entry_refiner import wait_for_entry

# ── 로그 설정 ─────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
log_file = f"logs/trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

STATE_FILE = "logs/trader_state.json"


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'peak': 0.0, 'trough': 0.0,
        'last_signal': None,
        'pending_signal': None,   # 5m 대기 중인 신호
        'entry_count': 0,
        'total_pnl': 0.0,
    }


def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def seconds_to_next_candle(interval_hours=4) -> int:
    now = datetime.now(timezone.utc)
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    interval_sec = interval_hours * 3600
    elapsed = seconds % interval_sec
    wait = interval_sec - elapsed + 10
    return wait


def run():
    mode = "테스트넷" if TESTNET else "실거래"
    log.info("=" * 60)
    log.info(f"BTC/USDT 자동매매 봇 v2 시작 [{mode}]")
    log.info(f"전략: EMA {EMA_FAST}/{EMA_SLOW}/{EMA_TREND} | trail={TRAIL_PCT:.0%} | base={BASE_PCT:.0%}")
    log.info(f"진입: 4h 신호 + 5m RSI 타이밍 정밀화")
    log.info("=" * 60)

    if not ping():
        log.error("바이낸스 연결 실패. API 키를 확인하세요.")
        return

    try:
        set_leverage(LEVERAGE)
        log.info(f"레버리지 {LEVERAGE}x 설정 완료")
    except Exception as e:
        log.warning(f"레버리지 설정 오류 (무시): {e}")

    state = load_state()
    log.info(f"상태 로드: {state}")

    while True:
        try:
            balance = get_balance()
            price   = get_price()
            pos     = get_position()
            fg      = get_fear_greed()
            df      = get_klines(interval=TIMEFRAME, limit=300)

            log.info(f"{'─'*55}")
            log.info(f"잔고={balance:,.2f}U | BTC=${price:,.0f} | FG={fg} | 거래={state['entry_count']}회")

            # ── 포지션 보유 중: 청산 조건 확인 ──────────────
            if pos:
                state['pending_signal'] = None  # 포지션 있으면 대기 신호 초기화

                if pos['side'] == 'long' and price > state['peak']:
                    state['peak'] = price
                    save_state(state)
                elif pos['side'] == 'short' and (state['trough'] == 0 or price < state['trough']):
                    state['trough'] = price
                    save_state(state)

                exit_check = check_exit(df, {
                    'side':    pos['side'],
                    'entry':   pos['entry'],
                    'peak':    state['peak'],
                    'trough':  state['trough'],
                })

                log.info(f"포지션: {pos['side'].upper()} {pos['size']}BTC @ ${pos['entry']:,.0f} | PnL={pos['pnl']:+.2f}U")
                log.info(f"청산 확인: {exit_check['reason']}")

                if exit_check['exit']:
                    log.info(f">>> 청산 실행: {exit_check['reason']}")
                    result = close_position()
                    log.info(f"청산 완료: {result}")

                    # PnL 누적
                    state['total_pnl'] += pos['pnl']
                    state['peak']         = 0.0
                    state['trough']       = 0.0
                    state['last_signal']  = None
                    save_state(state)
                    log.info(f"누적 PnL: {state['total_pnl']:+.2f}U")

            # ── 포지션 없음: 신호 확인 → 5m 타이밍 진입 ────
            else:
                signal = compute_signal(df, fg)
                log.info(f"4h 신호: {signal['signal'].upper()} | {signal['reason']}")

                if signal['signal'] in ('long', 'short'):
                    log.info(f">>> 4h 신호 발생! 5m 타이밍 대기 시작...")

                    # 5m 최적 진입 타이밍 대기
                    entry_info = wait_for_entry(signal['signal'])
                    log.info(f">>> 진입 조건 충족: {entry_info['reason']} | 대기={entry_info['waited_sec']//60}분")

                    # 주문 실행
                    current_balance = get_balance()
                    current_price   = get_price()
                    usdt_amount     = current_balance * signal['pos_pct']
                    qty             = calc_quantity(usdt_amount, current_price)

                    if qty >= 0.001:
                        side = 'BUY' if signal['signal'] == 'long' else 'SELL'
                        log.info(f">>> 주문: {side} {qty}BTC (${usdt_amount:,.0f} | {signal['pos_pct']:.0%})")
                        result = place_order(side, qty)
                        log.info(f"주문 완료: {result}")

                        state['peak']         = current_price if signal['signal'] == 'long' else 0.0
                        state['trough']       = current_price if signal['signal'] == 'short' else 0.0
                        state['last_signal']  = signal['signal']
                        state['entry_count'] += 1
                        save_state(state)
                    else:
                        log.warning(f"수량 부족: {qty}BTC (잔고={current_balance:.2f}U)")

            # ── 다음 4h 봉까지 대기 ───────────────────────────
            wait = seconds_to_next_candle(4)
            h, m = divmod(wait // 60, 60)
            log.info(f"다음 4h 확인까지 {h}시간 {m}분 대기...")
            time.sleep(wait)

        except KeyboardInterrupt:
            log.info("사용자 중단. 봇 종료.")
            break
        except Exception as e:
            log.error(f"오류 발생: {e}", exc_info=True)
            log.info("60초 후 재시도...")
            time.sleep(60)


if __name__ == "__main__":
    run()
