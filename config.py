# ── API 키 (테스트넷) ────────────────────────────────────────
# testnet.binancefuture.com 에서 발급
API_KEY    = "t4UsjRHMnRIWuLeiB7nnovL7mxc1GXXVKwGFaPniURi3eoxFQw7tAXAEGjjqMwsT"
API_SECRET = "bM2GmEzGQ465kpXy1rK7jftrm8U3A8aHyTxlYCoe5R16InDcxJdEbN7omPrBTIQ7"

# ── 거래 설정 ────────────────────────────────────────────────
SYMBOL     = "BTCUSDT"
TIMEFRAME  = "4h"
LEVERAGE   = 1       # 레버리지 (1 = 무레버리지, 실자본 투입 전 1로 유지)
TESTNET    = True    # True=테스트넷, False=실거래

# ── 전략 파라미터 (백테스트 최적값) ──────────────────────────
EMA_FAST   = 25
EMA_SLOW   = 50
EMA_TREND  = 200
TRAIL_PCT  = 0.05    # 트레일링 스탑 5%
BASE_PCT   = 0.60    # 기본 포지션 비율 60%
GAP_MIN    = 0.001   # EMA 간격 최소 0.1%
VOL_SCALE  = 0.5     # 변동성 역사이징 스케일
FG_UPPER   = 70      # 공포탐욕 > 70 → 롱 금지
FG_LOWER   = 20      # 공포탐욕 < 20 → 숏 금지

# ── 경로 설정 ────────────────────────────────────────────────
DATA_DIR   = "data"
LOG_DIR    = "logs"
