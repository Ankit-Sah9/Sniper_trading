from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from dashboard import write_html_dashboard
try:
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    from datetime import timezone as tz
    NY_TZ = tz(timedelta(hours=-4), "ET")

from FVGs import detect_fvgs, fvg_status
from FVGs import FVG as DetectedFVG
from history import METADATA_PATH, history_path, load_env, trading_symbol


PROJECT_ROOT   = Path(__file__).resolve().parent
OUTPUT_DIR     = PROJECT_ROOT / "data" / "backtest"
TRADE_LOG_PATH = OUTPUT_DIR / "trade_log.csv"
FOLD_LOG_PATH  = OUTPUT_DIR / "walk_forward_summary.csv"

# ── Strategy constants ────────────────────────────────────────────────────────
XAUUSD_PIP_SIZE      = 1.0
XAUUSD_PIP_VALUE_LOT = 1.0

VALID_WEEKDAYS = {1, 2, 3}
SESSION_START  = (2, 0)
SESSION_END    = (12, 0)

SL_PCT          = 0.01
MIN_RR          = 2.0
MAX_TRADES_WEEK = 3
MAX_WEEKLY_LOSS = 0.03
MAX_MONTHLY_DD  = 0.10
CPI_BUFFER_MINS = 10

TIMEFRAME_DURATIONS = {
    "W1": timedelta(days=7),
    "D1": timedelta(days=1),
    "H4": timedelta(hours=4),
    "H1": timedelta(hours=1),
}

TRADE_FIELDS = [
    "trade_id", "instrument", "fold",
    "entry_time_utc", "entry_time_ny",
    "exit_time_utc", "exit_time_ny",
    "day_of_week", "holding_hours",
    "direction",
    "entry_price", "sl_price", "tp_price", "close_price",
    "sl_pips", "rr_ratio",
    "trade_result", "r_gained", "pnl",
    "setup_quality", "confluence_score",
    "ttps_htf", "ttps_entry", "ttps_session", "ttps_risk", "ttps_total",
    "primary_fvg_tf", "primary_fvg_bottom", "primary_fvg_top",
    "swept_level", "risk_amount", "risk_multiplier",
    "week_number", "block_reason",
    # New bias engine columns
    "bias_trend_score", "bias_swing_score", "bias_zone_score",
    "bias_po3_score", "bias_bull_total", "bias_bear_total",
    "bias_conviction",
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candle:
    time_str:  str
    opened_at: datetime
    opened_ny: datetime
    open:  float
    high:  float
    low:   float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class SwingPoint:
    time_str: str
    price:    float
    kind:     str


@dataclass(frozen=True)
class BiasResult:
    """
    Output of the 4-variable bias engine.
    Stores scores for each variable and the final bias + conviction.
    """
    # Variable 1 — Market Trend
    trend_bull: int
    trend_bear: int

    # Variable 2 — Swing Point Respect
    swing_bull: int
    swing_bear: int

    # Variable 3 — Premium / Discount Zone
    zone_bull: int
    zone_bear: int

    # Variable 4 — Power of 3 (weekly)
    po3_bull: int
    po3_bear: int

    # Final totals
    bull_total: int
    bear_total: int
    final_bias: str       # "BULLISH", "BEARISH", "NO_TRADE"
    conviction: str       # "HIGH", "MEDIUM", "LOW", "NONE"


@dataclass(frozen=True)
class Signal:
    direction:       str
    entry_time:      datetime
    entry_price:     float
    sl_price:        float
    tp_price:        float
    sl_pips:         float
    rr_ratio:        float
    setup_quality:   str
    confluence_score: int
    ttps_htf:        int
    ttps_entry:      int
    ttps_session:    int
    ttps_risk:       int
    ttps_total:      int
    primary_fvg_tf:  str
    primary_fvg_bottom: float
    primary_fvg_top:    float
    swept_level:     float | None
    block_reason:    str
    bias:            BiasResult | None = None


@dataclass(frozen=True)
class Trade:
    trade_id:        str
    instrument:      str
    fold:            str
    entry_time_utc:  datetime
    exit_time_utc:   datetime | None
    direction:       str
    entry_price:     float
    sl_price:        float
    tp_price:        float
    close_price:     float | None
    sl_pips:         float
    rr_ratio:        float
    result:          str
    r_gained:        float
    pnl:             float
    setup_quality:   str
    confluence_score: int
    ttps_htf:        int
    ttps_entry:      int
    ttps_session:    int
    ttps_risk:       int
    ttps_total:      int
    primary_fvg_tf:  str
    primary_fvg_bottom: float
    primary_fvg_top:    float
    swept_level:     float | None
    risk_amount:     float
    risk_multiplier: float
    block_reason:    str
    bias:            BiasResult | None = None


@dataclass
class DayState:
    date:          object
    trades_today:  int = 0
    last_result:   str = ""
    wins_today:    int = 0
    losses_today:  int = 0

    def may_trade(self) -> tuple[bool, str]:
        if self.trades_today == 0:
            return True, ""
        if self.last_result == "LOSS":
            return False, "Daily block: last trade was a loss"
        return True, ""

    def record(self, result: str) -> None:
        self.trades_today += 1
        self.last_result   = result
        if result == "WIN":
            self.wins_today += 1
        elif result == "LOSS":
            self.losses_today += 1


@dataclass
class WeekState:
    trades_used:   int   = 0
    days_traded:   set   = None
    weekly_pnl:    float = 0.0
    start_balance: float = 0.0
    kill_active:   bool  = False
    weekly_wins:   int   = 0
    weekly_losses: int   = 0
    current_day:   DayState | None = None

    def __post_init__(self):
        if self.days_traded is None:
            self.days_traded = set()

    def get_day(self, trade_date) -> DayState:
        if self.current_day is None or self.current_day.date != trade_date:
            self.current_day = DayState(date=trade_date)
        return self.current_day

    def risk_multiplier(self) -> float:
        closed = self.weekly_wins + self.weekly_losses
        if closed == 0:
            return 1.0
        if self.weekly_wins >= 1:
            return 0.5
        if self.weekly_losses >= 2:
            return 0.5
        return 1.0


@dataclass
class BacktestState:
    account_balance:     float
    month_start_balance: float
    monthly_kill:        bool = False
    week:                WeekState = None

    def __post_init__(self):
        if self.week is None:
            self.week = WeekState(start_balance=self.account_balance)


@dataclass(frozen=True)
class Metrics:
    trades:          int
    closed:          int
    wins:            int
    losses:          int
    win_rate:        float
    profit_factor:   float
    total_r:         float
    average_r:       float
    max_drawdown_pct: float
    sharpe:          float
    final_balance:   float


# ── Time helpers ──────────────────────────────────────────────────────────────

def utc_to_ny(dt: datetime) -> datetime:
    return dt.astimezone(NY_TZ)


def is_valid_trading_time(
    candle: Candle,
    entry_candle: Candle | None = None,
) -> tuple[bool, str]:
    ny = candle.opened_ny
    if ny.weekday() not in VALID_WEEKDAYS:
        return False, f"Invalid day: {ny.strftime('%A')}"
    t = (ny.hour, ny.minute)
    if t < SESSION_START:
        return False, f"Before session: {ny.strftime('%H:%M')} NY"
    if t >= (11, 0):
        return False, f"Signal too late: {ny.strftime('%H:%M')} NY"
    if entry_candle is not None:
        eny = entry_candle.opened_ny
        if (eny.hour, eny.minute) >= (12, 0):
            return False, f"Entry candle outside window: {eny.strftime('%H:%M')} NY"
    return True, ""


def week_key(dt: datetime) -> str:
    ny = utc_to_ny(dt)
    iso = ny.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def month_key(dt: datetime) -> str:
    ny = utc_to_ny(dt)
    return f"{ny.year}-{ny.month:02d}"


def format_ny(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return utc_to_ny(dt).strftime("%Y-%m-%d %H:%M:%S")


def format_utc(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── CSV loading ───────────────────────────────────────────────────────────────

def parse_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def load_history(timeframe: str) -> list[Candle]:
    path = history_path(timeframe)
    if not path.exists():
        return []
    candles: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                raw = str(row.get("time", "")).strip()
                opened_utc = parse_time(raw)
                opened_ny  = utc_to_ny(opened_utc)
                candles.append(Candle(
                    time_str=raw,
                    opened_at=opened_utc,
                    opened_ny=opened_ny,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0.0),
                ))
            except (KeyError, ValueError):
                continue
    return sorted(candles, key=lambda c: c.opened_at)


def closed_as_of(
    candles: list[Candle],
    timeframe: str,
    as_of: datetime,
) -> list[Candle]:
    dur = TIMEFRAME_DURATIONS[timeframe]
    return [c for c in candles if c.opened_at + dur <= as_of]


# ── News filter ───────────────────────────────────────────────────────────────

NEWS_FULL_DAY    = {"NFP", "NONFARM", "NON-FARM", "FOMC", "FED RATE",
                    "RATE DECISION", "INTEREST RATE", "MONETARY POLICY"}
NEWS_PRE_RELEASE = {"CPI", "CORE CPI", "PCE", "GDP", "RETAIL SALES"}


@dataclass(frozen=True)
class NewsEvent:
    name:         str
    event_type:   str
    release_time: datetime


def classify_news(name: str) -> str:
    u = name.upper()
    if any(k in u for k in NEWS_FULL_DAY):    return "FULL_DAY"
    if any(k in u for k in NEWS_PRE_RELEASE): return "PRE_RELEASE"
    return "IGNORE"


def load_news(news_csv: Path) -> list[NewsEvent]:
    events: list[NewsEvent] = []
    if not news_csv.exists():
        return events
    with news_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name  = str(row.get("event", "")).strip()
            raw   = str(row.get("time_ny", "")).strip()
            etype = classify_news(name)
            if etype == "IGNORE":
                continue
            try:
                ny_naive = datetime.fromisoformat(raw)
                ny_aware = ny_naive.replace(tzinfo=NY_TZ)
                events.append(NewsEvent(
                    name=name,
                    event_type=etype,
                    release_time=ny_aware.astimezone(timezone.utc),
                ))
            except ValueError:
                continue
    return events


def get_cpi_this_week(
    candle_time: datetime,
    events: list[NewsEvent],
) -> datetime | None:
    days_since_mon = candle_time.weekday()
    week_start = (candle_time - timedelta(days=days_since_mon)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)
    for e in events:
        if e.event_type == "PRE_RELEASE" and "CPI" in e.name.upper():
            if week_start <= e.release_time < week_end:
                return e.release_time
    return None


def is_news_blocked(
    candle_time: datetime,
    events: list[NewsEvent],
    cpi_this_week: datetime | None,
) -> tuple[bool, str]:
    date = candle_time.date()
    todays    = [e for e in events if e.event_type == "FULL_DAY"
                 and e.release_time.date() == date]
    yesterdays = [e for e in events if e.event_type == "FULL_DAY"
                  and (date - e.release_time.date()).days == 1]
    names_today     = [e.name.upper() for e in todays]
    names_yesterday = [e.name.upper() for e in yesterdays]

    if any("NFP" in n or "NONFARM" in n or "NON-FARM" in n for n in names_today):
        return True, "NFP day"

    fomc_today = any("FOMC" in n for n in names_today)
    rate_today = any("RATE" in n or "INTEREST" in n for n in names_today)
    if fomc_today and rate_today:
        return True, "FOMC + Rate Decision"

    fomc_yst = any("FOMC" in n for n in names_yesterday)
    rate_yst = any("RATE" in n or "INTEREST" in n for n in names_yesterday)
    if fomc_yst and rate_yst:
        return True, "Day after FOMC + Rate Decision"

    if cpi_this_week is not None:
        clear = cpi_this_week + timedelta(minutes=CPI_BUFFER_MINS)
        if candle_time < clear:
            return True, f"CPI block — clears {format_ny(clear)} NY"

    return False, ""


# ── Technical helpers ─────────────────────────────────────────────────────────

def detect_swings(
    candles: list[Candle],
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    highs: list[SwingPoint] = []
    lows:  list[SwingPoint] = []
    for i in range(1, len(candles) - 1):
        p, c, n = candles[i-1], candles[i], candles[i+1]
        if c.high > p.high and c.high > n.high:
            highs.append(SwingPoint(c.time_str, c.high, "HIGH"))
        if c.low < p.low and c.low < n.low:
            lows.append(SwingPoint(c.time_str, c.low, "LOW"))
    return highs, lows


def average_true_range(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = [
        max(c.high - c.low,
            abs(c.high - p.close),
            abs(c.low  - p.close))
        for p, c in zip(candles[-(period+1):-1], candles[-period:])
    ]
    return sum(trs) / len(trs) if trs else 0.0


# ── NEW BIAS ENGINE (4-variable scoring system) ───────────────────────────────

def _three_months_candles(candles: list[Candle]) -> list[Candle]:
    """Return candles from the last 3 months (approx 90 days)."""
    cutoff = candles[-1].opened_at - timedelta(days=90)
    result = [c for c in candles if c.opened_at >= cutoff]
    return result if result else candles


def _three_weeks_candles(candles: list[Candle]) -> list[Candle]:
    """Return candles from the last 3 weeks (approx 21 days)."""
    cutoff = candles[-1].opened_at - timedelta(days=21)
    result = [c for c in candles if c.opened_at >= cutoff]
    return result if result else candles


# ── Variable 1: Market Trend ──────────────────────────────────────────────────

def _trend_score(candles: list[Candle]) -> tuple[int, int]:
    """
    Score trend direction using last 3 months of data.
    Looks for HH+HL sequence (bullish) or LH+LL sequence (bearish).
    Uses the last 40 candles within that window for swing detection
    to keep it sensitive enough for D1 and H4.

    Returns (bull_score, bear_score) — max 1 point per timeframe.
    """
    window = _three_months_candles(candles)
    highs, lows = detect_swings(window[-40:])
    if len(highs) < 2 or len(lows) < 2:
        return 0, 0
    rh, rl = highs[-2:], lows[-2:]
    hh = rh[1].price > rh[0].price
    hl = rl[1].price > rl[0].price
    lh = rh[1].price < rh[0].price
    ll = rl[1].price < rl[0].price
    if hh and hl: return 1, 0
    if lh and ll: return 0, 1
    return 0, 0


def variable1_trend(
    d1: list[Candle],
    h4: list[Candle],
) -> tuple[int, int]:
    """
    Variable 1 — Market Trend. Weight: 3 points total.

    D1 contributes up to 2 points (stronger weight).
    H4 contributes up to 1 point (supporting weight).
    Both use last 3 months of data.

    Score table:
      D1 bullish trend    → bull +2
      D1 bearish trend    → bear +2
      H4 bullish trend    → bull +1
      H4 bearish trend    → bear +1
    """
    d1_bull, d1_bear = _trend_score(d1)
    h4_bull, h4_bear = _trend_score(h4)

    bull = d1_bull * 2 + h4_bull * 1
    bear = d1_bear * 2 + h4_bear * 1
    return bull, bear


# ── Variable 2: Swing Point Respect ──────────────────────────────────────────

def _swing_respect_pct(
    candles: list[Candle],
    bias: str,
) -> float:
    """
    Check what percentage of recent swing points are being respected
    (not violated) using the last 3 weeks of data.

    For BULLISH bias: check swing lows — are they holding (price hasn't
    closed below them)?
    For BEARISH bias: check swing highs — are they holding (price hasn't
    closed above them)?

    A swing point is VIOLATED when a candle closes beyond it.
    Returns the percentage of swing points that are intact (0.0 to 1.0).
    """
    window = _three_weeks_candles(candles)
    if len(window) < 5:
        return 0.0

    highs, lows = detect_swings(window)

    if bias == "BULLISH":
        pivots = lows[-6:] if lows else []   # last 6 swing lows
        if not pivots:
            return 0.0
        latest_close = candles[-1].close
        # Respected = close has not gone below this swing low
        respected = sum(1 for p in pivots if latest_close > p.price)
        return respected / len(pivots)

    else:  # BEARISH
        pivots = highs[-6:] if highs else []  # last 6 swing highs
        if not pivots:
            return 0.0
        latest_close = candles[-1].close
        # Respected = close has not gone above this swing high
        respected = sum(1 for p in pivots if latest_close < p.price)
        return respected / len(pivots)


def variable2_swing_respect(
    d1: list[Candle],
    h4: list[Candle],
    current_bias_hint: str = "BULLISH",
) -> tuple[int, int]:
    """
    Variable 2 — Swing Point Respect. Weight: 3 points total.

    Checks both D1 and H4 swing points using last 3 weeks.
    Equal weight between D1 and H4 (1.5 points each, rounded).

    Threshold: 60-70% respected = confirms the bias.
    Score table (per timeframe, per direction):
      70%+ respected     → +2 (strong confirmation)
      60-70% respected   → +1 (moderate confirmation)
      below 60%          → 0
    """
    bull_score = 0
    bear_score = 0

    for candles in (d1, h4):
        bull_pct = _swing_respect_pct(candles, "BULLISH")
        bear_pct = _swing_respect_pct(candles, "BEARISH")

        # Bullish swing respect (swing lows holding)
        if bull_pct >= 0.70:
            bull_score += 2
        elif bull_pct >= 0.60:
            bull_score += 1

        # Bearish swing respect (swing highs holding)
        if bear_pct >= 0.70:
            bear_score += 2
        elif bear_pct >= 0.60:
            bear_score += 1

    # Cap at 3 per side (two timeframes each contributing up to 2 = 4 max,
    # but we cap at 3 to keep weights balanced with Variable 1)
    return min(bull_score, 3), min(bear_score, 3)


# ── Variable 3: Premium / Discount Zone ──────────────────────────────────────

def variable3_premium_discount(
    d1: list[Candle],
    h4: list[Candle],
    current_price: float,
) -> tuple[int, int]:
    """
    Variable 3 — Premium/Discount Zone. Weight: 2 points total.

    Uses 3-month high/low range to calculate the 50% Fibonacci level.
    Checks both D1 and H4 independently.

    Score table (per timeframe):
      Price at or below 50% (ideal discount)   → bull +1
      Price between 50-60% (valid discount)    → bull +0  (neutral zone)
      Price at or above 50% (ideal premium)    → bear +1
      Price between 40-50% (valid premium)     → bear +0  (neutral zone)

    Total max = 2 bull or 2 bear (one point per timeframe).
    """
    bull_score = 0
    bear_score = 0

    for candles in (d1, h4):
        window = _three_months_candles(candles)
        if not window:
            continue
        range_high = max(c.high for c in window)
        range_low  = min(c.low  for c in window)
        span = range_high - range_low
        if span <= 0:
            continue

        # Fibonacci level as percentage from low
        pct_from_low = (current_price - range_low) / span

        # Below 50% = discount (ideal buy zone)
        if pct_from_low <= 0.50:
            bull_score += 1
        # Above 50% = premium (ideal sell zone)
        elif pct_from_low >= 0.50:
            bear_score += 1
        # Exactly at 50% = neutral, no score for either

    return bull_score, bear_score


# ── Variable 4: Power of 3 — Weekly ──────────────────────────────────────────

def variable4_power_of_3(
    d1: list[Candle],
    w1: list[Candle],
    current_price: float,
    as_of: datetime,
) -> tuple[int, int]:
    """
    Variable 4 — Power of 3 Weekly. Weight: 1 point (with -1 penalty).

    Checks whether the current week's price action has taken out last
    week's high or low first. Dynamically updated as the week progresses.

    What "taken out" means: a candle CLOSED beyond the level.

    For BULLISH context:
      If price FIRST swept the weekly LOW before taking the high
      → bullish favour (+1 bull)
      If price FIRST swept the weekly HIGH before taking the low
      → decreases bullish probability (-1 penalty on bull)

    For BEARISH context: vice versa.

    We look at last week's high and low from D1/W1 candles.
    Then check current week's D1 candles to see which was taken first.
    """
    # Get last week's high and low from W1
    if len(w1) < 2:
        return 0, 0

    last_week_candle = w1[-2]   # previous completed week
    last_week_high   = last_week_candle.high
    last_week_low    = last_week_candle.low

    # Find start of current week (most recent Monday)
    now_ny = utc_to_ny(as_of)
    days_since_mon = now_ny.weekday()
    week_start_ny = (now_ny - timedelta(days=days_since_mon)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    week_start_utc = week_start_ny.astimezone(timezone.utc)

    # Get current week's D1 candles (candles that opened after week start)
    current_week_d1 = [c for c in d1 if c.opened_at >= week_start_utc]
    if not current_week_d1:
        return 0, 0

    # Walk through current week's candles in order
    # Check which level was taken out first (close beyond level = taken out)
    high_taken_first = False
    low_taken_first  = False
    for c in current_week_d1:
        if not high_taken_first and not low_taken_first:
            if c.close > last_week_high:
                high_taken_first = True
                break
            if c.close < last_week_low:
                low_taken_first = True
                break

    # Score:
    # Low taken first → bullish favour (liquidity below swept, reversal up)
    # High taken first → bearish favour (liquidity above swept, reversal down)
    # In bullish context, high taken first DECREASES probability → penalty
    # In bearish context, low taken first DECREASES probability → penalty

    if low_taken_first:
        return 1, 0    # bullish favour
    elif high_taken_first:
        return 0, 1    # bearish favour
    else:
        return 0, 0    # neither taken out yet — neutral


# ── Bias Engine: combine all 4 variables ─────────────────────────────────────

def calculate_bias(
    d1: list[Candle],
    h4: list[Candle],
    w1: list[Candle],
    current_price: float,
    as_of: datetime,
) -> BiasResult:
    """
    4-variable bias scoring system.

    Weights:
      Variable 1 — Trend:          max 3 points (D1=2, H4=1)
      Variable 2 — Swing respect:  max 3 points (D1+H4 combined, capped at 3)
      Variable 3 — Premium/Disc:   max 2 points (D1+H4, 1 each)
      Variable 4 — Power of 3:     max 1 point  (with -1 penalty possible)

    Maximum possible: 9 bull or 9 bear.

    Final bias decision:
      Bull ≥ 7 and beats bear by 2+              → BULLISH HIGH
      Bull 5-6 and beats bear by 2+              → BULLISH MEDIUM
      Bear ≥ 7 and beats bear by 2+              → BEARISH HIGH
      Bear 5-6 and beats bear by 2+              → BEARISH MEDIUM
      Scores within 2 of each other              → NO_TRADE (too close)
      Both below 5                               → NO_TRADE (no conviction)
    """
    # Variable 1
    v1_bull, v1_bear = variable1_trend(d1, h4)

    # Variable 2
    v2_bull, v2_bear = variable2_swing_respect(d1, h4)

    # Variable 3
    v3_bull, v3_bear = variable3_premium_discount(d1, h4, current_price)

    # Variable 4
    v4_bull, v4_bear = variable4_power_of_3(d1, w1, current_price, as_of)

    # Apply Power of 3 penalty
    # If price took out last week HIGH first (bearish signal) in a bullish context
    # → subtract 1 from bull total (reduce probability as per notes)
    bull_total = v1_bull + v2_bull + v3_bull + v4_bull
    bear_total = v1_bear + v2_bear + v3_bear + v4_bear

    # Apply cross-penalty: PO3 against the direction reduces that direction's score
    if v4_bear > 0:   # high taken first — penalise bulls
        bull_total = max(0, bull_total - 1)
    if v4_bull > 0:   # low taken first — penalise bears
        bear_total = max(0, bear_total - 1)

    # Determine final bias
    gap = abs(bull_total - bear_total)

    final_bias = "NO_TRADE"
    conviction = "NONE"

    if bull_total > bear_total and gap >= 2:
        final_bias = "BULLISH"
        conviction = "HIGH" if bull_total >= 7 else "MEDIUM" if bull_total >= 5 else "LOW"
    elif bear_total > bull_total and gap >= 2:
        final_bias = "BEARISH"
        conviction = "HIGH" if bear_total >= 7 else "MEDIUM" if bear_total >= 5 else "LOW"
    # else: too close → NO_TRADE

    return BiasResult(
        trend_bull=v1_bull,  trend_bear=v1_bear,
        swing_bull=v2_bull,  swing_bear=v2_bear,
        zone_bull=v3_bull,   zone_bear=v3_bear,
        po3_bull=v4_bull,    po3_bear=v4_bear,
        bull_total=bull_total,
        bear_total=bear_total,
        final_bias=final_bias,
        conviction=conviction,
    )


# ── FVG helpers ───────────────────────────────────────────────────────────────

def as_fvg_candles(candles: list[Candle]) -> list:
    result = []
    for c in candles:
        obj = type('C', (), {
            'time': c.time_str, 'open': c.open,
            'high': c.high, 'low': c.low, 'close': c.close,
            'body_size': abs(c.close - c.open),
            'range_size': c.high - c.low,
        })()
        result.append(obj)
    return result


def as_fvg_candles_simple(candles: list[Candle]) -> list:
    return as_fvg_candles(candles)


def build_fvg_cache(
    history: dict[str, list[Candle]],
) -> dict[str, list[DetectedFVG]]:
    return {
        tf: detect_fvgs(as_fvg_candles(history[tf]), tf)
        for tf in ("W1", "D1", "H4", "H1")
    }


def fvg_still_valid(
    fvg: DetectedFVG,
    candles: list[Candle],
    timeframe: str,
    as_of: datetime,
) -> bool:
    formed_at = parse_time(fvg.formed_time)
    if formed_at + TIMEFRAME_DURATIONS[timeframe] > as_of:
        return False
    for c in candles:
        if c.opened_at <= formed_at:
            continue
        if c.opened_at + TIMEFRAME_DURATIONS[timeframe] > as_of:
            continue
        if fvg.direction == "BULLISH" and c.close < fvg.bottom - 5.0:
            return False
        if fvg.direction == "BEARISH" and c.close > fvg.top + 5.0:
            return False
    return True


def choose_htf_area(
    history: dict[str, list[Candle]],
    fvg_cache: dict[str, list[DetectedFVG]],
    as_of: datetime,
    bias: str,
) -> DetectedFVG | None:
    candidates = []
    for tf, weight in (("W1", 4), ("D1", 3), ("H4", 2)):
        for fvg in fvg_cache.get(tf, []):
            if fvg.direction != bias:
                continue
            formed = parse_time(fvg.formed_time)
            if formed < as_of - timedelta(days=180):
                continue
            if fvg_still_valid(fvg, history[tf], tf, as_of):
                candidates.append((weight, formed, fvg))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)[0][2]


def recent_price_leg(
    candles: list[Candle],
    lookback: int = 20,
) -> tuple[float, float, float]:
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    high = max(c.high for c in recent)
    low  = min(c.low  for c in recent)
    mid  = (high + low) / 2
    return high, low, mid


def fvg_in_optimal_zone(
    fvg: DetectedFVG,
    bias: str,
    leg_high: float,
    leg_low: float,
    leg_mid: float,
) -> bool:
    fvg_center = (fvg.top + fvg.bottom) / 2
    if bias == "BULLISH":
        return fvg_center <= leg_mid
    return fvg_center >= leg_mid


def find_entry_fvg(
    history: dict[str, list[Candle]],
    fvg_cache: dict[str, list[DetectedFVG]],
    as_of: datetime,
    bias: str,
    current_price: float,
    htf_area: DetectedFVG | None,
    h1: list[Candle],
    h4: list[Candle],
) -> DetectedFVG | None:
    leg_high, leg_low, leg_mid = recent_price_leg(h4[-20:] if h4 else h1[-80:])
    entry_candidates = []

    for tf, weight in (("H4", 3), ("H1", 2)):
        for fvg in fvg_cache.get(tf, []):
            if fvg.direction != bias:
                continue
            formed = parse_time(fvg.formed_time)
            if formed < as_of - timedelta(days=14):
                continue
            if not fvg_still_valid(fvg, history[tf], tf, as_of):
                continue
            if not fvg.overlaps_price(current_price, tolerance=10.0):
                continue

            zone_bonus = 1 if fvg_in_optimal_zone(fvg, bias, leg_high, leg_low, leg_mid) else 0

            htf_bonus = 0
            if htf_area is not None:
                overlap = (
                    max(fvg.bottom, htf_area.bottom) 
                    -min(fvg.top, htf_area.top)
                )
                if overlap:
                    htf_bonus = 2

            days_old   = (as_of - formed).days
            week_bonus = 1 if days_old <= 7 else 0

            entry_candidates.append(
                (weight + htf_bonus + zone_bonus + week_bonus, formed, fvg)
            )

    if not entry_candidates:
        return None
    return sorted(entry_candidates, key=lambda x: (x[0], x[1]), reverse=True)[0][2]


# ── Sweep detection ───────────────────────────────────────────────────────────

def detect_liquidity_sweep(
    h1: list[Candle],
    bias: str,
    d1: list[Candle] | None = None,
) -> tuple[bool, float | None]:
    if len(h1) < 5:
        return False, None

    latest    = h1[-1]
    tolerance = 2.0
    levels: list[float] = []

    if len(h1) >= 25:
        prev_day = h1[-25:-1]
        levels.append(min(c.low  for c in prev_day))
        levels.append(max(c.high for c in prev_day))

    if len(h1) >= 121:
        prev_week = h1[-121:-1]
        levels.append(min(c.low  for c in prev_week))
        levels.append(max(c.high for c in prev_week))

    all_highs, all_lows = detect_swings(h1[-120:])
    eq_tol = 0.0015

    for i in range(len(all_highs)):
        for j in range(i + 1, len(all_highs)):
            a, b = all_highs[i].price, all_highs[j].price
            avg  = (a + b) / 2
            if avg > 0 and abs(a - b) / avg <= eq_tol:
                levels.append(avg)

    for i in range(len(all_lows)):
        for j in range(i + 1, len(all_lows)):
            a, b = all_lows[i].price, all_lows[j].price
            avg  = (a + b) / 2
            if avg > 0 and abs(a - b) / avg <= eq_tol:
                levels.append(avg)

    if bias == "BULLISH":
        ssl = [l for l in levels if l < latest.close]
        for level in sorted(ssl, reverse=True):
            if latest.low <= (level + tolerance) and latest.close > level:
                return True, level
    else:
        bsl = [l for l in levels if l > latest.close]
        for level in sorted(bsl):
            if latest.high >= (level - tolerance) and latest.close < level:
                return True, level

    return False, None


# ── SL / TP ───────────────────────────────────────────────────────────────────

def find_sl_price(
    bias: str,
    entry: float,
    h1: list[Candle],
    entry_fvg: DetectedFVG | None = None,
) -> tuple[float | None, float]:
    sl_distance = round(entry * SL_PCT, 2)
    if bias == "BULLISH":
        sl = round(entry - sl_distance, 2)
    else:
        sl = round(entry + sl_distance, 2)
    return sl, sl_distance


def find_tp_price(
    bias: str,
    entry: float,
    sl: float,
    d1: list[Candle],
    h4: list[Candle],
    h1: list[Candle],
) -> tuple[float | None, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        return None, 0.0

    targets: list[tuple[float, int]] = []

    d1_highs, d1_lows = detect_swings(d1[-40:])
    h4_highs, h4_lows = detect_swings(h4[-80:])
    h1_highs, h1_lows = detect_swings(h1[-120:])

    all_highs = d1_highs + h4_highs + h1_highs
    all_lows  = d1_lows  + h4_lows  + h1_lows
    tol_pct   = 0.0015

    if bias == "BULLISH":
        for i in range(len(all_highs)):
            for j in range(i + 1, len(all_highs)):
                a, b = all_highs[i].price, all_highs[j].price
                avg  = (a + b) / 2
                if avg <= entry: continue
                if abs(a - b) / avg <= tol_pct:
                    targets.append((avg, 1))
    else:
        for i in range(len(all_lows)):
            for j in range(i + 1, len(all_lows)):
                a, b = all_lows[i].price, all_lows[j].price
                avg  = (a + b) / 2
                if avg >= entry: continue
                if abs(a - b) / avg <= tol_pct:
                    targets.append((avg, 1))

    if bias == "BULLISH":
        targets += [(s.price, 2) for s in d1_highs if s.price > entry]
    else:
        targets += [(s.price, 2) for s in d1_lows  if s.price < entry]

    if bias == "BULLISH":
        targets += [(s.price, 3) for s in h4_highs if s.price > entry]
    else:
        targets += [(s.price, 3) for s in h4_lows  if s.price < entry]

    if len(d1) >= 6:
        prev_week = d1[-6:-1]
        if bias == "BULLISH":
            pw_high = max(c.high for c in prev_week)
            if pw_high > entry:
                targets.append((pw_high, 4))
        else:
            pw_low = min(c.low for c in prev_week)
            if pw_low < entry:
                targets.append((pw_low, 4))

    valid: list[tuple[int, float, float]] = []
    for price, priority in targets:
        rr = abs(price - entry) / risk
        if rr >= MIN_RR:
            valid.append((priority, abs(price - entry), price))

    if valid:
        valid.sort(key=lambda x: (x[0], -x[1]))
        priority, dist, tp = valid[0]
        rr = dist / risk
        # Cap at 5R to avoid unrealistic targets
        if rr > 5.0:
            tp = (entry + risk * 5.0) if bias == "BULLISH" else (entry - risk * 5.0)
            rr = 5.0
        return tp, rr

    min_tp = (entry + risk * MIN_RR) if bias == "BULLISH" else (entry - risk * MIN_RR)
    return min_tp, MIN_RR


# ── TTPS score ────────────────────────────────────────────────────────────────

def calculate_ttps(
    bias: str,
    swept: bool,
    primary_fvg: DetectedFVG | None,
    confluence_score: int,
    sl_pips: float,
    rr: float,
    candle: Candle,
    w1: list[Candle],
    d1: list[Candle],
    h4: list[Candle],
    h1: list[Candle],
    entry_price: float,
    bias_result: BiasResult | None = None,
) -> tuple[int, int, int, int, int, str]:
    """
    TTPS scoring — 4 dimensions, 5 points each = 20 total.

    HTF Bias dimension now uses the bias engine scores instead of
    re-running classify_trend, for consistency.
    """
    # Dim 1: HTF Bias — use bias engine if available
    htf = 0
    if bias_result is not None:
        # Scale bias engine scores into TTPS 0-5 range
        if bias == "BULLISH":
            htf = min(5, bias_result.bull_total)
        else:
            htf = min(5, bias_result.bear_total)
    else:
        # Fallback: simple swing check
        d1_highs, d1_lows = detect_swings(d1[-40:])
        h4_highs, h4_lows = detect_swings(h4[-40:])
        def simple_trend(highs, lows):
            if len(highs) < 2 or len(lows) < 2: return "NO_TRADE"
            hh = highs[-1].price > highs[-2].price
            hl = lows[-1].price  > lows[-2].price
            lh = highs[-1].price < highs[-2].price
            ll = lows[-1].price  < lows[-2].price
            if hh and hl: return "BULLISH"
            if lh and ll: return "BEARISH"
            return "NO_TRADE"
        d1_b = simple_trend(d1_highs, d1_lows)
        h4_b = simple_trend(h4_highs, h4_lows)
        if d1_b == bias: htf += 2
        if h4_b == bias: htf += 2
        if w1 and (bias == "BULLISH" and w1[-1].close > w1[-1].open) or \
                  (bias == "BEARISH" and w1[-1].close < w1[-1].open): htf += 1

    # Dim 2: Entry Confluence
    entry_score = 0
    if swept:         entry_score += 2
    if primary_fvg:   entry_score += 1
    if primary_fvg and primary_fvg.timeframe in {"H4", "D1", "W1"}:
        entry_score += 1

    # Dim 3: Session Quality
    hour = candle.opened_ny.hour
    session = 0
    if 2 <= hour < 5:     session = 2
    elif 8 <= hour <= 11: session = 3
    elif 5 <= hour < 8:   session = 1

    # Dim 4: Risk Quality
    risk_score = 0
    if rr >= 3.0:      risk_score += 3
    elif rr >= 2.5:    risk_score += 2
    elif rr >= MIN_RR: risk_score += 1
    sl_price_calc = entry_price - sl_pips if bias == "BULLISH" else entry_price + sl_pips
    _, lows_c  = detect_swings(h1[-50:])
    highs_c, _ = detect_swings(h1[-50:])
    sl_aligned = False
    if bias == "BULLISH":
        sl_aligned = any(abs(sw.price - sl_price_calc) <= 3.0 for sw in lows_c)
    else:
        sl_aligned = any(abs(sw.price - sl_price_calc) <= 3.0 for sw in highs_c)
    if sl_aligned:
        risk_score += 1

    total = htf + entry_score + session + risk_score

    if total >= 14:   grade = "A+"
    elif total >= 11: grade = "A"
    elif total >= 8:  grade = "B"
    elif total >= 5:  grade = "C"
    else:             grade = "WEAK"

    return htf, entry_score, session, risk_score, total, grade


# ── Open filter / weekly-daily open bonus ────────────────────────────────────

def weekly_daily_open_bonus(
    bias: str,
    entry_price: float,
    w1: list[Candle],
    d1: list[Candle],
) -> int:
    score = 0
    if w1:
        weekly_open = w1[-1].open
        if bias == "BULLISH" and entry_price < weekly_open:
            score += 1
        elif bias == "BEARISH" and entry_price > weekly_open:
            score += 1
    if d1:
        daily_open = d1[-1].open
        if bias == "BULLISH" and entry_price < daily_open:
            score += 1
        elif bias == "BEARISH" and entry_price > daily_open:
            score += 1
    return score


# ── Signal finder ─────────────────────────────────────────────────────────────

def find_signal(
    history: dict[str, list[Candle]],
    fvg_cache: dict[str, list[DetectedFVG]],
    h1_index: int,
    news_events: list[NewsEvent],
) -> Signal | None:
    h1_all = history["H1"]
    candle  = h1_all[h1_index]

    if h1_index + 1 >= len(h1_all):
        return None

    next_candle = h1_all[h1_index + 1]

    # ── Rule 1: Valid day and session ─────────────────────────────────────
    time_ok, _ = is_valid_trading_time(candle, next_candle)
    if not time_ok:
        return None

    as_of = candle.opened_at + TIMEFRAME_DURATIONS["H1"]

    w1 = closed_as_of(history["W1"], "W1", as_of)
    d1 = closed_as_of(history["D1"], "D1", as_of)
    h4 = closed_as_of(history["H4"], "H4", as_of)
    h1 = closed_as_of(h1_all[:h1_index + 1], "H1", as_of)

    if min(len(w1), len(d1), len(h4), len(h1)) < 20:
        return None

    # ── Rule 2: News filter ───────────────────────────────────────────────
    cpi_this_week = get_cpi_this_week(candle.opened_at, news_events)
    news_blocked, _ = is_news_blocked(candle.opened_at, news_events, cpi_this_week)
    if news_blocked:
        return None

    # ── Rule 3: Bias (new 4-variable engine) ──────────────────────────────
    current_price = candle.close
    bias_result   = calculate_bias(d1, h4, w1, current_price, as_of)
    bias          = bias_result.final_bias

    if bias not in {"BULLISH", "BEARISH"}:
        return None

    # Require at least MEDIUM conviction to trade
    if bias_result.conviction not in {"HIGH", "MEDIUM"}:
        return None

    # ── Layer 1: HTF area of interest ────────────────────────────────────
    htf_area = choose_htf_area(history, fvg_cache, as_of, bias)

    # ── Layer 2: H1/H4 entry FVG ─────────────────────────────────────────
    entry_price = next_candle.open

    entry_fvg = find_entry_fvg(
        history, fvg_cache, as_of, bias, current_price, htf_area, h1, h4
    )
    if entry_fvg is None:
        return None

    # ── Rule 4: SL placement ─────────────────────────────────────────────
    sl_price, sl_pips = find_sl_price(bias, entry_price, h1, entry_fvg)
    if sl_price is None:
        return None

    # ── Rule 5: Minimum 1:2 R:R ──────────────────────────────────────────
    tp_price, rr = find_tp_price(bias, entry_price, sl_price, d1, h4, h1)
    if tp_price is None or rr < MIN_RR:
        return None

    if bias == "BULLISH" and (tp_price <= entry_price or sl_price >= entry_price):
        return None
    if bias == "BEARISH" and (tp_price >= entry_price or sl_price <= entry_price):
        return None

    # ── Sweep detection ───────────────────────────────────────────────────
    swept, swept_level = detect_liquidity_sweep(h1, bias, d1)
    sweep_bonus = 2 if swept else 0

    # ── Confluence score ──────────────────────────────────────────────────
    open_score     = weekly_daily_open_bonus(bias, entry_price, w1, d1)
    entry_tf_wt    = 3 if entry_fvg.timeframe == "H4" else 2
    htf_bonus      = 2 if htf_area is not None else 0
    # Conviction bonus from bias engine
    conv_bonus     = 2 if bias_result.conviction == "HIGH" else 1
    score = open_score + entry_tf_wt + htf_bonus + sweep_bonus + conv_bonus

    setup_quality = (
        "A" if score >= 9 else
        "B" if score >= 7 else
        "C" if score >= 4 else "WEAK"
    )
    if setup_quality == "WEAK":
        return None

    # ── TTPS score ────────────────────────────────────────────────────────
    ttps_htf, ttps_entry, ttps_session, ttps_risk, ttps_total, ttps_grade = calculate_ttps(
        bias=bias, swept=swept, primary_fvg=entry_fvg,
        confluence_score=score, sl_pips=sl_pips, rr=rr,
        candle=candle, w1=w1, d1=d1, h4=h4, h1=h1,
        entry_price=entry_price, bias_result=bias_result,
    )

    if ttps_risk < 1:
        return None

    return Signal(
        direction=bias,
        entry_time=next_candle.opened_at,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        sl_pips=sl_pips,
        rr_ratio=round(rr, 2),
        setup_quality=setup_quality,
        confluence_score=score,
        ttps_htf=ttps_htf,
        ttps_entry=ttps_entry,
        ttps_session=ttps_session,
        ttps_risk=ttps_risk,
        ttps_total=ttps_total,
        primary_fvg_tf=entry_fvg.timeframe,
        primary_fvg_bottom=entry_fvg.bottom,
        primary_fvg_top=entry_fvg.top,
        swept_level=swept_level,
        block_reason="Signal confirmed",
        bias=bias_result,
    )


# ── Trade exit simulation ─────────────────────────────────────────────────────

def simulate_exit(
    signal: Signal,
    future_h1: list[Candle],
) -> tuple[str, float, float | None, datetime | None]:
    for c in future_h1:
        if c.opened_at < signal.entry_time:
            continue
        if signal.direction == "BULLISH":
            sl_hit = c.low  <= signal.sl_price
            tp_hit = c.high >= signal.tp_price
        else:
            sl_hit = c.high >= signal.sl_price
            tp_hit = c.low  <= signal.tp_price
        if sl_hit and tp_hit:
            return "LOSS", -1.0, signal.sl_price, c.opened_at
        if sl_hit:
            return "LOSS", -1.0, signal.sl_price, c.opened_at
        if tp_hit:
            return "WIN", signal.rr_ratio, signal.tp_price, c.opened_at
    return "OPEN", 0.0, None, None


# ── Kill switch checks ────────────────────────────────────────────────────────

def check_weekly_kill(state: BacktestState) -> bool:
    limit = state.week.start_balance * MAX_WEEKLY_LOSS
    return state.week.weekly_pnl <= -limit


def check_monthly_kill(state: BacktestState) -> bool:
    dd = (state.month_start_balance - state.account_balance) / max(state.month_start_balance, 1)
    return dd >= MAX_MONTHLY_DD


# ── Core backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    start: datetime | None = None,
    end:   datetime | None = None,
    fold:  str = "full",
    write_log: bool = True,
) -> list[Trade]:
    symbol        = exported_symbol()
    start_balance = env_float("ACCOUNT_BALANCE", 100_000.0)
    risk_pct      = env_float("RISK_PER_TRADE", 0.01)

    history     = {tf: load_history(tf) for tf in ("W1", "D1", "H4", "H1")}
    fvg_cache   = build_fvg_cache(history)
    news_csv    = PROJECT_ROOT / "data" / "news_events_sample.csv"
    news_events = load_news(news_csv)

    h1_all = history["H1"]
    trades: list[Trade] = []

    state = BacktestState(
        account_balance=start_balance,
        month_start_balance=start_balance,
    )
    state.week.start_balance = start_balance

    last_week  = ""
    last_month = ""

    for idx in range(25, len(h1_all) - 2):
        candle = h1_all[idx]
        as_of  = candle.opened_at + TIMEFRAME_DURATIONS["H1"]

        if start and candle.opened_at.date() < start.date(): continue
        if end   and candle.opened_at.date() >= end.date():  break

        wk = week_key(candle.opened_at)
        if wk != last_week:
            state.week = WeekState(start_balance=state.account_balance)
            last_week  = wk

        mk = month_key(candle.opened_at)
        if mk != last_month:
            state.month_start_balance = state.account_balance
            state.monthly_kill        = False
            last_month = mk

        if check_weekly_kill(state):  state.week.kill_active = True
        if check_monthly_kill(state): state.monthly_kill     = True
        if state.week.kill_active or state.monthly_kill:
            continue

        if state.week.trades_used >= MAX_TRADES_WEEK:
            continue

        trade_date      = candle.opened_ny.date()
        day             = state.week.get_day(trade_date)
        day_allowed, _  = day.may_trade()
        if not day_allowed:
            continue

        signal = find_signal(history, fvg_cache, idx, news_events)
        if signal is None:
            continue

        risk_multiplier = state.week.risk_multiplier()
        risk_amount     = start_balance * risk_pct
        lot_size        = math.floor(
            risk_amount / (signal.sl_pips * XAUUSD_PIP_VALUE_LOT) * 100
        ) / 100
        if lot_size <= 0:
            continue

        future = [c for c in h1_all[idx+1:] if c.opened_at >= signal.entry_time]
        result, r_gained, close_price, close_time = simulate_exit(signal, future)

        if result == "WIN":
            pnl = risk_amount * signal.rr_ratio
        elif result == "LOSS":
            pnl = -risk_amount
        else:
            pnl = 0.0

        holding_hours = 0.0
        if close_time:
            holding_hours = (close_time - signal.entry_time).total_seconds() / 3600

        trade = Trade(
            trade_id=f"{fold}-{len(trades)+1:04d}",
            instrument=symbol,
            fold=fold,
            entry_time_utc=signal.entry_time,
            exit_time_utc=close_time,
            direction=signal.direction,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            close_price=close_price,
            sl_pips=signal.sl_pips,
            rr_ratio=signal.rr_ratio,
            result=result,
            r_gained=r_gained,
            pnl=pnl,
            setup_quality=signal.setup_quality,
            confluence_score=signal.confluence_score,
            ttps_htf=signal.ttps_htf,
            ttps_entry=signal.ttps_entry,
            ttps_session=signal.ttps_session,
            ttps_risk=signal.ttps_risk,
            ttps_total=signal.ttps_total,
            primary_fvg_tf=signal.primary_fvg_tf,
            primary_fvg_bottom=signal.primary_fvg_bottom,
            primary_fvg_top=signal.primary_fvg_top,
            swept_level=signal.swept_level,
            risk_amount=risk_amount,
            risk_multiplier=risk_multiplier,
            block_reason=signal.block_reason,
            bias=signal.bias,
        )
        trades.append(trade)

        if result in {"WIN", "LOSS"}:
            state.account_balance   += pnl
            state.week.weekly_pnl   += pnl
            state.week.trades_used  += 1
            state.week.days_traded.add(trade_date)
            if result == "WIN":   state.week.weekly_wins   += 1
            else:                 state.week.weekly_losses += 1
            day.record(result)
            if check_weekly_kill(state):  state.week.kill_active = True
            if check_monthly_kill(state): state.monthly_kill     = True
        else:
            state.week.trades_used += 1
            state.week.days_traded.add(trade_date)

    if write_log:
        equity_path  = OUTPUT_DIR / "equity_curve.csv"
        metrics_path = OUTPUT_DIR / "metrics.json"
        write_trades(TRADE_LOG_PATH, trades)
        write_equity_curve(trades, equity_path, start_balance)
        write_metrics_json(trades, metrics_path, start_balance)
        try:
            write_html_dashboard()
        except Exception as e:
            print(f"Dashboard generation failed: {e}")

    return trades


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward() -> list[dict[str, object]]:
    h1 = load_history("H1")
    if not h1:
        print("No H1 data. Run: python main.py export-history")
        return []

    data_start = h1[0].opened_at
    data_end   = h1[-1].opened_at

    def add_months(dt: datetime, n: int) -> datetime:
        import calendar
        m = dt.month - 1 + n
        y = dt.year + m // 12
        m = m % 12 + 1
        d = min(dt.day, calendar.monthrange(y, m)[1])
        return dt.replace(year=y, month=m, day=d)

    folds  = []
    anchor = data_start
    fold_n = 1
    while True:
        oos_start = add_months(anchor, 6)
        oos_end   = add_months(oos_start, 2)
        if oos_end > data_end:
            break
        folds.append((
            f"fold{fold_n}",
            anchor.strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        ))
        anchor = add_months(anchor, 2)
        fold_n += 1

    if not folds:
        print("Not enough data for walk-forward.")
        return []

    rows: list[dict[str, object]] = []
    all_oos: list[Trade] = []

    for fold, train_start, val_start, val_end in folds:
        s      = parse_date(val_start)
        e      = parse_date(val_end)
        trades = run_backtest(s, e, fold=fold, write_log=False)
        all_oos.extend(trades)
        m = calculate_metrics(trades)
        rows.append({
            "fold": fold,
            "train_start":    train_start,
            "validate_start": val_start,
            "validate_end":   val_end,
            **metrics_to_row(m),
        })

    write_trades(TRADE_LOG_PATH, all_oos)
    write_dicts(FOLD_LOG_PATH, rows)
    sb = env_float("ACCOUNT_BALANCE", 100_000.0)
    write_equity_curve(all_oos, OUTPUT_DIR / "equity_curve.csv", sb)
    write_metrics_json(all_oos, OUTPUT_DIR / "metrics.json", sb)
    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(
    trades: list[Trade],
    starting_balance: float | None = None,
) -> Metrics:
    sb      = env_float("ACCOUNT_BALANCE", 100_000.0) if starting_balance is None else starting_balance
    closed  = [t for t in trades if t.result in {"WIN", "LOSS"}]
    wins    = [t for t in closed  if t.result == "WIN"]
    losses  = [t for t in closed  if t.result == "LOSS"]
    gp      = sum(t.pnl for t in wins)
    gl      = abs(sum(t.pnl for t in losses))
    total_r = sum(t.r_gained for t in closed)
    wr      = len(wins) / len(closed) * 100.0 if closed else 0.0
    pf      = gp / gl if gl else (math.inf if gp else 0.0)

    balance = sb
    peak    = sb
    max_dd  = 0.0
    returns: list[float] = []
    for t in closed:
        before   = balance
        balance += t.pnl
        peak     = max(peak, balance)
        max_dd   = max(max_dd, (peak - balance) / peak * 100 if peak else 0.0)
        if before:
            returns.append((balance - before) / before)

    sharpe = 0.0
    if len(returns) > 1:
        mean   = sum(returns) / len(returns)
        var    = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
        std    = math.sqrt(var)
        sharpe = mean / std * math.sqrt(len(returns)) if std else 0.0

    return Metrics(
        trades=len(trades), closed=len(closed),
        wins=len(wins), losses=len(losses),
        win_rate=wr, profit_factor=pf,
        total_r=total_r,
        average_r=total_r / len(closed) if closed else 0.0,
        max_drawdown_pct=max_dd, sharpe=sharpe,
        final_balance=balance,
    )


def monte_carlo(r_values: list[float], runs: int = 1000) -> dict[str, float]:
    if not r_values:
        return {"runs": float(runs), "worst_drawdown_r": 0.0, "median_drawdown_r": 0.0}
    dds: list[float] = []
    for _ in range(runs):
        shuffled = list(r_values)
        random.shuffle(shuffled)
        eq, peak, max_dd = 0.0, 0.0, 0.0
        for r in shuffled:
            eq    += r
            peak   = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        dds.append(max_dd)
    dds.sort()
    return {
        "runs":              float(runs),
        "worst_drawdown_r":  dds[-1],
        "median_drawdown_r": dds[len(dds) // 2],
    }


# ── Writers ───────────────────────────────────────────────────────────────────

def write_equity_curve(
    trades: list[Trade],
    path: Path,
    start_balance: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    balance = start_balance
    rows = [{"timestamp": "Start", "balance": round(balance, 2)}]
    for t in sorted(trades, key=lambda x: x.entry_time_utc):
        if t.result in {"WIN", "LOSS"}:
            balance += t.pnl
            rows.append({
                "timestamp": format_ny(t.exit_time_utc),
                "balance":   round(balance, 2),
            })
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp", "balance"])
        w.writeheader()
        w.writerows(rows)


def write_metrics_json(
    trades: list[Trade],
    path: Path,
    start_balance: float,
) -> None:
    m      = calculate_metrics(trades, start_balance)
    closed = [t for t in trades if t.result in {"WIN", "LOSS"}]
    weekly_r: dict[str, float] = {}
    for t in closed:
        wk = week_key(t.entry_time_utc)
        weekly_r[wk] = round(weekly_r.get(wk, 0.0) + t.r_gained, 3)
    payload = {
        "start_balance":          start_balance,
        "end_balance":            round(m.final_balance, 2),
        "total_trades":           m.trades,
        "wins":                   m.wins,
        "losses":                 m.losses,
        "win_rate":               round(m.win_rate, 2),
        "total_r":                round(m.total_r, 2),
        "profit_factor":          "inf" if math.isinf(m.profit_factor) else round(m.profit_factor, 3),
        "average_r":              round(m.average_r, 3),
        "max_drawdown_pct":       round(m.max_drawdown_pct, 2),
        "sharpe":                 round(m.sharpe, 3),
        "max_consecutive_losses": len([t for t in closed if t.result == "LOSS"]),
        "weekly_r":               weekly_r,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_trades(path: Path, trades: list[Trade]) -> None:
    write_dicts(path, [trade_to_row(t) for t in trades], TRADE_FIELDS)


def write_dicts(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else [])
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def trade_to_row(t: Trade) -> dict[str, object]:
    ny_entry = utc_to_ny(t.entry_time_utc)
    b = t.bias
    return {
        "trade_id":          t.trade_id,
        "instrument":        t.instrument,
        "fold":              t.fold,
        "entry_time_utc":    format_utc(t.entry_time_utc),
        "entry_time_ny":     format_ny(t.entry_time_utc),
        "exit_time_utc":     format_utc(t.exit_time_utc),
        "exit_time_ny":      format_ny(t.exit_time_utc),
        "day_of_week":       ny_entry.strftime("%A"),
        "holding_hours":     round(
            (t.exit_time_utc - t.entry_time_utc).total_seconds() / 3600, 1
        ) if t.exit_time_utc else "",
        "direction":         t.direction,
        "entry_price":       f"{t.entry_price:.2f}",
        "sl_price":          f"{t.sl_price:.2f}",
        "tp_price":          f"{t.tp_price:.2f}",
        "close_price":       f"{t.close_price:.2f}" if t.close_price else "",
        "sl_pips":           f"{t.sl_pips:.1f}",
        "rr_ratio":          f"{t.rr_ratio:.2f}",
        "trade_result":      t.result,
        "r_gained":          f"{t.r_gained:.2f}",
        "pnl":               f"{t.pnl:.2f}",
        "setup_quality":     t.setup_quality,
        "confluence_score":  t.confluence_score,
        "ttps_htf":          t.ttps_htf,
        "ttps_entry":        t.ttps_entry,
        "ttps_session":      t.ttps_session,
        "ttps_risk":         t.ttps_risk,
        "ttps_total":        t.ttps_total,
        "primary_fvg_tf":    t.primary_fvg_tf,
        "primary_fvg_bottom": f"{t.primary_fvg_bottom:.2f}",
        "primary_fvg_top":   f"{t.primary_fvg_top:.2f}",
        "swept_level":       f"{t.swept_level:.2f}" if t.swept_level else "",
        "risk_amount":       f"{t.risk_amount:.2f}",
        "risk_multiplier":   f"{t.risk_multiplier:.1f}",
        "week_number":       week_key(t.entry_time_utc),
        "block_reason":      t.block_reason,
        # Bias engine columns
        "bias_trend_score":  f"{b.trend_bull}/{b.trend_bear}" if b else "",
        "bias_swing_score":  f"{b.swing_bull}/{b.swing_bear}" if b else "",
        "bias_zone_score":   f"{b.zone_bull}/{b.zone_bear}"   if b else "",
        "bias_po3_score":    f"{b.po3_bull}/{b.po3_bear}"     if b else "",
        "bias_bull_total":   b.bull_total if b else "",
        "bias_bear_total":   b.bear_total if b else "",
        "bias_conviction":   b.conviction if b else "",
    }


def metrics_to_row(m: Metrics) -> dict[str, object]:
    return {
        "trades":           m.trades,
        "closed":           m.closed,
        "wins":             m.wins,
        "losses":           m.losses,
        "win_rate":         f"{m.win_rate:.2f}",
        "profit_factor":    "inf" if math.isinf(m.profit_factor) else f"{m.profit_factor:.2f}",
        "total_r":          f"{m.total_r:.2f}",
        "average_r":        f"{m.average_r:.2f}",
        "max_drawdown_pct": f"{m.max_drawdown_pct:.2f}",
        "sharpe":           f"{m.sharpe:.2f}",
        "final_balance":    f"{m.final_balance:.2f}",
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_metrics(title: str, trades: list[Trade]) -> None:
    m = calculate_metrics(trades)
    print(f"\n{title}")
    print(f"  Trades:          {m.trades} (closed={m.closed})")
    print(f"  Wins / Losses:   {m.wins} / {m.losses}")
    print(f"  Win rate:        {m.win_rate:.1f}%")
    print(f"  Total R:         {m.total_r:+.2f}")
    print(f"  Avg R/trade:     {m.average_r:+.3f}")
    print(f"  Profit factor:   {'inf' if math.isinf(m.profit_factor) else f'{m.profit_factor:.2f}'}")
    print(f"  Max drawdown:    {m.max_drawdown_pct:.2f}%")
    print(f"  Sharpe:          {m.sharpe:.2f}")
    print(f"  Final balance:   ${m.final_balance:,.2f}")
    mc = monte_carlo([t.r_gained for t in trades if t.result in {"WIN", "LOSS"}])
    print(f"  MC worst DD R:   {mc['worst_drawdown_r']:.2f}")
    print(f"  MC median DD R:  {mc['median_drawdown_r']:.2f}")


def print_data_status() -> None:
    history = {tf: load_history(tf) for tf in ("W1", "D1", "H4", "H1")}
    print(f"Symbol: {exported_symbol()}")
    for tf, candles in history.items():
        if candles:
            print(f"  {tf}: {len(candles)} candles  "
                  f"{format_ny(candles[0].opened_at)} → {format_ny(candles[-1].opened_at)} (NY)")
        else:
            print(f"  {tf}: MISSING")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def env_float(name: str, default: float) -> float:
    try:
        return float(load_env().get(name, default))
    except (TypeError, ValueError):
        return default


def exported_symbol() -> str:
    if not METADATA_PATH.exists():
        return trading_symbol()
    try:
        return str(json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        ).get("selected_symbol") or trading_symbol())
    except json.JSONDecodeError:
        return trading_symbol()


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sniper strategy backtester")
    p.add_argument("--mode", choices=("full", "walk-forward", "status"), default="full")
    p.add_argument("--start", help="Start date e.g. 2023-01-01")
    p.add_argument("--end",   help="End date e.g. 2025-01-01")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.mode == "status":
        print_data_status()
        return 0

    print_data_status()
    print()

    if args.mode == "walk-forward":
        rows = walk_forward()
        if not rows:
            return 1
        print("\nWalk-Forward Validation Results")
        print(f"  {'Fold':<8} {'Val Period':<24} {'Trades':>6} {'WR%':>6} "
              f"{'PF':>5} {'TotalR':>7} {'DD%':>6}")
        print("  " + "-" * 68)
        for row in rows:
            period = f"{row['validate_start']} → {row['validate_end']}"
            print(f"  {row['fold']:<8} {period:<24} {row['trades']:>6} "
                  f"{row['win_rate']:>6} {row['profit_factor']:>5} "
                  f"{row['total_r']:>7} {row['max_drawdown_pct']:>6}%")
        print(f"\n  Trade log:    {TRADE_LOG_PATH}")
        print(f"  Fold summary: {FOLD_LOG_PATH}")
        return 0

    start  = parse_date(args.start) if args.start else None
    end    = parse_date(args.end)   if args.end   else None
    trades = run_backtest(start, end)
    print_metrics("Full Backtest", trades)
    print(f"\n  Trade log: {TRADE_LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())