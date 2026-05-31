from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
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


PROJECT_ROOT  = Path(__file__).resolve().parent
OUTPUT_DIR    = PROJECT_ROOT / "data" / "backtest"
TRADE_LOG_PATH = OUTPUT_DIR / "trade_log.csv"
FOLD_LOG_PATH  = OUTPUT_DIR / "walk_forward_summary.csv"

# ── Strategy constants ────────────────────────────────────────────────────────
# XAUUSD: 1 pip = $1.00 price move
# ── Strategy constants ────────────────────────────────────────────────────────
XAUUSD_PIP_SIZE      = 1.0
XAUUSD_PIP_VALUE_LOT = 1.0

VALID_WEEKDAYS = {1, 2, 3}       # Tue=1, Wed=2, Thu=3
SESSION_START  = (2, 0)           # 02:00 NY
SESSION_END    = (11, 0)          # 11:00 NY

MAX_SL_PIPS     = 30.0   # $30 hard cap for XAUUSD
MIN_RR          = 2.0             # hard rule: 1:2 minimum
MAX_TRADES_WEEK = 3               # 3 ideas per week max
MAX_WEEKLY_LOSS = 0.03
MAX_MONTHLY_DD  = 0.10
CPI_BUFFER_MINS = 10              # rules doc: 10 min after CPI

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
    "swept_level", "risk_amount",
    "week_number", "block_reason",
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candle:
    time_str:  str
    opened_at: datetime       # UTC
    opened_ny: datetime       # NY local
    open:  float
    high:  float
    low:   float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class SwingPoint:
    time_str: str
    price:    float
    kind:     str   # "HIGH" or "LOW"


@dataclass(frozen=True)
class Signal:
    direction:       str
    entry_time:      datetime    # UTC
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
    block_reason:    str


@dataclass
class WeekState:
    """Resets every Sunday 18:00 NY."""
    trades_used:   int   = 0
    days_traded:   set   = None   # track which days traded
    weekly_pnl:    float = 0.0
    start_balance: float = 0.0
    kill_active:   bool  = False

    def __post_init__(self):
        if self.days_traded is None:
            self.days_traded = set()


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


def is_valid_trading_time(candle: Candle) -> tuple[bool, str]:
    """
    Enforce Tue-Thu only, 02:00-10:00 NY time.
    Capped at 10:00 (not 11:00) so that the entry candle
    (next candle after signal) still falls within the session window.
    Signal candle at 10:00 → entry on 11:00 candle open = still valid.
    Signal candle at 11:00 → entry on 12:00 candle open = outside session.
    """
    ny = candle.opened_ny
    if ny.weekday() not in VALID_WEEKDAYS:
        day_name = ny.strftime("%A")
        return False, f"Invalid day: {day_name} (must be Tue/Wed/Thu)"
    t = (ny.hour, ny.minute)
    if t < SESSION_START or t >= (10, 0):
        return False, f"Outside session: {ny.strftime('%H:%M')} NY (must be 02:00-10:00)"
    return True, ""


def week_key(dt: datetime) -> str:
    """Returns e.g. '2024-W03' using NY time."""
    ny = utc_to_ny(dt)
    iso = ny.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def month_key(dt: datetime) -> str:
    ny = utc_to_ny(dt)
    return f"{ny.year}-{ny.month:02d}"


def is_new_week(dt: datetime, last_week: str) -> bool:
    return week_key(dt) != last_week


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


def closed_as_of(candles: list[Candle], timeframe: str, as_of: datetime) -> list[Candle]:
    dur = TIMEFRAME_DURATIONS[timeframe]
    return [c for c in candles if c.opened_at + dur <= as_of]


# ── News filter ───────────────────────────────────────────────────────────────

NEWS_FULL_DAY    = {"NFP", "NONFARM", "NON-FARM", "FOMC", "FED RATE",
                    "RATE DECISION", "INTEREST RATE", "MONETARY POLICY"}
NEWS_PRE_RELEASE = {"CPI", "CORE CPI", "PCE", "GDP", "RETAIL SALES"}
CPI_BUFFER_MINS  = 5


@dataclass(frozen=True)
class NewsEvent:
    name:         str
    event_type:   str       # "FULL_DAY" or "PRE_RELEASE"
    release_time: datetime  # UTC


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
            name = str(row.get("event", "")).strip()
            raw  = str(row.get("time_ny", "")).strip()
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


def get_cpi_this_week(candle_time: datetime, events: list[NewsEvent]) -> datetime | None:
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
    """
    Rules from rules.docx:
      NFP day            → full day block
      FOMC + Rate same day → full day block + next day blocked
      FOMC alone         → tradeable (no block)
      CPI                → blocked until release + 10 min
      Major USD news     → tradeable
    """
    date = candle_time.date()

    # Collect today's events
    todays_events = [
        e for e in events
        if e.event_type == "FULL_DAY" and e.release_time.date() == date
    ]
    yesterdays_events = [
        e for e in events
        if e.event_type == "FULL_DAY"
        and (date - e.release_time.date()).days == 1
    ]
    event_names_today = [e.name.upper() for e in todays_events]
    event_names_yesterday = [e.name.upper() for e in yesterdays_events]

    # NFP — full day block
    if any("NFP" in n or "NONFARM" in n or "NON-FARM" in n
           for n in event_names_today):
        return True, "NFP day — no trading"

    # FOMC + Rate decision same day — block today AND next day
    has_fomc_today = any("FOMC" in n for n in event_names_today)
    has_rate_today = any("RATE" in n or "INTEREST" in n
                         for n in event_names_today)
    if has_fomc_today and has_rate_today:
        return True, "FOMC + Rate Decision — no trading"

    # Next day after FOMC + Rate decision — also blocked
    has_fomc_yesterday = any("FOMC" in n for n in event_names_yesterday)
    has_rate_yesterday = any("RATE" in n or "INTEREST" in n
                             for n in event_names_yesterday)
    if has_fomc_yesterday and has_rate_yesterday:
        return True, "Day after FOMC + Rate Decision — no trading"

    # CPI — block until release + 10 min buffer
    if cpi_this_week is not None:
        clear = cpi_this_week + timedelta(minutes=CPI_BUFFER_MINS)
        if candle_time < clear:
            return True, f"CPI block — clears {format_ny(clear)} NY"

    return False, ""


# ── Technical helpers ─────────────────────────────────────────────────────────

def detect_swings(candles: list[Candle]) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Simple 1-candle pivot detection — fast and sensitive."""
    highs: list[SwingPoint] = []
    lows:  list[SwingPoint] = []
    for i in range(1, len(candles) - 1):
        p, c, n = candles[i-1], candles[i], candles[i+1]
        if c.high > p.high and c.high > n.high:
            highs.append(SwingPoint(c.time_str, c.high, "HIGH"))
        if c.low < p.low and c.low < n.low:
            lows.append(SwingPoint(c.time_str, c.low, "LOW"))
    return highs, lows


def classify_trend(candles: list[Candle]) -> str:
    """
    Trend classification using last 40 candles (reduced from 120).
    Requires only 2 consecutive HH+HL or LH+LL — relaxed from 3.
    """
    highs, lows = detect_swings(candles[-40:])
    if len(highs) < 2 or len(lows) < 2:
        return "NO_TRADE"
    rh, rl = highs[-2:], lows[-2:]
    hh = rh[1].price > rh[0].price
    hl = rl[1].price > rl[0].price
    lh = rh[1].price < rh[0].price
    ll = rl[1].price < rl[0].price
    if hh and hl: return "BULLISH"
    if lh and ll: return "BEARISH"
    return "NO_TRADE"


def technical_bias(
    d1: list[Candle],
    h4: list[Candle],
    w1: list[Candle] | None = None,
) -> str:
    """
    D1 is primary. H4 is fallback. W1 candle direction as tiebreaker.
    """
    d1_bias = classify_trend(d1)
    if d1_bias in {"BULLISH", "BEARISH"}:
        return d1_bias

    h4_bias = classify_trend(h4)
    if h4_bias in {"BULLISH", "BEARISH"}:
        return h4_bias

    # W1 tiebreaker — last 3 weekly candles majority direction
    if w1 and len(w1) >= 3:
        recent = w1[-3:]
        bull = sum(1 for c in recent if c.close > c.open)
        bear = sum(1 for c in recent if c.close < c.open)
        if bull >= 2: return "BULLISH"
        if bear >= 2: return "BEARISH"

    return "NO_TRADE"


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


# ── FVG helpers ───────────────────────────────────────────────────────────────
def as_fvg_candles(candles: list[Candle]):
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
    """Convert backtest Candles to simple objects FVGs.py can consume."""
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


def build_fvg_cache(history: dict[str, list[Candle]]) -> dict[str, list[DetectedFVG]]:
    """Build FVG cache for ALL timeframes including H1 — needed for entry signals."""
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
    """
    FVG is invalid only if a candle CLOSED fully through the zone
    with a $5 buffer — partial closes inside the zone are normal
    price behaviour and do not invalidate the FVG.
    """
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
    """
    Layer 1 — Find the HTF area of interest (W1, D1, H4 FVG).
    This is NOT the entry — it defines the price area we expect
    price to react from. Entry happens on H1 FVG inside this area.
    If no HTF FVG exists, use the most recent significant swing level.
    """
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


def find_entry_fvg(
    history: dict[str, list[Candle]],
    fvg_cache: dict[str, list[DetectedFVG]],
    as_of: datetime,
    bias: str,
    current_price: float,
    htf_area: DetectedFVG | None,
) -> DetectedFVG | None:
    """
    Layer 2 — Find the H1 or H4 entry FVG.

    ICT entry model:
      - Scan H4 then H1 FVGs in bias direction
      - Price must be inside or have just entered the FVG
      - If an HTF area exists, prefer FVGs inside it (but don't require it)
      - Most recently formed FVG that price is currently touching = entry

    This is the actual entry zone where we place the trade.
    """
    entry_candidates = []

    for tf, weight in (("H4", 3), ("H1", 2)):
        for fvg in fvg_cache.get(tf, []):
            if fvg.direction != bias:
                continue
            formed = parse_time(fvg.formed_time)
            # Look back 14 days (2 weeks) for H4, 3 days for H1
            lookback_days = 14 if tf == "H4" else 3
            if formed < as_of - timedelta(days=lookback_days):
                continue
            if not fvg_still_valid(fvg, history[tf], tf, as_of):
                continue
            # Price must be inside the FVG (hard rule from rules doc)
            # Using $10 tolerance for XAUUSD spread and wick precision
            if not fvg.overlaps_price(current_price, tolerance=10.0):
                continue

            # Bonus if this entry FVG sits inside the HTF area of interest
            htf_bonus = 0
            if htf_area is not None:
                overlap = (
                   max(fvg.bottom, htf_area.bottom)
                   <=
                   min(fvg.top, htf_area.top)
                )
                if overlap:
                    htf_bonus = 2

            entry_candidates.append((weight + htf_bonus, formed, fvg))

    if not entry_candidates:
        return None
    return sorted(entry_candidates, key=lambda x: (x[0], x[1]), reverse=True)[0][2]


# ── Sweep detection ───────────────────────────────────────────────────────────

def detect_liquidity_sweep(
    h1: list[Candle],
    bias: str,
) -> tuple[bool, float | None]:
    """
    Sweep detection across three lookback windows:
      1. Previous session    — last 12 H1 candles (~half a day)
      2. Previous day        — last 24 H1 candles
      3. Previous week       — last 120 H1 candles (5 trading days)

    A sweep is confirmed when:
      - Price wicks through the lowest low (BULLISH) or highest high (BEARISH)
        within the window
      - The candle that swept CLOSES back on the other side (MSS confirmation)

    Prefers the most recent sweep found — closest in time = most relevant.
    """
    if len(h1) < 13:
        return False, None

    latest    = h1[-1]
    tolerance = 2.0   # $2 tolerance for XAUUSD

    # Define windows: (label, prior_candles)
    windows = [
        h1[-13:-1],   # previous session (~12 candles)
        h1[-25:-1],   # previous day (~24 candles)
        h1[-121:-1],  # previous week (~120 candles)
    ]

    for prior in windows:
        if not prior:
            continue

        if bias == "BULLISH":
            # Looking for SSL sweep: wick below the window's lowest low
            level = min(c.low for c in prior)
            if latest.low <= (level + tolerance) and latest.close > level:
                return True, level
        else:
            # Looking for BSL sweep: wick above the window's highest high
            level = max(c.high for c in prior)
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
    """
    SL always placed at a swing point relevant to the bias direction.
    No minimum pip rule — SL goes wherever the structure says.
    Hard maximum: $30 (MAX_SL_PIPS) to keep risk controlled.

    Priority:
      1. Most recent H1 swing low (BULLISH) or swing high (BEARISH)
         that is below/above entry and within $30
      2. FVG zone edge as fallback if no clean swing found
    """
    MAX_SL = MAX_SL_PIPS  # $30 hard cap

    highs, lows = detect_swings(h1[-100:])

    if bias == "BULLISH":
        # SL below the most recent swing low that is below entry
        candidates = [
            sw for sw in reversed(lows)
            if sw.price < entry and (entry - sw.price) <= MAX_SL
        ]
        if candidates:
            sl = candidates[0].price - 1.0  # $1 buffer below swing low
            return sl, abs(entry - sl)

        # Fallback: FVG bottom - buffer
        if entry_fvg is not None:
            sl = entry_fvg.bottom - 1.0
            dist = abs(entry - sl)
            if dist <= MAX_SL:
                return sl, dist

    else:  # BEARISH
        # SL above the most recent swing high that is above entry
        candidates = [
            sw for sw in reversed(highs)
            if sw.price > entry and (sw.price - entry) <= MAX_SL
        ]
        if candidates:
            sl = candidates[0].price + 1.0  # $1 buffer above swing high
            return sl, abs(entry - sl)

        # Fallback: FVG top + buffer
        if entry_fvg is not None:
            sl = entry_fvg.top + 1.0
            dist = abs(entry - sl)
            if dist <= MAX_SL:
                return sl, dist

    return None, 0.0


def find_tp_price(
    bias: str,
    entry: float,
    sl: float,
    d1: list[Candle],
    h4: list[Candle],
    h1: list[Candle],
) -> tuple[float | None, float]:
    """
    Flexible TP — searches for real liquidity targets in priority order:
      1. Equal highs (BULLISH) or equal lows (BEARISH) — highest probability
      2. D1 swing highs/lows — significant structural levels
      3. H4 swing highs/lows — intraday structural levels
      4. Previous week high/low — dealing range boundary
      5. Fallback: project minimum 1:2 from entry

    Minimum 1:2 R:R enforced on all targets.
    No maximum R:R cap — let the target be wherever liquidity sits.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return None, 0.0

    targets: list[tuple[float, int]] = []  # (price, priority)

    # ── Priority 1: Equal highs / equal lows ─────────────────────────────
    # These are the strongest ICT targets — clustered stop orders
    d1_highs, d1_lows = detect_swings(d1[-40:])
    h4_highs, h4_lows = detect_swings(h4[-80:])
    h1_highs, h1_lows = detect_swings(h1[-120:])

    all_highs = d1_highs + h4_highs + h1_highs
    all_lows  = d1_lows  + h4_lows  + h1_lows

    # Equal high/low tolerance: 0.15% of price (about $3 on $2000 gold)
    tolerance_pct = 0.0015

    if bias == "BULLISH":
        # Equal highs above entry = BSL targets
        for i in range(len(all_highs)):
            for j in range(i + 1, len(all_highs)):
                a, b = all_highs[i].price, all_highs[j].price
                avg  = (a + b) / 2
                if avg <= entry:
                    continue
                if abs(a - b) / avg <= tolerance_pct:
                    targets.append((avg, 1))  # priority 1 = equal level
    else:
        # Equal lows below entry = SSL targets
        for i in range(len(all_lows)):
            for j in range(i + 1, len(all_lows)):
                a, b = all_lows[i].price, all_lows[j].price
                avg  = (a + b) / 2
                if avg >= entry:
                    continue
                if abs(a - b) / avg <= tolerance_pct:
                    targets.append((avg, 1))

    # ── Priority 2: D1 swing points ──────────────────────────────────────
    if bias == "BULLISH":
        targets += [(s.price, 2) for s in d1_highs if s.price > entry]
    else:
        targets += [(s.price, 2) for s in d1_lows  if s.price < entry]

    # ── Priority 3: H4 swing points ──────────────────────────────────────
    if bias == "BULLISH":
        targets += [(s.price, 3) for s in h4_highs if s.price > entry]
    else:
        targets += [(s.price, 3) for s in h4_lows  if s.price < entry]

    # ── Priority 4: Previous week high / low ─────────────────────────────
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

    # ── Filter: must meet minimum R:R ────────────────────────────────────
    valid: list[tuple[float, float, int]] = []  # (distance, price, priority)
    for price, priority in targets:
        rr = abs(price - entry) / risk
        if rr >= MIN_RR:
            valid.append((abs(price - entry), price, priority, rr))

    if valid:
        # Sort by priority first (1=best), then by closest distance
        valid.sort(key=lambda x: (x[2], x[0]))
        _, tp, _, rr = valid[0]
        return tp, rr

    # ── Fallback: project minimum 1:2 ────────────────────────────────────
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
) -> tuple[int, int, int, int, int, str]:
    """
    TTPS scoring — 4 dimensions, 5 points each = 20 total.
    Returns (htf, entry, session, risk, total, grade)

    Dimension 1 — HTF Bias (0-5):
      +2  D1 trend matches bias
      +2  H4 trend matches bias
      +1  W1 candle direction matches (close > open = bullish)

    Dimension 2 — Entry Confluence (0-5):
      +3  Liquidity sweep confirmed
      +1  Price inside FVG at entry
      +1  H4 FVG also present at entry level

    Dimension 3 — Session Quality (0-5):
      +2  London session (02:00-05:00 NY)
      +3  London-NY overlap (08:00-11:00 NY)
      +1  NY morning (05:00-08:00 NY)

    Dimension 4 — Risk Quality (0-5):
      +2  SL within 20-25 pip range (tighter = better)
      +1  SL within 25-30 pip range
      +2  R:R >= 2.5
      +1  R:R >= 2.0 but < 2.5
    """

    # Dim 1: HTF Bias
    d1_bias = classify_trend(d1)
    h4_bias = classify_trend(h4)
    htf = 0
    if d1_bias == bias: htf += 2
    if h4_bias == bias: htf += 2
    if w1 and (bias == "BULLISH" and w1[-1].close > w1[-1].open) or \
              (bias == "BEARISH" and w1[-1].close < w1[-1].open): htf += 1
# Dim 2: Entry Confluence
    entry_score = 0
    if swept:
        entry_score += 2
    if primary_fvg is not None:
        entry_score += 1
    if primary_fvg is not None and primary_fvg.timeframe in {"H4", "D1", "W1"}:
        entry_score += 1

    # Dim 3: Session Quality
    hour = candle.opened_ny.hour
    session = 0
    if 2 <= hour < 5:  session = 2   # London open
    elif 8 <= hour <= 11: session = 3  # London-NY overlap
    elif 5 <= hour < 8:  session = 1   # NY morning

    # Dim 4: Risk Quality
    risk_score = 0
    if sl_pips <= 25:    risk_score += 2
    elif sl_pips <= 30:  risk_score += 1
    if rr >= 2.5:        risk_score += 2
    elif rr >= MIN_RR:   risk_score += 1

    total = htf + entry_score + session + risk_score

    # Grade
    if total >= 14:   grade = "A+"
    elif total >= 11: grade = "A"
    elif total >= 8:  grade = "B"
    elif total >= 5:  grade = "C"
    else:             grade = "WEAK"

    return htf, entry_score, session, risk_score, total, grade


# ── Open filter ───────────────────────────────────────────────────────────────

def open_filter_score(
    bias: str,
    entry: float,
    w1: list[Candle],
    d1: list[Candle],
) -> int:
    """0 = no confirms, 1 = one confirms, 2 = both confirm."""
    if not w1 or not d1:
        return 0
    w_ok = (entry >= w1[-1].open) if bias == "BULLISH" else (entry <= w1[-1].open)
    d_ok = (entry >= d1[-1].open) if bias == "BULLISH" else (entry <= d1[-1].open)
    return (1 if w_ok else 0) + (1 if d_ok else 0)


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

    # ── Rule 1: Valid day and session (NY time) ───────────────────────────
    time_ok, _ = is_valid_trading_time(candle)
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
    news_blocked, news_reason = is_news_blocked(
        candle.opened_at, news_events, cpi_this_week
    )
    if news_blocked:
        return None

    # ── Rule 3: Weekly bias from D1 / H4 ─────────────────────────────────
    bias = technical_bias(d1, h4, w1)
    if bias not in {"BULLISH", "BEARISH"}:
        return None

    # ── Layer 1: HTF area of interest ────────────────────────────────────
    # This gives us the macro zone — NOT the entry
    htf_area = choose_htf_area(history, fvg_cache, as_of, bias)
    # HTF area is preferred but not required
    # If no HTF FVG exists we still look for H1 entries in bias direction

    # ── Layer 2: H1/H4 entry FVG ─────────────────────────────────────────
    # Hard rule from rules.docx: price MUST be inside the FVG
    entry_price = next_candle.open
    current_price = candle.close

    entry_fvg = find_entry_fvg(
        history, fvg_cache, as_of, bias, current_price, htf_area
    )
    if entry_fvg is None:
        return None

    # ── Rule 4: SL placement ─────────────────────────────────────────────
    sl_price, sl_pips = find_sl_price(bias, entry_price, h1, entry_fvg)
    if sl_price is None:
        return None
    if sl_pips <= 0 or sl_pips > MAX_SL_PIPS:
        return None

    # ── Rule 5: Minimum 1:2 R:R ──────────────────────────────────────────
    tp_price, rr = find_tp_price(bias, entry_price, sl_price, d1, h4, h1)
    if tp_price is None or rr < MIN_RR:
        return None

    # Direction sanity
    if bias == "BULLISH" and (tp_price <= entry_price or sl_price >= entry_price):
        return None
    if bias == "BEARISH" and (tp_price >= entry_price or sl_price <= entry_price):
        return None

    # ── Sweep detection (optional — score bonus only) ─────────────────────
    swept, swept_level = detect_liquidity_sweep(h1, bias)
    sweep_bonus = 2 if swept else 0

    # ── Confluence score ──────────────────────────────────────────────────
    open_score = open_filter_score(bias, entry_price, w1, d1)
    entry_tf_weight = 3 if entry_fvg.timeframe == "H4" else 2  # H4 > H1
    htf_bonus = 2 if htf_area is not None else 0
    score = open_score + entry_tf_weight + htf_bonus + sweep_bonus

    # Grade — WEAK is the only hard rejection
    setup_quality = (
        "A" if score >= 8 else
        "B" if score >= 6 else
        "C" if score >= 3 else "WEAK"
    )
    if setup_quality == "WEAK":
        return None

    # ── TTPS score ────────────────────────────────────────────────────────
    ttps_htf, ttps_entry, ttps_session, ttps_risk, ttps_total, ttps_grade = calculate_ttps(
        bias=bias,
        swept=swept,
        primary_fvg=entry_fvg,
        confluence_score=score,
        sl_pips=sl_pips,
        rr=rr,
        candle=candle,
        w1=w1, d1=d1, h4=h4, h1=h1,
        entry_price=entry_price,
    )

    # Soft rule: risk dimension must be >= 1 (not a hard kill)
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
    )


# ── Trade exit simulation ─────────────────────────────────────────────────────

def simulate_exit(
    signal: Signal,
    future_h1: list[Candle],
) -> tuple[str, float, float | None, datetime | None]:
    """
    Walk forward candle by candle.
    Uses high/low touch — NOT close price.
    If SL and TP both touched on same candle → SL wins (conservative).
    """
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
            return "LOSS", -1.0, signal.sl_price, c.opened_at   # SL wins

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
    symbol = exported_symbol()
    start_balance = env_float("ACCOUNT_BALANCE", 100_000.0)
    risk_pct      = env_float("RISK_PER_TRADE", 0.01)

    history = {tf: load_history(tf) for tf in ("W1", "D1", "H4", "H1")}
    fvg_cache = build_fvg_cache(history)
    news_csv  = PROJECT_ROOT / "data" / "news_events_sample.csv"
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

        # Date range filter
        if start and as_of < start: continue
        if end   and as_of >= end:  break

        # ── Weekly reset (Sunday 18:00 NY) ────────────────────────────────
        wk = week_key(candle.opened_at)
        if wk != last_week:
            state.week = WeekState(start_balance=state.account_balance)
            last_week = wk

        # ── Monthly reset ─────────────────────────────────────────────────
        mk = month_key(candle.opened_at)
        if mk != last_month:
            state.month_start_balance = state.account_balance
            state.monthly_kill = False
            last_month = mk

        # ── Kill switch checks ────────────────────────────────────────────
        if check_weekly_kill(state):
            state.week.kill_active = True
        if check_monthly_kill(state):
            state.monthly_kill = True

        if state.week.kill_active or state.monthly_kill:
            continue

        # ── Max trades per week ───────────────────────────────────────────
        if state.week.trades_used >= MAX_TRADES_WEEK:
            continue

        # ── Find signal ───────────────────────────────────────────────────
        signal = find_signal(history, fvg_cache, idx, news_events)
        if signal is None:
            continue

        trade_day = candle.opened_ny.date()

        # ── Position sizing ───────────────────────────────────────────────
        risk_amount = state.account_balance * risk_pct
        lot_size    = math.floor(
            risk_amount / (signal.sl_pips * XAUUSD_PIP_VALUE_LOT) * 100
        ) / 100
        if lot_size <= 0:
            continue

        # ── Simulate exit ─────────────────────────────────────────────────
        future = [c for c in h1_all[idx+1:] if c.opened_at >= signal.entry_time]
        result, r_gained, close_price, close_time = simulate_exit(signal, future)

        if result == "WIN":
            pnl = risk_amount * signal.rr_ratio
        elif result == "LOSS":
            pnl = -risk_amount
        else:
            pnl = 0.0

        # ── Holding time ──────────────────────────────────────────────────
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
            block_reason=signal.block_reason,
        )
        trades.append(trade)

        # ── Update state ──────────────────────────────────────────────────
        if result in {"WIN", "LOSS"}:
            state.account_balance   += pnl
            state.week.weekly_pnl   += pnl
            state.week.trades_used  += 1
            state.week.days_traded.add(trade_day)

            if check_weekly_kill(state):
                state.week.kill_active = True
            if check_monthly_kill(state):
                state.monthly_kill = True
        else:
            # Open trade — still counts as used
            state.week.trades_used += 1
            state.week.days_traded.add(trade_day)

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
    """
    4 validation folds — 6 months IS, 2 months OOS each.
    Dates are anchored to the actual data range.
    """
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

    # Build folds dynamically from data range
    folds = []
    oos_months = 2
    is_months  = 6
    anchor = data_start
    fold_n = 1
    while True:
        oos_start = add_months(anchor, is_months)
        oos_end   = add_months(oos_start, oos_months)
        if oos_end > data_end:
            break
        folds.append((
            f"fold{fold_n}",
            anchor.strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        ))
        anchor = add_months(anchor, oos_months)
        fold_n += 1

    if not folds:
        print("Not enough data for walk-forward. Need at least 8 months.")
        return []

    rows: list[dict[str, object]] = []
    all_oos_trades: list[Trade] = []

    for fold, train_start, val_start, val_end in folds:
        s = parse_date(val_start)
        e = parse_date(val_end)
        trades = run_backtest(s, e, fold=fold, write_log=False)
        all_oos_trades.extend(trades)
        m = calculate_metrics(trades)
        rows.append({
            "fold": fold,
            "train_start":    train_start,
            "validate_start": val_start,
            "validate_end":   val_end,
            **metrics_to_row(m),
        })

    write_trades(TRADE_LOG_PATH, all_oos_trades)
    write_dicts(FOLD_LOG_PATH, rows)
    write_equity_curve(all_oos_trades, OUTPUT_DIR / "equity_curve.csv", env_float("ACCOUNT_BALANCE", 100_000.0))
    write_metrics_json(all_oos_trades, OUTPUT_DIR / "metrics.json", env_float("ACCOUNT_BALANCE", 100_000.0))
    return rows


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(
    trades: list[Trade],
    starting_balance: float | None = None,
) -> Metrics:
    sb = env_float("ACCOUNT_BALANCE", 100_000.0) if starting_balance is None else starting_balance
    closed   = [t for t in trades if t.result in {"WIN", "LOSS"}]
    wins     = [t for t in closed  if t.result == "WIN"]
    losses   = [t for t in closed  if t.result == "LOSS"]
    gp       = sum(t.pnl for t in wins)
    gl       = abs(sum(t.pnl for t in losses))
    total_r  = sum(t.r_gained for t in closed)
    wr       = len(wins) / len(closed) * 100.0 if closed else 0.0
    pf       = gp / gl if gl else (math.inf if gp else 0.0)

    balance  = sb
    peak     = sb
    max_dd   = 0.0
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
        mean = sum(returns) / len(returns)
        var  = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
        std  = math.sqrt(var)
        sharpe = mean / std * math.sqrt(len(returns)) if std else 0.0

    return Metrics(
        trades=len(trades),
        closed=len(closed),
        wins=len(wins),
        losses=len(losses),
        win_rate=wr,
        profit_factor=pf,
        total_r=total_r,
        average_r=total_r / len(closed) if closed else 0.0,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
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
            eq   += r
            peak  = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        dds.append(max_dd)
    dds.sort()
    return {
        "runs":               float(runs),
        "worst_drawdown_r":   dds[-1],
        "median_drawdown_r":  dds[len(dds) // 2],
    }


# ── Writers ───────────────────────────────────────────────────────────────────

def write_equity_curve(trades: list[Trade], path: Path, start_balance: float) -> None:
    """Write equity_curve.csv so the dashboard can draw an accurate chart."""
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


def write_metrics_json(trades: list[Trade], path: Path, start_balance: float) -> None:
    """Write metrics.json so the dashboard gets accurate summary stats."""
    m = calculate_metrics(trades, start_balance)
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
        "week_number":       week_key(t.entry_time_utc),
        "block_reason":      t.block_reason,
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
            first_ny = format_ny(candles[0].opened_at)
            last_ny  = format_ny(candles[-1].opened_at)
            print(f"  {tf}: {len(candles)} candles  {first_ny} → {last_ny} (NY)")
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
    p.add_argument("--start", help="UTC start date e.g. 2023-01-01")
    p.add_argument("--end",   help="UTC end date e.g. 2025-01-01")
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

    start = parse_date(args.start) if args.start else None
    end   = parse_date(args.end)   if args.end   else None
    trades = run_backtest(start, end)
    print_metrics("Full Backtest", trades)
    print(f"\n  Trade log: {TRADE_LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




