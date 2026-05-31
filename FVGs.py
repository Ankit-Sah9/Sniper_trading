from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import math


PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_DIR  = PROJECT_ROOT / "data" / "history"
ENV_PATH     = PROJECT_ROOT / ".env"

PRIMARY_TIMEFRAMES = ("W1", "D1", "H4")

TIMEFRAME_DURATIONS = {
    "W1": timedelta(days=7),
    "D1": timedelta(days=1),
    "H4": timedelta(hours=4),
    "H1": timedelta(hours=1),
}
MAX_CANDLE_GAPS = {
    "W1": timedelta(days=10),
    "D1": timedelta(days=4),
    "H4": timedelta(days=3),
    "H1": timedelta(hours=2),
}

# ── Config ────────────────────────────────────────────────────────────────────

def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_float(name: str, default: float) -> float:
    try:
        return float(load_env().get(name, default))
    except (TypeError, ValueError):
        return default


def history_path(timeframe: str) -> Path:
    return HISTORY_DIR / f"{timeframe.upper()}.csv"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Candle:
    time:  str
    open:  float
    high:  float
    low:   float
    close: float

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low


@dataclass
class FVG:
    """
    ICT Fair Value Gap — 3-candle imbalance zone.

    Fields:
      top / bottom   : zone boundaries (candle1.high to candle3.low for bullish)
      mid            : Consequent Encroachment — 50% of the zone (key entry level)
      displacement   : True if middle candle body > ATR (institutional-grade gap)
      status         : OPEN → MITIGATED → FILLED → INVERTED
      inversion_fvg  : when violated, this FVG flips to opposite direction
    """
    top:           float
    bottom:        float
    mid:           float          # Consequent Encroachment (CE) = 50% level
    timeframe:     str
    direction:     str            # "BULLISH" or "BEARISH"
    formed_time:   str
    candle1_time:  str
    candle2_time:  str
    candle3_time:  str
    candle1_high:  float
    candle1_low:   float
    candle3_high:  float
    candle3_low:   float
    candle1_row:   int
    candle2_row:   int
    candle3_row:   int
    gap_size:      float = 0.0   # raw gap in price units
    displacement:  bool  = False  # middle candle body > ATR
    status:        str   = "OPEN"
    # Inversion FVG — when this FVG is violated it becomes an opposite-direction zone
    inverted:      bool  = False
    inversion_direction: str = ""

    def overlaps_price(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.bottom - tolerance) <= price <= (self.top + tolerance)

    def price_near_ce(self, price: float, tolerance: float = 2.0) -> bool:
        """True if price is within tolerance of the Consequent Encroachment (50% midpoint)."""
        return abs(price - self.mid) <= tolerance

    def gap_width(self) -> float:
        return self.top - self.bottom


@dataclass(frozen=True)
class SwingPoint:
    time:  str
    price: float
    kind:  str    # "HIGH" or "LOW"
    row:   int


@dataclass(frozen=True)
class LiquidityLevel:
    """
    ICT BSL / SSL level — a significant swing high or low where
    stop orders cluster. Used as sweep targets and trade targets.
    """
    price:      float
    kind:       str       # "BSL" (above swing high) or "SSL" (below swing low)
    formed_time: str
    timeframe:  str
    equal:      bool = False   # True if this is an "equal high" or "equal low"
    swept:      bool = False
    sweep_time: str  = ""


@dataclass(frozen=True)
class BalancedPriceRange:
    """
    BPR — overlap zone between a bullish FVG and a bearish FVG.
    Very strong reaction zone — both buying and selling were inefficient here.
    """
    top:     float
    bottom:  float
    mid:     float
    bullish_fvg_formed: str
    bearish_fvg_formed: str


@dataclass(frozen=True)
class Phase1Output:
    final_bias:               str
    conviction:               str
    technical_bias:           str
    ai_bias:                  str
    ai_confidence:            str
    fib_high:                 float | None
    fib_low:                  float | None
    fib_50:                   float | None
    zone:                     str
    primary_fvg:              FVG | None
    no_trade_week:            bool
    structure_shift:          bool
    structure_shift_direction: str
    primary_fvg_score:        int = 0
    fvg_candidates:           tuple = ()
    liquidity_levels:         tuple = ()    # active BSL / SSL levels
    balanced_price_ranges:    tuple = ()    # active BPR zones


# ── Candle I/O ────────────────────────────────────────────────────────────────

def read_candles(timeframe: str) -> list[Candle]:
    path = history_path(timeframe)
    if not path.exists():
        return []
    candles: list[Candle] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                candles.append(Candle(
                    time=str(row.get("time", "")),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return candles


def parse_candle_time(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def closed_candles(candles: list[Candle], timeframe: str) -> list[Candle]:
    dur = TIMEFRAME_DURATIONS.get(timeframe.upper())
    if dur is None:
        return candles
    now = datetime.now(timezone.utc)
    return [
        c for c in candles
        if (t := parse_candle_time(c.time)) is not None and t + dur <= now
    ]


def recent_candles(candles: list[Candle], days: int = 120) -> list[Candle]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = [
        c for c in candles
        if (t := parse_candle_time(c.time)) is not None and t >= cutoff
    ]
    return result or candles


# ── ATR (for displacement filter) ─────────────────────────────────────────────

def compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Average True Range over last `period` candles."""
    if len(candles) < period + 1:
        if len(candles) < 2:
            return 0.0
        period = len(candles) - 1
    trs = [
        max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low  - candles[i - 1].close),
        )
        for i in range(len(candles) - period, len(candles))
    ]
    return sum(trs) / len(trs) if trs else 0.0


# ── Contiguity check ──────────────────────────────────────────────────────────

def is_valid_candle_gap(prev: datetime, curr: datetime, timeframe: str) -> bool:
    gap = curr - prev
    if gap <= timedelta(0):
        return False
    tf = timeframe.upper()
    if tf == "W1":
        return timedelta(days=6) <= gap <= timedelta(days=10)
    if tf == "D1":
        return gap <= timedelta(days=1, hours=4) or _weekend_gap(prev, curr, timedelta(days=4))
    if tf == "H4":
        return gap <= timedelta(hours=5) or _weekend_gap(prev, curr, timedelta(days=3))
    if tf == "H1":
        return gap <= timedelta(hours=2) or _weekend_gap(prev, curr, timedelta(days=3))
    return True


def _weekend_gap(prev: datetime, curr: datetime, max_gap: timedelta) -> bool:
    return prev.weekday() == 4 and curr.weekday() in {6, 0} and (curr - prev) <= max_gap


def is_contiguous_triplet(c1: Candle, c2: Candle, c3: Candle, timeframe: str) -> bool:
    t1, t2, t3 = parse_candle_time(c1.time), parse_candle_time(c2.time), parse_candle_time(c3.time)
    if None in (t1, t2, t3):
        return False
    return (
        is_valid_candle_gap(t1, t2, timeframe) and
        is_valid_candle_gap(t2, t3, timeframe)
    )


# ── FVG Detection (ICT-accurate) ──────────────────────────────────────────────

def detect_fvgs(
    candles: list[Candle],
    timeframe: str,
    min_gap: float | None = None,
    require_displacement: bool = False,
    atr_multiplier: float = 0.5,
) -> list[FVG]:
    """
    ICT-accurate FVG detection using strict 3-candle rule.

    Bullish FVG:
      candle3.low > candle1.high  (gap between C1 high and C3 low)
      The imbalance zone = bottom: C1.high, top: C3.low
      Consequent Encroachment (CE) = midpoint = 50% level

    Bearish FVG:
      candle1.low > candle3.high  (gap between C3 high and C1 low)
      The imbalance zone = bottom: C3.high, top: C1.low
      CE = midpoint

    Displacement filter (optional):
      Middle candle body must exceed ATR * atr_multiplier
      This ensures only institutional-grade moves are captured

    Args:
      min_gap           : minimum gap size in price units (default from env)
      require_displacement : if True, reject FVGs without displacement candle
      atr_multiplier    : how large middle candle body must be vs ATR
    """
    if min_gap is None:
        min_gap = get_float("GOLD_FVG_MIN_GAP", 3.0)  # relaxed from 5.0 to 3.0

    fvgs: list[FVG] = []
    if len(candles) < 3:
        return fvgs

    # Pre-compute ATR for displacement filter
    atr = compute_atr(candles) if require_displacement else 0.0
    displacement_threshold = atr * atr_multiplier

    for i in range(2, len(candles)):
        c1, c2, c3 = candles[i - 2], candles[i - 1], candles[i]

        if not is_contiguous_triplet(c1, c2, c3, timeframe):
            continue

        # Displacement check on middle candle (C2)
        is_displaced = c2.body_size >= displacement_threshold if require_displacement else True

        # ── Bullish FVG ───────────────────────────────────────────────────
        bullish_gap = c3.low - c1.high
        if bullish_gap >= min_gap:
            bottom = c1.high
            top    = c3.low
            mid    = (top + bottom) / 2   # Consequent Encroachment
            fvgs.append(FVG(
                top=top, bottom=bottom, mid=mid,
                timeframe=timeframe,
                direction="BULLISH",
                formed_time=c3.time,
                candle1_time=c1.time, candle2_time=c2.time, candle3_time=c3.time,
                candle1_high=c1.high, candle1_low=c1.low,
                candle3_high=c3.high, candle3_low=c3.low,
                candle1_row=i - 1, candle2_row=i, candle3_row=i + 1,
                gap_size=bullish_gap,
                displacement=is_displaced,
            ))

        # ── Bearish FVG ───────────────────────────────────────────────────
        bearish_gap = c1.low - c3.high
        if bearish_gap >= min_gap:
            bottom = c3.high
            top    = c1.low
            mid    = (top + bottom) / 2
            fvgs.append(FVG(
                top=top, bottom=bottom, mid=mid,
                timeframe=timeframe,
                direction="BEARISH",
                formed_time=c3.time,
                candle1_time=c1.time, candle2_time=c2.time, candle3_time=c3.time,
                candle1_high=c1.high, candle1_low=c1.low,
                candle3_high=c3.high, candle3_low=c3.low,
                candle1_row=i - 1, candle2_row=i, candle3_row=i + 1,
                gap_size=bearish_gap,
                displacement=is_displaced,
            ))

    return fvgs


# ── FVG Status ────────────────────────────────────────────────────────────────

def fvg_status(candles: list[Candle], fvg: FVG) -> str:
    """
    ICT FVG status tracking:
      OPEN      — price has not entered the zone
      MITIGATED — price entered (touched) the zone but close did not violate it
      FILLED    — price closed through the CE (50% midpoint) — zone is used
      INVERTED  — price closed BEYOND the full zone — FVG flips direction (IFVG)

    Key distinction from old logic:
      Old: FILLED when close passes bottom/top (full zone)
      New: FILLED when close passes CE (mid) — this is ICT's actual rule
           INVERTED when close passes beyond the entire zone
    """
    after = [c for c in candles if c.time > fvg.formed_time]
    if not after:
        return "OPEN"

    if fvg.direction == "BULLISH":
        for c in after:
            if c.close < fvg.bottom:   return "INVERTED"  # full close-through → IFVG
            if c.close < fvg.mid:      return "FILLED"    # closed through CE
            if c.low   < fvg.top:      return "MITIGATED" # touched but held
        return "OPEN"
    else:  # BEARISH
        for c in after:
            if c.close > fvg.top:      return "INVERTED"
            if c.close > fvg.mid:      return "FILLED"
            if c.high  > fvg.bottom:   return "MITIGATED"
        return "OPEN"


def formed_within_days(fvg: FVG, days: int = 120) -> bool:
    formed_at = parse_candle_time(fvg.formed_time)
    if formed_at is None:
        return False
    return formed_at >= datetime.now(timezone.utc) - timedelta(days=days)


def unfilled_fvgs(
    timeframe: str,
    recent_only: bool = True,
    include_mitigated: bool = True,
    require_displacement: bool = False,
) -> list[FVG]:
    """
    Returns active FVGs (OPEN and MITIGATED).
    MITIGATED FVGs are still valid entry zones — price touched but didn't fill.
    INVERTED FVGs are returned separately via inverted_fvgs().
    """
    candles = closed_candles(read_candles(timeframe), timeframe)
    fvgs = detect_fvgs(candles, timeframe, require_displacement=require_displacement)
    active: list[FVG] = []
    for fvg in fvgs:
        if recent_only and not formed_within_days(fvg):
            continue
        status = fvg_status(candles, fvg)
        fvg.status = status
        if status == "OPEN" or (include_mitigated and status == "MITIGATED"):
            active.append(fvg)
        elif status == "INVERTED":
            # Mark as inverted for use as IFVG
            fvg.inverted = True
            fvg.inversion_direction = "BEARISH" if fvg.direction == "BULLISH" else "BULLISH"
    return active


def inverted_fvgs(timeframe: str) -> list[FVG]:
    """
    Returns Inversion FVGs (IFVGs) — violated FVGs that now act as
    opposite-direction zones. In ICT, a broken bullish FVG becomes
    bearish resistance, and vice versa.
    """
    candles = closed_candles(read_candles(timeframe), timeframe)
    fvgs    = detect_fvgs(candles, timeframe)
    result: list[FVG] = []
    for fvg in fvgs:
        if not formed_within_days(fvg):
            continue
        status = fvg_status(candles, fvg)
        if status == "INVERTED":
            fvg.status = "INVERTED"
            fvg.inverted = True
            fvg.inversion_direction = "BEARISH" if fvg.direction == "BULLISH" else "BULLISH"
            result.append(fvg)
    return result


# ── Balanced Price Range (BPR) ────────────────────────────────────────────────

def detect_balanced_price_ranges(
    bullish_fvgs: list[FVG],
    bearish_fvgs: list[FVG],
) -> list[BalancedPriceRange]:
    """
    BPR = overlap zone between a bullish FVG and a bearish FVG.
    These are very high-probability reaction zones — both buying
    and selling were inefficient at the same price level.
    """
    bprs: list[BalancedPriceRange] = []
    for bull in bullish_fvgs:
        for bear in bearish_fvgs:
            overlap_bottom = max(bull.bottom, bear.bottom)
            overlap_top    = min(bull.top,    bear.top)
            if overlap_top > overlap_bottom:
                bprs.append(BalancedPriceRange(
                    top=overlap_top,
                    bottom=overlap_bottom,
                    mid=(overlap_top + overlap_bottom) / 2,
                    bullish_fvg_formed=bull.formed_time,
                    bearish_fvg_formed=bear.formed_time,
                ))
    return bprs


# ── Liquidity Level Detection ─────────────────────────────────────────────────

def detect_swings(candles: list[Candle], lookback: int = 5) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Detect swing highs and lows using a configurable lookback window.
    lookback=5 means the pivot is the highest/lowest among 5 candles
    on each side — more significant than a simple 1-candle comparison.
    """
    highs: list[SwingPoint] = []
    lows:  list[SwingPoint] = []
    n = len(candles)
    for i in range(lookback, n - lookback):
        window = candles[i - lookback: i + lookback + 1]
        c = candles[i]
        if c.high == max(w.high for w in window):
            highs.append(SwingPoint(c.time, c.high, "HIGH", i + 2))
        if c.low == min(w.low for w in window):
            lows.append(SwingPoint(c.time, c.low, "LOW", i + 2))
    return highs, lows


def detect_equal_levels(
    swings: list[SwingPoint],
    tolerance_pct: float = 0.001,  # 0.1% tolerance
) -> list[SwingPoint]:
    """
    Find equal highs or equal lows — levels where price touched
    the same price area multiple times, creating clustered stop orders.
    These are the highest-probability ICT liquidity pools.
    """
    equal: list[SwingPoint] = []
    for i in range(len(swings)):
        for j in range(i + 1, len(swings)):
            avg   = (swings[i].price + swings[j].price) / 2
            diff  = abs(swings[i].price - swings[j].price) / avg if avg else 0
            if diff <= tolerance_pct:
                # Mark the more recent one as the equal level
                newer = swings[j] if swings[j].time > swings[i].time else swings[i]
                if newer not in equal:
                    equal.append(newer)
    return equal


def build_liquidity_levels(
    candles: list[Candle],
    timeframe: str,
    lookback: int = 5,
) -> list[LiquidityLevel]:
    """
    Build BSL (buy-side liquidity) and SSL (sell-side liquidity) levels.

    BSL = stop orders above swing highs (equal highs get priority)
    SSL = stop orders below swing lows  (equal lows get priority)

    Equal highs/lows are weighted more heavily because more retail
    stops cluster at visually obvious double tops/bottoms.
    """
    highs, lows = detect_swings(candles, lookback=lookback)
    equal_highs = {s.time for s in detect_equal_levels(highs)}
    equal_lows  = {s.time for s in detect_equal_levels(lows)}

    levels: list[LiquidityLevel] = []

    for swing in highs[-20:]:   # last 20 significant highs
        levels.append(LiquidityLevel(
            price=swing.price,
            kind="BSL",
            formed_time=swing.time,
            timeframe=timeframe,
            equal=swing.time in equal_highs,
        ))

    for swing in lows[-20:]:    # last 20 significant lows
        levels.append(LiquidityLevel(
            price=swing.price,
            kind="SSL",
            formed_time=swing.time,
            timeframe=timeframe,
            equal=swing.time in equal_lows,
        ))

    return levels


# ── ICT Liquidity Sweep Detection ─────────────────────────────────────────────

def detect_liquidity_sweep_ict(
    candles: list[Candle],
    bias: str,
    lookback: int = 20,
    require_mss: bool = True,
) -> tuple[bool, float | None, str]:
    """
    ICT-accurate liquidity sweep detection.

    Process:
      1. Build SSL (for bullish) or BSL (for bearish) levels from recent swings
      2. Check if the most recent candle(s) swept through a level
      3. Confirm the sweep with a Market Structure Shift (MSS) — price must
         close back through the swept level (showing rejection, not continuation)
      4. Prefer equal highs/lows as they have more stop orders clustered there

    Returns: (swept, swept_level_price, sweep_type)
      swept_level_price: the price that was swept
      sweep_type: "EQUAL_HIGH", "EQUAL_LOW", "SWING_HIGH", "SWING_LOW"

    For BULLISH bias (looking to go long):
      - Look for SSL sweep: price wicks below a swing low / equal low
      - Then closes back above it (stop hunt complete, reversal likely)

    For BEARISH bias (looking to go short):
      - Look for BSL sweep: price wicks above a swing high / equal high
      - Then closes back below it
    """
    if len(candles) < lookback + 3:
        return False, None, ""

    # Build levels from prior candles (excluding last 3 — those are the sweep candles)
    prior   = candles[-(lookback + 3): -3]
    recent  = candles[-3:]
    latest  = candles[-1]
    highs, lows = detect_swings(prior, lookback=3)

    if bias == "BULLISH":
        # Looking for SSL sweep (sweep below swing lows → bullish reversal)
        if not lows:
            return False, None, ""

        # Find all SSL levels, prefer equal lows
        equal_lows = {s.time for s in detect_equal_levels(lows)}
        ssl_candidates = sorted(lows, key=lambda s: (s.time in equal_lows, s.time), reverse=True)

        for level in ssl_candidates[-5:]:   # check last 5 SSL levels
            # Did any recent candle wick below this level?
            swept = any(c.low < level.price for c in recent)
            if not swept:
                continue

            if require_mss:
                # MSS confirmation: latest candle closes ABOVE the swept level
                mss = latest.close > level.price
                if not mss:
                    continue

            sweep_type = "EQUAL_LOW" if level.time in equal_lows else "SWING_LOW"
            return True, level.price, sweep_type

        return False, None, ""

    else:  # BEARISH
        # Looking for BSL sweep (sweep above swing highs → bearish reversal)
        if not highs:
            return False, None, ""

        equal_highs = {s.time for s in detect_equal_levels(highs)}
        bsl_candidates = sorted(highs, key=lambda s: (s.time in equal_highs, s.time), reverse=True)

        for level in bsl_candidates[-5:]:
            swept = any(c.high > level.price for c in recent)
            if not swept:
                continue

            if require_mss:
                mss = latest.close < level.price
                if not mss:
                    continue

            sweep_type = "EQUAL_HIGH" if level.time in equal_highs else "SWING_HIGH"
            return True, level.price, sweep_type

        return False, None, ""


# ── Trend / Bias ──────────────────────────────────────────────────────────────

def classify_trend(candles: list[Candle]) -> str:
    highs, lows = detect_swings(recent_candles(candles), lookback=3)
    if len(highs) < 3 or len(lows) < 3:
        return "NO_TRADE"
    rh, rl = highs[-3:], lows[-3:]
    hh = sum(1 for a, b in zip(rh, rh[1:]) if b.price > a.price)
    hl = sum(1 for a, b in zip(rl, rl[1:]) if b.price > a.price)
    lh = sum(1 for a, b in zip(rh, rh[1:]) if b.price < a.price)
    ll = sum(1 for a, b in zip(rl, rl[1:]) if b.price < a.price)
    if hh >= 2 and hl >= 2: return "BULLISH"
    if lh >= 2 and ll >= 2: return "BEARISH"
    return "NO_TRADE"


def technical_bias() -> str:
    daily = classify_trend(closed_candles(read_candles("D1"), "D1"))
    if daily in {"BULLISH", "BEARISH"}:
        return daily
    h4 = classify_trend(closed_candles(read_candles("H4"), "H4"))
    return h4 if h4 in {"BULLISH", "BEARISH"} else "NO_TRADE"


def detect_structure_shift(prior_trend: str) -> tuple[bool, str]:
    prior_trend = prior_trend.upper()
    if prior_trend not in {"BULLISH", "BEARISH"}:
        return False, "NONE"
    for timeframe in ("D1", "W1"):
        candles = closed_candles(read_candles(timeframe), timeframe)
        if len(candles) < 5:
            continue
        highs, lows = detect_swings(recent_candles(candles))
        latest_close = candles[-1].close
        if prior_trend == "BEARISH" and highs and latest_close >= highs[-1].price:
            return True, "BULLISH"
        if prior_trend == "BULLISH" and lows and latest_close <= lows[-1].price:
            return True, "BEARISH"
    return False, "NONE"


# ── Fibonacci ─────────────────────────────────────────────────────────────────

def fibonacci_zone(
    candles: list[Candle],
    current_price: float | None = None,
) -> tuple[float | None, float | None, float | None, str]:
    candles = recent_candles(candles)
    if not candles:
        return None, None, None, "NEUTRAL"
    fib_high = max(c.high for c in candles)
    fib_low  = min(c.low  for c in candles)
    fib_50   = fib_low + (fib_high - fib_low) * 0.50
    price    = candles[-1].close if current_price is None else current_price
    tolerance = get_float("GOLD_FVG_MIN_GAP", 3.0)
    if abs(price - fib_50) <= tolerance:
        return fib_high, fib_low, fib_50, "NEUTRAL"
    return fib_high, fib_low, fib_50, "DISCOUNT" if price < fib_50 else "PREMIUM"


# ── Bias combine ──────────────────────────────────────────────────────────────

def combine_bias(
    technical: str,
    ai_bias: str = "CONSOLIDATION",
    ai_confidence: str = "LOW",
    zone: str = "NEUTRAL",
) -> tuple[str, str]:
    technical    = technical.upper()
    ai_bias      = ai_bias.upper()
    ai_confidence = ai_confidence.upper()
    zone         = zone.upper()

    if technical not in {"BULLISH", "BEARISH"}:
        return "NO_TRADE", "NONE"
    if ai_bias == technical:
        return technical, "HIGH" if ai_confidence == "HIGH" else "MEDIUM"
    if ai_bias == "CONSOLIDATION":
        return technical, "MEDIUM"

    zone_bias = "BULLISH" if zone == "DISCOUNT" else "BEARISH" if zone == "PREMIUM" else "NO_TRADE"
    if zone_bias == technical:
        return technical, "MEDIUM"
    if zone_bias == ai_bias and ai_confidence == "HIGH":
        return ai_bias, "MEDIUM"
    return "NO_TRADE", "NONE"


def configured_ai_bias() -> tuple[str, str]:
    env = load_env()
    bias       = env.get("AI_BIAS", env.get("FUNDAMENTAL_BIAS", "CONSOLIDATION")).strip().upper()
    confidence = env.get("AI_CONFIDENCE", env.get("FUNDAMENTAL_CONFIDENCE", "LOW")).strip().upper()
    if bias not in {"BULLISH", "BEARISH", "CONSOLIDATION"}:
        bias = "CONSOLIDATION"
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"
    return bias, confidence


# ── FVG scoring ───────────────────────────────────────────────────────────────

def fvg_weight(timeframe: str) -> int:
    return {"W1": 4, "D1": 3, "H4": 2, "H1": 1}.get(timeframe.upper(), 1)


def fvg_zone_valid(fvg: FVG, bias: str, fib_50: float | None) -> bool:
    if fib_50 is None:
        return True
    if bias == "BULLISH":
        return fvg.top <= fib_50
    if bias == "BEARISH":
        return fvg.bottom >= fib_50
    return False


def fvg_overlap(left: FVG, right: FVG) -> bool:
    return max(left.bottom, right.bottom) <= min(left.top, right.top)


def score_fvgs(bias: str, fib_50: float | None) -> list[tuple[FVG, int]]:
    """
    Score all active FVGs. Key improvements:
    - Displacement bonus: +2 if middle candle had displacement (institutional move)
    - Overlap bonus: FVGs that overlap with others on different timeframes score higher
    - BPR bonus: +3 if this FVG overlaps with an opposite FVG (Balanced Price Range)
    - Recency: more recent FVGs preferred (tie-breaker)
    - H1 FVGs now included (previously excluded) to generate more signals
    """
    all_fvgs: list[FVG] = []

    # Include H1 FVGs for more signal opportunities
    for tf in ("W1", "D1", "H4", "H1"):
        for fvg in unfilled_fvgs(tf, include_mitigated=True):
            if fvg.direction == bias and fvg_zone_valid(fvg, bias, fib_50):
                all_fvgs.append(fvg)

    # Also collect opposite-direction FVGs for BPR detection
    opposite_fvgs: list[FVG] = []
    opposite_bias = "BEARISH" if bias == "BULLISH" else "BULLISH"
    for tf in ("D1", "H4"):
        for fvg in unfilled_fvgs(tf, include_mitigated=True):
            if fvg.direction == opposite_bias:
                opposite_fvgs.append(fvg)

    scored: list[tuple[FVG, int]] = []
    for fvg in all_fvgs:
        score = fvg_weight(fvg.timeframe)

        # Overlap bonus (multi-timeframe confluence)
        for other in all_fvgs:
            if other is not fvg and fvg_overlap(fvg, other):
                score += fvg_weight(other.timeframe)

        # Displacement bonus
        if fvg.displacement:
            score += 2

        # BPR bonus — overlaps with opposite direction FVG
        for opp in opposite_fvgs:
            if fvg_overlap(fvg, opp):
                score += 3
                break

        # Status bonus — OPEN is better than MITIGATED
        if fvg.status == "OPEN":
            score += 1

        scored.append((fvg, score))

    return sorted(
        scored,
        key=lambda item: (
            item[1],
            -abs(item[0].mid - fib_50) if fib_50 is not None else 0,
            item[0].formed_time,
        ),
        reverse=True,
    )


def select_scored_primary_fvg(
    bias: str,
    fib_50: float | None,
) -> tuple[FVG | None, int, tuple]:
    scored = score_fvgs(bias, fib_50)
    candidates = []
    for fvg, score in scored:
        row = fvg_to_dict(fvg) or {}
        row["score"] = score
        candidates.append(row)
    if not scored:
        return None, 0, tuple(candidates)
    primary, score = scored[0]
    return primary, score, tuple(candidates)


def select_primary_fvg(bias: str | None = None) -> FVG | None:
    bias = bias.upper() if bias else None
    # Try timeframes from highest to lowest
    for tf in ("W1", "D1", "H4", "H1"):
        fvgs = unfilled_fvgs(tf, include_mitigated=True)
        if bias in {"BULLISH", "BEARISH"}:
            fvgs = [f for f in fvgs if f.direction == bias]
        if fvgs:
            return sorted(fvgs, key=lambda f: f.formed_time, reverse=True)[0]
    return None


# ── Phase 1 engine ────────────────────────────────────────────────────────────

def weekly_bias_engine(
    ai_bias: str | None = None,
    ai_confidence: str | None = None,
) -> Phase1Output:
    if ai_bias is None or ai_confidence is None:
        configured_bias, configured_confidence = configured_ai_bias()
        ai_bias       = configured_bias       if ai_bias       is None else ai_bias
        ai_confidence = configured_confidence if ai_confidence is None else ai_confidence

    tech    = technical_bias()
    shifted, shift_direction = detect_structure_shift(tech)
    daily   = closed_candles(read_candles("D1"), "D1")
    fib_high, fib_low, fib_50, zone = fibonacci_zone(daily)
    final_bias, conviction = combine_bias(tech, ai_bias, ai_confidence, zone)
    no_trade_week = shifted or final_bias == "NO_TRADE" or conviction == "NONE"

    primary, primary_score, candidates = (
        (None, 0, ()) if no_trade_week
        else select_scored_primary_fvg(final_bias, fib_50)
    )

    if primary is None and not no_trade_week:
        final_bias    = "NO_TRADE"
        conviction    = "NONE"
        no_trade_week = True

    # Build liquidity levels for the output
    d1_candles = closed_candles(read_candles("D1"), "D1")
    h4_candles = closed_candles(read_candles("H4"), "H4")
    liq_levels = (
        build_liquidity_levels(d1_candles, "D1") +
        build_liquidity_levels(h4_candles, "H4")
    ) if d1_candles else []

    # BPR detection
    bull_fvgs = [f for f in (unfilled_fvgs("D1") + unfilled_fvgs("H4")) if f.direction == "BULLISH"]
    bear_fvgs = [f for f in (unfilled_fvgs("D1") + unfilled_fvgs("H4")) if f.direction == "BEARISH"]
    bprs = detect_balanced_price_ranges(bull_fvgs, bear_fvgs)

    return Phase1Output(
        final_bias=final_bias,
        conviction=conviction,
        technical_bias=tech,
        ai_bias=ai_bias.upper(),
        ai_confidence=ai_confidence.upper(),
        fib_high=fib_high,
        fib_low=fib_low,
        fib_50=fib_50,
        zone=zone,
        primary_fvg=primary,
        no_trade_week=no_trade_week,
        structure_shift=shifted,
        structure_shift_direction=shift_direction,
        primary_fvg_score=primary_score,
        fvg_candidates=candidates,
        liquidity_levels=tuple(liq_levels),
        balanced_price_ranges=tuple(bprs),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def fvg_to_dict(fvg: FVG | None) -> dict | None:
    if fvg is None:
        return None
    return {
        "fvg_top": fvg.top, "fvg_bottom": fvg.bottom, "fvg_mid": fvg.mid,
        "fvg_tf": fvg.timeframe, "direction": fvg.direction,
        "status": fvg.status, "formed_time": fvg.formed_time,
        "gap_size": fvg.gap_size, "displacement": fvg.displacement,
        "inverted": fvg.inverted,
    }


def fvg_summary() -> list[dict]:
    rows = []
    for tf in PRIMARY_TIMEFRAMES:
        fvgs = unfilled_fvgs(tf, include_mitigated=True)
        rows.append({
            "timeframe": tf,
            "file": str(history_path(tf)),
            "active_fvgs": len(fvgs),
            "displaced_fvgs": sum(1 for f in fvgs if f.displacement),
            "latest_direction": fvgs[-1].direction if fvgs else "",
            "latest_status": fvgs[-1].status if fvgs else "",
        })
    return rows


def format_fvg(fvg: FVG) -> str:
    disp = " [DISPLACED]" if fvg.displacement else ""
    bpr  = " [BPR]" if fvg.inverted else ""
    return (
        f"{fvg.timeframe} {fvg.direction} {fvg.bottom:.2f}-{fvg.top:.2f} "
        f"CE={fvg.mid:.2f} gap={fvg.gap_size:.1f} {fvg.status}{disp}{bpr} "
        f"formed={fvg.formed_time}"
    )


def main() -> int:
    phase1 = weekly_bias_engine()
    print("Phase 1 Weekly Bias Engine")
    print(f"  final_bias={phase1.final_bias} conviction={phase1.conviction}")
    print(f"  technical_bias={phase1.technical_bias} zone={phase1.zone}")
    print(f"  fib_high={phase1.fib_high} fib_low={phase1.fib_low} fib_50={phase1.fib_50}")
    print(f"  structure_shift={phase1.structure_shift} direction={phase1.structure_shift_direction}")
    print(f"  no_trade_week={phase1.no_trade_week}")
    print(f"  primary_fvg_score={phase1.primary_fvg_score}")
    print(f"  liquidity_levels={len(phase1.liquidity_levels)}")
    print(f"  balanced_price_ranges={len(phase1.balanced_price_ranges)}")
    print()

    print("Active FVGs")
    for row in fvg_summary():
        print(f"  {row['timeframe']}: active={row['active_fvgs']} displaced={row['displaced_fvgs']}")
    if phase1.primary_fvg:
        print(f"\n  Primary: {format_fvg(phase1.primary_fvg)}")

    if phase1.balanced_price_ranges:
        print("\nBalanced Price Ranges (BPR — strongest zones)")
        for bpr in phase1.balanced_price_ranges[:3]:
            print(f"  {bpr.bottom:.2f} - {bpr.top:.2f}  CE={bpr.mid:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())