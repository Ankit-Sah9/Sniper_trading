from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta, timezone
from zoneinfo import ZoneInfo
from zoneinfo._common import ZoneInfoNotFoundError

from FVGs import Candle, SwingPoint, detect_swings, get_float, read_candles, weekly_bias_engine


try:
    NY_TZ = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:
    NY_TZ = timezone(timedelta(hours=-4), "New_York_Fallback")
SESSION_WINDOWS = {
    "ASIAN": (time(20, 0), time(23, 0)),
    "LONDON": (time(2, 0), time(5, 0)),
    "NY": (time(7, 0), time(11, 0)),
}


@dataclass(frozen=True)
class LiquidityCluster:
    level: float
    touches: int
    kind: str
    source: str


@dataclass(frozen=True)
class SessionRange:
    high: float
    low: float
    range: float
    label: str


@dataclass(frozen=True)
class Phase2Output:
    manipulation_zone_active: bool
    manipulation_zone_invalidated: bool
    session_game_score: int
    ssl_swept: bool
    bsl_swept: bool
    ssl_swept_level: float | None
    bsl_swept_level: float | None
    eq_highs: list[LiquidityCluster]
    eq_lows: list[LiquidityCluster]
    block_reason: str


def pip_size(symbol: str = "GOLD") -> float:
    symbol = symbol.upper()
    return 0.0001 if "EURUSD" in symbol else 0.10


def equal_tolerance(symbol: str = "GOLD") -> float:
    return pip_size(symbol) * 10


def min_sweep_distance(symbol: str = "GOLD") -> float:
    configured = get_float("GOLD_MIN_SWEEP_DISTANCE", 5.0)
    return configured if "EURUSD" not in symbol.upper() else pip_size(symbol) * 5


def cluster_swings(swings: list[SwingPoint], kind: str, source: str, tolerance: float) -> list[LiquidityCluster]:
    clusters: list[list[SwingPoint]] = []
    for swing in sorted(swings, key=lambda item: item.price):
        for cluster in clusters:
            average = sum(item.price for item in cluster) / len(cluster)
            if abs(swing.price - average) <= tolerance:
                cluster.append(swing)
                break
        else:
            clusters.append([swing])

    result: list[LiquidityCluster] = []
    for cluster in clusters:
        if len(cluster) >= 2:
            result.append(
                LiquidityCluster(
                    level=sum(item.price for item in cluster) / len(cluster),
                    touches=len(cluster),
                    kind=kind,
                    source=source,
                )
            )
    return sorted(result, key=lambda item: item.touches, reverse=True)


def detect_equal_highs_lows(symbol: str = "GOLD") -> tuple[list[LiquidityCluster], list[LiquidityCluster]]:
    tolerance = equal_tolerance(symbol)
    highs: list[LiquidityCluster] = []
    lows: list[LiquidityCluster] = []
    for timeframe in ("D1", "H4", "H1"):
        swing_highs, swing_lows = detect_swings(read_candles(timeframe))
        highs.extend(cluster_swings(swing_highs, "EQ_HIGH", timeframe, tolerance))
        lows.extend(cluster_swings(swing_lows, "EQ_LOW", timeframe, tolerance))
    return highs, lows


def session_range(candles: list[Candle], session: str) -> SessionRange | None:
    start, end = SESSION_WINDOWS[session]
    selected: list[Candle] = []
    for candle in candles:
        from FVGs import parse_candle_time

        opened_at = parse_candle_time(candle.time)
        if opened_at is None:
            continue
        local_time = opened_at.astimezone(NY_TZ).time()
        if start <= local_time <= end:
            selected.append(candle)
    if not selected:
        return None
    high = max(candle.high for candle in selected)
    low = min(candle.low for candle in selected)
    return SessionRange(high=high, low=low, range=high - low, label=session)


def calculate_session_game_score(final_bias: str, h1_candles: list[Candle]) -> int:
    asian = session_range(h1_candles, "ASIAN")
    london = session_range(h1_candles, "LONDON")
    if asian is None or london is None:
        return 0

    bias = final_bias.upper()
    if bias == "BULLISH":
        if london.low < asian.low and london.range <= max(asian.range * 1.5, asian.range):
            return 2
        if london.low < asian.low:
            return 1
    if bias == "BEARISH":
        if london.high > asian.high and london.range <= max(asian.range * 1.5, asian.range):
            return 2
        if london.high > asian.high:
            return 1
    return 0


def confirm_liquidity_sweep(candle: Candle, clusters: list[LiquidityCluster], side: str, symbol: str = "GOLD") -> tuple[bool, float | None]:
    distance = min_sweep_distance(symbol)
    for cluster in clusters:
        if side == "SELL_SIDE":
            if cluster.level - candle.low >= distance and candle.close >= cluster.level:
                return True, cluster.level
        if side == "BUY_SIDE":
            if candle.high - cluster.level >= distance and candle.close <= cluster.level:
                return True, cluster.level
    return False, None


def evaluate_manipulation_zone(symbol: str = "GOLD") -> Phase2Output:
    phase1 = weekly_bias_engine()
    if phase1.final_bias not in {"BULLISH", "BEARISH"} or phase1.no_trade_week or phase1.primary_fvg is None:
        return Phase2Output(False, False, 0, False, False, None, None, [], [], "Phase 1 is not tradable")

    h1 = read_candles("H1")
    latest = h1[-1] if h1 else None
    eq_highs, eq_lows = detect_equal_highs_lows(symbol)
    score = calculate_session_game_score(phase1.final_bias, h1)
    if latest is None:
        return Phase2Output(False, False, score, False, False, None, None, eq_highs, eq_lows, "No H1 candles")

    fvg = phase1.primary_fvg
    if phase1.final_bias == "BULLISH":
        active = latest.low <= fvg.top
        invalidated = latest.close < fvg.bottom
        ssl_swept, ssl_level = confirm_liquidity_sweep(latest, eq_lows, "SELL_SIDE", symbol)
        bsl_swept, bsl_level = False, None
    else:
        active = latest.high >= fvg.bottom
        invalidated = latest.close > fvg.top
        ssl_swept, ssl_level = False, None
        bsl_swept, bsl_level = confirm_liquidity_sweep(latest, eq_highs, "BUY_SIDE", symbol)

    reason = "Manipulation zone active" if active else "Waiting for price to enter primary FVG"
    if invalidated:
        reason = "Primary FVG invalidated"

    return Phase2Output(active, invalidated, score, ssl_swept, bsl_swept, ssl_level, bsl_level, eq_highs, eq_lows, reason)


def main() -> int:
    output = evaluate_manipulation_zone()
    print("Phase 2 Manipulation Zone")
    print(f"active={output.manipulation_zone_active} invalidated={output.manipulation_zone_invalidated}")
    print(f"session_game_score={output.session_game_score}")
    print(f"ssl_swept={output.ssl_swept} ssl_level={output.ssl_swept_level}")
    print(f"bsl_swept={output.bsl_swept} bsl_level={output.bsl_swept_level}")
    print(f"eq_highs={len(output.eq_highs)} eq_lows={len(output.eq_lows)}")
    print(f"reason={output.block_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
