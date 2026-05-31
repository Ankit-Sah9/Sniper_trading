from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from FVGs import classify_trend, get_float, read_candles, weekly_bias_engine, unfilled_fvgs


@dataclass(frozen=True)
class TTPSScore:
    htf_bias: int
    entry_confluence: int
    session_quality: int
    risk_quality: int
    total: int
    grade: str
    action: str
    position_multiplier: float
    proceed: bool
    reason: str


def grade_total(total: int) -> tuple[str, str, float, bool]:
    if total >= 18:
        return "A", "Elite setup - full position size", 1.0, True
    if total >= 14:
        return "B", "Good setup - standard position size", 1.0, True
    if total >= 10:
        return "C", "Average setup - half position size", 0.5, True
    return "D", "Weak setup - no trade, mandatory skip", 0.0, False


def open_respected(final_bias: str) -> bool:
    weekly = read_candles("W1")
    daily = read_candles("D1")
    if not weekly or not daily:
        return False
    latest = daily[-1].close
    if final_bias == "BULLISH":
        return latest >= weekly[-1].open and latest >= daily[-1].open
    if final_bias == "BEARISH":
        return latest <= weekly[-1].open and latest <= daily[-1].open
    return False


def htf_bias_score(phase1: Any) -> int:
    final_bias = str(phase1.final_bias)
    if final_bias not in {"BULLISH", "BEARISH"}:
        return 0

    score = 0
    if classify_trend(read_candles("W1")) == final_bias:
        score += 2
    if classify_trend(read_candles("D1")) == final_bias:
        score += 1
    if classify_trend(read_candles("H4")) == final_bias:
        score += 1
    if open_respected(final_bias):
        score += 1
    return min(score, 5)


def entry_confluence_score(phase4: Any) -> int:
    score = 0
    if bool(phase4.entry_signal):
        score += 1
    if bool(phase4.swept_level is not None):
        score += 1
    if bool(phase4.fvg_at_entry_1h):
        score += 1
    if bool(phase4.fvg_at_entry_4h):
        score += 1
    if bool(phase4.both_opens_confirm or phase4.one_open_confirms):
        score += 1
    return min(score, 5)


def session_quality_score(phase2: Any) -> int:
    score = 0
    if bool(phase2.manipulation_zone_active):
        score += 1
    if not bool(phase2.manipulation_zone_invalidated):
        score += 1
    score += min(int(getattr(phase2, "session_game_score", 0)), 2)
    if bool(phase2.ssl_swept or phase2.bsl_swept):
        score += 1
    return min(score, 5)


def no_major_htf_level_between(final_bias: str, entry_price: float, tp_price: float | None) -> bool:
    if tp_price is None:
        return False
    low = min(entry_price, tp_price)
    high = max(entry_price, tp_price)
    opposing = "BEARISH" if final_bias == "BULLISH" else "BULLISH"
    for timeframe in ("W1", "D1"):
        for fvg in unfilled_fvgs(timeframe):
            if fvg.direction == opposing and low <= fvg.mid <= high:
                return False
    return True


def risk_quality_score(
    final_bias: str,
    entry_price: float | None,
    sl_price: float | None,
    tp_price: float | None,
    sl_distance: float,
    actual_rr: float,
    phase3: Any,
) -> int:
    score = 0
    if actual_rr >= 2.0:
        score += 1
    if actual_rr >= 3.0:
        score += 1
    if sl_price is not None:
        score += 1
    if 0 < sl_distance <= get_float("GOLD_MAX_SL_DISTANCE", 30.0):
        score += 1
    if bool(getattr(phase3, "target_valid", False)):
        score += 1
    if entry_price is not None and no_major_htf_level_between(final_bias, entry_price, tp_price):
        score += 1
    return min(score, 5)


def calculate_ttps(
    phase1: Any,
    phase2: Any,
    phase3: Any,
    phase4: Any,
    sl_price: float | None,
    tp_price: float | None,
    sl_distance: float,
    actual_rr: float,
) -> TTPSScore:
    htf = htf_bias_score(phase1)
    entry = entry_confluence_score(phase4)
    session = session_quality_score(phase2)
    risk = risk_quality_score(
        str(phase1.final_bias),
        phase4.entry_price,
        sl_price,
        tp_price,
        sl_distance,
        actual_rr,
        phase3,
    )
    total = htf + entry + session + risk
    grade, action, multiplier, proceed = grade_total(total)

    reason = action
    if htf < 3:
        proceed = False
        multiplier = 0.0
        reason = "HTF Bias score below 3/5 - no trade"
    elif risk < 3:
        proceed = False
        multiplier = 0.0
        reason = "Risk Quality score below 3/5 - no trade"
    elif total < 10:
        proceed = False
        multiplier = 0.0
        reason = action

    return TTPSScore(htf, entry, session, risk, total, grade, action, multiplier, proceed, reason)


def main() -> int:
    phase1 = weekly_bias_engine()
    print("TTPS Master Score")
    print(f"HTF Bias: {htf_bias_score(phase1)}/5")
    print("Run Phase 5 for full TTPS after entry, SL, and TP are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
