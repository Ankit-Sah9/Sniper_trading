from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

from FVGs import get_float, read_candles, weekly_bias_engine, unfilled_fvgs


PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = PROJECT_ROOT / "data" / "trade_state.json"


@dataclass(frozen=True)
class TargetCandidate:
    target_price: float
    target_type: str
    source: str
    rr_ratio: float
    confluence_score: int


@dataclass(frozen=True)
class Phase3Output:
    target_valid: bool
    rr_valid: bool
    target_price: float | None
    target_type: str
    rr_ratio: float
    confluence_score: int
    trade_idea_count: int
    weekly_ideas_exhausted: bool
    block_reason: str
    candidates: list[TargetCandidate]
    target_reached: bool = False
    target_invalidated: bool = False


def load_trade_state() -> dict[str, object]:
    if not STATE_PATH.exists():
        return {"trade_idea_count": 0, "weekly_ideas_exhausted": False}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"trade_idea_count": 0, "weekly_ideas_exhausted": False}


def save_trade_state(state: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def state_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def mark_trade_idea_used() -> dict[str, object]:
    state = load_trade_state()
    count = state_int(state.get("trade_idea_count", 0)) + 1
    state["trade_idea_count"] = count
    state["weekly_ideas_exhausted"] = count >= 3
    save_trade_state(state)
    return state


def rr_ratio(entry_estimate: float, sl_estimate: float, target: float) -> float:
    risk = abs(entry_estimate - sl_estimate)
    reward = abs(target - entry_estimate)
    if risk <= 0:
        return 0.0
    return reward / risk


def previous_high_low(timeframe: str) -> tuple[float | None, float | None]:
    candles = read_candles(timeframe)
    if len(candles) < 2:
        return None, None
    previous = candles[-2]
    return previous.high, previous.low


def target_reached(final_bias: str, target_price: float) -> bool:
    candles = read_candles("H1")
    if not candles:
        return False
    latest = candles[-1]
    if final_bias == "BULLISH":
        return latest.high >= target_price
    if final_bias == "BEARISH":
        return latest.low <= target_price
    return False


def target_invalidated(final_bias: str, target_price: float, entry_estimate: float) -> bool:
    phase1 = weekly_bias_engine()
    if phase1.structure_shift or phase1.final_bias not in {final_bias, "NO_TRADE"}:
        return True
    candles = read_candles("H1")
    if not candles:
        return False
    latest = candles[-1]
    if final_bias == "BULLISH":
        return latest.close < entry_estimate or latest.close > target_price
    if final_bias == "BEARISH":
        return latest.close > entry_estimate or latest.close < target_price
    return True


def candidate_targets(phase2_output, symbol: str = "GOLD") -> list[TargetCandidate]:
    phase1 = weekly_bias_engine()
    fvg = phase1.primary_fvg
    if fvg is None:
        return []

    max_sl = get_float("GOLD_MAX_SL_DISTANCE", 30.0)
    entry_estimate = fvg.top if phase1.final_bias == "BULLISH" else fvg.bottom
    sl_estimate = entry_estimate - max_sl if phase1.final_bias == "BULLISH" else entry_estimate + max_sl

    raw: list[tuple[float, str, str, int]] = []
    if phase1.final_bias == "BULLISH":
        raw.extend((item.level, item.kind, item.source, 4 + item.touches) for item in phase2_output.eq_highs)
        week_high, _ = previous_high_low("W1")
        day_high, _ = previous_high_low("D1")
        if week_high is not None:
            raw.append((week_high, "PREV_WEEK_HIGH", "W1", 5))
        if day_high is not None:
            raw.append((day_high, "PREV_DAY_HIGH", "D1", 3))
        raw.extend((item.bottom, "OPPOSING_FVG", item.timeframe, 2) for item in unfilled_fvgs("H4") if item.direction == "BEARISH")
    else:
        raw.extend((item.level, item.kind, item.source, 4 + item.touches) for item in phase2_output.eq_lows)
        _, week_low = previous_high_low("W1")
        _, day_low = previous_high_low("D1")
        if week_low is not None:
            raw.append((week_low, "PREV_WEEK_LOW", "W1", 5))
        if day_low is not None:
            raw.append((day_low, "PREV_DAY_LOW", "D1", 3))
        raw.extend((item.top, "OPPOSING_FVG", item.timeframe, 2) for item in unfilled_fvgs("H4") if item.direction == "BULLISH")

    candidates: list[TargetCandidate] = []
    for price, target_type, source, score in raw:
        if phase1.final_bias == "BULLISH" and price <= entry_estimate:
            continue
        if phase1.final_bias == "BEARISH" and price >= entry_estimate:
            continue
        rr = rr_ratio(entry_estimate, sl_estimate, price)
        if rr >= 2.0:
            candidates.append(TargetCandidate(price, target_type, source, rr, score))
    return sorted(candidates, key=lambda item: (item.confluence_score, item.rr_ratio), reverse=True)


def identify_target(phase2_output=None, symbol: str = "GOLD") -> Phase3Output:
    if phase2_output is None:
        import importlib.util

        module_path = PROJECT_ROOT / "Manipulation Zone.py"
        spec = importlib.util.spec_from_file_location("manipulation_zone", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load Manipulation Zone.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        phase2_output = module.evaluate_manipulation_zone(symbol)

    state = load_trade_state()
    count = state_int(state.get("trade_idea_count", 0))
    exhausted = bool(state.get("weekly_ideas_exhausted", False)) or count >= 3
    if exhausted:
        return Phase3Output(False, False, None, "", 0.0, 0, count, True, "Weekly trade ideas exhausted", [])
    if not phase2_output.manipulation_zone_active or phase2_output.manipulation_zone_invalidated:
        return Phase3Output(False, False, None, "", 0.0, 0, count, exhausted, "Manipulation zone is not valid", [])

    candidates = candidate_targets(phase2_output, symbol)
    if not candidates:
        return Phase3Output(False, False, None, "", 0.0, 0, count, exhausted, "No target passed 1:2 R:R", [])
    best = candidates[0]
    phase1 = weekly_bias_engine()
    fvg = phase1.primary_fvg
    entry_estimate = fvg.top if fvg is not None and phase1.final_bias == "BULLISH" else fvg.bottom if fvg is not None else best.target_price
    reached = target_reached(phase1.final_bias, best.target_price)
    invalidated = target_invalidated(phase1.final_bias, best.target_price, entry_estimate)
    if reached:
        return Phase3Output(False, False, best.target_price, best.target_type, best.rr_ratio, best.confluence_score, count, exhausted, "Target already reached", candidates, True, False)
    if invalidated:
        state = mark_trade_idea_used()
        new_count = state_int(state.get("trade_idea_count", count))
        return Phase3Output(False, False, best.target_price, best.target_type, best.rr_ratio, best.confluence_score, new_count, new_count >= 3, "Target invalidated; re-run Phase 1", candidates, False, True)
    return Phase3Output(True, True, best.target_price, best.target_type, best.rr_ratio, best.confluence_score, count, exhausted, "Target selected", candidates, False, False)


def main() -> int:
    output = identify_target()
    print("Phase 3 Target Identification")
    print(f"target_valid={output.target_valid} rr_valid={output.rr_valid}")
    print(f"target_price={output.target_price} target_type={output.target_type} rr={output.rr_ratio:.2f}")
    print(f"score={output.confluence_score} trade_idea_count={output.trade_idea_count}")
    print(f"weekly_ideas_exhausted={output.weekly_ideas_exhausted}")
    print(f"target_reached={output.target_reached} target_invalidated={output.target_invalidated}")
    print(f"reason={output.block_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
