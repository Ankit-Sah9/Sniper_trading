from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4


Command = tuple[str, Callable[[], int]]

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
TRADE_LOG_PATH = PROJECT_ROOT / "data" / "trade_log.csv"
SIGNAL_STATE_PATH = PROJECT_ROOT / "data" / "last_signal.json"

DEFAULT_ENV = {
    "TRADING_SYMBOL": "GOLD",
    "DEMO_ONLY": "true",
    "LIVE_TRADING": "false",
    "PAPER_TRADING": "true",
    "TRADE_EXECUTION_ENABLED": "false",
    "RISK_PER_TRADE": "0.01",
    "MAX_WEEKLY_LOSS": "0.03",
    "MAX_MONTHLY_DRAWDOWN": "0.10",
    "MAX_TRADES_PER_WEEK": "3",
    "MAX_OPEN_TRADES": "1",
    "MAX_SPREAD_PRICE": "3.0",
    "GOLD_FVG_MIN_GAP": "5.0",
    "GOLD_MIN_SWEEP_DISTANCE": "5.0",
    "GOLD_MIN_SL_DISTANCE": "20.0",
    "GOLD_MAX_SL_DISTANCE": "30.0",
    "NEWS_FILTER_ENABLED": "true",
    "NEWS_CSV": "data/news_events_sample.csv",
    "SCAN_INTERVAL_SECONDS": "60",
}


def load_env() -> dict[str, str]:
    values = dict(DEFAULT_ENV)
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def run_env() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    for key, value in sorted(load_env().items()):
        safe_value = "***" if "PASSWORD" in key or "TOKEN" in key else value
        print(f"{key}: {safe_value}")
    return 0


def run_risk() -> int:
    from risk_manager import main

    return main()


def run_filter() -> int:
    from filter import main

    return main()


def print_news_status(limit: int = 5) -> None:
    from filter import classify_event, is_blocked, news_events_path

    blocked, reason = is_blocked()
    print("=== News Filter ===")
    print(f"Blocked: {blocked}")
    print(f"Reason: {reason}")
    if "NEWS_FILTER_ENABLED=false" in reason:
        print("Action: set NEWS_FILTER_ENABLED=true in .env to make news block trades.")

    path = news_events_path()
    if not path.exists():
        print(f"Calendar: missing ({path})")
        print()
        return

    rows: list[tuple[datetime, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("event", "")).strip()
            raw_time = str(row.get("time_ny", "")).strip()
            try:
                event_time = datetime.fromisoformat(raw_time)
            except ValueError:
                continue
            event_type = classify_event(name)
            if event_type != "IGNORE":
                rows.append((event_time, name, event_type))

    now = datetime.now()
    upcoming = [(event_time, name, event_type) for event_time, name, event_type in rows if event_time >= now]
    upcoming.sort(key=lambda item: item[0])
    if not upcoming:
        print(f"Calendar: {path}")
        print("Upcoming high-impact news: none loaded")
        print()
        return

    print(f"Calendar: {path}")
    print("Upcoming high-impact news:")
    for event_time, name, event_type in upcoming[:limit]:
        print(f"- {event_time:%Y-%m-%d %H:%M} NY | {event_type:<11} | {name}")
    print()


def run_train() -> int:
    from train import main

    return main()


def run_dashboard() -> int:
    from dashboard import main

    return main()


def run_history() -> int:
    from history import main

    return main()


def run_fvgs() -> int:
    from FVGs import main

    return main()


def load_python_file(filename: str, module_name: str):
    path = PROJECT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_phase2() -> int:
    return load_python_file("Manipulation Zone.py", "manipulation_zone").main()


def run_phase3() -> int:
    from targetIdentification import main

    return main()


@dataclass(frozen=True)
class Phase4Output:
    entry_signal: bool
    entry_price: float | None
    entry_time: str
    swept_level: float | None
    swept_level_type: str
    confluence_score: int
    setup_quality: str
    fvg_at_entry_1h: bool
    fvg_at_entry_4h: bool
    both_opens_confirm: bool
    one_open_confirms: bool
    no_open_confirms: bool
    block_reason: str


@dataclass(frozen=True)
class Phase5Output:
    trade_id: str
    sl_price: float | None
    tp_price: float | None
    sl_distance: float
    actual_rr: float
    sl_valid: bool
    tp_valid: bool
    trade_open: bool
    trade_result: str
    r_gained: float
    mandatory_recheck: bool
    recommended_recheck: bool
    block_reason: str
    close_time: str = ""
    close_price: float | None = None
    pips_gained: float = 0.0
    ttps_total: int = 0
    ttps_grade: str = ""


def open_filters_confirm(final_bias: str, entry_price: float) -> tuple[bool, bool, bool]:
    from FVGs import read_candles

    weekly = read_candles("W1")
    daily = read_candles("D1")
    weekly_ok = bool(weekly) and ((final_bias == "BULLISH" and entry_price >= weekly[-1].open) or (final_bias == "BEARISH" and entry_price <= weekly[-1].open))
    daily_ok = bool(daily) and ((final_bias == "BULLISH" and entry_price >= daily[-1].open) or (final_bias == "BEARISH" and entry_price <= daily[-1].open))
    return weekly_ok and daily_ok, weekly_ok or daily_ok, not weekly_ok and not daily_ok


def fvg_at_price(timeframe: str, price: float) -> bool:
    from FVGs import unfilled_fvgs

    return any(fvg.overlaps_price(price) for fvg in unfilled_fvgs(timeframe))


def evaluate_entry_conditions(symbol: str = "GOLD") -> Phase4Output:
    from FVGs import read_candles, weekly_bias_engine
    from targetIdentification import identify_target

    phase1 = weekly_bias_engine()
    phase2 = load_python_file("Manipulation Zone.py", "manipulation_zone_phase4").evaluate_manipulation_zone(symbol)
    phase3 = identify_target(phase2, symbol)
    h1 = read_candles("H1")
    latest = h1[-1] if h1 else None

    hard_checks = [
        (phase1.final_bias in {"BULLISH", "BEARISH"}, "Phase 1 has no tradable bias"),
        (phase1.conviction in {"HIGH", "MEDIUM"}, "Phase 1 conviction is too low"),
        (not phase1.no_trade_week, "No-trade week is active"),
        (phase2.manipulation_zone_active, "Manipulation zone is not active"),
        (not phase2.manipulation_zone_invalidated, "Manipulation zone invalidated"),
        (phase3.target_valid and phase3.rr_valid, "No valid Phase 3 target"),
        (not phase3.weekly_ideas_exhausted, "Weekly trade ideas exhausted"),
    ]
    for passed, reason in hard_checks:
        if not passed:
            return Phase4Output(False, None, "", None, "", 0, "WEAK", False, False, False, False, True, reason)
    if latest is None:
        return Phase4Output(False, None, "", None, "", 0, "WEAK", False, False, False, False, True, "No H1 candle for entry check")

    if phase1.final_bias == "BULLISH":
        swept = phase2.ssl_swept
        swept_level = phase2.ssl_swept_level
        swept_type = "SELL_SIDE"
    else:
        swept = phase2.bsl_swept
        swept_level = phase2.bsl_swept_level
        swept_type = "BUY_SIDE"

    entry_price = latest.close
    both_opens, one_open, no_open = open_filters_confirm(phase1.final_bias, entry_price)
    fvg_1h = fvg_at_price("H1", entry_price)
    fvg_4h = fvg_at_price("H4", entry_price)

    score = 0
    score += 3 if swept else 0
    score += min(phase2.session_game_score, 2)
    score += 2 if both_opens else 1 if one_open else 0
    score += 1 if fvg_1h else 0
    score += 2 if fvg_4h else 0
    score += 1 if phase1.ai_bias == phase1.final_bias else 0

    if score >= 8:
        quality = "A"
    elif score >= 6:
        quality = "B"
    elif score >= 4:
        quality = "C"
    else:
        quality = "WEAK"

    entry_signal = swept and score >= 4
    reason = "Entry signal confirmed" if entry_signal else "Waiting for sweep/confluence"
    return Phase4Output(entry_signal, entry_price, latest.time, swept_level, swept_type, score, quality, fvg_1h, fvg_4h, both_opens, one_open, no_open, reason)


def run_phase4() -> int:
    output = evaluate_entry_conditions()
    print("Phase 4 Entry Conditions")
    print(f"entry_signal={output.entry_signal} entry_price={output.entry_price} entry_time={output.entry_time}")
    print(f"swept_level={output.swept_level} swept_type={output.swept_level_type}")
    print(f"score={output.confluence_score} setup_quality={output.setup_quality}")
    print(f"fvg_1h={output.fvg_at_entry_1h} fvg_4h={output.fvg_at_entry_4h}")
    print(f"reason={output.block_reason}")
    return 0


def find_stop_loss(final_bias: str, entry_price: float) -> tuple[float | None, float]:
    from FVGs import detect_swings, get_float, read_candles

    min_sl = get_float("GOLD_MIN_SL_DISTANCE", 20.0)
    max_sl = get_float("GOLD_MAX_SL_DISTANCE", 30.0)
    highs, lows = detect_swings(read_candles("H1"))
    swings = lows if final_bias == "BULLISH" else highs
    ordered = sorted(swings, key=lambda item: item.time, reverse=True)
    for swing in ordered:
        distance = abs(entry_price - swing.price)
        if min_sl <= distance <= max_sl:
            return swing.price, distance
    sl_price = entry_price - min_sl if final_bias == "BULLISH" else entry_price + min_sl
    return sl_price, min_sl


def calculate_take_profit(final_bias: str, entry_price: float, sl_price: float, target_price: float) -> tuple[float | None, float]:
    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None, 0.0
    early = 10.0
    late = 15.0
    candidates = []
    if final_bias == "BULLISH":
        candidates = [target_price - early, target_price, target_price + late]
    else:
        candidates = [target_price + early, target_price, target_price - late]
    for tp_price in candidates:
        rr = abs(tp_price - entry_price) / risk
        if rr >= 2.0:
            return tp_price, rr
    return None, 0.0


def pip_size(symbol: str) -> float:
    symbol = symbol.upper()
    return 0.0001 if "EURUSD" in symbol else 0.10


def simulate_trade_close(final_bias: str, sl_price: float, tp_price: float, entry_time: str) -> tuple[str, float, float | None, str]:
    from FVGs import read_candles

    candles = [candle for candle in read_candles("H1") if candle.time > entry_time]
    for candle in candles:
        if final_bias == "BULLISH":
            if candle.low <= sl_price:
                return "LOSS", -1.0, sl_price, candle.time
            if candle.high >= tp_price:
                return "WIN", 0.0, tp_price, candle.time
        else:
            if candle.high >= sl_price:
                return "LOSS", -1.0, sl_price, candle.time
            if candle.low <= tp_price:
                return "WIN", 0.0, tp_price, candle.time
    return "OPEN", 0.0, None, ""


def place_mt5_order(symbol: str, final_bias: str, lot_size: float, sl_price: float, tp_price: float) -> tuple[bool, str]:
    env = load_env()
    execution_enabled = env.get("TRADE_EXECUTION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    paper_trading = env.get("PAPER_TRADING", "true").strip().lower() in {"1", "true", "yes", "on"}
    if paper_trading:
        return True, "PAPER_ORDER"
    if not execution_enabled:
        return False, "Trade execution disabled. Set TRADE_EXECUTION_ENABLED=true and PAPER_TRADING=false to send MT5 orders."

    import MetaTrader5

    mt5 = cast(Any, MetaTrader5)

    if not mt5.initialize(path=env.get("MT5_PATH") or None):
        return False, f"MT5 initialize failed: {mt5.last_error()}"
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, f"No tick data for {symbol}"
        order_type = mt5.ORDER_TYPE_BUY if final_bias == "BULLISH" else mt5.ORDER_TYPE_SELL
        price = tick.ask if final_bias == "BULLISH" else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 260525,
            "comment": "sniper_phase5",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return False, f"MT5 order_send failed: {mt5.last_error()}"
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"MT5 rejected order: {result.retcode} {result.comment}"
        return True, str(result.order)
    finally:
        mt5.shutdown()


def append_trade_log(row: dict[str, object]) -> None:
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = TRADE_LOG_PATH.exists()
    with TRADE_LOG_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def signal_key(
    symbol: str,
    final_bias: str,
    entry_time: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    target_price: float,
) -> str:
    return "|".join(
        (
            symbol,
            final_bias,
            entry_time,
            f"{entry_price:.5f}",
            f"{sl_price:.5f}",
            f"{tp_price:.5f}",
            f"{target_price:.5f}",
        )
    )


def load_last_signal() -> dict[str, object]:
    if not SIGNAL_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SIGNAL_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_last_signal(key: str, trade_id: str, order_ref: str) -> None:
    SIGNAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_STATE_PATH.write_text(
        json.dumps(
            {
                "signal_key": key,
                "trade_id": trade_id,
                "order_ref": order_ref,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def execute_trade(symbol: str = "GOLD") -> Phase5Output:
    from FVGs import weekly_bias_engine
    from risk_manger import register_trade_open, trade_permission_gate, update_after_trade_close
    from targetIdentification import identify_target, mark_trade_idea_used
    from ttps import calculate_ttps

    phase1 = weekly_bias_engine()
    phase2 = load_python_file("Manipulation Zone.py", "manipulation_zone_phase5").evaluate_manipulation_zone(symbol)
    phase3 = identify_target(phase2, symbol)
    phase4 = evaluate_entry_conditions(symbol)
    if not phase4.entry_signal or phase4.entry_price is None or phase3.target_price is None:
        return Phase5Output("", None, None, 0.0, 0.0, False, False, False, "", 0.0, False, False, phase4.block_reason)

    sl_price, sl_distance = find_stop_loss(phase1.final_bias, phase4.entry_price)
    if sl_price is None:
        return Phase5Output("", None, None, sl_distance, 0.0, False, False, False, "", 0.0, False, False, "No valid 1H swing stop loss")

    tp_price, actual_rr = calculate_take_profit(phase1.final_bias, phase4.entry_price, sl_price, phase3.target_price)
    if tp_price is None:
        return Phase5Output("", sl_price, None, sl_distance, actual_rr, True, False, False, "", 0.0, False, False, "Target does not maintain 1:2 R:R")

    ttps = calculate_ttps(phase1, phase2, phase3, phase4, sl_price, tp_price, sl_distance, actual_rr)
    if not ttps.proceed:
        return Phase5Output("", sl_price, tp_price, sl_distance, actual_rr, True, True, False, "", 0.0, False, False, ttps.reason, "", None, 0.0, ttps.total, ttps.grade)

    permission = trade_permission_gate(phase1.final_bias, phase1.no_trade_week, sl_distance, symbol)
    if not permission.trade_permitted:
        return Phase5Output("", sl_price, tp_price, sl_distance, actual_rr, True, True, False, "", 0.0, False, False, permission.block_reason, "", None, 0.0, ttps.total, ttps.grade)

    key = signal_key(symbol, phase1.final_bias, phase4.entry_time, phase4.entry_price, sl_price, tp_price, phase3.target_price)
    if load_last_signal().get("signal_key") == key:
        return Phase5Output("", sl_price, tp_price, sl_distance, actual_rr, True, True, False, "", 0.0, False, False, "Duplicate signal already handled")

    trade_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    lot_size = permission.lot_size * ttps.position_multiplier
    order_ok, order_ref = place_mt5_order(symbol, phase1.final_bias, lot_size, sl_price, tp_price)
    if not order_ok:
        return Phase5Output("", sl_price, tp_price, sl_distance, actual_rr, True, True, False, "", 0.0, False, False, order_ref, "", None, 0.0, ttps.total, ttps.grade)
    save_last_signal(key, trade_id, order_ref)
    register_trade_open()
    mark_trade_idea_used()
    result, r_gained, close_price, close_time = simulate_trade_close(phase1.final_bias, sl_price, tp_price, phase4.entry_time)
    if result == "WIN":
        r_gained = actual_rr
    pips_gained = 0.0
    if close_price is not None:
        pips_gained = abs(close_price - phase4.entry_price) / pip_size(symbol)
    if result in {"WIN", "LOSS"}:
        pnl = permission.risk_amount * r_gained
        update_after_trade_close(pnl)

    append_trade_log(
        {
            "trade_id": trade_id,
            "instrument": symbol,
            "entry_time": phase4.entry_time,
            "entry_price": phase4.entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_distance": sl_distance,
            "actual_rr": actual_rr,
            "target_price": phase3.target_price,
            "target_type": phase3.target_type,
            "swept_level": phase4.swept_level,
            "swept_level_type": phase4.swept_level_type,
            "confluence_score": phase4.confluence_score,
            "setup_quality": phase4.setup_quality,
            "final_bias": phase1.final_bias,
            "conviction": phase1.conviction,
            "lot_size": lot_size,
            "ttps_total": ttps.total,
            "ttps_grade": ttps.grade,
            "ttps_htf_bias": ttps.htf_bias,
            "ttps_entry_confluence": ttps.entry_confluence,
            "ttps_session_quality": ttps.session_quality,
            "ttps_risk_quality": ttps.risk_quality,
            "ttps_action": ttps.action,
            "order_ref": order_ref,
            "close_time": close_time,
            "close_price": close_price,
            "trade_result": result,
            "r_gained": r_gained,
            "pips_gained": pips_gained,
        }
    )
    return Phase5Output(trade_id, sl_price, tp_price, sl_distance, actual_rr, True, True, result == "OPEN", result, r_gained, result == "LOSS", result == "WIN", "Trade logged", close_time, close_price, pips_gained, ttps.total, ttps.grade)


def run_phase5() -> int:
    output = execute_trade()
    print("Phase 5 Trade Execution")
    print(f"trade_id={output.trade_id}")
    print(f"sl={output.sl_price} tp={output.tp_price} sl_distance={output.sl_distance:.2f} rr={output.actual_rr:.2f}")
    print(f"trade_open={output.trade_open} result={output.trade_result} r={output.r_gained:.2f}")
    print(f"ttps={output.ttps_total}/20 grade={output.ttps_grade}")
    print(f"close_time={output.close_time} close_price={output.close_price} pips={output.pips_gained:.1f}")
    print(f"mandatory_recheck={output.mandatory_recheck} recommended_recheck={output.recommended_recheck}")
    print(f"reason={output.block_reason}")
    return 0


def run_ttps() -> int:
    from ttps import main

    return main()


def run_backtest(mode: str = "full", start: str | None = None, end: str | None = None) -> int:
    from backtest import main

    original_argv = sys.argv
    try:
        new_argv = [original_argv[0], "--mode", mode]
        if start:
            new_argv += ["--start", start]
        if end:
            new_argv += ["--end", end]
        sys.argv = new_argv
        return main()
    finally:
        sys.argv = original_argv


def run_controller() -> int:
    print("=== Sniper MT5 Bot Controller ===")
    print_news_status()
    steps: list[tuple[str, Callable[[], int]]] = [
        ("filter", run_filter),
        ("fvgs", run_fvgs),
        ("phase2", run_phase2),
        ("phase3", run_phase3),
        ("phase4", run_phase4),
        ("phase5", run_phase5),
    ]
    failures = 0
    for name, handler in steps:
        print()
        print(f"--- Running {name} ---")
        try:
            result = int(handler() or 0)
        except Exception as exc:
            failures += 1
            print(f"{name} failed: {exc}")
            continue
        if result != 0 and name != "filter":
            failures += 1
            print(f"{name} returned exit code {result}")

    print()
    if failures:
        print(f"Controller finished with {failures} issue(s).")
        return 1
    print("Controller finished successfully.")
    return 0


def scan_interval_seconds() -> int:
    try:
        return max(int(float(load_env().get("SCAN_INTERVAL_SECONDS", "60"))), 5)
    except ValueError:
        return 60


def run_scanner() -> int:
    interval = scan_interval_seconds()
    cycle = 1
    print("=== Sniper MT5 Bot Market Scanner ===")
    print(f"Scanning every {interval} seconds. Press Ctrl+C to stop.")
    while True:
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print()
        print(f"===== Scan cycle {cycle} | {started_at} =====")
        try:
            run_controller()
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
            return 0
        except Exception as exc:
            print(f"Scan cycle {cycle} failed: {exc}")

        print()
        print(f"Next scan in {interval} seconds...")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nScanner stopped by user.")
            return 0
        cycle += 1


def run_menu() -> int:
    return interactive_menu()


COMMANDS: dict[str, Command] = {
    "scan": ("Keep scanning the market until stopped", run_scanner),
    "all": ("Run full controller pipeline", run_controller),
    "menu": ("Open interactive launcher menu", run_menu),
    "env": ("Print sanitized runtime configuration", run_env),
    "risk": ("Print risk manager rules", run_risk),
    "filter": ("Check active news/filter block", run_filter),
    "train": ("Build training summary CSV", run_train),
    "backtest": ("Summarize trade log performance", run_backtest),
    "dashboard": ("Show trading dashboard summary", run_dashboard),
    "export-history": ("Show history file status", run_history),
    "fvgs": ("Detect FVGs from saved history", run_fvgs),
    "phase2": ("Run Phase 2 manipulation zone checks", run_phase2),
    "phase3": ("Run Phase 3 target identification", run_phase3),
    "phase4": ("Run Phase 4 entry conditions", run_phase4),
    "phase5": ("Run Phase 5 trade execution", run_phase5),
    "ttps": ("Run TTPS master score", run_ttps),
}


ALIASES = {
    "config": "env",
    "settings": "env",
    "risk-manager": "risk",
    "risk-manger": "risk",
    "news": "filter",
    "history": "export-history",
    "export": "export-history",
    "dash": "dashboard",
    "fvg": "fvgs",
    "manipulation": "phase2",
    "target": "phase3",
    "entry": "phase4",
    "execute": "phase5",
    "trade": "phase5",
    "score": "ttps",
    "start": "all",
    "run": "all",
    "bot": "all",
    "scanner": "scan",
    "loop": "scan",
}


def normalize_command(command: str) -> str:
    key = command.strip().lower()
    return ALIASES.get(key, key)


def print_menu() -> None:
    print("=== Sniper MT5 Bot Launcher ===")
    for index, (name, (description, _)) in enumerate(COMMANDS.items(), start=1):
        print(f"{index}. {name:<14} {description}")
    print("q. quit without running anything")


def prompt_for_command() -> str | None:
    names = list(COMMANDS)
    while True:
        print_menu()
        choice = input("Select what to run [Enter=all]: ").strip().lower()

        if choice == "":
            return "all"

        if choice in {"q", "quit", "exit"}:
            return None

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(names):
                return names[index]

        command = normalize_command(choice)
        if command in COMMANDS:
            return command

        print(f"Unknown choice: {choice}")


def dispatch(command: str) -> int:
    command = normalize_command(command)
    if command not in COMMANDS:
        valid = ", ".join(COMMANDS)
        raise SystemExit(f"Unknown command '{command}'. Valid commands: {valid}")

    _, handler = COMMANDS[command]
    print(f"Starting {command}...")
    result = handler()
    return int(result or 0)


def wait_for_next_choice() -> None:
    if sys.stdin.isatty():
        input("\nPress Enter to return to the menu...")


def interactive_menu() -> int:
    print_news_status()
    while True:
        command = prompt_for_command()
        if command is None:
            print("Exiting launcher.")
            return 0

        try:
            dispatch(command)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as exc:
            print(f"\nError while running {command}: {exc}")

        wait_for_next_choice()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sniper MT5 Bot launcher."
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="What to run: scan, all, menu, env, risk, filter, train, backtest, dashboard, export-history, fvgs, phase2, phase3, phase4, phase5",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "walk-forward", "status"),
        default="full",
        help="Backtest mode: full, walk-forward, or status (only used with backtest command)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Backtest start date in UTC e.g. 2024-01-01 (only used with backtest command)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Backtest end date in UTC e.g. 2025-01-01 (only used with backtest command)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = args.command or "scan"
    command = normalize_command(command)

    # Backtest gets special handling to pass through its extra args
    if command == "backtest":
        return run_backtest(
            mode=args.mode,
            start=args.start,
            end=args.end,
        )

    return dispatch(command)


if __name__ == "__main__":
    raise SystemExit(main())