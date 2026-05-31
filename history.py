from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
import argparse
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_DIR = PROJECT_ROOT / "data" / "history"
METADATA_PATH = HISTORY_DIR / "export_metadata.json"
ENV_PATH = PROJECT_ROOT / ".env"

TIMEFRAMES = ("W1", "D1", "H4", "H1")
PRICE_FIELDS = ("time", "open", "high", "low", "close", "volume", "bid", "ask")
TICK_FIELDS = ("time", "bid", "ask", "last", "volume")
EXPORT_RANGES = {
    "W1": 8,
    "D1": 8,
    "H4": 8,
    "H1": 8,
}
MIN_EXPECTED_ROWS = {
    "W1": 140,
    "D1": 700,
    "H4": 3500,
    "H1": 14000,
}
CHUNK_DAYS = {
    "W1": 365,
    "D1": 180,
    "H4": 45,
    "H1": 30,
}
DEFAULT_MT5_PATH = Path("C:/Program Files/MetaTrader 5/terminal64.exe")


def import_mt5() -> Any:
    import MetaTrader5

    return cast(Any, MetaTrader5)


def history_path(timeframe: str) -> Path:
    return HISTORY_DIR / f"{timeframe.upper()}.csv"


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


def trading_symbol() -> str:
    return load_env().get("TRADING_SYMBOL", "GOLD")


def symbol_candidates() -> list[str]:
    env = load_env()
    configured = trading_symbol()
    raw_aliases = env.get("TRADING_SYMBOL_ALIASES", "GOLD,XAUUSD,GOLDm,XAUUSDm")
    candidates = [configured]
    candidates.extend(item.strip() for item in raw_aliases.split(",") if item.strip())

    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def mt5_terminal_path() -> str | None:
    configured = load_env().get("MT5_PATH", "").strip()
    if configured:
        return configured
    if DEFAULT_MT5_PATH.exists():
        return str(DEFAULT_MT5_PATH)
    return None


def mt5_login_params() -> dict[str, object]:
    env = load_env()
    params: dict[str, object] = {}

    login = env.get("MT5_LOGIN", "").strip()
    password = env.get("MT5_PASSWORD", "").strip()
    server = env.get("MT5_SERVER", "").strip()

    if login:
        try:
            params["login"] = int(login)
        except ValueError as exc:
            raise RuntimeError("MT5_LOGIN must be a number") from exc
    if password:
        params["password"] = password
    if server:
        params["server"] = server

    return params


def check_history_files() -> dict[str, bool]:
    return {timeframe: history_path(timeframe).exists() for timeframe in TIMEFRAMES}


def mt5_timeframe(name: str) -> int:
    mt5 = import_mt5()

    mapping = {
        "W1": mt5.TIMEFRAME_W1,
        "D1": mt5.TIMEFRAME_D1,
        "H4": mt5.TIMEFRAME_H4,
        "H1": mt5.TIMEFRAME_H1,
    }
    return mapping[name.upper()]


def rate_time(rate) -> int:
    return int(rate["time"])


def timeframe_duration(timeframe: str) -> timedelta:
    mapping = {
        "W1": timedelta(days=7),
        "D1": timedelta(days=1),
        "H4": timedelta(hours=4),
        "H1": timedelta(hours=1),
    }
    return mapping[timeframe.upper()]


def rate_spread_points(rate) -> float:
    try:
        return float(rate["spread"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def symbol_point(mt5, symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    point = getattr(info, "point", None) if info is not None else None
    try:
        return float(point or 0.0)
    except (TypeError, ValueError):
        return 0.0


def write_rates_csv(path: Path, rates, point: float = 0.0) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rates, key=rate_time)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PRICE_FIELDS)
        for rate in ordered:
            timestamp = int(rate["time"])
            candle_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            close_price = float(rate["close"])
            bid = close_price
            ask = close_price + rate_spread_points(rate) * point
            writer.writerow(
                (
                    candle_time.strftime("%Y-%m-%d %H:%M:%S"),
                    float(rate["open"]),
                    float(rate["high"]),
                    float(rate["low"]),
                    close_price,
                    float(rate["tick_volume"]) if "tick_volume" in rate.dtype.names else 0.0,
                    bid,
                    ask,
                )
            )
    return len(ordered)


def copy_rates_chunked(symbol: str, timeframe: str, start: datetime, end: datetime):
    mt5 = import_mt5()

    chunk = timedelta(days=CHUNK_DAYS.get(timeframe.upper(), 30))
    current = start
    by_time = {}
    while current < end:
        chunk_end = min(current + chunk, end)
        rates = mt5.copy_rates_range(symbol, mt5_timeframe(timeframe), current, chunk_end)
        if rates is None:
            code, message = mt5.last_error()
            raise RuntimeError(f"MT5 failed for {timeframe}: {code} {message}")
        for rate in rates:
            by_time[rate_time(rate)] = rate
        current = chunk_end + timedelta(seconds=1)
    return list(by_time.values())


def export_timeframe(mt5, symbol: str, timeframe: str, years: int) -> int:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years + 7)
    rates = copy_rates_chunked(symbol, timeframe, start, end)
    return write_rates_csv(history_path(timeframe), rates, symbol_point(mt5, symbol))


def export_history() -> tuple[dict[str, int], str]:
    mt5 = import_mt5()

    requested_symbol = trading_symbol()
    terminal_path = mt5_terminal_path()
    login_params = mt5_login_params()
    initialized = (
        mt5.initialize(path=terminal_path, timeout=120000, **login_params)
        if terminal_path
        else mt5.initialize(timeout=120000, **login_params)
    )
    if not initialized:
        code, message = mt5.last_error()
        path_hint = f" path={terminal_path}" if terminal_path else ""
        login_hint = " using MT5_LOGIN/MT5_SERVER" if login_params else ""
        raise RuntimeError(
            f"MT5 initialize failed{path_hint}{login_hint}: {code} {message}. "
            "Open MetaTrader 5, log in, enable the symbol in Market Watch, "
            "then run this command again."
        )

    try:
        terminal_info = mt5.terminal_info()
        max_bars = getattr(terminal_info, "maxbars", None) if terminal_info is not None else None
        if max_bars is not None:
            print(f"MT5 max bars in chart: {max_bars}")

        symbol = ""
        errors: list[str] = []
        for candidate in symbol_candidates():
            if mt5.symbol_select(candidate, True):
                symbol = candidate
                break
            code, message = mt5.last_error()
            errors.append(f"{candidate}: {code} {message}")
        if not symbol:
            raise RuntimeError(f"MT5 could not select any symbol. Tried: {'; '.join(errors)}")
        if symbol != requested_symbol:
            print(f"Selected symbol fallback: {symbol} (requested {requested_symbol})")

        counts = {
            timeframe: export_timeframe(mt5, symbol, timeframe, years)
            for timeframe, years in EXPORT_RANGES.items()
        }
        write_export_metadata(requested_symbol, symbol, counts)
        return counts, symbol
    finally:
        mt5.shutdown()


def initialize_mt5():
    mt5 = import_mt5()
    terminal_path = mt5_terminal_path()
    login_params = mt5_login_params()
    initialized = (
        mt5.initialize(path=terminal_path, timeout=120000, **login_params)
        if terminal_path
        else mt5.initialize(timeout=120000, **login_params)
    )
    if not initialized:
        code, message = mt5.last_error()
        path_hint = f" path={terminal_path}" if terminal_path else ""
        login_hint = " using MT5_LOGIN/MT5_SERVER" if login_params else ""
        raise RuntimeError(
            f"MT5 initialize failed{path_hint}{login_hint}: {code} {message}. "
            "Open MetaTrader 5, log in, enable the symbol in Market Watch, "
            "then run this command again."
        )
    return mt5


def select_mt5_symbol(mt5) -> str:
    requested_symbol = trading_symbol()
    errors: list[str] = []
    for candidate in symbol_candidates():
        if mt5.symbol_select(candidate, True):
            if candidate != requested_symbol:
                print(f"Selected symbol fallback: {candidate} (requested {requested_symbol})")
            return str(candidate)
        code, message = mt5.last_error()
        errors.append(f"{candidate}: {code} {message}")
    raise RuntimeError(f"MT5 could not select any symbol. Tried: {'; '.join(errors)}")


def write_export_metadata(requested_symbol: str, selected_symbol: str, counts: dict[str, int]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "requested_symbol": requested_symbol,
                "selected_symbol": selected_symbol,
                "exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "ranges_years": EXPORT_RANGES,
                "rows": counts,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_candle_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def read_exported_times(timeframe: str) -> list[datetime]:
    path = history_path(timeframe)
    if not path.exists():
        return []
    times: list[datetime] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = parse_candle_time(str(row.get("time", "")))
            if parsed is not None:
                times.append(parsed)
    return sorted(times)


def is_weekend_gap(previous: datetime, current: datetime, max_gap: timedelta) -> bool:
    return previous.weekday() == 4 and current.weekday() in {6, 0} and current - previous <= max_gap


def daily_maintenance_gap(previous: datetime, current: datetime) -> bool:
    if current - previous > timedelta(hours=4):
        return False
    return previous.hour in {20, 21, 22, 23} and current.hour in {0, 1, 2, 3}


def known_daily_market_closure(previous: datetime, current: datetime) -> bool:
    return (previous.date(), current.date()) in {
        (datetime(2023, 12, 22).date(), datetime(2023, 12, 26).date()),
        (datetime(2023, 12, 29).date(), datetime(2024, 1, 2).date()),
        (datetime(2024, 3, 28).date(), datetime(2024, 4, 1).date()),
        (datetime(2024, 12, 24).date(), datetime(2024, 12, 26).date()),
        (datetime(2024, 12, 31).date(), datetime(2025, 1, 2).date()),
        (datetime(2025, 4, 17).date(), datetime(2025, 4, 21).date()),
        (datetime(2025, 12, 24).date(), datetime(2025, 12, 26).date()),
        (datetime(2025, 12, 31).date(), datetime(2026, 1, 2).date()),
        (datetime(2026, 4, 2).date(), datetime(2026, 4, 6).date()),
    }


def known_intraday_market_closure(previous: datetime, current: datetime, max_gap: timedelta) -> bool:
    if current - previous > max_gap:
        return False
    return known_daily_market_closure(
        previous.replace(hour=0, minute=0, second=0, microsecond=0),
        current.replace(hour=0, minute=0, second=0, microsecond=0),
    ) or (previous.hour in {3, 4, 20, 21, 22, 23} and current.hour in {0, 1, 10, 12})


def is_valid_gap(previous: datetime, current: datetime, timeframe: str) -> bool:
    gap = current - previous
    if gap <= timedelta(0):
        return False
    if timeframe == "W1":
        return timedelta(days=6) <= gap <= timedelta(days=10)
    if timeframe == "D1":
        return (
            gap <= timedelta(days=1, hours=4)
            or is_weekend_gap(previous, current, timedelta(days=4))
            or known_daily_market_closure(previous, current)
        )
    if timeframe == "H4":
        return (
            gap <= timedelta(hours=5)
            or is_weekend_gap(previous, current, timedelta(days=3))
            or known_intraday_market_closure(previous, current, timedelta(days=4))
        )
    if timeframe == "H1":
        return (
            gap <= timedelta(hours=2)
            or daily_maintenance_gap(previous, current)
            or is_weekend_gap(previous, current, timedelta(days=3))
            or known_intraday_market_closure(previous, current, timedelta(days=4))
        )
    return True


def count_gaps(times: list[datetime], timeframe: str) -> int:
    return sum(
        1
        for previous, current in zip(times, times[1:])
        if not is_valid_gap(previous, current, timeframe)
    )


def validate_exports(counts: dict[str, int]) -> list[str]:
    warnings: list[str] = []
    now = datetime.now(timezone.utc)
    for timeframe, years in EXPORT_RANGES.items():
        times = read_exported_times(timeframe)
        rows = read_exported_rows(timeframe)
        expected = MIN_EXPECTED_ROWS.get(timeframe, 0)
        if counts.get(timeframe, 0) < expected:
            warnings.append(
                f"{timeframe}: only {counts.get(timeframe, 0)} rows; expected at least {expected}. "
                "Increase MT5 Max bars in chart/history and re-export."
            )
        if not times:
            warnings.append(f"{timeframe}: exported file is empty")
            continue
        invalid_rows = count_invalid_price_rows(rows)
        if invalid_rows:
            warnings.append(f"{timeframe}: {invalid_rows} invalid OHLCV price row(s)")
        oldest_allowed = now - timedelta(days=365 * years - 14)
        if times[0] > oldest_allowed:
            warnings.append(
                f"{timeframe}: starts at {times[0]:%Y-%m-%d}, not a full {years}-year export"
            )
        gaps = count_gaps(times, timeframe)
        if gaps:
            warnings.append(f"{timeframe}: {gaps} non-adjacent candle gap(s)")
    return warnings


def read_exported_rows(timeframe: str) -> list[dict[str, str]]:
    path = history_path(timeframe)
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({field: str(row.get(field, "")).strip() for field in PRICE_FIELDS})
    return rows


def count_invalid_price_rows(rows: list[dict[str, str]]) -> int:
    invalid = 0
    for row in rows:
        try:
            values = [float(row[field]) for field in ("open", "high", "low", "close")]
            float(row["volume"])
        except ValueError:
            invalid += 1
            continue
        if min(values) <= 0:
            invalid += 1
        elif values[1] < max(values[0], values[3]) or values[2] > min(values[0], values[3]):
            invalid += 1
    return invalid


def print_history_status() -> int:
    counts = {timeframe: len(read_exported_rows(timeframe)) for timeframe in TIMEFRAMES}
    for timeframe in TIMEFRAMES:
        path = history_path(timeframe)
        rows = read_exported_rows(timeframe)
        first = rows[0]["time"] if rows else ""
        last = rows[-1]["time"] if rows else ""
        print(f"{timeframe}: rows={len(rows)} first={first} last={last} file={path}")
    warnings = validate_exports(counts)
    if warnings:
        print("Backtest data warnings")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Backtest data validation: OK")
    return 0


def print_history_preview(timeframe: str, limit: int) -> int:
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAMES:
        raise SystemExit(f"Unknown timeframe '{timeframe}'. Use: {', '.join(TIMEFRAMES)}")
    rows = read_exported_rows(timeframe)
    print(",".join(PRICE_FIELDS))
    for row in rows[:limit]:
        print(",".join(row[field] for field in PRICE_FIELDS))
    return 0


def parse_utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise SystemExit(f"Invalid datetime '{value}'. Use YYYY-MM-DD HH:MM:SS in UTC.") from exc
    return parsed.replace(tzinfo=timezone.utc)


def print_tick_preview(limit: int, minutes: int, start: str | None = None, end: str | None = None) -> int:
    mt5 = initialize_mt5()
    try:
        symbol = select_mt5_symbol(mt5)
        end_dt = parse_utc_datetime(end) if end else datetime.now(timezone.utc)
        start_dt = parse_utc_datetime(start) if start else end_dt - timedelta(minutes=minutes)
        ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_ALL)
        if ticks is None:
            code, message = mt5.last_error()
            raise RuntimeError(f"MT5 tick export failed: {code} {message}")

        print(",".join(TICK_FIELDS))
        ordered = sorted(ticks, key=lambda tick: int(tick["time_msc"]) if "time_msc" in tick.dtype.names else int(tick["time"]))
        for tick in ordered[:limit]:
            tick_time = datetime.fromtimestamp(int(tick["time"]), tz=timezone.utc)
            print(
                ",".join(
                    (
                        tick_time.strftime("%Y-%m-%d %H:%M:%S"),
                        str(float(tick["bid"]) if "bid" in tick.dtype.names else 0.0),
                        str(float(tick["ask"]) if "ask" in tick.dtype.names else 0.0),
                        str(float(tick["last"]) if "last" in tick.dtype.names else 0.0),
                        str(float(tick["volume"]) if "volume" in tick.dtype.names else 0.0),
                    )
                )
            )
        return 0
    finally:
        mt5.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and read MT5 OHLCV history CSV files")
    parser.add_argument("--mode", choices=("export", "status", "preview", "ticks"), default="export")
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default="W1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--start", help="UTC start datetime for ticks: YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", help="UTC end datetime for ticks: YYYY-MM-DD HH:MM:SS")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "status":
        return print_history_status()
    if args.mode == "preview":
        return print_history_preview(args.timeframe, args.limit)
    if args.mode == "ticks":
        return print_tick_preview(args.limit, args.minutes, args.start, args.end)

    print("Exporting history from MT5")
    print(f"Symbol: {trading_symbol()}")
    counts, selected_symbol = export_history()
    print(f"Exported symbol: {selected_symbol}")
    for timeframe, years in EXPORT_RANGES.items():
        print(
            f"{timeframe}: {counts[timeframe]} rows "
            f"({years} years) -> {history_path(timeframe)}"
        )
    warnings = validate_exports(counts)
    if warnings:
        print()
        print("Backtest data warnings")
        for warning in warnings:
            print(f"- {warning}")
        print("MT5 tip: Tools -> Options -> Charts -> Max bars in chart, set a very high value, restart MT5, then export again.")
    else:
        print("Backtest data validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
