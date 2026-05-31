from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from math import floor
from pathlib import Path
from zoneinfo import ZoneInfo
from zoneinfo._common import ZoneInfoNotFoundError
import json
import urllib.request

from filter import is_blocked


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
STATE_PATH = PROJECT_ROOT / "data" / "risk_state.json"
try:
    NY_TZ = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:
    NY_TZ = timezone(timedelta(hours=-4), "New_York_Fallback")


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


def get_int(name: str, default: int) -> int:
    try:
        return int(float(load_env().get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RiskRules:
    account_balance: float = get_float("ACCOUNT_BALANCE", 100000.0)
    risk_per_trade: float = get_float("RISK_PER_TRADE", 0.01)
    max_weekly_loss: float = get_float("MAX_WEEKLY_LOSS", 0.03)
    max_monthly_drawdown: float = get_float("MAX_MONTHLY_DRAWDOWN", 0.10)
    max_trades_per_week: int = get_int("MAX_TRADES_PER_WEEK", 3)
    max_open_trades: int = get_int("MAX_OPEN_TRADES", 1)
    max_spread_price: float = get_float("MAX_SPREAD_PRICE", 3.0)
    min_sl_distance: float = get_float("GOLD_MIN_SL_DISTANCE", 20.0)
    max_sl_distance: float = get_float("GOLD_MAX_SL_DISTANCE", 30.0)


@dataclass(frozen=True)
class PositionSize:
    lot_size: float
    risk_amount: float
    pip_value_per_lot: float
    sl_pips: float


@dataclass(frozen=True)
class TradePermission:
    trade_permitted: bool
    block_reason: str
    lot_size: float
    risk_amount: float
    trade_ideas_used: int
    weekly_pnl: float
    weekly_kill_active: bool
    monthly_kill_active: bool


def default_risk_state(balance: float | None = None) -> dict[str, object]:
    rules = RiskRules()
    current_balance = rules.account_balance if balance is None else balance
    now = datetime.now(NY_TZ)
    return {
        "trade_ideas_used": 0,
        "weekly_pnl": 0.0,
        "week_start_balance": current_balance,
        "weekly_kill_active": False,
        "month_start_balance": current_balance,
        "monthly_kill_active": False,
        "open_trades": 0,
        "last_week_reset": now.date().isoformat(),
        "last_month": now.strftime("%Y-%m"),
    }


def load_risk_state(balance: float | None = None) -> dict[str, object]:
    if not STATE_PATH.exists():
        return default_risk_state(balance)
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_risk_state(balance)
    defaults = default_risk_state(balance)
    defaults.update(state)
    return defaults


def save_risk_state(state: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def maybe_reset_periods(state: dict[str, object], account_balance: float, now: datetime | None = None) -> dict[str, object]:
    now = (now or datetime.now(NY_TZ)).astimezone(NY_TZ)
    if now.weekday() == 6 and now.time() >= time(18, 0) and state.get("last_week_reset") != now.date().isoformat():
        state["trade_ideas_used"] = 0
        state["weekly_pnl"] = 0.0
        state["week_start_balance"] = account_balance
        state["weekly_kill_active"] = False
        state["last_week_reset"] = now.date().isoformat()

    month_key = now.strftime("%Y-%m")
    if state.get("last_month") != month_key:
        state["month_start_balance"] = account_balance
        state["monthly_kill_active"] = False
        state["last_month"] = month_key
    return state


def pip_value_per_lot(symbol: str) -> float:
    symbol = symbol.upper()
    if "EURUSD" in symbol:
        return 10.0
    if "XAU" in symbol or "GOLD" in symbol:
        return 1.0
    return get_float("DEFAULT_PIP_VALUE_PER_LOT", 1.0)


def calculate_risk_amount(balance: float, risk_per_trade: float | None = None) -> float:
    rules = RiskRules()
    return balance * (rules.risk_per_trade if risk_per_trade is None else risk_per_trade)


def calculate_lot_size(account_balance: float, sl_pips: float, symbol: str = "GOLD", risk_pct: float | None = None) -> PositionSize:
    rules = RiskRules()
    risk = calculate_risk_amount(account_balance, rules.risk_per_trade if risk_pct is None else risk_pct)
    pip_value = pip_value_per_lot(symbol)
    if sl_pips <= 0 or pip_value <= 0:
        return PositionSize(0.0, risk, pip_value, sl_pips)

    raw_lot_size = risk / (sl_pips * pip_value)
    lot_size = floor(raw_lot_size * 100) / 100
    while lot_size > 0 and lot_size * sl_pips * pip_value > risk:
        lot_size = floor((lot_size - 0.01) * 100) / 100
    return PositionSize(max(lot_size, 0.0), risk, pip_value, sl_pips)


def valid_trading_time(now: datetime | None = None) -> tuple[bool, str]:
    current = (now or datetime.now(NY_TZ)).astimezone(NY_TZ)
    if current.weekday() not in {1, 2, 3}:
        return False, "Not a valid trading day"
    if not (time(2, 0) <= current.time() <= time(11, 0)):
        return False, "Outside trading window"
    return True, "Trading window open"


def fetch_calendar_events() -> list[dict[str, object]]:
    api_url = load_env().get("ECONOMIC_CALENDAR_API_URL", "").strip()
    if not api_url:
        return []
    try:
        with urllib.request.urlopen(api_url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        events = payload.get("events", [])
        if isinstance(events, list):
            return [item for item in events if isinstance(item, dict)]
    return []


def parse_event_datetime(event: dict[str, object]) -> datetime | None:
    raw = event.get("time_ny") or event.get("datetime") or event.get("time") or event.get("date")
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=NY_TZ)
    return parsed.astimezone(NY_TZ)


def live_news_block(now: datetime | None = None) -> tuple[bool, str]:
    from filter import classify_event, event_rule

    current = (now or datetime.now(NY_TZ)).astimezone(NY_TZ)
    for event in fetch_calendar_events():
        name = str(event.get("event") or event.get("name") or event.get("title") or "").upper()
        if classify_event(name) == "IGNORE":
            continue
        event_time = parse_event_datetime(event)
        if event_time is None:
            continue
        rule = event_rule(name, event_time.replace(tzinfo=None), current.replace(tzinfo=None))
        if rule.blocked:
            return True, rule.reason
    return False, "Clear"


def news_permission(now: datetime | None = None) -> tuple[bool, str]:
    if load_env().get("ECONOMIC_CALENDAR_API_URL", "").strip():
        return live_news_block(now)
    return is_blocked(None if now is None else now.replace(tzinfo=None))


def update_after_trade_close(pnl_usd: float, account_balance: float | None = None) -> dict[str, object]:
    rules = RiskRules()
    balance = rules.account_balance if account_balance is None else account_balance
    state = maybe_reset_periods(load_risk_state(balance), balance)
    state["weekly_pnl"] = float(state.get("weekly_pnl", 0.0)) + pnl_usd
    state["open_trades"] = max(int(state.get("open_trades", 0)) - 1, 0)
    if float(state["weekly_pnl"]) <= -(float(state.get("week_start_balance", balance)) * rules.max_weekly_loss):
        state["weekly_kill_active"] = True
    month_start = float(state.get("month_start_balance", balance))
    if month_start > 0 and (month_start - balance) / month_start >= rules.max_monthly_drawdown:
        state["monthly_kill_active"] = True
    save_risk_state(state)
    return state


def register_trade_open(account_balance: float | None = None) -> dict[str, object]:
    rules = RiskRules()
    balance = rules.account_balance if account_balance is None else account_balance
    state = maybe_reset_periods(load_risk_state(balance), balance)
    state["trade_ideas_used"] = int(state.get("trade_ideas_used", 0)) + 1
    state["open_trades"] = int(state.get("open_trades", 0)) + 1
    save_risk_state(state)
    return state


def trade_permission_gate(
    final_bias: str,
    no_trade_week: bool,
    sl_pips: float,
    symbol: str = "GOLD",
    account_balance: float | None = None,
    now: datetime | None = None,
) -> TradePermission:
    rules = RiskRules()
    balance = rules.account_balance if account_balance is None else account_balance
    state = maybe_reset_periods(load_risk_state(balance), balance, now)
    save_risk_state(state)

    blocked, news_reason = news_permission(now)
    time_ok, time_reason = valid_trading_time(now)
    position = calculate_lot_size(balance, sl_pips, symbol)

    checks = [
        (final_bias != "NO_TRADE", "No bias this week"),
        (not no_trade_week, "Structure shift or no-trade week active"),
        (not blocked, news_reason),
        (not bool(state.get("weekly_kill_active", False)), "Weekly loss limit reached"),
        (not bool(state.get("monthly_kill_active", False)), "Monthly kill switch active"),
        (int(state.get("trade_ideas_used", 0)) < rules.max_trades_per_week, "3 trade ideas used this week"),
        (int(state.get("open_trades", 0)) < rules.max_open_trades, "Maximum open trades reached"),
        (time_ok, time_reason),
        (position.lot_size > 0 and sl_pips > 0, "Position sizing error"),
    ]
    for passed, reason in checks:
        if not passed:
            return TradePermission(
                False,
                reason,
                position.lot_size,
                position.risk_amount,
                int(state.get("trade_ideas_used", 0)),
                float(state.get("weekly_pnl", 0.0)),
                bool(state.get("weekly_kill_active", False)),
                bool(state.get("monthly_kill_active", False)),
            )

    return TradePermission(
        True,
        "",
        position.lot_size,
        position.risk_amount,
        int(state.get("trade_ideas_used", 0)),
        float(state.get("weekly_pnl", 0.0)),
        bool(state.get("weekly_kill_active", False)),
        bool(state.get("monthly_kill_active", False)),
    )


def risk_summary() -> RiskRules:
    return RiskRules()


def main() -> int:
    rules = risk_summary()
    position = calculate_lot_size(rules.account_balance, rules.min_sl_distance)
    state = load_risk_state(rules.account_balance)
    print("Phase 6 Risk Management Engine")
    print(f"Risk per trade: {rules.risk_per_trade:.2%}")
    print(f"Risk amount: {position.risk_amount:.2f}")
    print(f"Example lot size at min SL: {position.lot_size:.2f}")
    print(f"Max weekly loss: {rules.max_weekly_loss:.2%}")
    print(f"Max monthly drawdown: {rules.max_monthly_drawdown:.2%}")
    print(f"Max trades per week: {rules.max_trades_per_week}")
    print(f"State: trades={state.get('trade_ideas_used')} weekly_pnl={state.get('weekly_pnl')}")
    print(f"Weekly kill: {state.get('weekly_kill_active')} Monthly kill: {state.get('monthly_kill_active')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
