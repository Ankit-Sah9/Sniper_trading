from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import csv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


FULL_DAY_KEYWORDS = (
    "NFP",
    "NONFARM",
    "NON-FARM",
    "FOMC",
    "FED RATE",
    "RATE DECISION",
    "INTEREST RATE",
    "MONETARY POLICY",
    "POWELL",
    "FED CHAIR",
    "ECB RATE",
    "BOE RATE",
)
PRE_RELEASE_KEYWORDS = (
    "CPI",
    "CORE CPI",
    "PCE",
    "CORE PCE",
    "GDP",
    "RETAIL SALES",
    "UNEMPLOYMENT",
    "JOBS REPORT",
    "AVERAGE HOURLY EARNINGS",
    "ISM",
    "PMI",
    "ADP",
)
WATCH_ONLY_KEYWORDS = (
    "INITIAL JOBLESS CLAIMS",
    "CONSUMER CONFIDENCE",
    "JOLTS",
    "PPI",
)


@dataclass(frozen=True)
class NewsRule:
    blocked: bool
    reason: str
    importance: str


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


def get_bool(name: str, default: bool = False) -> bool:
    value = load_env().get(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def news_events_path() -> Path:
    values = load_env()
    return PROJECT_ROOT / values.get("NEWS_CSV", "data/news_events_sample.csv")


def parse_event_time(raw_time: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw_time)
    except ValueError:
        return None


def classify_event(name: str) -> str:
    upper_name = name.upper()
    if any(keyword in upper_name for keyword in FULL_DAY_KEYWORDS):
        return "FULL_DAY"
    if any(keyword in upper_name for keyword in PRE_RELEASE_KEYWORDS):
        return "PRE_RELEASE"
    if any(keyword in upper_name for keyword in WATCH_ONLY_KEYWORDS):
        return "WATCH_ONLY"
    return "IGNORE"


def event_rule(name: str, event_time: datetime, now: datetime) -> NewsRule:
    event_class = classify_event(name)
    upper_name = name.upper()
    if event_class == "FULL_DAY" and event_time.date() == now.date():
        return NewsRule(True, f"blocked by full-day high-impact news: {upper_name}", "HIGH")
    if event_class == "PRE_RELEASE":
        block_start = datetime.combine(event_time.date(), time.min)
        clear_time = event_time + timedelta(minutes=5)
        if block_start <= now <= clear_time:
            return NewsRule(True, f"blocked until release + 5 min: {upper_name}", "HIGH")
    if event_class == "WATCH_ONLY":
        block_start = event_time - timedelta(minutes=30)
        clear_time = event_time + timedelta(minutes=15)
        if block_start <= now <= clear_time:
            return NewsRule(True, f"blocked around medium/high-impact news: {upper_name}", "MEDIUM")
    return NewsRule(False, "clear", "LOW")


def csv_news_block(now: datetime | None = None) -> tuple[bool, str]:
    now = now or datetime.now()
    path = news_events_path()
    if not path.exists():
        return False, f"news file not found: {path}"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("event", ""))
            if classify_event(name) == "IGNORE":
                continue
            event_time = parse_event_time(str(row.get("time_ny", "")))
            if event_time is None:
                continue
            rule = event_rule(name, event_time, now)
            if rule.blocked:
                return True, rule.reason
    return False, "no active news block"


def is_blocked(now: datetime | None = None) -> tuple[bool, str]:
    if not get_bool("NEWS_FILTER_ENABLED", False):
        return False, "news filter disabled because NEWS_FILTER_ENABLED=false in .env"
    return csv_news_block(now)


def main() -> int:
    blocked, reason = is_blocked()
    print(f"Blocked: {blocked}")
    print(f"Reason: {reason}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
