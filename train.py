from __future__ import annotations

import csv
from pathlib import Path

from history import HISTORY_DIR, TIMEFRAMES, history_path


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def build_training_summary() -> list[dict[str, object]]:
    return [
        {
            "timeframe": timeframe,
            "path": str(history_path(timeframe)),
            "rows": count_rows(history_path(timeframe)),
            "available": history_path(timeframe).exists(),
        }
        for timeframe in TIMEFRAMES
    ]


def main() -> int:
    rows = build_training_summary()
    print(f"History folder: {HISTORY_DIR}")
    for row in rows:
        print(f"{row['timeframe']}: rows={row['rows']} available={row['available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
