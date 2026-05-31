from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
BACKTEST_DIR = ROOT / "data" / "backtest"
DASHBOARD_HTML = ROOT / "dashboard.html"

TRADES_CANDIDATES = (
    RESULTS_DIR / "trades_oos.csv",
    RESULTS_DIR / "trade_log.csv",
    BACKTEST_DIR / "trade_log.csv",
)
EQUITY_CANDIDATES = (
    RESULTS_DIR / "equity_curve.csv",
    BACKTEST_DIR / "equity_curve.csv",
)
METRICS_CANDIDATES = (
    RESULTS_DIR / "metrics.json",
    BACKTEST_DIR / "metrics.json",
)


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        if value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return default


def parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def fmt_r(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}R"


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def trade_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return str(value)
    return ""


def normalize_result(value: str) -> str:
    text = value.strip().upper()
    if text in {"WIN", "WON", "TP", "PROFIT"}:
        return "Win"
    if text in {"LOSS", "LOST", "SL"}:
        return "Loss"
    if text in {"OPEN", "RUNNING"}:
        return "Open"
    return text.title() if text else "Open"


def normalize_direction(value: str) -> str:
    text = value.strip().upper()
    if text in {"BULLISH", "BUY", "LONG"}:
        return "Long"
    if text in {"BEARISH", "SELL", "SHORT"}:
        return "Short"
    return text.title()


def sl_pips(instrument: str, entry: float, sl: float) -> float:
    pip = 0.1 if "XAU" in instrument.upper() or "GOLD" in instrument.upper() else 0.0001
    return abs(entry - sl) / pip if pip else 0.0


def enrich_trades(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        entry_dt = parse_dt(trade_value(
    row,
    "entry_time_utc",
    "entry_time_ny",
    "entry_time",
    "open_time",
    "date",
))
        exit_dt = parse_dt(trade_value(
    row,
    "exit_time_utc",
    "exit_time_ny",
    "exit_time",
    "close_time",
))
        instrument = trade_value(row, "instrument", "symbol")
        entry = safe_float(trade_value(row, "entry_price", "entry"))
        sl = safe_float(trade_value(row, "sl_price", "stop_loss", "sl"))
        tp = safe_float(trade_value(row, "tp_price", "take_profit", "tp"))
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        r_gained = safe_float(trade_value(row, "r_gained", "r", "R"))
        pnl = safe_float(trade_value(row, "pnl", "pnl_usd", "profit"))
        result = normalize_result(trade_value(row, "trade_result", "result"))
        direction = normalize_direction(trade_value(row, "direction", "side"))
        holding = ""
        if entry_dt and exit_dt:
            delta = exit_dt - entry_dt
            hours = delta.total_seconds() / 3600
            holding = f"{hours:.1f}h" if hours < 48 else f"{hours / 24:.1f}d"
        week = entry_dt.isocalendar().week if entry_dt else ""
        enriched.append(
            {
                "trade_id": trade_value(row, "trade_id", "id") or f"#{index}",
                "date": entry_dt.strftime("%Y-%m-%d") if entry_dt else "",
                "day": entry_dt.strftime("%A") if entry_dt else "",
                "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S") if entry_dt else trade_value(row, "entry_time"),
                "exit_time": exit_dt.strftime("%Y-%m-%d %H:%M:%S") if exit_dt else trade_value(row, "exit_time"),
                "holding_time": holding,
                "instrument": instrument,
                "direction": direction,
                "entry_price": entry,
                "sl_price": sl,
                "tp_price": tp,
                "sl_pips": sl_pips(instrument, entry, sl),
                "rr_ratio": reward / risk if risk else 0.0,
                "ttps_score": int(safe_float(trade_value(row, "ttps_score", "confluence_score"))),
                "setup_quality": trade_value(row, "setup_quality", "grade") or "Q",
                "result": result,
                "r_gained": r_gained,
                "pnl": pnl,
                "week": f"{entry_dt.isocalendar().year}-W{week:02d}" if entry_dt else "",
                "sort_time": entry_dt.timestamp() if entry_dt else index,
            }
        )
    return enriched


def max_consecutive_losses(trades: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for trade in trades:
        if trade["result"] == "Loss":
            current += 1
            best = max(best, current)
        elif trade["result"] == "Win":
            current = 0
    return best


def equity_from_trades(trades: list[dict[str, Any]], start_balance: float) -> list[dict[str, Any]]:
    balance = start_balance
    rows = [{"time": "Start", "balance": balance}]
    for trade in sorted(trades, key=lambda item: item["sort_time"]):
        balance += float(trade["pnl"])
        rows.append({"time": trade["exit_time"] or trade["entry_time"], "balance": balance})
    return rows


def equity_from_csv(path: Path | None) -> list[dict[str, Any]]:
    rows = read_csv(path)
    equity: list[dict[str, Any]] = []
    for row in rows:
        time = trade_value(row, "time", "date", "timestamp", "exit_time")
        balance = safe_float(trade_value(row, "balance", "equity", "account_balance"))
        if time and balance:
            equity.append({"time": time, "balance": balance})
    return equity


def drawdown(equity: list[dict[str, Any]]) -> tuple[float, float]:
    peak = None
    max_dd = 0.0
    max_dd_pct = 0.0
    for row in equity:
        balance = float(row["balance"])
        peak = balance if peak is None else max(peak, balance)
        dd = peak - balance
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd / peak * 100 if peak else 0.0)
    return max_dd_pct, max_dd


def weekly_breakdown(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[trade["week"] or "Unknown"].append(trade)

    rows: list[dict[str, Any]] = []
    for week, week_trades in sorted(grouped.items()):
        wins = sum(1 for trade in week_trades if trade["result"] == "Win")
        losses = sum(1 for trade in week_trades if trade["result"] == "Loss")
        total_r = sum(float(trade["r_gained"]) for trade in week_trades)
        pnl = sum(float(trade["pnl"]) for trade in week_trades)
        rows.append(
            {
                "week": week,
                "trades": len(week_trades),
                "wins": wins,
                "losses": losses,
                "total_r": total_r,
                "pnl": pnl,
            }
        )
    return rows


def calculate_metrics(trades: list[dict[str, Any]], metrics_json: dict[str, Any], equity: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [trade for trade in trades if trade["result"] in {"Win", "Loss"}]
    wins = [trade for trade in closed if trade["result"] == "Win"]
    losses = [trade for trade in closed if trade["result"] == "Loss"]
    total_r = sum(float(trade["r_gained"]) for trade in closed)
    pnl = sum(float(trade["pnl"]) for trade in closed)
    gross_profit = sum(float(trade["pnl"]) for trade in wins)
    gross_loss = abs(sum(float(trade["pnl"]) for trade in losses))
    start_balance = safe_float(metrics_json.get("start_balance"), 100000.0)
    if equity:
        start_balance = safe_float(metrics_json.get("start_balance"), float(equity[0]["balance"]))
    end_balance = safe_float(metrics_json.get("end_balance"), start_balance + pnl)
    if equity:
        end_balance = float(equity[-1]["balance"])
    max_dd_pct, max_dd = drawdown(equity)
    average_win_r = sum(float(trade["r_gained"]) for trade in wins) / len(wins) if wins else 0.0
    average_loss_r = sum(float(trade["r_gained"]) for trade in losses) / len(losses) if losses else 0.0
    returns = [float(trade["pnl"]) / start_balance for trade in closed if start_balance]
    sharpe = 0.0
    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        sharpe = mean / std * math.sqrt(len(returns)) if std else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) * 100 if closed else 0.0,
        "total_r": total_r,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit else 0.0,
        "net_pnl": pnl,
        "average_r": total_r / len(closed) if closed else 0.0,
        "average_win_r": average_win_r,
        "average_loss_r": average_loss_r,
        "max_dd_pct": max_dd_pct,
        "max_dd": max_dd,
        "max_consecutive_losses": max_consecutive_losses(closed),
        "sharpe": safe_float(metrics_json.get("sharpe"), sharpe),
        "start_balance": start_balance,
        "end_balance": end_balance,
    }


def metric_card(title: str, value: str, tone: str = "") -> str:
    return f"""
    <div class="metric {tone}">
      <div class="metric-label">{esc(title)}</div>
      <div class="metric-value">{esc(value)}</div>
    </div>
    """


def render_dashboard_html() -> str:
    trade_path = first_existing(TRADES_CANDIDATES)
    equity_path = first_existing(EQUITY_CANDIDATES)
    metrics_path = first_existing(METRICS_CANDIDATES)
    raw_trades = read_csv(trade_path)
    trades = enrich_trades(raw_trades)
    metrics_json = read_json(metrics_path)
    equity = equity_from_csv(equity_path)
    if not equity:
        start_balance = safe_float(metrics_json.get("start_balance"), 100000.0)
        equity = equity_from_trades(trades, start_balance)
    metrics = calculate_metrics(trades, metrics_json, equity)
    weeks = weekly_breakdown(trades)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cards = "".join(
        [
            metric_card("Total Trades", str(metrics["total_trades"])),
            metric_card("Wins / Losses", f"{metrics['wins']} / {metrics['losses']}"),
            metric_card("Win Rate", f"{metrics['win_rate']:.2f}%", "good" if metrics["win_rate"] >= 50 else "bad"),
            metric_card("Total R", fmt_r(metrics["total_r"]), "good" if metrics["total_r"] >= 0 else "bad"),
            metric_card("Profit Factor", "inf" if math.isinf(metrics["profit_factor"]) else f"{metrics['profit_factor']:.2f}", "good" if metrics["profit_factor"] >= 1 else "bad"),
            metric_card("Net P&L", fmt_money(metrics["net_pnl"]), "good" if metrics["net_pnl"] >= 0 else "bad"),
            metric_card("Average R", fmt_r(metrics["average_r"]), "good" if metrics["average_r"] >= 0 else "bad"),
            metric_card("Avg Win / Loss R", f"{metrics['average_win_r']:.2f} / {metrics['average_loss_r']:.2f}"),
            metric_card("Max Drawdown", f"{metrics['max_dd_pct']:.2f}% / {fmt_money(metrics['max_dd'])}", "bad" if metrics["max_dd"] > 0 else "good"),
            metric_card("Max Consecutive Losses", str(metrics["max_consecutive_losses"])),
            metric_card("Sharpe", f"{metrics['sharpe']:.2f}", "good" if metrics["sharpe"] > 0 else "bad"),
            metric_card("Start / End Balance", f"{fmt_money(metrics['start_balance'])} / {fmt_money(metrics['end_balance'])}"),
        ]
    )

    payload = {
        "trades": trades,
        "weeks": weeks,
        "equity": equity,
        "sources": {
            "trades": str(trade_path) if trade_path else "missing",
            "equity": str(equity_path) if equity_path else "computed from trades",
            "metrics": str(metrics_path) if metrics_path else "computed from trades",
        },
    }
    data_json = json.dumps(payload)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Dashboard</title>
  <style>
    :root {{
      --bg: #101214;
      --panel: #171b20;
      --panel-2: #1f252c;
      --text: #e8eef5;
      --muted: #97a3b2;
      --line: #2c343d;
      --green: #31d07f;
      --green-bg: rgba(49, 208, 127, .12);
      --red: #ff5b5b;
      --red-bg: rgba(255, 91, 91, .12);
      --amber: #f4bf4f;
      --blue: #5aa7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      font-size: 14px;
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #12161a;
    }}
    h1 {{ margin: 0; font-size: 24px; }}
    .sub {{ color: var(--muted); margin-top: 6px; }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 17px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }}
    .metric-label {{ color: var(--muted); font-size: 12px; }}
    .metric-value {{ margin-top: 8px; font-size: 22px; font-weight: 700; }}
    .good .metric-value, .pos {{ color: var(--green); }}
    .bad .metric-value, .neg {{ color: var(--red); }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
    input, select {{
      background: #11161b;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      padding: 8px 10px;
      min-width: 150px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    th {{
      color: var(--muted);
      text-align: left;
      cursor: pointer;
      user-select: none;
      background: #151a1f;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr.win td {{ background: var(--green-bg); }}
    tr.loss td {{ background: var(--red-bg); }}
    .table-wrap {{ overflow: auto; max-height: 620px; border: 1px solid var(--line); border-radius: 8px; }}
    .badge {{ padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .badge.win, .badge.long, .badge.yes {{ color: var(--green); background: var(--green-bg); }}
    .badge.loss, .badge.short, .badge.no {{ color: var(--red); background: var(--red-bg); }}
    .badge.open {{ color: var(--amber); background: rgba(244, 191, 79, .12); }}
    canvas {{ width: 100%; height: 320px; background: #11161b; border: 1px solid var(--line); border-radius: 8px; }}
    .source {{ color: var(--muted); font-size: 12px; line-height: 1.6; }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Backtest Dashboard</h1>
    <div class="sub">Generated {esc(generated_at)}. Reads result files only; no backtest rerun.</div>
  </header>
  <main>
    <section>
      <h2>Summary Stats</h2>
      <div class="metrics">{cards}</div>
    </section>

    <section class="grid-2">
      <div>
        <h2>Equity Curve</h2>
        <canvas id="equityChart" width="900" height="320"></canvas>
      </div>
      <div>
        <h2>Weekly R</h2>
        <canvas id="weeklyChart" width="900" height="320"></canvas>
      </div>
    </section>

    <section>
      <h2>Trade Log</h2>
      <div class="toolbar">
        <input id="searchBox" placeholder="Search trades">
        <select id="resultFilter"><option value="">All results</option><option>Win</option><option>Loss</option><option>Open</option></select>
        <select id="directionFilter"><option value="">All directions</option><option>Long</option><option>Short</option></select>
        <select id="gradeFilter"><option value="">All grades</option></select>
      </div>
      <div class="table-wrap"><table id="tradeTable"></table></div>
    </section>

    <section>
      <h2>Weekly Breakdown</h2>
      <div class="table-wrap"><table id="weeklyTable"></table></div>
    </section>

    <section>
      <h2>Sources</h2>
      <div class="source" id="sources"></div>
    </section>
  </main>

  <script>
    const DATA = {data_json};
    const tradeColumns = [
      ['trade_id', 'Trade #'], ['date', 'Date'], ['day', 'Day'], ['entry_time', 'Open Time'],
      ['exit_time', 'Close Time'], ['holding_time', 'Holding'], ['instrument', 'Instrument'],
      ['direction', 'Direction'], ['entry_price', 'Entry'], ['sl_price', 'SL'], ['tp_price', 'TP'],
      ['sl_pips', 'SL Pips'], ['rr_ratio', 'R:R'], ['ttps_score', 'TTPS /20'], ['setup_quality', 'Grade'],
      ['result', 'Result'], ['r_gained', 'R Gained'], ['pnl', 'P&L'], ['week', 'Week']
    ];
    const weeklyColumns = [
      ['week', 'Week'], ['trades', 'Trades'], ['wins', 'Wins'], ['losses', 'Losses'],
      ['total_r', 'Total R'], ['pnl', 'Weekly P&L']
    ];
    let tradeSort = ['sort_time', true];
    let weeklySort = ['week', true];

    function money(v) {{ return '$' + Number(v || 0).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}); }}
    function num(v, d=2) {{ return Number(v || 0).toFixed(d); }}
    function cls(v) {{ return Number(v) > 0 ? 'pos' : Number(v) < 0 ? 'neg' : ''; }}
    function badge(value) {{ return `<span class="badge ${{String(value).toLowerCase()}}">${{value}}</span>`; }}

    function cell(key, value) {{
      if (key === 'result' || key === 'direction') return badge(value);
      if (key === 'pnl') return `<span class="${{cls(value)}}">${{money(value)}}</span>`;
      if (key === 'r_gained' || key === 'total_r') return `<span class="${{cls(value)}}">${{Number(value) > 0 ? '+' : ''}}${{num(value)}}R</span>`;
      if (['entry_price','sl_price','tp_price'].includes(key)) return num(value, 2);
      if (['sl_pips','rr_ratio'].includes(key)) return num(value, 2);
      return value ?? '';
    }}

    function sortRows(rows, sortSpec) {{
      const [key, asc] = sortSpec;
      return [...rows].sort((a,b) => {{
        const av = a[key], bv = b[key];
        const an = Number(av), bn = Number(bv);
        let result = !Number.isNaN(an) && !Number.isNaN(bn) ? an - bn : String(av ?? '').localeCompare(String(bv ?? ''));
        return asc ? result : -result;
      }});
    }}

    function renderTable(id, rows, columns, sortSpec, onSort) {{
      const table = document.getElementById(id);
      table.innerHTML = '<thead><tr>' + columns.map(([key,label]) => `<th data-key="${{key}}">${{label}}</th>`).join('') + '</tr></thead><tbody></tbody>';
      table.querySelectorAll('th').forEach(th => th.addEventListener('click', () => onSort(th.dataset.key)));
      const body = table.querySelector('tbody');
      body.innerHTML = sortRows(rows, sortSpec).map(row => {{
        const klass = row.result === 'Win' ? 'win' : row.result === 'Loss' ? 'loss' : '';
        return `<tr class="${{klass}}">` + columns.map(([key]) => `<td>${{cell(key, row[key])}}</td>`).join('') + '</tr>';
      }}).join('');
    }}

    function filteredTrades() {{
      const q = document.getElementById('searchBox').value.toLowerCase();
      const result = document.getElementById('resultFilter').value;
      const direction = document.getElementById('directionFilter').value;
      const grade = document.getElementById('gradeFilter').value;
      return DATA.trades.filter(t => {{
        const haystack = Object.values(t).join(' ').toLowerCase();
        return (!q || haystack.includes(q)) && (!result || t.result === result) && (!direction || t.direction === direction) && (!grade || t.setup_quality === grade);
      }});
    }}

    function refreshTrades() {{
      renderTable('tradeTable', filteredTrades(), tradeColumns, tradeSort, key => {{
        tradeSort = [key, tradeSort[0] === key ? !tradeSort[1] : true];
        refreshTrades();
      }});
    }}

    function refreshWeekly() {{
      renderTable('weeklyTable', DATA.weeks, weeklyColumns, weeklySort, key => {{
        weeklySort = [key, weeklySort[0] === key ? !weeklySort[1] : true];
        refreshWeekly();
      }});
    }}

    function drawEquity() {{
      const canvas = document.getElementById('equityChart'), ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height, pad = 34;
      ctx.clearRect(0,0,w,h);
      const values = DATA.equity.map(x => Number(x.balance));
      if (!values.length) return;
      const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
      ctx.strokeStyle = '#2c343d'; ctx.lineWidth = 1;
      for (let i=0;i<5;i++) {{ const y=pad+i*(h-2*pad)/4; ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(w-pad,y); ctx.stroke(); }}
      ctx.strokeStyle = '#5aa7ff'; ctx.lineWidth = 3; ctx.beginPath();
      values.forEach((v,i) => {{
        const x = pad + i * (w - 2*pad) / Math.max(values.length-1,1);
        const y = h - pad - (v - min) / span * (h - 2*pad);
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }});
      ctx.stroke();
      ctx.fillStyle = '#97a3b2'; ctx.fillText(money(max), 8, pad); ctx.fillText(money(min), 8, h-pad);
    }}

    function drawWeekly() {{
      const canvas = document.getElementById('weeklyChart'), ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height, pad = 34;
      ctx.clearRect(0,0,w,h);
      const values = DATA.weeks.map(x => Number(x.total_r));
      if (!values.length) return;
      const maxAbs = Math.max(...values.map(Math.abs), 1);
      const zero = h / 2;
      const barW = (w - 2*pad) / values.length * .72;
      ctx.strokeStyle = '#2c343d'; ctx.beginPath(); ctx.moveTo(pad, zero); ctx.lineTo(w-pad, zero); ctx.stroke();
      values.forEach((v,i) => {{
        const x = pad + i * (w - 2*pad) / values.length + barW*.2;
        const barH = Math.abs(v) / maxAbs * (h/2 - pad);
        ctx.fillStyle = v >= 0 ? '#31d07f' : '#ff5b5b';
        ctx.fillRect(x, v >= 0 ? zero - barH : zero, barW, barH);
      }});
    }}

    function init() {{
      const grades = [...new Set(DATA.trades.map(t => t.setup_quality).filter(Boolean))].sort();
      document.getElementById('gradeFilter').innerHTML += grades.map(g => `<option>${{g}}</option>`).join('');
      ['searchBox','resultFilter','directionFilter','gradeFilter'].forEach(id => document.getElementById(id).addEventListener('input', refreshTrades));
      refreshTrades();
      refreshWeekly();
      document.getElementById('sources').innerHTML = Object.entries(DATA.sources).map(([k,v]) => `<div>${{k}}: ${{v}}</div>`).join('');
      drawEquity(); drawWeekly();
    }}
    init();
  </script>
</body>
</html>"""


def write_html_dashboard() -> Path:
    DASHBOARD_HTML.write_text(render_dashboard_html(), encoding="utf-8")
    return DASHBOARD_HTML


def main() -> int:
    output_path = write_html_dashboard()
    print("Backtest dashboard generated")
    print(f"Open this file in your browser: {output_path}")
    print(f"Trades source: {first_existing(TRADES_CANDIDATES) or 'missing'}")
    print(f"Equity source: {first_existing(EQUITY_CANDIDATES) or 'computed from trades'}")
    print(f"Metrics source: {first_existing(METRICS_CANDIDATES) or 'computed from trades'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
