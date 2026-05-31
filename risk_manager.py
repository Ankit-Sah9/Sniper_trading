from __future__ import annotations

from risk_manger import (
    PositionSize,
    RiskRules,
    TradePermission,
    calculate_lot_size,
    calculate_risk_amount,
    main,
    register_trade_open,
    risk_summary,
    trade_permission_gate,
    update_after_trade_close,
)

__all__ = [
    "PositionSize",
    "RiskRules",
    "TradePermission",
    "calculate_lot_size",
    "calculate_risk_amount",
    "main",
    "register_trade_open",
    "risk_summary",
    "trade_permission_gate",
    "update_after_trade_close",
]


if __name__ == "__main__":
    raise SystemExit(main())
