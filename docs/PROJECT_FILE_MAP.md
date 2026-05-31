# Reduced Project File Map

This project was reduced to these files only:

- `main.py`: command launcher.
- `.env`: runtime settings.
- `risk_manager.py`: prints risk rules and exposes basic risk amount calculation.
- `risk_manger.py`: compatibility alias for the misspelled risk manager name.
- `filter.py`: checks the CSV news filter if enabled.
- `FVGs.py`: detects fair value gaps from saved W1, D1, and H4 history.
- `history.py`: checks for saved history CSV files.
- `train.py`: prints a simple history availability summary.
- `dashboard.py`: lightweight dashboard placeholder/status command.
- `docs/PROJECT_FILE_MAP.md`: this file.

Supported commands:

```powershell
python main.py env
python main.py risk
python main.py filter
python main.py train
python main.py backtest
python main.py dashboard
python main.py export-history
python main.py fvgs
```
