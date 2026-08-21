# Live track record — season 2026/27

Real FPL team managed by the model. Two numbers are tracked separately:

- **model XI pts** — points the model's exact squad/captain/bench would have scored. This is the model's true track record.
- **account pts** — what the live FPL team actually scored (includes human execution at the deadline).

| GW | model | model XI pts | account pts | overall rank | notes |
|---|---|---|---|---|---|
| 1 | ML v0.2 | TBD (pred 72.4) | TBD | TBD | execution error, see log below |

## Execution log

- **GW1**: manual entry error minutes before the deadline — Schade (model: XI) and Walle Egeli (model: bench) were entered swapped. Model output unchanged and timestamped in `predictions/`; *model XI pts* for GW1 computed from the intended lineup. This is exactly why the roadmap ends with full automation — the human proved to be the least reliable component on day one.
