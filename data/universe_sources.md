# Universe Sources (Starter)

`data/universe.txt` is the editable universe used for daily momentum ranking.

## Included buckets
- Broad U.S. large-cap and liquid mid-cap equities
- High-liquidity U.S. ETFs (core index, sector, style, rates, commodities, international)
- Additional symbols merged from local scanner watchlists

## File format
Text file with one ticker per line:

```text
# comment lines are ignored
AAPL
MSFT
NVDA
```

- Lines beginning with `#` are treated as comments.
- Empty lines are ignored.
- Keep symbols in standard Yahoo/TradingView style (for example `BRK-B`).

## Expanding to a full institutional universe
To make this universe even better:
1. Append full S&P 500 + Nasdaq-100 constituent lists.
2. Add a Russell 1000/3000 liquid subset.
3. Keep ADRs/microcaps only if they pass your liquidity filters.
4. Re-run `scripts/build_data.py` to regenerate `data/snapshot.json`, `data/meta.json`, and `data/events.json`.
