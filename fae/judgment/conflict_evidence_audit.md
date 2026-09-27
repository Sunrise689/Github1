# FAE Conflict Evidence Audit

- Status: **PASS**
- Conflict rules checked: **21**
- Scope: declarative evidence conditions in `conflict_rules.json`

## Evidence fields used by rules

- `bearish_bar_count`
- `bullish_bar_count`
- `consecutive_bearish_bars`
- `continuation_down`
- `continuation_up`

## Availability

- Static extractor contract: `bearish_bar_count, bullish_bar_count, consecutive_bearish_bars, continuation_down, continuation_up`
- Runtime top-level fields: `bearish_bar_count, bullish_bar_count, consecutive_bearish_bars, continuation_down, continuation_up`
- Runtime signal-local fields: `bearish_bar_count, bullish_bar_count, consecutive_bearish_bars, continuation_down, continuation_up`
- Missing statically: `none`
- Missing at runtime: `none`

## Semantics

`bullish_bar_count`, `bearish_bar_count`, and consecutive counts are calculated on the candidate span when signal-local evidence is available. `continuation_up/down` requires at least two post-candidate closes moving strictly in that direction; a candidate at the end of the data is not invalidated merely because the global trend points the other way.
