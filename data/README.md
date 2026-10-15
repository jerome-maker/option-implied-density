# Live option-chain snapshot

`experiments/run_wp4.py` reads the most recent folder `data/wp4_*/`, which must
contain one file per chain named `<TICKER>_<DDD>d.json` and a `rates.json`.

- Chain files: the JSON returned by Yahoo Finance's option-chain endpoint for a
  single expiry (`optionChain.result[0]` with `quote` and `options[0].calls/puts`).
  `src/real_chains.load_otm_chain` reads the spot price, quote time, expiry,
  dividend yield and each contract's strike, bid and ask.
- `rates.json`: `{"irx_pct": 4.08, "fetched_utc": "2026-10-15T15:23:51.840939+00:00"}`.

The raw quotes are not redistributed. The files used in the paper
(folder `wp4_20261015_1523UTC/`) have these SHA-256 fingerprints:

```
925fcac213b56bf92ee9dd08dbe0759a86fee9ecf7b487653840a368d7611ff4  IWM_015d.json
d631862f4e4b1ca3d5564af470464bfc229970e1a13d76ad6b7dccf6edb2b0d9  IWM_029d.json
03ff70e4145ea420cb84667f8ad8a80ac41f386e9f4a482b671c5143b54b78dc  IWM_064d.json
35b4cba60f6720e35f5f1d8d2cc7cbfa75e62b94a76836900b5c1d3dd8778549  IWM_092d.json
88f99cb92e5ebfd9f8b4ce4631e89dbd149c8bc3179b9c7c5eb921f68c9a0b4b  QQQ_015d.json
8dd17344c3da825b13b58a380fc614e1fd23531a4dd82f0c39e47d62cf8e1e56  QQQ_029d.json
96045a615dd87abc92b6acdf8ef5f6881342ab105cc8e099047054531b49dd89  QQQ_064d.json
bc4a646933edac44748984561c773cc3433a1a62b4723addb7faf6778d794034  QQQ_092d.json
65b4a13716b48f76b92d5065f13956e6198aa508d66246fbef2f3b6cf2f63d91  SPY_015d.json
e4b92e0594fe9dd693c31f12b94fadf347fdb7efd49385d4632f48351dd965fa  SPY_029d.json
d16968c60ae5ff0acbbfe7d49ecaa0588b47b8df9285859e4ebb4495521a3e4c  SPY_064d.json
3de5bcaf9efcf7e081f9b8069e7b6ae2aae68a6f68814edc2c503aa7c8005e1a  SPY_092d.json
98e53f8fcfcf1b18461121781a0a122a0c64c8b3a93084f9256e619ce353a149  rates.json
```
