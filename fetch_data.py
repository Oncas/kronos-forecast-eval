"""Download BTCUSDT daily candles from Binance's public API (no key needed)."""
import json
import urllib.request
from datetime import datetime, timezone

import csv

URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start}&limit=1000"
START = datetime(2022, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 2, 1, tzinfo=timezone.utc)  # exclusive

rows = []
start_ms = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000)
while start_ms < end_ms:
    with urllib.request.urlopen(URL.format(start=start_ms), timeout=30) as r:
        batch = json.load(r)
    if not batch:
        break
    for k in batch:
        if k[0] >= end_ms:
            break
        rows.append(k)
    start_ms = batch[-1][0] + 1

with open("btcusdt_1d.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamps", "open", "high", "low", "close", "volume", "amount"])
    for k in rows:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        w.writerow([ts, k[1], k[2], k[3], k[4], k[5], k[7]])
print(len(rows), rows[0][0], rows[-1][0])
