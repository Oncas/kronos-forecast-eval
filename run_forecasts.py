"""Walk-forward BTC daily forecasts with Kronos, keeping every sample path.

For each target day D in [--start, --end], the context is the `--lookback`
fully closed daily candles strictly before D. The model then forecasts
`--pred-len` days starting at D. KronosPredictor.predict() averages its samples
internally, so to keep the distribution we call predict_batch() with
sample_count=1 and repeat each series `--samples` times in the batch.

Output: forecasts_<model>.npz with
    samples   float32 [n_origins, n_samples, pred_len, 6]  (open, high, low, close, volume, amount)
    targets   str     [n_origins]  first forecast day (YYYY-MM-DD)
    last_close float  [n_origins]  close of the day before the target
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}
COLS = ["open", "high", "low", "close", "volume", "amount"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="small")
    p.add_argument("--data", default=str(Path(__file__).with_name("btcusdt_1d.csv")))
    p.add_argument("--start", default="2025-09-01")
    p.add_argument("--end", default="2026-01-31")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--pred-len", type=int, default=7)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--origins-per-batch", type=int, default=8)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="only run the first N origins (smoke test)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    model_id, tok_id, max_ctx = MODELS[args.model]
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model = Kronos.from_pretrained(model_id)
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=max_ctx)

    df = pd.read_csv(args.data)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    assert df["timestamps"].diff().dropna().eq(pd.Timedelta(days=1)).all(), "gap in daily data"

    targets = pd.date_range(args.start, args.end, freq="D")
    if args.limit:
        targets = targets[: args.limit]
    idx_of = {ts: i for i, ts in enumerate(df["timestamps"])}

    jobs = []
    for d in targets:
        i = idx_of[d]
        assert i >= args.lookback, f"not enough history before {d.date()}"
        ctx = df.iloc[i - args.lookback : i]
        assert ctx["timestamps"].iloc[-1] == d - pd.Timedelta(days=1)
        y_ts = pd.Series(pd.date_range(d, periods=args.pred_len, freq="D"))
        jobs.append((d, ctx, y_ts))

    out = np.empty((len(jobs), args.samples, args.pred_len, len(COLS)), dtype=np.float32)
    t0 = time.time()
    for b in range(0, len(jobs), args.origins_per_batch):
        chunk = jobs[b : b + args.origins_per_batch]
        df_list, x_list, y_list = [], [], []
        for _, ctx, y_ts in chunk:
            for _ in range(args.samples):
                df_list.append(ctx[COLS].reset_index(drop=True))
                x_list.append(ctx["timestamps"].reset_index(drop=True))
                y_list.append(y_ts)
        preds = predictor.predict_batch(
            df_list, x_list, y_list, pred_len=args.pred_len,
            T=args.T, top_k=0, top_p=args.top_p, sample_count=1, verbose=False,
        )
        for j in range(len(chunk)):
            for s in range(args.samples):
                out[b + j, s] = preds[j * args.samples + s][COLS].to_numpy()
        done = b + len(chunk)
        print(f"[{args.model}] {done}/{len(jobs)} origins  {time.time() - t0:.0f}s", flush=True)

    last_close = np.array([ctx["close"].iloc[-1] for _, ctx, _ in jobs], dtype=np.float64)
    out_path = args.out or str(Path(__file__).with_name(f"forecasts_{args.model}.npz"))
    np.savez_compressed(
        out_path,
        samples=out,
        targets=np.array([d.strftime("%Y-%m-%d") for d, _, _ in jobs]),
        last_close=last_close,
        meta=np.array([f"model={args.model} lookback={args.lookback} pred_len={args.pred_len} "
                       f"samples={args.samples} T={args.T} top_p={args.top_p} seed={args.seed}"]),
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
