"""Score Kronos BTC daily forecasts against what actually happened.

Reads forecasts_<model>.npz (from run_forecasts.py) and btcusdt_1d.csv, and
compares Kronos to two naive baselines:
  * random walk: tomorrow's close = today's close (point forecast), and a
    Gaussian with the trailing 30-day volatility of log returns (distribution)
  * trailing 20-day mean range for the high-low range

Writes results.json, a printed summary, and plots per model.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}
FEE = 0.001  # per side, per position change
# Evaluation names that pool several forecast files (all out of sample, back to back).
COMBINED = {"small_all": ["small_ext", "small"], "small_all_lb90": ["small_ext_lb90", "small_lb90"]}
COMBINED.update({f"{m}_all_lb40": [f"{m}_ext_lb40", f"{m}_lb40"] for m in ("mini", "small", "base")})


def binom_two_sided_p(k, n, p=0.5):
    logpmf = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
              + i * math.log(p) + (n - i) * math.log(1 - p) for i in range(n + 1)]
    pmf = [math.exp(x) for x in logpmf]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] * (1 + 1e-9)))


def rankdata(a):
    return pd.Series(a).rank().to_numpy()


def crps_samples(samples, y):
    """Fair energy-form CRPS estimator for an ensemble: E|X-y| - 0.5 E|X-X'|.

    E|X-X'| excludes the zero self-pairs so a finite ensemble isn't penalised
    against the closed-form Gaussian baseline.
    """
    s = np.asarray(samples, dtype=np.float64)
    n = len(s)
    term1 = np.mean(np.abs(s - y))
    term2 = np.abs(s[:, None] - s[None, :]).sum() / (n * (n - 1))
    return term1 - 0.5 * term2


def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + math.erf(z / np.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def strategy(pos, act_ret):
    """Daily returns of holding `pos` (1 long, 0 flat, -1 short) each day, with fees."""
    pos = np.asarray(pos, dtype=float)
    prev = np.concatenate([[0.0], pos[:-1]])
    fees = np.abs(pos - prev) * FEE
    return pos * act_ret - fees


def summarize_equity(r):
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    sharpe = np.mean(r) / np.std(r) * np.sqrt(365) if np.std(r) > 0 else float("nan")
    return {"total_return": float(eq[-1] - 1), "max_drawdown": float(np.min(eq / peak - 1)), "sharpe_ann": float(sharpe)}


def evaluate_model(name, npz, df):
    samples = npz["samples"].astype(np.float64)  # [N, S, H, 6]
    targets = pd.to_datetime(npz["targets"])
    last_close = npz["last_close"]
    by_date = df.set_index("timestamps")
    N, S, H, _ = samples.shape
    log_ret = np.log(df["close"]).diff()
    res = {"model": name, "meta": str(npz["meta"][0]), "n_origins": int(N), "n_samples": int(S), "horizons": {}}

    for h in sorted({1, min(7, H)}):
        rows = []
        for i, d in enumerate(targets):
            day = d + pd.Timedelta(days=h - 1)
            if day not in by_date.index:
                continue
            act = by_date.loc[day]
            s = samples[i, :, h - 1, :]
            # Trailing 30-day vol of log returns, known at the close before the target.
            prev_idx = df.index[df["timestamps"] == d - pd.Timedelta(days=1)][0]
            vol = log_ret.iloc[prev_idx - 29 : prev_idx + 1].std()
            # For h>1 the relevant high/low is over the whole window, not the single day.
            win = by_date.loc[d:day]
            rows.append(dict(
                date=day, origin=d - pd.Timedelta(days=1), lc=last_close[i],
                act_close=act["close"], act_high=win["high"].max(), act_low=win["low"].min(),
                pred_close_mean=s[:, C["close"]].mean(), pred_close_median=np.median(s[:, C["close"]]),
                pred_high=samples[i, :, :h, C["high"]].max(axis=1).mean(),
                pred_low=samples[i, :, :h, C["low"]].min(axis=1).mean(),
                p_up=(s[:, C["close"]] > last_close[i]).mean(),
                pred_open_d1=samples[i, :, 0, C["open"]].mean(),
                q05=np.quantile(s[:, C["close"]], 0.05), q10=np.quantile(s[:, C["close"]], 0.10),
                q90=np.quantile(s[:, C["close"]], 0.90), q95=np.quantile(s[:, C["close"]], 0.95),
                crps=crps_samples(s[:, C["close"]], act["close"]),
                vol=vol,
                base_range=(df["high"] - df["low"]).iloc[prev_idx - 19 : prev_idx + 1].mean() * (h ** 0.5),
            ))
        r = pd.DataFrame(rows)
        act_ret = r["act_close"] / r["lc"] - 1
        pred_ret = r["pred_close_mean"] / r["lc"] - 1
        pred_ret_med = r["pred_close_median"] / r["lc"] - 1

        # Point accuracy on close
        mae_k = (r["pred_close_mean"] - r["act_close"]).abs().mean()
        mae_k_med = (r["pred_close_median"] - r["act_close"]).abs().mean()
        mae_rw = (r["lc"] - r["act_close"]).abs().mean()
        mape_k = ((r["pred_close_mean"] - r["act_close"]).abs() / r["act_close"]).mean()
        mape_rw = ((r["lc"] - r["act_close"]).abs() / r["act_close"]).mean()

        # Direction (ties where actual return is exactly 0 are dropped). For h>1 the
        # windows of consecutive origins overlap, so significance and correlations
        # use only every h-th origin to keep observations independent.
        indep = np.zeros(len(r), dtype=bool)
        indep[::h] = True
        mask = (act_ret != 0).to_numpy()
        hit = (np.sign(pred_ret) == np.sign(act_ret)).to_numpy()
        hits = int((hit & mask).sum())
        n_dir = int(mask.sum())
        hits_i = int((hit & mask & indep).sum())
        n_dir_i = int((mask & indep).sum())
        # Alternative signal: predicted day-h close vs predicted day-1 open, both
        # through the tokenizer, so a systematic reconstruction bias cancels out.
        pred_ret_oc = r["pred_close_mean"] / r["pred_open_d1"] - 1
        hits_oc = int(((np.sign(pred_ret_oc) == np.sign(act_ret)).to_numpy() & mask).sum())
        hits_oc_i = int(((np.sign(pred_ret_oc) == np.sign(act_ret)).to_numpy() & mask & indep).sum())
        tok_bias = r["pred_open_d1"] / r["lc"] - 1
        up_days = int((act_ret > 0).sum())
        pred_up_days = int((pred_ret > 0).sum())
        brier_k = ((r["p_up"] - (act_ret > 0)) ** 2).mean()
        brier_half = 0.25

        # Range / volatility skill
        pred_range = r["pred_high"] - r["pred_low"]
        act_range = r["act_high"] - r["act_low"]
        range_mae_k = (pred_range - act_range).abs().mean()
        range_mae_base = (r["base_range"] - act_range).abs().mean()

        # Distribution
        sig = r["vol"] * np.sqrt(h) * r["lc"]
        crps_rw = np.mean([crps_gaussian(lc, sg, y) for lc, sg, y in zip(r["lc"], sig, r["act_close"])])
        cov80 = ((r["act_close"] >= r["q10"]) & (r["act_close"] <= r["q90"])).mean()
        cov90 = ((r["act_close"] >= r["q05"]) & (r["act_close"] <= r["q95"])).mean()
        width80_k = ((r["q90"] - r["q10"]) / r["lc"]).mean()
        z80 = 1.2815515655446004
        cov80_rw = ((r["act_close"] - r["lc"]).abs() <= z80 * sig).mean()
        width80_rw = (2 * z80 * sig / r["lc"]).mean()

        hres = {
            "n": int(len(r)),
            "close_mae_kronos_mean": float(mae_k),
            "close_mae_kronos_median": float(mae_k_med),
            "close_mae_random_walk": float(mae_rw),
            "close_mae_ratio_vs_rw": float(mae_k / mae_rw),
            "close_mape_kronos": float(mape_k),
            "close_mape_random_walk": float(mape_rw),
            "direction_hits": hits, "direction_n": n_dir,
            "direction_accuracy": hits / n_dir,
            "direction_independent_n": n_dir_i,
            "direction_accuracy_independent": hits_i / n_dir_i,
            "direction_p_value_vs_coinflip": binom_two_sided_p(hits_i, n_dir_i),
            "direction_accuracy_open_to_close_signal": hits_oc / n_dir,
            "direction_p_value_open_to_close_signal": binom_two_sided_p(hits_oc_i, n_dir_i),
            "tokenizer_open_bias_mean_pct": float(tok_bias.mean() * 100),
            "tokenizer_open_bias_corr_with_actual_ret": float(np.corrcoef(tok_bias, act_ret)[0, 1]),
            "actual_up_fraction": up_days / len(r),
            "predicted_up_fraction": pred_up_days / len(r),
            "brier_kronos_p_up": float(brier_k), "brier_coinflip": brier_half,
            "ret_corr_pearson": float(np.corrcoef(pred_ret[indep], act_ret[indep])[0, 1]),
            "ret_corr_spearman": float(np.corrcoef(rankdata(pred_ret[indep]), rankdata(act_ret[indep]))[0, 1]),
            "ret_corr_pearson_median_forecast": float(np.corrcoef(pred_ret_med[indep], act_ret[indep])[0, 1]),
            "mean_pred_return": float(pred_ret.mean()), "mean_actual_return": float(act_ret.mean()),
            "range_mae_kronos": float(range_mae_k), "range_mae_trailing20": float(range_mae_base),
            "range_corr_kronos": float(np.corrcoef(pred_range, act_range)[0, 1]),
            "range_corr_trailing20": float(np.corrcoef(r["base_range"], act_range)[0, 1]),
            "crps_kronos": float(r["crps"].mean()), "crps_random_walk_gauss": float(crps_rw),
            "coverage80_kronos": float(cov80), "coverage90_kronos": float(cov90),
            "width80_kronos_pct": float(width80_k),
            "coverage80_random_walk": float(cov80_rw), "width80_random_walk_pct": float(width80_rw),
        }
        if h == 1:
            ar = act_ret.to_numpy()
            long_flat = np.where(pred_ret > 0, 1, 0)
            long_short = np.where(pred_ret > 0, 1, -1)
            hres["strategy"] = {
                "buy_and_hold": summarize_equity(strategy(np.ones(len(ar)), ar)),
                "kronos_long_flat": summarize_equity(strategy(long_flat, ar)),
                "kronos_long_short": summarize_equity(strategy(long_short, ar)),
                "trades_long_flat": int(np.abs(np.diff(np.concatenate([[0], long_flat]))).sum()),
            }
            res["_daily"] = r.assign(pred_ret=pred_ret, act_ret=act_ret)
        res["horizons"][f"h{h}"] = hres
    return res


def plot(res, df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = res["_daily"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 2, 2]})
    ax = axes[0]
    ax.fill_between(r["date"], r["q10"], r["q90"], alpha=0.25, label="Kronos 10-90% band")
    ax.plot(r["date"], r["pred_close_mean"], label="Kronos 1-day-ahead close (mean)", lw=1.2)
    ax.plot(r["date"], r["act_close"], label="Actual close", color="k", lw=1.2)
    ax.set_title(f"BTCUSDT daily, 1-day-ahead forecasts - Kronos-{res['model']}")
    ax.legend()
    ax = axes[1]
    ax.scatter(r["pred_ret"] * 100, r["act_ret"] * 100, s=10)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Predicted next-day return %")
    ax.set_ylabel("Actual next-day return %")
    ax = axes[2]
    ar = r["act_ret"].to_numpy()
    for label, pos in [("Buy & hold", np.ones(len(ar))),
                       ("Kronos long/flat", np.where(r["pred_ret"] > 0, 1, 0)),
                       ("Kronos long/short", np.where(r["pred_ret"] > 0, 1, -1))]:
        ax.plot(r["date"], np.cumprod(1 + strategy(pos, ar)), label=label)
    ax.set_title("Equity (0.1% fee per side)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=["mini", "small", "base", "small_ext", "small_all",
                                                  "mini_lb40", "small_lb40", "base_lb40",
                                                  "mini_all_lb40", "small_all_lb40", "base_all_lb40",
                                                  "small_lb40_topp1", "small_lb90", "small_all_lb90", "small_7d_lb40"])
    args = p.parse_args()
    df = pd.read_csv(HERE / "btcusdt_1d.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    out_json = HERE / "results.json"
    all_res = json.loads(out_json.read_text()) if out_json.exists() else {}
    for m in args.models:
        parts = COMBINED.get(m, [m])
        files = [HERE / f"forecasts_{x}.npz" for x in parts]
        if not all(f.exists() for f in files):
            print(f"skip {m}: missing {[f.name for f in files if not f.exists()]}")
            continue
        loaded = [np.load(f) for f in files]
        npz = {k: np.concatenate([z[k] for z in loaded]) for k in ("samples", "targets", "last_close")}
        npz["meta"] = loaded[0]["meta"]
        res = evaluate_model(m, npz, df)
        plot(res, df, HERE / f"eval_{m}.png")
        res["_daily"].to_csv(HERE / f"daily_{m}.csv", index=False)
        del res["_daily"]
        all_res[m] = res
        print(json.dumps(res, indent=2))
    out_json.write_text(json.dumps(all_res, indent=2))


if __name__ == "__main__":
    main()
