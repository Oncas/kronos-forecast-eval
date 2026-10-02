"""Print one comparison table from results.json."""
import json
from pathlib import Path

r = json.loads((Path(__file__).with_name("results.json")).read_text())
# Direction accuracy, its p-value and the correlation use only non-overlapping
# windows (every h-th origin), so for h>1 they are computed on far fewer days.
cols = [("n", "n", "{:d}"), ("MAE/RW", "close_mae_ratio_vs_rw", "{:.2f}"),
        ("dir n", "direction_independent_n", "{:d}"), ("dir acc", "direction_accuracy_independent", "{:.1%}"), ("p", "direction_p_value_vs_coinflip", "{:.2f}"),
        ("corr", "ret_corr_pearson", "{:+.3f}"), ("pred up", "predicted_up_fraction", "{:.0%}"),
        ("act up", "actual_up_fraction", "{:.0%}"), ("cov80", "coverage80_kronos", "{:.0%}"),
        ("CRPS/RW", None, "{:.2f}"), ("rangeMAE/base", None, "{:.2f}"),
        ("L/S ret", None, "{:+.0%}"), ("B&H ret", None, "{:+.0%}")]
print(f"{'run':18s}" + "".join(f"{c[0]:>14s}" for c in cols))
for name, res in r.items():
    for hk, h in res["horizons"].items():
        vals = []
        for label, key, fmt in cols:
            if label == "CRPS/RW":
                v = h["crps_kronos"] / h["crps_random_walk_gauss"]
            elif label == "rangeMAE/base":
                v = h["range_mae_kronos"] / h["range_mae_trailing20"]
            elif label in ("L/S ret", "B&H ret"):
                st = h.get("strategy")
                if not st:
                    vals.append(f"{'-':>14s}"); continue
                v = st["kronos_long_short" if label == "L/S ret" else "buy_and_hold"]["total_return"]
            else:
                v = h[key]
            vals.append(f"{fmt.format(v):>14s}")
        print(f"{name + '/' + hk:18s}" + "".join(vals))
