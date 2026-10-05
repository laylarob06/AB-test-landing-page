"""A/B test analysis: does a redesigned landing page raise e-commerce conversion?

Pipeline: CSV -> SQLite (raw tables) -> SQL integrity checks -> clean one-row-per-user table
-> SQL metrics -> Python statistics (SRM check, two-proportion z-test, confidence interval,
power / minimum detectable effect, logistic regression for country-level effects) -> chart.

Run:  python analysis.py
"""
import json
import re
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
ALPHA, POWER = 0.05, 0.80


def load_queries(path):
    """Split queries.sql into {name: sql} using '-- name: <key>' tags."""
    parts = re.split(r"^-- name: (\w+)\s*$", path.read_text(), flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def show(title, df):
    print(f"\n== {title}\n{df.to_string(index=False)}")


# ---------------------------------------------------------------- load + SQL
con = sqlite3.connect(":memory:")
pd.read_csv(ROOT / "data/ab_data.csv").to_sql("raw_events", con, index=False)
pd.read_csv(ROOT / "data/countries.csv").to_sql("raw_countries", con, index=False)
q = load_queries(ROOT / "sql/queries.sql")
sql = lambda name: pd.read_sql(q[name], con)

for name in ["raw_row_counts", "missing_values", "assignment_matrix", "experiment_window"]:
    show(name, sql(name))

con.executescript(q["build_experiment_users"])
cleaning = sql("cleaning_summary")
show("cleaning_summary", cleaning)

by_group = sql("conversion_by_group").set_index("grp")
show("conversion_by_group", by_group.reset_index())
df = pd.read_sql("SELECT * FROM experiment_users", con)

# ---------------------------------------------------------------- statistics
n_c, n_t = by_group.loc["control", "users"], by_group.loc["treatment", "users"]
x_c, x_t = by_group.loc["control", "conversions"], by_group.loc["treatment", "conversions"]
p_c, p_t = x_c / n_c, x_t / n_t
diff = p_t - p_c
z_crit = stats.norm.ppf(1 - ALPHA / 2)

# 1) Sample-ratio mismatch: was traffic really split 50/50?
srm_chi2, srm_p = stats.chisquare([n_c, n_t])

# 2) Two-proportion z-test (pooled SE) + 95% CI on the difference (unpooled SE)
p_pool = (x_c + x_t) / (n_c + n_t)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
z = diff / se_pool
p_two_sided = 2 * (1 - stats.norm.cdf(abs(z)))
p_one_sided = 1 - stats.norm.cdf(z)            # H1: new page converts better
se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

# 3) Power: smallest absolute lift this sample could reliably detect (80% power, alpha 0.05)
mde = (z_crit + stats.norm.ppf(POWER)) * np.sqrt(p_c * (1 - p_c) * (1 / n_c + 1 / n_t))

# 4) Logistic regression: does the treatment effect differ by country?
df["treatment"] = (df["grp"] == "treatment").astype(int)
base = smf.logit("converted ~ treatment + C(country)", data=df).fit(disp=0)
inter = smf.logit("converted ~ treatment * C(country)", data=df).fit(disp=0)
lr_stat = 2 * (inter.llf - base.llf)
lr_df = inter.df_model - base.df_model
lr_p = 1 - stats.chi2.cdf(lr_stat, lr_df)

results = {
    "clean_users": int(n_c + n_t),
    "control": {"users": int(n_c), "conversions": int(x_c), "rate": round(p_c, 5)},
    "treatment": {"users": int(n_t), "conversions": int(x_t), "rate": round(p_t, 5)},
    "abs_diff_pp": round(diff * 100, 3),
    "relative_lift_pct": round(diff / p_c * 100, 2),
    "ci95_abs_diff_pp": [round(ci[0] * 100, 3), round(ci[1] * 100, 3)],
    "z": round(z, 3),
    "p_two_sided": round(p_two_sided, 4),
    "p_one_sided_new_better": round(p_one_sided, 4),
    "srm_chi2": round(srm_chi2, 3),
    "srm_p": round(srm_p, 4),
    "mde_abs_pp": round(mde * 100, 3),
    "mde_relative_pct": round(mde / p_c * 100, 2),
    "country_interaction_lr_p": round(lr_p, 4),
    "treatment_odds_ratio": round(float(np.exp(base.params["treatment"])), 4),
    "treatment_coef_p": round(float(base.pvalues["treatment"]), 4),
}
results["decision"] = (
    "Ship new page" if (p_two_sided < ALPHA and diff > 0)
    else "Do not ship: no statistically significant lift"
)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("\n== results\n" + json.dumps(results, indent=2))

# ---------------------------------------------------------------- chart
daily = sql("daily_conversion")
daily["day"] = pd.to_datetime(daily["day"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})

labels, rates = ["Control\n(old page)", "Treatment\n(new page)"], [p_c, p_t]
errs = [z_crit * np.sqrt(p * (1 - p) / n) for p, n in [(p_c, n_c), (p_t, n_t)]]
for i, (r, e, c) in enumerate(zip(rates, errs, ["#9aa5b1", "#1652f0"])):
    a1.errorbar(i, r * 100, yerr=e * 100, fmt="o", ms=9, capsize=8, color=c, lw=2)
    a1.text(i + 0.12, r * 100, f"{r:.2%}", va="center", fontsize=10)
a1.set_xticks([0, 1], labels)
a1.set_xlim(-0.5, 1.7)
a1.set_ylim(11.4, 12.5)
a1.set_ylabel("Conversion rate (%)")
a1.set_title(f"Conversion with 95% CIs  (p = {p_two_sided:.2f})", fontsize=11)

for grp, color, label in [("control", "#9aa5b1", "Control"), ("treatment", "#1652f0", "Treatment")]:
    d = daily[daily["grp"] == grp]
    a2.plot(d["day"], d["conversion_rate"] * 100, marker="o", ms=3, color=color, label=label)
a2.set_ylabel("Daily conversion rate (%)")
a2.set_title("Daily conversion: no sustained gap between arms", fontsize=11)
a2.legend(frameon=False)
a2.tick_params(axis="x", rotation=30)
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "conversion_results.png", dpi=150)
print(f"\nSaved {OUT / 'conversion_results.png'} and {OUT / 'results.json'}")
