"""Phase E - rating factor (relativity) tables from the fitted GLMs.

This reproduces the core output of pricing tools such as WTW Radar / Emblem: one
multiplicative relativity per level of each rating factor, relative to a base
level (the level with the most exposure), which multiply together with a base
rate to give the technical price:

    technical price = base rate x age x NCD x vehicle group x region x mileage x young-VG4-5 loading

Both GLMs use a log link, so frequency and severity relativities for the same
factor multiply into a single combined (pure premium) relativity. The severity
model's under-25 flag is mapped onto the 17-21 and 22-24 age bands.

The script checks that the table reproduces the GLM technical price exactly,
then writes CSV/markdown tables, bar charts and a worked example.

Run:  python -m pricing.rating_factors   (after glm_frequency and glm_severity)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing.config import (
    COLOR_BASE, COLOR_PRIMARY, COLOR_SURFACE, COLOR_TEXT, COLOR_TEXT_MUTED, FACTOR_LABELS,
    FACTOR_LEVELS, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, base_levels, ensure_dirs,
    load_policies, style_axes,
)
from pricing.glm_severity import load_glms, technical_price

Z95 = 1.959964
YOUNG_BANDS = {"17-21", "22-24"}


def coef_lookup(res, factor, level, base):
    """(coefficient, std error) for one factor level; the base level is (0, 0)."""
    if level == base:
        return 0.0, 0.0
    key = f"C({factor}, Treatment(reference='{base}'))[T.{level}]"
    if key not in res.params:
        return 0.0, 0.0
    return res.params[key], res.bse[key]


def build_rating_table(freq_res, sev_res, bases, train):
    rows = []
    sev_factors = {name.split(",")[0][2:] for name in sev_res.params.index if name.startswith("C(")}
    for factor, levels in FACTOR_LEVELS.items():
        for level in levels:
            b_f, se_f = coef_lookup(freq_res, factor, level, bases[factor])
            if factor in sev_factors:
                b_s, se_s = coef_lookup(sev_res, factor, level, bases[factor])
            elif factor == "age_band" and level in YOUNG_BANDS:
                b_s, se_s = sev_res.params["young_driver"], sev_res.bse["young_driver"]
            else:
                b_s, se_s = 0.0, 0.0
            rows.append(_row(factor, level, level == bases[factor], b_f, se_f, b_s, se_s,
                             train[train[factor] == level]))
    # The interaction is a frequency-only loading, presented as its own factor.
    b_i, se_i = freq_res.params["young_high_vg"], freq_res.bse["young_high_vg"]
    for flag, label in [(0, "No"), (1, "Yes")]:
        rows.append(_row("young_high_vg", label, flag == 0,
                         b_i * flag, se_i * flag, 0.0, 0.0,
                         train[train["young_high_vg"] == flag]))
    return pd.DataFrame(rows)


def _row(factor, level, is_base, b_f, se_f, b_s, se_s, segment):
    # Frequency and severity are fitted separately; treat their estimates as independent.
    b_c, se_c = b_f + b_s, np.sqrt(se_f ** 2 + se_s ** 2)
    exposure = segment["exposure"].sum()
    return {
        "factor": factor,
        "level": level,
        "is_base": is_base,
        "exposure": exposure,
        "claims": int(segment["claim_count"].sum()),
        "oneway_frequency": segment["claim_count"].sum() / exposure,
        "frequency_relativity": np.exp(b_f),
        "severity_relativity": np.exp(b_s),
        "relativity": np.exp(b_c),
        "relativity_lower_95": np.exp(b_c - Z95 * se_c),
        "relativity_upper_95": np.exp(b_c + Z95 * se_c),
    }


def rate_policy(policy, table, base_rate):
    """Price a single policy from the rating table: base rate x one relativity per factor."""
    lookup = table.set_index(["factor", "level"])["relativity"]
    steps = [("Base rate", "", base_rate)]
    price = base_rate
    for factor in FACTOR_LEVELS:
        rel = lookup[(factor, policy[factor])]
        price *= rel
        steps.append((FACTOR_LABELS[factor], policy[factor], rel))
    flag = "Yes" if policy["young_high_vg"] == 1 else "No"
    rel = lookup[("young_high_vg", flag)]
    price *= rel
    steps.append((FACTOR_LABELS["young_high_vg"], flag, rel))
    return price, steps


def check_reproduces_glm(table, base_rate, test, freq_res, sev_res):
    tariff = np.array([rate_policy(p, table, base_rate)[0] for _, p in test.iterrows()])
    glm = technical_price(test, freq_res, sev_res)["glm_pure_premium"].to_numpy()
    max_diff = np.max(np.abs(tariff / glm - 1))
    assert max_diff < 1e-9, f"rating table does not reproduce GLM (max rel diff {max_diff})"
    return max_diff


def plot_relativities(table, path):
    factors = list(FACTOR_LEVELS) + ["young_high_vg"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    total_exposure = table.loc[table["factor"] == "age_band", "exposure"].sum()
    ymax = table["relativity_upper_95"].max() * 1.12

    for ax, factor in zip(axes.flat, factors):
        t = table[table["factor"] == factor].reset_index(drop=True)
        x = np.arange(len(t))
        colors = [COLOR_BASE if b else COLOR_PRIMARY for b in t["is_base"]]
        ax.bar(x, t["relativity"], width=0.62, color=colors, zorder=2)
        err = [t["relativity"] - t["relativity_lower_95"], t["relativity_upper_95"] - t["relativity"]]
        ax.errorbar(x, t["relativity"], yerr=err, fmt="none", ecolor=COLOR_TEXT_MUTED,
                    elinewidth=1, capsize=3, zorder=3)
        ax.axhline(1.0, color=COLOR_TEXT_MUTED, lw=1, ls="--", zorder=1)
        for xi, r, hi in zip(x, t["relativity"], t["relativity_upper_95"]):
            ax.text(xi, max(hi, 1.0) + ymax * 0.015, f"{r:.2f}", ha="center", va="bottom",
                    fontsize=8, color=COLOR_TEXT)
        labels = [f"{lvl}\n{e / total_exposure:.0%}" for lvl, e in zip(t["level"], t["exposure"])]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.6, max(len(t), 4) - 0.4)
        ax.set_title(FACTOR_LABELS[factor], loc="left", fontsize=10.5)
        style_axes(ax)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Motor technical price: rating factor relativities (frequency x severity GLM)",
                 x=0.01, ha="left", fontsize=13, color=COLOR_TEXT)
    fig.text(0.01, 0.935,
             "Grey bar = base level (1.00). Whiskers = 95% CI. Label under each level = share of exposure.",
             fontsize=9, color=COLOR_TEXT_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=COLOR_SURFACE)
    plt.close(fig)


def to_markdown(df):
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def markdown_report(table, base_rate, bases, example_price, example_steps, max_diff):
    lines = [
        "# Rating factor tables",
        "",
        "_Generated by `pricing/rating_factors.py` - do not edit by hand._",
        "",
        "Multiplicative relativities from the frequency (Negative Binomial) and severity "
        "(Gamma) GLMs, both log link, normalised to the base level of each factor "
        "(the level with the most exposure).",
        "",
        f"**Base rate: £{base_rate:,.2f}** per vehicle-year: the technical price for the "
        f"base risk (age {bases['age_band']}, NCD {bases['ncd_band']}, {bases['vehicle_group']}, "
        f"{bases['region']}, {bases['mileage_band']} miles).",
        "",
        f"Check: base rate x relativities reproduces the GLM technical price on every "
        f"holdout policy (max relative difference {max_diff:.1e}).",
        "",
        "![Rating factor relativities](figures/rating_factor_relativities.png)",
        "",
    ]
    for factor in list(FACTOR_LEVELS) + ["young_high_vg"]:
        t = table[table["factor"] == factor]
        out = pd.DataFrame({
            "Level": t["level"] + np.where(t["is_base"], " (base)", ""),
            "Exposure": t["exposure"].map("{:,.0f}".format),
            "Claims": t["claims"],
            "One-way freq": t["oneway_frequency"].map("{:.3f}".format),
            "Freq rel": t["frequency_relativity"].map("{:.3f}".format),
            "Sev rel": t["severity_relativity"].map("{:.3f}".format),
            "**Relativity**": t["relativity"].map("**{:.3f}**".format),
            "95% CI": [f"{lo:.2f} - {hi:.2f}" for lo, hi in
                       zip(t["relativity_lower_95"], t["relativity_upper_95"])],
        })
        lines += [f"## {FACTOR_LABELS[factor]}", "", to_markdown(out), ""]

    lines += [
        "## Worked example",
        "",
        "Pricing one policy straight off the tables:",
        "",
        "| Step | Level | Multiplier |",
        "|---|---|---|",
    ]
    for name, level, value in example_steps:
        value_text = f"£{value:,.2f}" if name == "Base rate" else f"× {value:.3f}"
        lines.append(f"| {name} | {level} | {value_text} |")
    lines += ["", f"**Technical price = £{example_price:,.2f} per vehicle-year.**", ""]
    return "\n".join(lines)


def main():
    ensure_dirs()
    df = load_policies()
    train, test = df[df["is_train"]], df[~df["is_train"]]
    bases = base_levels(train)
    freq_res, sev_res = load_glms()

    table = build_rating_table(freq_res, sev_res, bases, train)
    base_rate = np.exp(freq_res.params["Intercept"] + sev_res.params["Intercept"])
    max_diff = check_reproduces_glm(table, base_rate, test, freq_res, sev_res)
    print(f"Base rate £{base_rate:,.2f}; table reproduces GLM (max rel diff {max_diff:.1e})")

    example = pd.Series({"age_band": "17-21", "ncd_band": "1-2", "vehicle_group": "VG4",
                         "region": "London", "mileage_band": "10-15k", "young_high_vg": 1})
    example_price, example_steps = rate_policy(example, table, base_rate)
    print(f"Example (19yo, NCD 1, VG4, London, 10-15k): £{example_price:,.2f}")

    table.to_csv(TABLES_DIR / "rating_factors.csv", index=False)
    plot_relativities(table, FIGURES_DIR / "rating_factor_relativities.png")
    (REPORTS_DIR / "rating_factors.md").write_text(
        markdown_report(table, base_rate, bases, example_price, example_steps, max_diff),
        encoding="utf-8")
    print(table[["factor", "level", "frequency_relativity", "severity_relativity", "relativity"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
