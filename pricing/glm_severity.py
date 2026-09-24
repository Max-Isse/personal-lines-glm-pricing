"""Phase C - claim severity GLM (Gamma, log link) and the combined technical price.

Severity model, fitted on policies with at least one claim:
    avg_cost = claim_cost / claim_count
    avg_cost ~ Gamma(mu, phi / claim_count),   log(mu) = b0 + sum_j b_j x_j

The response is the policy's average claim cost, weighted by claim count. An
average of n claims has 1/n the variance of a single claim, so this is
equivalent to modelling each claim individually.

Specifications (nested, most to least complex):
    S1  age_band + ncd_band + vehicle_group + region + mileage_band
    S2  age_band + vehicle_group + region
    S3  young_driver + vehicle_group + region
The Gamma dispersion is estimated, so nested models are compared with F-tests on
the deviance (not chi-square). The simplest model that is not rejected wins.

Technical (risk) price = E[frequency] x E[severity] per vehicle-year. Both
models use a log link, so this equals exp(freq linear predictor + sev linear
predictor), and the rating factor relativities multiply.

Run:  python -m pricing.glm_severity   (after pricing.glm_frequency)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from pricing import generate_data as gen
from pricing.config import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_TEXT_MUTED, FIGURES_DIR,
    MODELS_DIR, TABLES_DIR, base_levels, ensure_dirs, factor_term, load_policies,
    style_axes,
)
from pricing.glm_frequency import FREQUENCY_MODEL_FILE

SEVERITY_MODEL_FILE = MODELS_DIR / "severity_glm.pickle"


def severity_specs(bases):
    t = {f: factor_term(f, bases[f]) for f in bases}
    return {
        "S1": ("All five factors",
               [t["age_band"], t["ncd_band"], t["vehicle_group"], t["region"], t["mileage_band"]]),
        "S2": ("Age band + vehicle group + region",
               [t["age_band"], t["vehicle_group"], t["region"]]),
        "S3": ("Young driver flag + vehicle group + region",
               ["young_driver", t["vehicle_group"], t["region"]]),
    }


def fit_gamma(claims, terms):
    formula = "avg_severity ~ " + " + ".join(terms)
    return smf.glm(
        formula, data=claims,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=claims["claim_count"],
    ).fit(scale="X2")


def f_test(reduced, full):
    """Deviance F-test for nested GLMs with an estimated dispersion."""
    df_diff = reduced.df_resid - full.df_resid
    f_stat = (reduced.deviance - full.deviance) / df_diff / full.scale
    return f_stat, df_diff, stats.f.sf(f_stat, df_diff, full.df_resid)


def technical_price(df, freq_res, sev_res):
    """Annualised expected frequency, severity and pure premium per policy."""
    freq = freq_res.predict(df, offset=np.zeros(len(df)))
    sev = sev_res.predict(df)
    return pd.DataFrame({"glm_frequency": freq, "glm_severity": sev,
                         "glm_pure_premium": freq * sev}, index=df.index)


def load_glms():
    return sm.load(FREQUENCY_MODEL_FILE), sm.load(SEVERITY_MODEL_FILE)


def weighted_corr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def recovery_metrics(pred, true, w):
    ape = np.abs(pred / true - 1)
    return {
        "total predicted / total true": np.sum(pred * w) / np.sum(true * w),
        "correlation of log price with log true": weighted_corr(np.log(pred), np.log(true), w),
        "mean abs % error vs true": np.average(ape, weights=w),
        "share of policies within 10% of true": np.average(ape <= 0.10, weights=w),
        "share of policies within 20% of true": np.average(ape <= 0.20, weights=w),
    }


def decile_table(test, pred_col):
    t = test.copy()
    t["decile"] = pd.qcut(t[pred_col].rank(method="first"), 10, labels=range(1, 11))
    g = t.groupby("decile", observed=True)
    out = pd.DataFrame({
        "exposure": g["exposure"].sum(),
        "predicted": g.apply(lambda d: np.average(d[pred_col], weights=d["exposure"]), include_groups=False),
        "true": g.apply(lambda d: np.average(d["true_pure_premium"], weights=d["exposure"]), include_groups=False),
        "observed": g["claim_cost"].sum() / g["exposure"].sum(),
    })
    return out


def band_truth(df, band_col, raw_col, true_fn, base):
    """Band-level true relativity: exposure-weighted mean of the continuous true curve."""
    d = df.assign(_rel=true_fn(df[raw_col]))
    g = d.groupby(band_col).apply(
        lambda x: np.average(x["_rel"], weights=x["exposure"]), include_groups=False
    )
    return g / g[base]


def plot_deciles(dec, path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = dec.index.astype(int)
    ax.plot(x, dec["true"], color=COLOR_SECONDARY, lw=2, marker="o", ms=5, label="True expected (generator)")
    ax.plot(x, dec["predicted"], color=COLOR_PRIMARY, lw=2, marker="o", ms=5, label="GLM technical price")
    ax.scatter(x, dec["observed"], color=COLOR_TERTIARY, s=36, zorder=3, label="Observed burning cost")
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xlabel("Decile of GLM technical price (holdout)", color=COLOR_TEXT_MUTED)
    ax.set_ylabel("£ per vehicle-year", color=COLOR_TEXT_MUTED)
    ax.set_title("GLM technical price recovers the true risk across the book", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ensure_dirs()
    df = load_policies()
    df["avg_severity"] = np.where(df["claim_count"] > 0,
                                  df["claim_cost"] / df["claim_count"].clip(lower=1), np.nan)
    train, test = df[df["is_train"]], df[~df["is_train"]]
    bases = base_levels(train)
    claims = train[train["claim_count"] > 0]
    print(f"Severity training data: {len(claims):,} policies, {claims['claim_count'].sum():,} claims")

    specs = severity_specs(bases)
    fits = {name: fit_gamma(claims, terms) for name, (_, terms) in specs.items()}

    n = len(claims)
    rows = []
    for name, (desc, _) in specs.items():
        r = fits[name]
        k = r.df_model + 1
        rows.append({"model": name, "specification": desc, "parameters": int(k),
                     "deviance": r.deviance, "dispersion": r.scale,
                     "AIC": r.aic, "BIC": r.deviance + np.log(n) * k})
    selection = pd.DataFrame(rows)

    tests = []
    for reduced, full in [("S2", "S1"), ("S3", "S2")]:
        f_stat, dfd, p = f_test(fits[reduced], fits[full])
        tests.append({"test": f"{full} -> {reduced}", "F": f_stat, "df": dfd, "p_value": p})
    tests = pd.DataFrame(tests)

    # Simplest model not rejected at 5%.
    final_name = "S1"
    for _, row in tests.iterrows():
        if row["p_value"] < 0.05:
            break
        final_name = row["test"].split("-> ")[1]
    sev_res = fits[final_name]

    print("\nSeverity model selection:")
    print(selection.round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(f"Selected: {final_name}")
    print(sev_res.summary())

    selection.to_csv(TABLES_DIR / "severity_model_selection.csv", index=False)
    tests.to_csv(TABLES_DIR / "severity_tests.csv", index=False)
    pd.DataFrame({"coefficient": sev_res.params, "std_error": sev_res.bse,
                  "p_value": sev_res.pvalues, "relativity": np.exp(sev_res.params)}
                 ).to_csv(TABLES_DIR / "severity_coefficients.csv")
    sev_res.save(SEVERITY_MODEL_FILE, remove_data=True)

    # ---- Technical price and sense-check on the holdout ---------------------
    freq_res = sm.load(FREQUENCY_MODEL_FILE)
    test = test.join(technical_price(test, freq_res, sev_res))
    w = test["exposure"]

    flat_rate = train["claim_cost"].sum() / train["exposure"].sum()
    metrics = pd.DataFrame({
        "GLM technical price": recovery_metrics(test["glm_pure_premium"], test["true_pure_premium"], w),
        "Flat rate (no rating factors)": recovery_metrics(
            np.full(len(test), flat_rate), test["true_pure_premium"], w),
    })
    metrics.loc["correlation of log price with log true", "Flat rate (no rating factors)"] = np.nan
    print("\nHoldout recovery of the true pure premium:")
    print(metrics.round(4))
    metrics.to_csv(TABLES_DIR / "technical_price_recovery.csv")

    freq_comp = pd.DataFrame({
        "glm_frequency": np.average(test["glm_frequency"], weights=w),
        "true_frequency": np.average(test["true_frequency"], weights=w),
        "observed_frequency": test["claim_count"].sum() / w.sum(),
    }, index=["holdout"])
    print(freq_comp.round(4))

    dec = decile_table(test, "glm_pure_premium")
    print("\nDeciles:")
    print(dec.round(1))
    dec.to_csv(TABLES_DIR / "technical_price_deciles.csv")
    plot_deciles(dec, FIGURES_DIR / "technical_price_deciles.png")

    # Band-level truth for age and NCD vs fitted frequency relativities.
    rows = []
    for factor, raw, fn in [("age_band", "driver_age", gen.true_age_relativity),
                            ("ncd_band", "ncd_years", gen.true_ncd_relativity)]:
        truth = band_truth(train, factor, raw, fn, bases[factor])
        for level, true_rel in truth.items():
            key = f"C({factor}, Treatment(reference='{bases[factor]}'))[T.{level}]"
            fitted = np.exp(freq_res.params.get(key, 0.0))
            rows.append({"factor": factor, "level": level, "fitted_relativity": fitted,
                         "true_band_relativity": true_rel})
    band_check = pd.DataFrame(rows)
    print("\nAge / NCD band relativities, fitted vs true:")
    print(band_check.round(3).to_string(index=False))
    band_check.to_csv(TABLES_DIR / "frequency_band_truth_check.csv", index=False)

    true_sev = {"VG": {k: v / gen.SEV_VEHICLE_GROUP[bases["vehicle_group"]]
                       for k, v in gen.SEV_VEHICLE_GROUP.items()},
                "region": {k: v / gen.SEV_REGION[bases["region"]] for k, v in gen.SEV_REGION.items()}}
    print("\nTrue severity relativities (rebased):", {k: {l: round(x, 3) for l, x in v.items()}
                                                     for k, v in true_sev.items()})


if __name__ == "__main__":
    main()
