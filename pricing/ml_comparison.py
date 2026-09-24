"""Phase D - gradient-boosted Tweedie model vs the GLM technical price.

XGBoost with `reg:tweedie` models pure premium (claim cost per vehicle-year) in
a single model. The Tweedie distribution with 1 < p < 2 is a compound
Poisson-Gamma: a Poisson number of Gamma-sized claims. It is the one-model
counterpart of the GLM frequency x severity split. Exposure is handled the
standard way: the target is cost / exposure and the sample weight is exposure.

The GBM gets *raw* age and NCD years (no banding) and the same other factors.
It can learn the smooth age curve and any interactions without being told.
It gets no hand-built young x VG4-5 flag.

Tree depth and number of trees are chosen by 5-fold CV on the training split
only; the holdout is used purely for the comparison.

Run:  python -m pricing.ml_comparison   (after glm_frequency and glm_severity)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_tweedie_deviance

from pricing.config import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_TEXT_MUTED, FIGURES_DIR,
    MILEAGE_BANDS, REGIONS, SEED, TABLES_DIR, ensure_dirs, load_policies, style_axes,
)
from pricing.glm_severity import load_glms, recovery_metrics, technical_price

TWEEDIE_POWER = 1.5
GBM_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": TWEEDIE_POWER,
    "eta": 0.03,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": f"tweedie-nloglik@{TWEEDIE_POWER}",
    "seed": SEED,
}
MAX_DEPTH_GRID = [1, 2, 3, 4]


def gbm_features(df):
    return pd.DataFrame({
        "driver_age": df["driver_age"],
        "ncd_years": df["ncd_years"],
        "vehicle_group": df["vehicle_group"].str[2:].astype(int),
        "region": pd.Categorical(df["region"], categories=REGIONS),
        "mileage": pd.Categorical(df["mileage_band"], categories=MILEAGE_BANDS).codes,
    }, index=df.index)


def fit_gbm(train):
    """Pick tree depth and number of trees by 5-fold CV on the training data, then refit.

    Returns the booster, a balance factor, and the CV table. Early-stopped boosting
    with shrinkage does not reproduce the training total exactly (a GLM with a log
    link almost does), so predictions are rescaled to the training loss cost - the
    usual "balance correction" before comparing price levels.
    """
    dtrain = xgb.DMatrix(gbm_features(train), label=train["claim_cost"] / train["exposure"],
                         weight=train["exposure"], enable_categorical=True)
    cv_rows = []
    for depth in MAX_DEPTH_GRID:
        params = {**GBM_PARAMS, "max_depth": depth}
        cv = xgb.cv(params, dtrain, num_boost_round=6000, nfold=5,
                    early_stopping_rounds=200, seed=SEED)
        cv_rows.append({"max_depth": depth, "trees": len(cv),
                        "cv_tweedie_nloglik": cv.iloc[-1, 2]})
    cv_table = pd.DataFrame(cv_rows)
    best = cv_table.loc[cv_table["cv_tweedie_nloglik"].idxmin()]
    print(cv_table.round(5).to_string(index=False))
    print(f"Selected max_depth={int(best['max_depth'])}, {int(best['trees'])} trees")

    params = {**GBM_PARAMS, "max_depth": int(best["max_depth"])}
    booster = xgb.train(params, dtrain, num_boost_round=int(best["trees"]))
    balance = train["claim_cost"].sum() / np.sum(booster.predict(dtrain) * train["exposure"])
    print(f"Balance correction factor: {balance:.4f}")
    return booster, balance, cv_table


def gbm_predict(booster, balance, df):
    d = xgb.DMatrix(gbm_features(df), enable_categorical=True)
    return booster.predict(d) * balance


def normalised_gini(actual, pred, weight):
    """Exposure-weighted Gini of the Lorenz curve ordered by prediction, / the perfect model's."""
    def gini(order_by):
        idx = np.argsort(order_by)
        w, a = weight[idx], actual[idx] * weight[idx]
        cum_w, cum_a = np.cumsum(w) / w.sum(), np.cumsum(a) / a.sum()
        return 1 - 2 * np.trapezoid(cum_a, cum_w)
    return gini(pred) / gini(actual)


def plot_age_curve(test, path):
    g = test.groupby("driver_age")
    curve = pd.DataFrame({
        c: g.apply(lambda d: np.average(d[c], weights=d["exposure"]), include_groups=False)
        for c in ("true_pure_premium", "glm_pure_premium", "gbm_pure_premium")
    })
    curve = curve[g["exposure"].sum() >= 50]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(curve.index, curve["true_pure_premium"], color=COLOR_SECONDARY, lw=2, label="True expected")
    ax.plot(curve.index, curve["glm_pure_premium"], color=COLOR_PRIMARY, lw=2,
            drawstyle="steps-mid", label="GLM (banded age)")
    ax.plot(curve.index, curve["gbm_pure_premium"], color=COLOR_TERTIARY, lw=2, label="XGBoost Tweedie (raw age)")
    style_axes(ax)
    ax.set_xlabel("Driver age", color=COLOR_TEXT_MUTED)
    ax.set_ylabel("Mean pure premium, £ per vehicle-year", color=COLOR_TEXT_MUTED)
    ax.set_title("Holdout pure premium by driver age: the GLM steps, the GBM curves",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return curve


def main():
    ensure_dirs()
    df = load_policies()
    train, test = df[df["is_train"]], df[~df["is_train"]].copy()

    freq_res, sev_res = load_glms()
    test = test.join(technical_price(test, freq_res, sev_res))

    booster, balance, cv_table = fit_gbm(train)
    cv_table.to_csv(TABLES_DIR / "gbm_cv_depth_search.csv", index=False)
    test["gbm_pure_premium"] = gbm_predict(booster, balance, test)

    w = test["exposure"].to_numpy()
    observed_pp = (test["claim_cost"] / test["exposure"]).to_numpy()
    true_pp = test["true_pure_premium"].to_numpy()

    rows = {}
    for label, col in [("GLM (freq x sev)", "glm_pure_premium"), ("XGBoost Tweedie", "gbm_pure_premium")]:
        pred = test[col].to_numpy()
        m = recovery_metrics(pred, true_pp, w)
        m["Tweedie deviance vs observed (p=1.5)"] = mean_tweedie_deviance(
            observed_pp, pred, sample_weight=w, power=TWEEDIE_POWER)
        m["normalised Gini vs observed"] = normalised_gini(observed_pp, pred, w)
        m["normalised Gini vs true"] = normalised_gini(true_pp, pred, w)
        rows[label] = m
    comparison = pd.DataFrame(rows)
    print("\nHoldout comparison:")
    print(comparison.round(4))
    comparison.to_csv(TABLES_DIR / "glm_vs_gbm_metrics.csv")

    # Agreement between the two models.
    ratio = test["gbm_pure_premium"] / test["glm_pure_premium"]
    agreement = {
        "share within 5% of each other": np.average(np.abs(ratio - 1) <= 0.05, weights=w),
        "share within 10% of each other": np.average(np.abs(ratio - 1) <= 0.10, weights=w),
        "share within 20% of each other": np.average(np.abs(ratio - 1) <= 0.20, weights=w),
        "share where GBM is closer to true": np.average(
            np.abs(np.log(test["gbm_pure_premium"] / true_pp))
            < np.abs(np.log(test["glm_pure_premium"] / true_pp)), weights=w),
    }
    print(pd.Series(agreement).round(3))
    pd.Series(agreement).to_csv(TABLES_DIR / "glm_vs_gbm_agreement.csv", header=["value"])

    # Where they diverge: segments by single-year age for the young, bands otherwise.
    test["segment_age"] = np.where(test["driver_age"] < 25, test["driver_age"].astype(str), test["age_band"])
    seg = test.groupby(["segment_age", "young_high_vg"]).apply(
        lambda d: pd.Series({
            "exposure": d["exposure"].sum(),
            "true": np.average(d["true_pure_premium"], weights=d["exposure"]),
            "glm": np.average(d["glm_pure_premium"], weights=d["exposure"]),
            "gbm": np.average(d["gbm_pure_premium"], weights=d["exposure"]),
        }), include_groups=False)
    seg["glm_vs_true"] = seg["glm"] / seg["true"]
    seg["gbm_vs_true"] = seg["gbm"] / seg["true"]
    seg["gbm_vs_glm"] = seg["gbm"] / seg["glm"]
    print("\nSegment comparison (age x young-high-VG flag):")
    print(seg.round(3).to_string())
    seg.to_csv(TABLES_DIR / "glm_vs_gbm_segments.csv")

    importance = pd.Series(booster.get_score(importance_type="total_gain"))
    importance = (importance / importance.sum()).sort_values(ascending=False)
    print("\nGBM feature importance (share of total gain):")
    print(importance.round(3))
    importance.to_csv(TABLES_DIR / "gbm_feature_importance.csv", header=["gain_share"])

    plot_age_curve(test, FIGURES_DIR / "glm_vs_gbm_by_age.png")


if __name__ == "__main__":
    main()
