"""Phase B - claim frequency GLM (Poisson / Negative Binomial, log link, exposure offset).

Model:  log E[claims] = log(exposure) + b0 + sum_j b_j x_j

`log(exposure)` enters as an *offset*: a term with its coefficient fixed at 1.
The model therefore predicts claims per unit exposure, so a half-year policy is
expected to have half the claims of an otherwise identical full-year one.
Putting exposure in as an ordinary feature would let the model estimate that
coefficient freely. It would also stop the linear predictor from being a rate,
and the relativities would no longer read as frequency multipliers.

Specifications compared (all fitted on the training split):
    F1  Poisson, main effects only
    F2  Poisson, main effects + young-driver x VG4-5 interaction
    F3  Negative Binomial (NB2), same terms as F2, alpha estimated by ML

Selection uses nested likelihood-ratio (deviance) tests, AIC/BIC, and the
Pearson dispersion statistic. Out-of-sample Poisson deviance is a check.

Run:  python -m pricing.glm_frequency
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import mean_poisson_deviance

from pricing.config import (
    FACTOR_LEVELS, MODELS_DIR, TABLES_DIR, base_levels, ensure_dirs, factor_term,
    load_policies,
)

FREQUENCY_MODEL_FILE = MODELS_DIR / "frequency_glm.pickle"


def frequency_formula(bases, interaction=True):
    terms = [factor_term(f, bases[f]) for f in FACTOR_LEVELS]
    if interaction:
        terms.append("young_high_vg")
    return "claim_count ~ " + " + ".join(terms)


def fit_poisson(train, formula):
    return smf.glm(
        formula, data=train, family=sm.families.Poisson(),
        offset=np.log(train["exposure"]),
    ).fit()


def estimate_nb_alpha(train, formula):
    """Maximum-likelihood NB2 dispersion. statsmodels' GLM NB family needs alpha fixed."""
    nb = smf.negativebinomial(
        formula, data=train, offset=np.log(train["exposure"]), loglike_method="nb2",
    ).fit(disp=0, maxiter=200)
    return nb.params["alpha"], nb.bse["alpha"]


def fit_negative_binomial(train, formula, alpha):
    return smf.glm(
        formula, data=train, family=sm.families.NegativeBinomial(alpha=alpha),
        offset=np.log(train["exposure"]),
    ).fit()


def pearson_dispersion(res):
    return res.pearson_chi2 / res.df_resid


def holdout_deviance(res, test):
    """Mean Poisson deviance of predicted claim counts on the test split."""
    mu = res.predict(test, offset=np.log(test["exposure"]))
    return mean_poisson_deviance(test["claim_count"], mu)


def offset_check(train, bases):
    """Fit log(exposure) as a free covariate. A coefficient near 1 supports the offset."""
    formula = frequency_formula(bases) + " + np.log(exposure)"
    res = smf.glm(formula, data=train, family=sm.families.Poisson()).fit()
    return res.params["np.log(exposure)"], res.bse["np.log(exposure)"]


def main():
    ensure_dirs()
    df = load_policies()
    train, test = df[df["is_train"]], df[~df["is_train"]]
    bases = base_levels(train)
    print("Base levels (highest exposure):", bases)

    f_main = frequency_formula(bases, interaction=False)
    f_int = frequency_formula(bases, interaction=True)

    f1 = fit_poisson(train, f_main)
    f2 = fit_poisson(train, f_int)
    alpha, alpha_se = estimate_nb_alpha(train, f_int)
    f3 = fit_negative_binomial(train, f_int, alpha)

    # AIC/BIC: the NB model has one extra estimated parameter (alpha).
    n = len(train)
    rows = []
    for name, desc, res, extra in [
        ("F1", "Poisson, main effects", f1, 0),
        ("F2", "Poisson, + young x VG4-5", f2, 0),
        ("F3", f"NegBin (alpha={alpha:.3f}), + young x VG4-5", f3, 1),
    ]:
        k = res.df_model + 1 + extra
        rows.append({
            "model": name,
            "specification": desc,
            "parameters": int(k),
            "deviance": res.deviance,
            "log_likelihood": res.llf,
            "AIC": -2 * res.llf + 2 * k,
            "BIC": -2 * res.llf + np.log(n) * k,
            "pearson_dispersion": pearson_dispersion(res),
            "holdout_poisson_deviance": holdout_deviance(res, test),
        })
    selection = pd.DataFrame(rows)

    # Nested LR test F1 -> F2. For Poisson (phi = 1) the deviance drop is chi-square.
    lr_stat = f1.deviance - f2.deviance
    lr_df = f2.df_model - f1.df_model
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    # LR test Poisson -> NB (alpha = 0 is on the boundary, so halve the p-value).
    nb_stat = 2 * (f3.llf - f2.llf)
    nb_p = 0.5 * stats.chi2.sf(nb_stat, 1)

    final_name = selection.loc[selection["AIC"].idxmin(), "model"]
    final = {"F1": f1, "F2": f2, "F3": f3}[final_name]

    exp_coef, exp_se = offset_check(train, bases)

    pd.set_option("display.width", 160)
    print("\nModel selection (training data):")
    print(selection.round(4).to_string(index=False))
    print(f"\nLR test F1 vs F2 (interaction): chi2={lr_stat:.1f}, df={lr_df:.0f}, p={lr_p:.2e}")
    print(f"LR test F2 vs F3 (overdispersion): chi2={nb_stat:.1f}, p={nb_p:.2e}")
    print(f"NB alpha = {alpha:.3f} (se {alpha_se:.3f}); true value in generator = 0.5")
    print(f"Offset check: free log(exposure) coefficient = {exp_coef:.3f} (se {exp_se:.3f})")
    print(f"Selected: {final_name}")
    print(final.summary())

    # Save outputs.
    selection.to_csv(TABLES_DIR / "frequency_model_selection.csv", index=False)
    tests = pd.DataFrame([
        {"test": "F1 vs F2: young x VG4-5 interaction", "statistic": lr_stat,
         "df": lr_df, "p_value": lr_p},
        {"test": "F2 vs F3: overdispersion (Poisson vs NB)", "statistic": nb_stat,
         "df": 1, "p_value": nb_p},
        {"test": "Offset check: free log(exposure) coefficient", "statistic": exp_coef,
         "df": np.nan, "p_value": stats.norm.sf(abs(exp_coef - 1) / exp_se) * 2},
    ])
    tests.to_csv(TABLES_DIR / "frequency_tests.csv", index=False)

    coefs = pd.DataFrame({
        "coefficient": final.params, "std_error": final.bse, "p_value": final.pvalues,
        "relativity": np.exp(final.params),
    })
    coefs.to_csv(TABLES_DIR / "frequency_coefficients.csv")

    final.save(FREQUENCY_MODEL_FILE, remove_data=True)
    print(f"\nSaved {final_name} to {FREQUENCY_MODEL_FILE}")


if __name__ == "__main__":
    main()
