# Phase B: claim frequency GLM

Script: `pricing/glm_frequency.py` · Outputs: `reports/tables/frequency_*.csv`

## Model form

```
claims ~ NegativeBinomial(mu, alpha)
log(mu) = log(exposure) + b0 + age_band + ncd_band + vehicle_group + region + mileage_band + young_high_vg
```

- **Log link.** Effects are multiplicative, so `exp(coef)` is a relativity. This
  is what makes a GLM directly usable as a rating structure.
- **`log(exposure)` is an offset** (coefficient fixed at 1), not a feature. The
  model therefore predicts claims *per vehicle-year*, and a 3-month policy is
  expected to have a quarter of the claims of an identical annual policy. As a
  check, `log(exposure)` was also fitted as a free covariate. Its estimated
  coefficient was **1.006 (se 0.045)**, which is consistent with the offset
  assumption. Had it come out well below 1, that would point to something like
  early-cancellation adverse selection.
- **Base levels** are the levels with the most exposure (40-49, NCD 9+, VG2,
  North, 5-10k miles). This is the Emblem/Radar convention and keeps the
  intercept's standard error small.
- Fitted on an 80% training split (120,057 policies). The other 20% is held out.

## Specification comparison

| Model | Specification | Params | AIC | BIC | Pearson χ²/df | Holdout Poisson deviance |
|---|---|---|---|---|---|---|
| F1 | Poisson, main effects | 24 | 54,901.2 | 55,133.9 | 1.037 | 0.3270 |
| F2 | Poisson + young × VG4-5 | 25 | 54,888.0 | 55,130.4 | 1.035 | 0.3267 |
| **F3** | **Negative Binomial + young × VG4-5** | 26 | **54,790.1** | **55,042.1** | **0.999** | 0.3267 |

**Nested tests**

| Test | Statistic | Result |
|---|---|---|
| F1 → F2: add young × VG4-5 interaction | Δdeviance = 15.1 on 1 df | p = 1.0e-4: keep the interaction |
| F2 → F3: Poisson → Negative Binomial | LR = 100.0 (boundary-adjusted) | p = 7.7e-24: data is overdispersed |

**Selected: F3.**

- **The interaction earns its place.** It is significant, and both AIC (−13) and
  BIC (−3) prefer it. Under BIC's heavier penalty that is a narrow margin. In
  practice I would also ask whether underwriters can explain it (young drivers in
  high-performance cars is a well-understood risk) and whether it is stable
  across years before adding it to the tariff.
- **Negative Binomial over Poisson.** The estimated alpha is **0.568 (se 0.069)**.
  The generator's true value is 0.5. The Poisson fits' fixed coefficients are
  almost the same as the NB fit's, because Poisson GLM estimates stay unbiased
  under overdispersion. The Poisson *standard errors*, however, are too small,
  which makes rating factors look more significant than they are. NB (or
  quasi-Poisson) fixes this. At a 7% claim frequency, overdispersion barely
  shows in the Pearson statistic (1.035). The likelihood-ratio test detects it
  clearly.
- **Holdout deviance is almost the same across all three.** That is expected:
  the choice is about correct inference and a parsimonious tariff, not a few
  basis points of predictive accuracy. Note that the NB model's own deviance
  (36,612) is on a different scale from the Poisson deviance and cannot be
  compared with it directly. Compare AIC or log-likelihood instead.

## Fitted relativities vs the truth built into the generator

| Factor | Level | Fitted | True* |
|---|---|---|---|
| Vehicle group (vs VG2) | VG1 / VG3 / VG4 / VG5 | 0.90 / 1.04 / 1.17 / 1.39 | 0.89 / 1.05 / 1.21 / 1.42 |
| Region (vs North) | London / Scotland / Wales | 1.17 / 0.75 / 0.79 | 1.20 / 0.79 / 0.83 |
| Mileage (vs 5-10k) | <5k / 10-15k / 15k+ | 0.76 / 1.19 / 1.52 | 0.75 / 1.20 / 1.45 |
| Interaction | Young × VG4-5 | 1.27 | 1.35 |

\* True relativities rebased to the GLM's base level. Age and NCD are continuous
in the generator; the band-level comparison is in the Phase C sense-check.

All fitted relativities are within about two standard errors of the truth.
London is the best example of why the model has to be multivariate: London's
true frequency effect (×1.30 vs Midlands) is partly hidden in one-way
statistics, because London drivers declare lower mileage. The GLM separates
the region effect from the mileage mix.
