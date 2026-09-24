# Personal Lines GLM Pricing: Motor frequency-severity model

This project builds a technical price with GLMs, the way UK Personal Lines
pricing teams do. It uses a simulated Motor portfolio, fits frequency and
severity GLMs, combines them into a technical price, compares that price with
a gradient-boosted model, and outputs the rating factor relativity tables that
tools such as Radar and Emblem are designed to produce.

It is a companion to my **Consensus Price Guardian** project, an ML pipeline
that flags anomalous financial valuations. The two cover different sides of
pricing work:

| | Consensus Price Guardian | This project |
|---|---|---|
| Technique | Modern ML (anomaly detection) | Classical actuarial GLM pricing |
| Question answered | "Is this price out of line?" | "What should the price be, and why?" |
| Output | Flags and scores | A multiplicative tariff: base rate × rating factor relativities |

Pricing teams generally use both. The GLM gives a transparent, regulator-ready
tariff; ML finds missing structure and checks it. Phase D shows where each
approach does better on the same data.

> **The data is synthetic.** It was generated with known relationships so the
> models can be checked against the true risk. See
> [data/personal_lines_README.md](data/personal_lines_README.md).

## Headline results

![Rating factor relativities](reports/figures/rating_factor_relativities.png)

- **The technical price recovers the true risk.** On a 20% holdout it has a
  mean absolute error of 6.7% against the true pure premium, compared with 66.8%
  for a flat rate. 79% of policies are priced within 10% of the truth.
  ([Phase C](reports/severity_and_technical_price.md))
- **Model selection recovers the true structure.** LR tests keep the young
  driver × high vehicle group interaction (p = 1e-4) and prefer Negative
  Binomial to Poisson (estimated alpha 0.57; the true value is 0.5). F-tests drop
  NCD and mileage from the severity model, which matches the generator, where
  they have no severity effect. ([Phase B](reports/frequency_model.md))
- **GLM vs XGBoost is practically a tie overall.** The Gini coefficient is 0.383
  for both. The GBM does better on the smooth within-band age curve. The GLM
  does better on the highest-risk segment: the GBM under-prices young drivers
  in VG4-5 cars by 23%, while the GLM is 7% under.
  ([Phase D](reports/glm_vs_gbm_comparison.md))
- **Rating table.** Base rate £112.99 × one relativity per factor, and the table
  reproduces the GLM price exactly. A 19-year-old in London with 1 year's NCD,
  driving a VG4 car 10-15k miles a year, prices at **£1,819**.
  ([Phase E](reports/rating_factors.md))

## What's in it

| Phase | Script | Write-up |
|---|---|---|
| A. Simulated Motor portfolio (150k policy-years) | [pricing/generate_data.py](pricing/generate_data.py) | [data/personal_lines_README.md](data/personal_lines_README.md) |
| B. Frequency GLM: Poisson / Negative Binomial, log link, exposure offset | [pricing/glm_frequency.py](pricing/glm_frequency.py) | [reports/frequency_model.md](reports/frequency_model.md) |
| C. Severity GLM (Gamma), technical price, check against true risk | [pricing/glm_severity.py](pricing/glm_severity.py) | [reports/severity_and_technical_price.md](reports/severity_and_technical_price.md) |
| D. XGBoost Tweedie comparison | [pricing/ml_comparison.py](pricing/ml_comparison.py) | [reports/glm_vs_gbm_comparison.md](reports/glm_vs_gbm_comparison.md) |
| E. Rating factor relativity tables | [pricing/rating_factors.py](pricing/rating_factors.py) | [reports/rating_factors.md](reports/rating_factors.md) |

Shared settings (factor bands, base-level rule, train/test split, chart style)
are in [pricing/config.py](pricing/config.py).

## Run it

```bash
pip install -r requirements.txt
```

```bash
python -m pricing.run_all
```

The whole pipeline runs in a few minutes and is deterministic (seed 42). Each
phase can also be run on its own, for example `python -m pricing.glm_frequency`.
Data and fitted models are regenerated, not committed. Tables and figures are
written to `reports/`.

## Modelling approach

```
Frequency:  claims ~ NegBin(mu),  log(mu) = log(exposure) + b0 + age + NCD + vehicle group + region + mileage + young×VG4-5
Severity:   avg cost ~ Gamma,     log(mu) = b0 + young driver + vehicle group + region       (weight = claim count)
Technical price = E[frequency] × E[severity] = base rate × product of relativities
```

Key decisions, with the reasoning behind each in the phase write-ups:

- **Exposure is an offset, not a feature.** Its coefficient is fixed at 1, so the
  model predicts a rate per vehicle-year. When fitted as a free coefficient it
  came out at 1.006 (se 0.045).
- **Frequency and severity are modelled separately.** Each factor enters only
  the component it affects: mileage and NCD drive how often people claim, not
  how much a claim costs. Each set of relativities is also easier to explain.
- **Selection uses actuarial tests**: nested LR / deviance tests (χ² for
  Poisson, F for Gamma, where dispersion is estimated), AIC/BIC, and dispersion
  diagnostics. Holdout deviance is a check, not the main criterion.
- **Base level = the level with the most exposure.** This is the standard
  convention and keeps the intercept stable.
- **Checked against the truth.** Because the data is simulated, every fitted
  relativity and the combined price can be compared with the generator's true
  values. Where recovery is imperfect (age/NCD aliasing, banding a smooth curve),
  the reports explain why.

## Radar and Emblem: what this project does and doesn't do

WTW's **Emblem** (GLM fitting) and **Radar** (rating, price optimisation and
deployment) are widely used pricing tools in UK Personal Lines. I don't have
access to either, so **I have not used them here and don't claim experience
with the software.**

What I have built is the modelling those tools are designed around:

- multiplicative GLMs with exposure offsets and Poisson/NB and Gamma error
  structures, which is what Emblem fits;
- factor selection by deviance tests and information criteria;
- the **rating factor relativity table**: one relativity per level, a base level
  at 1.00, confidence intervals and exposure by level. This is the main object a
  pricing actuary reviews, smooths and adjusts in Emblem before it goes into a
  Radar rating structure;
- a check that base rate × relativities reproduces the model price exactly,
  which is what makes a GLM deployable as a rating table.

I'd expect to learn the tools' interfaces and workflows (interactive factor
grouping and smoothing, model comparison views, Radar's deployment and
optimisation layers) on the job. The concepts underneath them are what this
project demonstrates.

## Simplifications, stated up front

This is a demonstration of method, not a production pricing model. It does not
include claims development / IBNR, large-loss capping, a peril-level split,
trend and inflation, or expense, commission and profit loadings. It also does
not cover demand modelling or price optimisation. It produces a technical
(risk) price. More detail is in the
[data README](data/personal_lines_README.md#what-this-data-deliberately-does-not-model).
