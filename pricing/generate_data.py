"""Phase A - simulate a synthetic UK Motor portfolio at policy-year level.

The generator builds in *known* multiplicative frequency and severity effects so
the fitted GLMs can be checked against the truth. The true expected frequency,
severity and pure premium are saved alongside each policy (columns prefixed
`true_`); they are never used as model inputs, only for the sense-check.

Run:  python -m pricing.generate_data
"""

import numpy as np
import pandas as pd

from pricing.config import POLICY_FILE, REGIONS, SEED, VEHICLE_GROUPS, ensure_dirs

N_POLICIES = 150_000

# ---------------------------------------------------------------------------
# True frequency structure (claims per policy-year)
# ---------------------------------------------------------------------------
BASE_FREQUENCY = 0.085

FREQ_VEHICLE_GROUP = {"VG1": 0.85, "VG2": 0.95, "VG3": 1.00, "VG4": 1.15, "VG5": 1.35}
FREQ_REGION = {
    "London": 1.30, "South East": 1.05, "Midlands": 1.00,
    "North": 1.08, "Scotland": 0.85, "Wales": 0.90,
}
FREQ_MILEAGE = {"<5k": 0.75, "5-10k": 1.00, "10-15k": 1.20, "15k+": 1.45}
# Young drivers in powerful/expensive cars: extra loading on top of the main effects.
FREQ_YOUNG_HIGH_VG = 1.35

# Unobserved risk heterogeneity: each policy's Poisson mean is multiplied by a
# Gamma(shape, 1/shape) draw (mean 1) -> Negative Binomial counts, alpha = 1/shape.
HETEROGENEITY_SHAPE = 2.0

# ---------------------------------------------------------------------------
# True severity structure (cost per claim, GBP)
# ---------------------------------------------------------------------------
BASE_SEVERITY = 2_800.0
SEV_VEHICLE_GROUP = {"VG1": 0.80, "VG2": 0.90, "VG3": 1.00, "VG4": 1.20, "VG5": 1.50}
SEV_REGION = {
    "London": 1.35, "South East": 1.15, "Midlands": 1.00,
    "North": 0.92, "Scotland": 0.85, "Wales": 0.90,
}
SEV_YOUNG_DRIVER = 1.15  # driver age < 25
SEVERITY_GAMMA_SHAPE = 1.5  # CV of individual claim cost = 1/sqrt(1.5) ~ 0.82


def true_age_relativity(age):
    """Smooth U-shape: steep decline from 17, flat through middle age, gentle rise after 70.

    Deliberately continuous and non-linear, so banding in the GLM is an
    approximation - this is where the GBM comparison in Phase D gets interesting.
    """
    age = np.asarray(age, dtype=float)
    young = 1.8 * np.exp(-(age - 17) / 5.0)
    old = 0.5 * np.clip((age - 70) / 10.0, 0, None)
    rel = 1.0 + young + old
    ref = 1.0 + 1.8 * np.exp(-(45 - 17) / 5.0)
    return rel / ref


def true_ncd_relativity(ncd):
    """Each no-claims year reduces frequency by ~7%, capped at 9 years."""
    return np.exp(-0.07 * np.minimum(np.asarray(ncd, dtype=float), 9))


def true_frequency(df):
    return (
        BASE_FREQUENCY
        * true_age_relativity(df["driver_age"])
        * true_ncd_relativity(df["ncd_years"])
        * df["vehicle_group"].map(FREQ_VEHICLE_GROUP).to_numpy()
        * df["region"].map(FREQ_REGION).to_numpy()
        * df["mileage_band"].map(FREQ_MILEAGE).to_numpy()
        * np.where(
            (df["driver_age"] < 25) & df["vehicle_group"].isin(["VG4", "VG5"]),
            FREQ_YOUNG_HIGH_VG, 1.0,
        )
    )


def true_severity(df):
    return (
        BASE_SEVERITY
        * df["vehicle_group"].map(SEV_VEHICLE_GROUP).to_numpy()
        * df["region"].map(SEV_REGION).to_numpy()
        * np.where(df["driver_age"] < 25, SEV_YOUNG_DRIVER, 1.0)
    )


def simulate_portfolio(n=N_POLICIES, seed=SEED):
    rng = np.random.default_rng(seed)

    # Driver age: main adult population plus a young-driver segment.
    adult = np.clip(rng.normal(47, 14, n), 25, 88)
    young = rng.uniform(17, 25, n)
    driver_age = np.where(rng.random(n) < 0.10, young, adult).astype(int)

    # No-claims years are capped by driving history (can't exceed years since 17).
    years_driving = driver_age - 17
    ncd_years = np.minimum(years_driving, rng.poisson(0.35 * years_driving + 0.3))
    ncd_years = np.minimum(ncd_years, 15)

    # Young drivers skew toward cheaper vehicle groups.
    vg_adult = rng.choice(VEHICLE_GROUPS, n, p=[0.18, 0.30, 0.26, 0.16, 0.10])
    vg_young = rng.choice(VEHICLE_GROUPS, n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    vehicle_group = np.where(driver_age < 25, vg_young, vg_adult)

    region = rng.choice(REGIONS, n, p=[0.15, 0.20, 0.20, 0.22, 0.13, 0.10])

    # London drivers cover fewer miles.
    mileage_other = rng.choice(["<5k", "5-10k", "10-15k", "15k+"], n, p=[0.18, 0.40, 0.27, 0.15])
    mileage_london = rng.choice(["<5k", "5-10k", "10-15k", "15k+"], n, p=[0.35, 0.40, 0.18, 0.07])
    mileage_band = np.where(region == "London", mileage_london, mileage_other)

    # Exposure: 75% full-year policies, 25% part-year (new business mid-term, cancellations).
    exposure = np.where(rng.random(n) < 0.75, 1.0, rng.uniform(0.05, 1.0, n)).round(4)

    df = pd.DataFrame({
        "policy_id": np.arange(1, n + 1),
        "driver_age": driver_age,
        "ncd_years": ncd_years,
        "vehicle_group": vehicle_group,
        "region": region,
        "mileage_band": mileage_band,
        "exposure": exposure,
    })

    df["true_frequency"] = true_frequency(df)
    df["true_severity"] = true_severity(df)
    df["true_pure_premium"] = df["true_frequency"] * df["true_severity"]

    # Claim counts: Poisson with gamma-mixed mean -> overdispersed (Negative Binomial).
    frailty = rng.gamma(HETEROGENEITY_SHAPE, 1.0 / HETEROGENEITY_SHAPE, n)
    df["claim_count"] = rng.poisson(df["true_frequency"] * df["exposure"] * frailty)

    # Claim costs: sum of independent Gamma claims per policy.
    total_cost = np.zeros(n)
    has_claim = df["claim_count"].to_numpy() > 0
    counts = df["claim_count"].to_numpy()[has_claim]
    means = df["true_severity"].to_numpy()[has_claim]
    # Sum of k iid Gamma(shape, scale) = Gamma(k*shape, scale).
    total_cost[has_claim] = rng.gamma(
        counts * SEVERITY_GAMMA_SHAPE, means / SEVERITY_GAMMA_SHAPE
    )
    df["claim_cost"] = total_cost.round(2)

    return df


def main():
    ensure_dirs()
    df = simulate_portfolio()
    df.to_csv(POLICY_FILE, index=False)

    claims = df["claim_count"].sum()
    print(f"Wrote {len(df):,} policies to {POLICY_FILE}")
    print(f"  exposure (years):     {df['exposure'].sum():,.0f}")
    print(f"  claims:               {claims:,}")
    print(f"  claim frequency:      {claims / df['exposure'].sum():.4f}")
    print(f"  mean severity:        {df['claim_cost'].sum() / claims:,.0f}")
    print(f"  burning cost / year:  {df['claim_cost'].sum() / df['exposure'].sum():,.1f}")


if __name__ == "__main__":
    main()
