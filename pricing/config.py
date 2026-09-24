"""Shared paths, rating-factor definitions and helpers for the pricing module."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"

POLICY_FILE = DATA_DIR / "motor_policies.csv"

SEED = 42
TEST_FRACTION = 0.2

# Rating factors used by the GLMs, in the order they appear in tables and charts.
# Each is a categorical band; the GLM base level is the band with the most exposure.
AGE_BANDS = ["17-21", "22-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"]
NCD_BANDS = ["0", "1-2", "3-4", "5-8", "9+"]
VEHICLE_GROUPS = ["VG1", "VG2", "VG3", "VG4", "VG5"]
REGIONS = ["London", "South East", "Midlands", "North", "Scotland", "Wales"]
MILEAGE_BANDS = ["<5k", "5-10k", "10-15k", "15k+"]

FACTOR_LEVELS = {
    "age_band": AGE_BANDS,
    "ncd_band": NCD_BANDS,
    "vehicle_group": VEHICLE_GROUPS,
    "region": REGIONS,
    "mileage_band": MILEAGE_BANDS,
}

FACTOR_LABELS = {
    "age_band": "Driver age",
    "ncd_band": "No-claims discount (years)",
    "vehicle_group": "Vehicle group",
    "region": "Region",
    "mileage_band": "Annual mileage",
    "young_driver": "Young driver (<25)",
    "young_high_vg": "Young driver x VG4-5",
}

# Chart styling (single-hue bars, recessive axes).
COLOR_PRIMARY = "#2a78d6"
COLOR_SECONDARY = "#eb6834"
COLOR_TERTIARY = "#1baf7a"
COLOR_BASE = "#a3a29c"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e4e3de"


def band_age(age):
    return pd.cut(
        age,
        bins=[16, 21, 24, 29, 39, 49, 59, 69, 200],
        labels=AGE_BANDS,
    ).astype(str)


def band_ncd(ncd):
    return pd.cut(ncd, bins=[-1, 0, 2, 4, 8, 100], labels=NCD_BANDS).astype(str)


def add_bands(df):
    """Derive the banded rating factors and the interaction flag from raw columns."""
    df = df.copy()
    df["age_band"] = band_age(df["driver_age"])
    df["ncd_band"] = band_ncd(df["ncd_years"])
    df["young_driver"] = (df["driver_age"] < 25).astype(int)
    df["young_high_vg"] = (
        (df["driver_age"] < 25) & df["vehicle_group"].isin(["VG4", "VG5"])
    ).astype(int)
    return df


def load_policies():
    """Load the simulated portfolio with bands and a fixed train/test split."""
    if not POLICY_FILE.exists():
        raise FileNotFoundError(
            f"{POLICY_FILE} not found - run `python -m pricing.generate_data` first."
        )
    df = pd.read_csv(POLICY_FILE)
    df = add_bands(df)
    rng = np.random.default_rng(SEED)
    df["is_train"] = rng.random(len(df)) >= TEST_FRACTION
    return df


def base_levels(df, weight="exposure"):
    """Base level per factor = the level carrying the most exposure (Emblem/Radar convention)."""
    return {
        f: df.groupby(f)[weight].sum().idxmax() for f in FACTOR_LEVELS
    }


def factor_term(factor, base):
    return f"C({factor}, Treatment(reference='{base}'))"


def ensure_dirs():
    for d in (DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def style_axes(ax):
    ax.set_facecolor(COLOR_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(COLOR_TEXT)
