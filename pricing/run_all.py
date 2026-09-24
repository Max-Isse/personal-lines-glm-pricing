"""Run the full pricing pipeline end to end (Phases A-E).

Run:  python -m pricing.run_all
"""

from pricing import generate_data, glm_frequency, glm_severity, ml_comparison, rating_factors

STEPS = [
    ("A  Simulate portfolio", generate_data.main),
    ("B  Frequency GLM", glm_frequency.main),
    ("C  Severity GLM + technical price", glm_severity.main),
    ("D  GLM vs XGBoost", ml_comparison.main),
    ("E  Rating factor tables", rating_factors.main),
]


def main():
    for name, step in STEPS:
        print(f"\n{'=' * 70}\nPhase {name}\n{'=' * 70}")
        step()


if __name__ == "__main__":
    main()
