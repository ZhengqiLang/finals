from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

CSV_PATH = Path(
"pt/date=2026-06-16_21-20-13_model=TripleIntegrator_p=0.8_seed=4_alg=PPO_JAX_distance=l1_weights=inverse_sampleNbr=10/loss_history.csv"
)

OUTPUT_PATH = CSV_PATH.parent / "valid_fraction_curve.png"

SMOOTH_SPAN = 1000
SHOW_RAW = False
RAW_ALPHA = 0.10


def ema_smooth(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=1,
    ).mean()


def add_cegis_boundaries(
    ax: plt.Axes,
    data: pd.DataFrame,
) -> None:
    cegis_ids = data["cegis_iteration"].astype(int).to_numpy()
    steps = data["global_step"].to_numpy()

    for cegis_id in np.unique(cegis_ids)[1:]:
        indices = np.flatnonzero(cegis_ids == cegis_id)

        if len(indices) == 0:
            continue

        ax.axvline(
            steps[indices[0]],
            linestyle="--",
            linewidth=0.8,
            alpha=0.25,
        )


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find CSV file:\n{CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)
    data = data.sort_values("global_step").reset_index(drop=True)

    required_columns = {
        "global_step",
        "cegis_iteration",
        "valid_init_frac",
        "valid_unsafe_frac",
        "valid_target_frac",
        "valid_decrease_frac",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {sorted(missing)}"
        )

    steps = data["global_step"]

    fraction_columns = {
        "Initial": "valid_init_frac",
        "Unsafe": "valid_unsafe_frac",
        "Target": "valid_target_frac",
        "Decrease": "valid_decrease_frac",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, column in fraction_columns.items():
        raw = (
            pd.to_numeric(data[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
        )

        smooth = ema_smooth(raw, span=SMOOTH_SPAN)

        if SHOW_RAW:
            ax.plot(
                steps,
                raw,
                linewidth=0.6,
                alpha=RAW_ALPHA,
            )

        ax.plot(
            steps,
            smooth,
            linewidth=2.0,
            label=label,
        )

    add_cegis_boundaries(ax, data)

    ax.set_xlabel("Learner update step")
    ax.set_ylabel("Valid sample fraction")

    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.grid(alpha=0.3)
    ax.legend()

    ax.set_title(
        f"Valid local-sample fractions "
        f"(EMA span = {SMOOTH_SPAN})"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved valid-fraction curve to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()