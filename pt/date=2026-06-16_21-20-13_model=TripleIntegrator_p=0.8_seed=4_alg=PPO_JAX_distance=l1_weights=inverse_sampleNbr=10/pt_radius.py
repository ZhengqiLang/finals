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

REGION_OUTPUT_PATH = CSV_PATH.parent / "radius_curve_regions_smooth.png"
DECREASE_OUTPUT_PATH = CSV_PATH.parent / "radius_curve_decrease_smooth.png"

# 越大越平滑。你的曲线约 9000 个 update，150 比较合适。
SMOOTH_SPAN = 1000

# 是否显示透明的原始曲线
SHOW_RAW = False
RAW_ALPHA = 0.08


def ema_smooth(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=1,
    ).mean()


def clean_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert a column to a finite numeric series."""
    return (
        pd.to_numeric(data[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
    )


def add_cegis_boundaries(
    ax: plt.Axes,
    data: pd.DataFrame,
) -> None:
    """Draw a vertical line at the beginning of each CEGIS iteration."""
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


def plot_region_mean_radii(data: pd.DataFrame) -> None:
    """Plot the mean radius for each sampled region."""
    steps = data["global_step"]

    radius_columns = {
        "Initial": "n_init_mean",
        "Unsafe": "n_unsafe_mean",
        "Target": "n_target_mean",
        "Decrease": "n_decrease_mean",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, column in radius_columns.items():
        raw = clean_series(data, column)
        smooth = ema_smooth(raw, SMOOTH_SPAN)

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
    ax.set_ylabel("Predicted radius")
    ax.set_title(
        f"Mean predicted radii by region "
        f"(EMA span = {SMOOTH_SPAN})"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    # 根据你的代码半径范围是 [0.0002, 0.9]
    ax.set_ylim(0.0, 0.92)

    fig.tight_layout()
    fig.savefig(
        REGION_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved region radius curve to:\n{REGION_OUTPUT_PATH}")


def plot_decrease_radius_range(data: pd.DataFrame) -> None:
    """Plot mean, minimum, and maximum decrease radii."""
    steps = data["global_step"]

    raw_mean = clean_series(data, "n_decrease_mean")
    raw_min = clean_series(data, "n_decrease_min")
    raw_max = clean_series(data, "n_decrease_max")

    smooth_mean = ema_smooth(raw_mean, SMOOTH_SPAN)
    smooth_min = ema_smooth(raw_min, SMOOTH_SPAN)
    smooth_max = ema_smooth(raw_max, SMOOTH_SPAN)

    fig, ax = plt.subplots(figsize=(10, 6))

    if SHOW_RAW:
        ax.plot(
            steps,
            raw_mean,
            linewidth=0.5,
            alpha=RAW_ALPHA,
        )
        ax.plot(
            steps,
            raw_min,
            linewidth=0.5,
            alpha=RAW_ALPHA,
        )
        ax.plot(
            steps,
            raw_max,
            linewidth=0.5,
            alpha=RAW_ALPHA,
        )

    ax.plot(
        steps,
        smooth_mean,
        linewidth=2.2,
        label="Mean radius",
    )
    ax.plot(
        steps,
        smooth_min,
        linewidth=1.8,
        label="Minimum radius",
    )
    ax.plot(
        steps,
        smooth_max,
        linewidth=1.8,
        label="Maximum radius",
    )

    # 阴影表示 min-max 范围
    ax.fill_between(
        steps.to_numpy(),
        smooth_min.to_numpy(),
        smooth_max.to_numpy(),
        alpha=0.12,
        label="Min–max range",
    )

    add_cegis_boundaries(ax, data)

    ax.set_xlabel("Learner update step")
    ax.set_ylabel("Predicted decrease radius")
    ax.set_title(
        f"Predicted decrease-radius range "
        f"(EMA span = {SMOOTH_SPAN})"
    )

    ax.set_ylim(0.0, 0.92)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        DECREASE_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved decrease radius curve to:\n{DECREASE_OUTPUT_PATH}")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find CSV file:\n{CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    required_columns = {
        "global_step",
        "cegis_iteration",
        "n_init_mean",
        "n_unsafe_mean",
        "n_target_mean",
        "n_decrease_mean",
        "n_decrease_min",
        "n_decrease_max",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {sorted(missing)}"
        )

    data = (
        data.sort_values("global_step")
        .reset_index(drop=True)
    )

    plot_region_mean_radii(data)
    plot_decrease_radius_range(data)


if __name__ == "__main__":
    main()