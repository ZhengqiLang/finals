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

OUTPUT_PATH = CSV_PATH.parent / "learning_curve_smooth.png"

# EMA span 越大，曲线越平滑。
# 你的曲线约 9000 个 update，可以先用 100～200。
SMOOTH_SPAN = 1000

# 是否同时绘制淡色原始数据
SHOW_RAW = False

# 原始曲线透明度
RAW_ALPHA = 0.10

# 对数坐标不能显示 0
EPS = 1e-8


def ema_smooth(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average.

    adjust=False makes the result behave like the usual recursive EMA.
    """
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=1,
    ).mean()


def add_cegis_boundaries(
    ax: plt.Axes,
    data: pd.DataFrame,
) -> None:
    """
    Draw a vertical dashed line at the beginning of each new CEGIS iteration.
    """
    cegis_ids = data["cegis_iteration"].astype(int).to_numpy()
    steps = data["global_step"].to_numpy()

    unique_ids = np.unique(cegis_ids)

    # Skip the first CEGIS iteration because it starts at the left boundary.
    for cegis_id in unique_ids[1:]:
        indices = np.flatnonzero(cegis_ids == cegis_id)

        if len(indices) == 0:
            continue

        first_idx = indices[0]

        ax.axvline(
            steps[first_idx],
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

    required_columns = {
        "global_step",
        "cegis_iteration",
        "total_loss",
        "init_loss",
        "unsafe_loss",
        "decrease_loss",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {sorted(missing)}"
        )

    data = data.sort_values("global_step").reset_index(drop=True)

    steps = data["global_step"]

    loss_columns = {
        "Total": "total_loss",
        "Initial": "init_loss",
        "Unsafe": "unsafe_loss",
        "Decrease": "decrease_loss",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, column in loss_columns.items():
        raw = (
            pd.to_numeric(data[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )

        smooth = ema_smooth(raw, span=SMOOTH_SPAN)

        if SHOW_RAW:
            ax.plot(
                steps,
                raw + EPS,
                linewidth=0.6,
                alpha=RAW_ALPHA,
            )

        ax.plot(
            steps,
            smooth + EPS,
            linewidth=2.0,
            label=label,
        )

    add_cegis_boundaries(ax, data)

    ax.set_xlabel("Learner update step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.legend()

    ax.set_title(
        f"Smoothed learner losses "
        # f"(EMA span = {SMOOTH_SPAN})"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved smoothed loss curve to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()