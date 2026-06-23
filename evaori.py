from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT_DIR = Path("/Users/zhengqi/Desktop/Final-main/output/backup/local_sampling_2D_newLoss")

def find_info_files(root: Path):
    """
    Find info files recursively.
    Supports:
        info
        info.csv
        info.txt
    """
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower().startswith("info"):
            files.append(p)
    return files

def read_info_file(path: Path):
    """
    Read info file with lines:
        key,value
    """
    data = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, value = line.split(",", 1)
            data[key.strip()] = value.strip()
    return data

def parse_folder_params(folder_name: str):
    """
    Parse folder name like:
    date=2026-06-01_21-35-13_model=MyPendulum_p=0.8_seed=1_alg=PPO_JAX_distance=l1_weights=inverse_gaussian_sampleNbr=10

    This parser supports values containing underscores, such as:
        alg=PPO_JAX
        weights=inverse_gaussian
    """
    keys = ["date", "model", "p", "seed", "alg", "distance", "weights", "sampleNbr"]

    params = {}

    for i, key in enumerate(keys):
        start_token = key + "="
        start = folder_name.find(start_token)
        if start == -1:
            continue

        value_start = start + len(start_token)

        # find the next _key= after current value
        next_positions = []
        for next_key in keys[i + 1:]:
            pos = folder_name.find("_" + next_key + "=", value_start)
            if pos != -1:
                next_positions.append(pos)

        if next_positions:
            value_end = min(next_positions)
        else:
            value_end = len(folder_name)

        params[key] = folder_name[value_start:value_end]

    return params

def build_group_name(folder_params: dict):
    """
    Group by same parameters but different seed and date.
    Remove date and seed.
    """
    ignore_keys = {"date", "seed"}

    # keep order for readability
    preferred_order = ["model", "p", "alg", "distance", "weights", "sampleNbr"]

    parts = []
    for k in preferred_order:
        if k in folder_params and k not in ignore_keys:
            parts.append(f"{k}={folder_params[k]}")

    # add remaining keys if any
    for k, v in folder_params.items():
        if k not in ignore_keys and k not in preferred_order:
            parts.append(f"{k}={v}")

    return "_".join(parts)

def normalize_status(status):
    if status is None:
        return "missing"
    status = status.strip().lower()
    if status == "seccess":
        return "success"
    return status

info_files = find_info_files(ROOT_DIR)

if not info_files:
    print(f"No info files found under: {ROOT_DIR}")
    print()
    print("请先检查子文件夹里 info 文件到底叫什么。你可以在终端运行：")
    print(f"find {ROOT_DIR} -maxdepth 3 -type f | head -50")
    raise SystemExit

records = []

for info_path in info_files:
    run_dir = info_path.parent
    folder_name = run_dir.name

    info = read_info_file(info_path)
    folder_params = parse_folder_params(folder_name)

    seed = info.get("seed", folder_params.get("seed", ""))
    status = normalize_status(info.get("status", ""))

    try:
        cegis_time = float(info.get("total_CEGIS_time", np.nan))
    except ValueError:
        cegis_time = np.nan

    group_name = build_group_name(folder_params)

    records.append({
        "group_name": group_name,
        "folder": folder_name,
        "info_file": str(info_path),
        "model": info.get("model", folder_params.get("model", "")),
        "p": folder_params.get("p", info.get("probability_bound", "")),
        "seed": seed,
        "alg": folder_params.get("alg", info.get("algorithm", "")),
        "distance": folder_params.get("distance", ""),
        "weights": folder_params.get("weights", ""),
        "sampleNbr": folder_params.get("sampleNbr", ""),
        "status": status,
        "success": status == "success",
        "total_CEGIS_time": cegis_time,
        "verify_samples": info.get("verify_samples", ""),
    })

df = pd.DataFrame(records)

raw_out = ROOT_DIR / "summary_raw_runs.csv"
df.to_csv(raw_out, index=False)

def join_unique(x):
    return ", ".join(map(str, sorted(set(x), key=lambda v: str(v))))

def failed_seeds(group):
    failed = group[group["success"] == False]
    if len(failed) == 0:
        return ""
    return ", ".join(f"{row.seed}:{row.status}" for _, row in failed.iterrows())
def sample_variance(values):
    """
    Calculate sample variance using ddof=1.

    Returns NaN when fewer than two valid observations are available,
    because variance cannot be estimated from a single run.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    return float(values.var(ddof=1))


summary = (
    df.groupby("group_name", dropna=False)
    .apply(lambda g: pd.Series({
        "model": g["model"].iloc[0],
        "p": g["p"].iloc[0],
        "alg": g["alg"].iloc[0],
        "distance": g["distance"].iloc[0],
        "weights": g["weights"].iloc[0],
        "sampleNbr": g["sampleNbr"].iloc[0],

        "num_runs": len(g),
        "num_success": int(g["success"].sum()),
        "success_rate": float(g["success"].mean() * 100),

        # Statistics over successful runs only
        "mean_CEGIS_time_success": (
            g.loc[g["success"], "total_CEGIS_time"].mean()
        ),
        "variance_CEGIS_time_success": sample_variance(
            g.loc[g["success"], "total_CEGIS_time"]
        ),

        # Statistics over all runs that contain a recorded time
        "mean_CEGIS_time_all": (
            g["total_CEGIS_time"].mean()
        ),
        "variance_CEGIS_time_all": sample_variance(
            g["total_CEGIS_time"]
        ),

        "min_CEGIS_time": (
            g["total_CEGIS_time"].min()
        ),
        "max_CEGIS_time": (
            g["total_CEGIS_time"].max()
        ),

        "seeds": join_unique(g["seed"]),
        "failed_seeds": failed_seeds(g),
    }))
    .reset_index()
)


summary_out = ROOT_DIR / "summary_by_params.csv"
summary.to_csv(summary_out, index=False)


# ============================================================
# Generate LaTeX table rows
# ============================================================

def format_distance(value):
    """
    Convert distance names to LaTeX.
    """
    value = str(value).strip().lower()

    mapping = {
        "l1": "$L_1$",
        "1": "$L_1$",
        "l2": "$L_2$",
        "2": "$L_2$",
        "linf": "$L_\\infty$",
        "l_inf": "$L_\\infty$",
        "inf": "$L_\\infty$",
        "infinity": "$L_\\infty$",
    }

    return mapping.get(value, str(value))


def format_weight(value):
    """
    Convert internal weight names to table labels.
    """
    value = str(value).strip().lower()

    mapping = {
        "inverse": "Inverse",
        "gaussian": "Gaussian",
        "inverse_gaussian": "Inverse-Gaussian",
        "inverse-gaussian": "Inverse-Gaussian",
        "uniform": "Uniform",
    }

    return mapping.get(value, str(value).replace("_", "-").title())


def format_number(value, decimals=2):
    """
    Format a numeric value for LaTeX.
    Missing values are represented by --.
    """
    if pd.isna(value):
        return "--"

    return f"{float(value):.{decimals}f}"


def parse_sample_number(value):
    """
    Convert sampleNbr to a sortable numeric value when possible.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value)


# Convert sampleNbr to numeric where possible, so 5, 10, 20
# are sorted numerically rather than lexicographically.
summary["_sample_sort"] = summary["sampleNbr"].apply(parse_sample_number)

sample_numbers = sorted(
    summary["_sample_sort"].dropna().unique(),
    key=lambda x: (isinstance(x, str), x),
)

print(f"Detected sampleNbr values: {sample_numbers}")


# One LaTeX row is created for every combination of:
# model, probability, algorithm, distance and weight.
latex_group_columns = [
    "model",
    "p",
    "alg",
    "distance",
    "weights",
]

latex_lines = []

for _, group in summary.groupby(
    latex_group_columns,
    dropna=False,
    sort=False,
):
    first = group.iloc[0]

    cells = [
        "",  # Produces the leading "&" used inside a multirow table
        format_distance(first["distance"]),
        format_weight(first["weights"]),
    ]

    # Add four columns for every sample-number configuration:
    # Runs, Success, Variance, Mean Time
    for sample_nbr in sample_numbers:
        matched = group[group["_sample_sort"] == sample_nbr]

        if matched.empty:
            cells.extend([
                "--",  # Runs
                "--",  # Success
                "--",  # Variance
                "--",  # Mean time
            ])
            continue

        row = matched.iloc[0]

        num_runs = int(row["num_runs"])
        num_success = int(row["num_success"])

        cells.extend([
            str(num_runs),
            f"{num_success}/{num_runs}",
            format_number(
                row["variance_CEGIS_time_success"],
                decimals=2,
            ),
            format_number(
                row["mean_CEGIS_time_success"],
                decimals=2,
            ),
        ])

    latex_line = " & ".join(cells) + r" \\"
    latex_lines.append(latex_line)


latex_out = ROOT_DIR / "latex_rows.txt"

with latex_out.open("w", encoding="utf-8") as f:
    for line in latex_lines:
        f.write(line + "\n")


# Remove temporary sorting column before optional printing.
summary = summary.drop(columns=["_sample_sort"])


print(f"Found {len(info_files)} info files.")
print(f"Saved raw runs to: {raw_out}")
print(f"Saved grouped summary to: {summary_out}")
print(f"Saved LaTeX rows to: {latex_out}")
print()
print(summary.to_string(index=False))

print("\nLaTeX rows:")
for line in latex_lines:
    print(line)