from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

# 指向所有 samebudget2D_* 文件夹的共同父目录
ROOT_DIR = Path(
    "/Users/zhengqi/Desktop/Final-main/output/sto/triplesame"
)

RAW_OUTPUT_NAME = "summary_raw_runs.csv"
SUMMARY_OUTPUT_NAME = "summary_by_params.csv"
LATEX_OUTPUT_NAME = "latex_rows.txt"

# 支持：
# samebudget2D_5
# samebudget2D_10
# samebudget2D_20
# samebudget2D_100
BUDGET_FOLDER_PATTERN = re.compile(
    r"^samebudget2D_(\d+)$",
    re.IGNORECASE,
)


# ============================================================
# File discovery
# ============================================================

def find_info_files(root: Path):
    """
    Recursively find one info file per experiment directory.

    Supported exact filenames, in priority order:
        info
        info.csv
        info.txt

    This avoids counting the same run multiple times when one folder
    accidentally contains several files such as info and info.csv.
    """
    priority = {
        "info": 0,
        "info.csv": 1,
        "info.txt": 2,
    }

    files_by_run_directory = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        filename = path.name.lower()

        if filename not in priority:
            continue

        current = files_by_run_directory.get(path.parent)

        if current is None:
            files_by_run_directory[path.parent] = path
            continue

        current_priority = priority[current.name.lower()]
        new_priority = priority[filename]

        if new_priority < current_priority:
            files_by_run_directory[path.parent] = path

    return sorted(files_by_run_directory.values())


def read_info_file(path: Path):
    """
    Read an info file containing lines in the form:

        key,value
    """
    data = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            line = line.strip()

            if not line or "," not in line:
                continue

            key, value = line.split(",", 1)
            data[key.strip()] = value.strip()

    return data


# ============================================================
# Path and folder parsing
# ============================================================

def extract_budget_from_path(path: Path):
    """
    Extract budget from ancestor folders such as:

        samebudget2D_10
        samebudget2D_20
        tripleSameBudget_10
        tripleSameBudget_20
        collisionSameBudget_30

    Returns:
        int budget, or None if no matching folder is found.
    """
    pattern = re.compile(
        r"increase(?:2d)?[_-]?(\d+)$",
        re.IGNORECASE,
    )

    for candidate in [path] + list(path.parents):
        match = pattern.search(candidate.name)

        if match:
            return int(match.group(1))

    return None


def parse_folder_params(folder_name: str):
    """
    Parse a run directory name such as:

    date=2026-06-01_21-35-13_model=MyPendulum_p=0.8_seed=1_alg=PPO_JAX_distance=l1_weights=inverse_gaussian_sampleNbr=10

    Values may contain underscores, for example:
        PPO_JAX
        inverse_gaussian
    """
    keys = [
        "date",
        "model",
        "p",
        "seed",
        "alg",
        "distance",
        "weights",
        "sampleNbr",
    ]

    params = {}

    for index, key in enumerate(keys):
        start_token = key + "="
        start = folder_name.find(start_token)

        if start == -1:
            continue

        value_start = start + len(start_token)

        next_positions = []

        for next_key in keys[index + 1:]:
            position = folder_name.find(
                "_" + next_key + "=",
                value_start,
            )

            if position != -1:
                next_positions.append(position)

        if next_positions:
            value_end = min(next_positions)
        else:
            value_end = len(folder_name)

        params[key] = folder_name[value_start:value_end]

    return params


def normalize_status(status):
    """
    Normalize result status names.
    """
    if status is None:
        return "missing"

    normalized = str(status).strip().lower()

    # Handle typo present in some historical output files
    if normalized == "seccess":
        return "success"

    if normalized == "":
        return "missing"

    return normalized


def safe_float(value):
    """
    Convert a value to float, returning NaN on failure.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================
# Statistical helpers
# ============================================================

def join_unique(values):
    """
    Join unique values in a stable readable form.
    """
    cleaned = {
        str(value).strip()
        for value in values
        if str(value).strip() != ""
    }

    def sort_key(value):
        try:
            return 0, float(value)
        except ValueError:
            return 1, value

    return ", ".join(sorted(cleaned, key=sort_key))


def failed_seeds(group):
    """
    Return failed seeds and their statuses.
    """
    failed = group.loc[~group["success"]]

    if failed.empty:
        return ""

    entries = []

    for _, row in failed.iterrows():
        entries.append(
            f"{row['seed']}:{row['status']}"
        )

    return ", ".join(entries)


def sample_std(values):
    """
    Calculate sample standard deviation using ddof=1.

    At least two valid observations are required. When there are fewer
    than two observations, NaN is returned and later displayed as --.
    """
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(numeric_values) < 2:
        return np.nan

    return float(numeric_values.std(ddof=1))


# ============================================================
# LaTeX formatting helpers
# ============================================================

def format_distance(value):
    """
    Convert internal distance names into LaTeX labels.
    """
    normalized = str(value).strip().lower()

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

    return mapping.get(normalized, str(value))


def format_weight(value):
    """
    Convert internal weight names into readable table labels.
    """
    normalized = str(value).strip().lower()

    mapping = {
        "inverse": "Inverse",
        "gaussian": "Gaussian",
        "inverse_gaussian": "Inverse-Gaussian",
        "inverse-gaussian": "Inverse-Gaussian",
        "uniform": "Uniform",
    }

    return mapping.get(
        normalized,
        str(value).replace("_", "-").title(),
    )


def format_number(value, decimals=2):
    """
    Format a number for LaTeX.
    """
    if pd.isna(value):
        return "--"

    return f"{float(value):.{decimals}f}"


def latex_escape(value):
    """
    Escape common LaTeX special characters.
    """
    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================
# Collect raw run data
# ============================================================

info_files = find_info_files(ROOT_DIR)

if not info_files:
    print(f"No info files found under:\n{ROOT_DIR}")
    print()
    print("请检查目录结构，例如运行：")
    print(f'find "{ROOT_DIR}" -maxdepth 4 -type f | head -100')
    raise SystemExit(1)


records = []

for info_path in info_files:
    run_directory = info_path.parent
    folder_name = run_directory.name

    info = read_info_file(info_path)
    folder_params = parse_folder_params(folder_name)

    budget = extract_budget_from_path(info_path)

    model = info.get(
        "model",
        folder_params.get("model", ""),
    )

    probability = folder_params.get(
        "p",
        info.get("probability_bound", ""),
    )

    seed = info.get(
        "seed",
        folder_params.get("seed", ""),
    )

    algorithm = folder_params.get(
        "alg",
        info.get("algorithm", ""),
    )

    distance = folder_params.get(
        "distance",
        "",
    )

    weights = folder_params.get(
        "weights",
        "",
    )

    sample_nbr = folder_params.get(
        "sampleNbr",
        "",
    )

    status = normalize_status(
        info.get("status", "")
    )

    cegis_time = safe_float(
        info.get("total_CEGIS_time", np.nan)
    )

    verify_samples = safe_float(
        info.get("verify_samples", np.nan)
    )

    records.append({
        "budget": budget,
        "model": model,
        "p": probability,
        "seed": seed,
        "alg": algorithm,
        "distance": distance,
        "weights": weights,
        "sampleNbr": sample_nbr,
        "status": status,
        "success": status == "success",
        "total_CEGIS_time": cegis_time,
        "verify_samples": verify_samples,
        "folder": folder_name,
        "run_directory": str(run_directory),
        "info_file": str(info_path),
    })


df = pd.DataFrame(records)


# ============================================================
# Data validation
# ============================================================

print(f"Found {len(info_files)} experiment runs.")

missing_budget = df["budget"].isna()

if missing_budget.any():
    print()
    print("Warning: the following runs were not under a folder named")
    print("samebudget2D_<number> and will be excluded:")

    print(
        df.loc[
            missing_budget,
            [
                "model",
                "folder",
                "info_file",
            ],
        ].to_string(index=False)
    )

    df = df.loc[~missing_budget].copy()


if df.empty:
    raise RuntimeError(
        "No valid runs remain after budget-folder filtering."
    )


missing_model = (
    df["model"]
    .astype(str)
    .str.strip()
    .eq("")
)

if missing_model.any():
    print()
    print("Warning: runs with missing model names:")

    print(
        df.loc[
            missing_model,
            [
                "budget",
                "folder",
                "info_file",
            ],
        ].to_string(index=False)
    )


# Convert budget to integer
df["budget"] = pd.to_numeric(
    df["budget"],
    errors="coerce",
).astype("Int64")


# Detect accidental duplicates by run directory
duplicate_counts = (
    df.groupby("run_directory")
    .size()
)

duplicate_counts = duplicate_counts[
    duplicate_counts > 1
]

if not duplicate_counts.empty:
    print()
    print("Warning: duplicate run directories detected:")
    print(duplicate_counts.to_string())


# ============================================================
# Save raw run data
# ============================================================

raw_output = ROOT_DIR / RAW_OUTPUT_NAME

df.to_csv(
    raw_output,
    index=False,
)


# ============================================================
# Group statistics
# ============================================================

# Explicit grouping guarantees that environments and budgets
# are never mixed.
GROUP_COLUMNS = [
    "model",
    "p",
    "alg",
    "distance",
    "weights",
    "budget",
]


def summarize_group(group):
    successful_times = group.loc[
        group["success"],
        "total_CEGIS_time",
    ]

    all_times = group[
        "total_CEGIS_time"
    ]

    return pd.Series({
        "num_runs": len(group),
        "num_success": int(
            group["success"].sum()
        ),
        "success_rate": float(
            group["success"].mean() * 100
        ),

        "mean_CEGIS_time_success": (
            successful_times.mean()
        ),
        "std_CEGIS_time_success": sample_std(
            successful_times
        ),

        "mean_CEGIS_time_all": (
            all_times.mean()
        ),
        "std_CEGIS_time_all": sample_std(
            all_times
        ),

        "min_CEGIS_time_success": (
            successful_times.min()
        ),
        "max_CEGIS_time_success": (
            successful_times.max()
        ),

        "seeds": join_unique(
            group["seed"]
        ),
        "failed_seeds": failed_seeds(
            group
        ),

        # Useful for checking whether sampleNbr agrees with
        # the budget-folder name.
        "sampleNbr_values": join_unique(
            group["sampleNbr"]
        ),
    })


summary = (
    df.groupby(
        GROUP_COLUMNS,
        dropna=False,
        sort=False,
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)


# ============================================================
# Print raw times for diagnostic purposes
# ============================================================

print()
print("=" * 80)
print("Successful CEGIS times by group")
print("=" * 80)

for group_key, group in df.groupby(
    GROUP_COLUMNS,
    dropna=False,
    sort=True,
):
    successful = group.loc[
        group["success"],
        [
            "seed",
            "total_CEGIS_time",
            "folder",
        ],
    ].copy()

    print()
    print(f"Group: {group_key}")

    if successful.empty:
        print("  No successful runs.")
        continue

    print(successful.to_string(index=False))

    successful_times = pd.to_numeric(
        successful["total_CEGIS_time"],
        errors="coerce",
    ).dropna()

    mean_value = successful_times.mean()
    std_value = sample_std(successful_times)

    print(
        f"  Mean: {format_number(mean_value)} s"
    )
    print(
        f"  Std.: {format_number(std_value)} s"
    )


# ============================================================
# Sort summary consistently
# ============================================================

distance_order = {
    "l1": 0,
    "1": 0,
    "l2": 1,
    "2": 1,
    "inf": 2,
    "linf": 2,
    "l_inf": 2,
    "infinity": 2,
}

weight_order = {
    "inverse": 0,
    "gaussian": 1,
    "inverse_gaussian": 2,
    "inverse-gaussian": 2,
    "uniform": 3,
}


summary["_distance_sort"] = (
    summary["distance"]
    .astype(str)
    .str.lower()
    .map(distance_order)
    .fillna(99)
)

summary["_weight_sort"] = (
    summary["weights"]
    .astype(str)
    .str.lower()
    .map(weight_order)
    .fillna(99)
)

summary = summary.sort_values(
    by=[
        "model",
        "p",
        "alg",
        "_distance_sort",
        "_weight_sort",
        "budget",
    ],
    kind="stable",
).reset_index(drop=True)


# ============================================================
# Detect all available budgets automatically
# ============================================================

budgets = sorted(
    int(value)
    for value in summary["budget"].dropna().unique()
)

print()
print(f"Detected budgets: {budgets}")

if not budgets:
    raise RuntimeError(
        "No samebudget2D_<number> folders were detected."
    )


# ============================================================
# Save grouped CSV
# ============================================================

summary_output = ROOT_DIR / SUMMARY_OUTPUT_NAME

clean_summary = summary.drop(
    columns=[
        "_distance_sort",
        "_weight_sort",
    ]
)

clean_summary.to_csv(
    summary_output,
    index=False,
)


# ============================================================
# Generate LaTeX rows
#
# For every budget:
# Success | Std. (s) | Mean (s)
# ============================================================

latex_lines = []

# Columns:
# Model + Distance + Weight + 3 columns for each budget
total_table_columns = 3 + 3 * len(budgets)


for model_name, model_group in summary.groupby(
    "model",
    dropna=False,
    sort=False,
):
    if latex_lines:
        latex_lines.append(r"\midrule")

    formatted_model = latex_escape(model_name)

    latex_lines.append(
        rf"\multicolumn{{{total_table_columns}}}"
        rf"{{l}}{{\textbf{{{formatted_model}}}}} \\"
    )

    # Budget is deliberately excluded because it is expanded
    # horizontally into separate table-column groups.
    row_group_columns = [
        "p",
        "alg",
        "distance",
        "weights",
    ]

    for _, group in model_group.groupby(
        row_group_columns,
        dropna=False,
        sort=False,
    ):
        first = group.iloc[0]

        cells = [
            "",
            format_distance(
                first["distance"]
            ),
            format_weight(
                first["weights"]
            ),
        ]

        for budget in budgets:
            matched = group.loc[
                group["budget"] == budget
            ]

            if matched.empty:
                cells.extend([
                    "--",  # Success
                    "--",  # Std.
                    "--",  # Mean
                ])
                continue

            if len(matched) > 1:
                print()
                print(
                    "Warning: multiple summary rows matched "
                    f"model={model_name}, "
                    f"distance={first['distance']}, "
                    f"weights={first['weights']}, "
                    f"budget={budget}."
                )
                print(
                    "Only the first matching row is used in LaTeX."
                )

            row = matched.iloc[0]

            num_runs = int(
                row["num_runs"]
            )
            num_success = int(
                row["num_success"]
            )

            cells.extend([
                f"{num_success}/{num_runs}",

                format_number(
                    row["std_CEGIS_time_success"],
                    decimals=2,
                ),

                format_number(
                    row["mean_CEGIS_time_success"],
                    decimals=2,
                ),
            ])

        latex_lines.append(
            " & ".join(cells) + r" \\"
        )


# ============================================================
# Generate optional LaTeX table-header template
# ============================================================

header_lines = []

header_lines.append(
    "% Suggested LaTeX header"
)

header_lines.append(
    rf"\begin{{tabular}}{{lll{'rrr' * len(budgets)}}}"
)

header_lines.append(r"\toprule")

first_header_cells = [
    "Model",
    "Distance",
    "Weight",
]

for budget in budgets:
    first_header_cells.append(
        rf"\multicolumn{{3}}{{c}}{{Budget {budget}}}"
    )

header_lines.append(
    " & ".join(first_header_cells) + r" \\"
)


# Generate cmidrule ranges automatically.
# First three columns are Model, Distance, Weight.
cmidrules = []

for index, _ in enumerate(budgets):
    start_column = 4 + index * 3
    end_column = start_column + 2

    cmidrules.append(
        rf"\cmidrule(lr){{{start_column}-{end_column}}}"
    )

header_lines.append(
    "".join(cmidrules)
)


second_header_cells = [
    "",
    "",
    "",
]

for _ in budgets:
    second_header_cells.extend([
        "Success",
        "Std. (s)",
        "Mean (s)",
    ])

header_lines.append(
    " & ".join(second_header_cells) + r" \\"
)

header_lines.append(r"\midrule")


# ============================================================
# Save LaTeX output
# ============================================================

latex_output = ROOT_DIR / LATEX_OUTPUT_NAME

with latex_output.open(
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "% Automatically generated LaTeX rows\n"
    )

    file.write(
        "% Each budget group contains: "
        "Success, sample standard deviation, mean CEGIS time\n"
    )

    file.write(
        f"% Detected budgets: {budgets}\n\n"
    )

    for line in header_lines:
        file.write(line + "\n")

    file.write("\n")

    for line in latex_lines:
        file.write(line + "\n")

    file.write("\n")
    file.write(r"\bottomrule" + "\n")
    file.write(r"\end{tabular}" + "\n")


# ============================================================
# Final output summary
# ============================================================

print()
print("=" * 80)
print("Saved files")
print("=" * 80)

print(f"Raw runs:\n  {raw_output}")
print(f"Grouped summary:\n  {summary_output}")
print(f"LaTeX rows:\n  {latex_output}")

print()
print("=" * 80)
print("Grouped summary")
print("=" * 80)

print(
    clean_summary.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("LaTeX output")
print("=" * 80)

for line in header_lines:
    print(line)

print()

for line in latex_lines:
    print(line)

print(r"\bottomrule")
print(r"\end{tabular}")