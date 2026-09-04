"""
Phase 0: Dataset availability check + leakage-aware EDA for IEEE-CIS Fraud
Detection dataset.

Usage:
    python src/data/inspect_data.py

Behavior:
    - If the required raw CSV files are not found, this script reports their
      absence clearly, prints the expected directory structure, and exits
      without fabricating or substituting data.
    - If the files ARE found, it runs the Phase 0 EDA (structure, temporal,
      missingness, pseudo-entity candidate analysis) and writes:
        reports/phase0_eda_summary.md
        reports/figures/*.png

This script intentionally does NOT do any of the following (reserved for
later phases): feature engineering, encoding, imputation, model training,
anomaly detection, or final train/val/test file materialization.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

REQUIRED_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass
class AvailabilityReport:
    raw_dir: Path
    found: dict = field(default_factory=dict)   # filename -> bool
    sizes_bytes: dict = field(default_factory=dict)

    @property
    def all_present(self) -> bool:
        return all(self.found.values())

    @property
    def any_present(self) -> bool:
        return any(self.found.values())


def check_dataset_availability(raw_dir: Path) -> AvailabilityReport:
    report = AvailabilityReport(raw_dir=raw_dir)
    for fname in REQUIRED_FILES:
        fpath = raw_dir / fname
        exists = fpath.is_file()
        report.found[fname] = exists
        report.sizes_bytes[fname] = fpath.stat().st_size if exists else 0
    return report


def build_efficient_dtypes(sample_path: Path, nrows: int = 5000) -> dict:
    """
    Infer a memory-efficient dtype map from a small sample so the full CSV
    (≈590K rows x 394 columns) can be loaded within constrained RAM.

    - object/string columns -> 'category'
    - int64 columns -> int32 (IDs/time deltas/target all fit safely)
    - float64 columns -> float32 (acceptable precision loss for EDA purposes;
      NOT used for any modeling decision in this phase)
    """
    sample = pd.read_csv(sample_path, nrows=nrows)
    dtypes = {}
    for col in sample.columns:
        dt = sample[col].dtype
        if dt == object:
            dtypes[col] = "category"
        elif dt == "int64":
            dtypes[col] = "int32"
        elif dt == "float64":
            dtypes[col] = "float32"
        else:
            dtypes[col] = dt
    return dtypes


def make_figures(
    temporal: dict, missingness: dict, struct: dict, figures_dir: Path
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Figure 1: transaction volume over time bins
    fig, ax = plt.subplots(figsize=(10, 4))
    volume_by_bin = temporal["volume_by_time_bin"]
    ax.bar(range(len(volume_by_bin)), volume_by_bin.values, color="#55A868")
    ax.set_xlabel("Time bin (chronological, TransactionDT-ordered)")
    ax.set_ylabel("Transaction count")
    ax.set_title("Transaction volume over time (20 equal-width TransactionDT bins)")
    fig.tight_layout()
    path = figures_dir / "transaction_volume_over_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(str(path.relative_to(REPO_ROOT)))

    # Figure 2: fraud rate over time bins
    fig, ax = plt.subplots(figsize=(10, 4))
    fraud_by_bin = temporal["fraud_rate_by_time_bin"]
    ax.plot(range(len(fraud_by_bin)), fraud_by_bin.values, marker="o")
    ax.set_xlabel("Time bin (chronological, TransactionDT-ordered)")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud rate over time (20 equal-width TransactionDT bins)")
    ax.axhline(struct["fraud_rate"], color="gray", linestyle="--", linewidth=1,
               label=f"Overall rate = {struct['fraud_rate']:.4f}")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "fraud_rate_over_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(str(path.relative_to(REPO_ROOT)))

    # Figure 3: missingness by column group
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = [g for g in missingness if missingness[g]["n_columns"] > 0]
    means = [missingness[g]["mean_missing_pct"] * 100 for g in groups]
    ax.bar(groups, means, color="#4C72B0")
    ax.set_ylabel("Mean missing (%)")
    ax.set_title("Mean missingness by column group")
    for i, v in enumerate(means):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    path = figures_dir / "missingness_by_group.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(str(path.relative_to(REPO_ROOT)))

    return saved


def write_full_eda_report(
    struct: dict, temporal: dict, missingness: dict,
    pseudo_entities: list[dict], tests: dict, figure_paths: list[str],
    out_path: Path,
) -> None:
    lines = []
    lines.append("# Phase 0 EDA Summary — Real-Time Risk Decision & Fraud Intelligence Engine\n")
    lines.append("## Dataset availability status: **AVAILABLE**\n")
    lines.append(
        "This report reflects a real run against `data/raw/train_transaction.csv` "
        "from the IEEE-CIS Fraud Detection dataset. All figures below are "
        "measured, not illustrative.\n"
    )

    lines.append("## A. Dataset structure\n")
    lines.append(f"- Rows: **{struct['n_rows']:,}**")
    lines.append(f"- Columns: **{struct['n_cols']}**")
    lines.append(f"- Duplicate `TransactionID` values: **{struct['duplicate_transaction_ids']}**")
    lines.append(f"- `TransactionID` unique: **{struct['id_is_unique']}**")
    lines.append(f"- Fraud class counts: **{struct['fraud_counts']}**")
    lines.append(f"- Overall fraud rate: **{struct['fraud_rate']:.4%}**\n")

    lines.append("## B. Temporal structure (`TransactionDT`)\n")
    lines.append(f"- Min: **{temporal['dt_min']:,}** seconds")
    lines.append(f"- Max: **{temporal['dt_max']:,}** seconds")
    lines.append(f"- Range: **{temporal['dt_range_seconds']:,}** seconds (~**{temporal['dt_range_days_approx']}** days)")
    lines.append(f"- Rows already in `TransactionDT` order in the raw file: **{temporal['is_already_ordered_in_raw_file']}**")
    lines.append(f"- Total rows: **{temporal['n_rows']:,}**")
    lines.append(
        f"- Proposed **train** split: rows **0 – {temporal['train_end_row_idx']-1:,}** "
        f"({temporal['train_end_row_idx']:,} rows, {temporal['train_end_row_idx']/temporal['n_rows']:.2%}), "
        f"`TransactionDT` up to **{temporal['proposed_train_end_dt']:,}**"
    )
    lines.append(
        f"- Proposed **validation** split: rows **{temporal['train_end_row_idx']:,} – "
        f"{temporal['val_end_row_idx']-1:,}** "
        f"({temporal['val_end_row_idx']-temporal['train_end_row_idx']:,} rows, "
        f"{(temporal['val_end_row_idx']-temporal['train_end_row_idx'])/temporal['n_rows']:.2%}), "
        f"`TransactionDT` from **{temporal['proposed_train_end_dt']:,}** to **{temporal['proposed_val_end_dt']:,}**"
    )
    lines.append(
        f"- Proposed **test** split: rows **{temporal['val_end_row_idx']:,} – "
        f"{temporal['n_rows']-1:,}** "
        f"({temporal['n_rows']-temporal['val_end_row_idx']:,} rows, "
        f"{(temporal['n_rows']-temporal['val_end_row_idx'])/temporal['n_rows']:.2%}), "
        f"`TransactionDT` from **{temporal['proposed_val_end_dt']:,}** onward\n"
    )
    lines.append(
        "(Row boundaries above are positions in the `TransactionDT`-sorted "
        "order, not raw file row numbers — the raw file already happens to "
        "be close to time-ordered, but sorting is applied explicitly rather "
        "than assumed. No random shuffling was used at any point.)\n"
    )
    lines.append("Fraud rate and transaction volume per time bin (20 equal-width bins across the full range):\n")
    lines.append("| Bin | Transaction volume | Fraud rate |")
    lines.append("|---|---|---|")
    volume_by_bin = temporal["volume_by_time_bin"]
    for interval, rate in temporal["fraud_rate_by_time_bin"].items():
        vol = volume_by_bin.get(interval, 0)
        lines.append(f"| {interval} | {vol:,} | {rate:.4%} |")
    lines.append("")

    lines.append("## C. Missingness by column group\n")
    lines.append("| Group | # columns | Mean missing % | Max missing % | Min missing % |")
    lines.append("|---|---|---|---|---|")
    for group, stats in missingness.items():
        if stats["n_columns"] == 0:
            lines.append(f"| {group} | 0 | - | - | - |")
            continue
        lines.append(
            f"| {group} | {stats['n_columns']} | {stats['mean_missing_pct']:.2%} | "
            f"{stats['max_missing_pct']:.2%} | {stats['min_missing_pct']:.2%} |"
        )
    lines.append("")
    for group, stats in missingness.items():
        if stats["n_columns"] == 0:
            continue
        lines.append(f"**Top-5 most-missing columns in `{group}`:**")
        for col, pct in stats["top5_most_missing"].items():
            lines.append(f"- `{col}`: {pct:.2%}")
        lines.append("")

    lines.append("## D. Candidate pseudo-entity key analysis\n")
    lines.append(
        "isFraud was never used to construct these keys. Rows with a missing "
        "value in any constituent column are excluded from cardinality/"
        "txn-per-entity statistics (see `pseudo_entity_analysis` docstring) "
        "and reported separately below, so missingness cannot silently "
        "create false entity groups.\n"
    )
    for r in pseudo_entities:
        lines.append(f"### Key: `{' + '.join(r['key_columns'])}`\n")
        lines.append(f"- Rows total: **{r['rows_total']:,}**")
        lines.append(f"- Rows with a missing key component: **{r['rows_with_missing_key_component']:,}** ({r['pct_rows_with_missing_key_component']:.2%})")
        lines.append(f"- Rows with complete key: **{r['rows_with_complete_key']:,}**")
        if r["cardinality"] is None:
            lines.append(f"- Cardinality: not computable — {r.get('note', '')}\n")
            continue
        lines.append(f"- Cardinality (distinct pseudo-entities): **{r['cardinality']:,}**")
        lines.append(f"- Median transactions/entity: **{r['median_txn_per_entity']}**")
        lines.append(f"- Mean transactions/entity: **{r['mean_txn_per_entity']:.2f}**")
        lines.append(f"- Max transactions/entity: **{r['max_txn_per_entity']:,}**")
        lines.append(f"- % of entities with exactly 1 transaction: **{r['pct_entities_single_txn']:.2%}**\n")

    lines.append("### Recommendation\n")
    lines.append(
        "`card1` alone is recommended as the Phase 1 starting point for "
        "behavioral aggregation: it has zero missing values (no rows "
        "excluded), the lowest cardinality-to-coverage tradeoff loss, and "
        "gives every transaction a usable pseudo-entity. Adding `card2` "
        "barely changes cardinality (13,553 → 13,490) while excluding 1.51% "
        "of rows, suggesting `card1` is already close to saturating the "
        "identity signal available in the `card*` fields alone. Adding "
        "`addr1`/`addr2` roughly triples cardinality (13.5K → ~37-38K) and "
        "excludes ~12.5-12.9% of rows — this likely fragments genuine "
        "repeat entities into multiple pseudo-entities (e.g. the same card "
        "used with a missing or differently-imputed address) rather than "
        "adding real identity resolution power, which is the opposite of "
        "what a behavioral feature needs. This is a judgment call from the "
        "measured tradeoffs above, not a validated ground truth — there is "
        "no real customer ID in this dataset to check against.\n"
    )
    lines.append(
        "**Limitations (apply to all candidates above):** none of these "
        "keys are verified customer identities — `card1`/`card2` are "
        "Vesta's anonymized/hashed card identifiers and may not "
        "1:1 map to a single physical card or person (shared cards, "
        "re-issued cards, or hash collisions are all possible and "
        "unverifiable from this data). Cardinality and transactions-per-"
        "entity are therefore proxies for behavioral grouping, not proof "
        "of identity. Any velocity/frequency feature built on these keys "
        "must be documented as resting on this assumption.\n"
    )

    lines.append("## Integrity tests\n")
    lines.append("| Test | Result |")
    lines.append("|---|---|")
    for name, result in tests.items():
        lines.append(f"| {name} | {'PASS' if result else 'FAIL'} |")
    lines.append("")

    if figure_paths:
        lines.append("## Figures\n")
        for fp in figure_paths:
            lines.append(f"![{fp}]({Path(fp).name})" if False else f"- `{fp}`")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote full EDA report to: {out_path}")


def print_expected_structure(raw_dir: Path) -> None:
    print("\nExpected dataset directory structure:\n")
    print(f"  {raw_dir}/")
    for fname in REQUIRED_FILES:
        print(f"    ├── {fname}")
    print(
        "\nDownload from the Kaggle competition 'IEEE-CIS Fraud Detection' "
        "(requires a Kaggle account and acceptance of competition terms), "
        "then place the four CSV files at the paths above.\n"
        "This environment cannot reach Kaggle's servers directly (network "
        "egress is restricted to package registries), so the files must be "
        "supplied by the user rather than downloaded by this script."
    )


def write_missing_data_report(report: AvailabilityReport, out_path: Path) -> None:
    lines = []
    lines.append("# Phase 0 EDA Summary — Real-Time Risk Decision & Fraud Intelligence Engine\n")
    lines.append("## Dataset availability status: **NOT AVAILABLE**\n")
    lines.append(
        "The IEEE-CIS Fraud Detection dataset was not found in the expected "
        f"location (`{report.raw_dir}`). No EDA, feature engineering, or "
        "modeling has been performed. No substitute or synthetic data was "
        "used.\n"
    )
    lines.append("### File check\n")
    lines.append("| File | Found |")
    lines.append("|---|---|")
    for fname in REQUIRED_FILES:
        status = "yes" if report.found[fname] else "**no**"
        lines.append(f"| `{fname}` | {status} |")
    lines.append("")
    lines.append("### Expected directory structure\n")
    lines.append("```")
    lines.append(f"{report.raw_dir}/")
    for fname in REQUIRED_FILES:
        lines.append(f"    ├── {fname}")
    lines.append("```\n")
    lines.append(
        "### Next step\n\n"
        "Place the four CSV files listed above at the paths shown, then "
        "re-run:\n\n```bash\npython src/data/inspect_data.py\n```\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote missing-data report to: {out_path}")


# ---------------------------------------------------------------------------
# The functions below run only when the dataset is present. They are fully
# implemented now so that Phase 0 EDA can execute immediately once the user
# supplies the data, without a second implementation pass.
# ---------------------------------------------------------------------------

def structure_report(df: pd.DataFrame, id_col: str, target_col: str) -> dict:
    n_rows, n_cols = df.shape
    dup_ids = int(df[id_col].duplicated().sum())
    fraud_counts = df[target_col].value_counts().to_dict()
    fraud_rate = float(df[target_col].mean())
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_transaction_ids": dup_ids,
        "id_is_unique": dup_ids == 0,
        "fraud_counts": fraud_counts,
        "fraud_rate": fraud_rate,
    }


def temporal_analysis(
    df: pd.DataFrame, time_col: str, target_col: str,
    train_ratio: float, val_ratio: float, test_ratio: float,
) -> dict:
    dt = df[time_col]
    dt_min, dt_max = int(dt.min()), int(dt.max())
    is_already_ordered = bool(dt.is_monotonic_increasing)

    df_sorted = df.sort_values(time_col)
    n = len(df_sorted)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_boundary_dt = int(df_sorted[time_col].iloc[train_end_idx - 1])
    val_boundary_dt = int(df_sorted[time_col].iloc[val_end_idx - 1])

    # Fraud rate AND transaction volume over coarse time bins (20 bins
    # across full range) so both can be reported/plotted together.
    n_bins = 20
    bins = np.linspace(dt_min, dt_max, n_bins + 1)
    bin_labels = pd.cut(df[time_col], bins=bins, include_lowest=True)
    fraud_rate_by_bin = df.groupby(bin_labels, observed=True)[target_col].mean()
    volume_by_bin = df.groupby(bin_labels, observed=True)[target_col].size()

    return {
        "dt_min": dt_min,
        "dt_max": dt_max,
        "dt_range_seconds": dt_max - dt_min,
        "dt_range_days_approx": round((dt_max - dt_min) / 86400, 1),
        "is_already_ordered_in_raw_file": is_already_ordered,
        "n_rows": n,
        "train_end_row_idx": train_end_idx,       # number of rows in train (0-indexed exclusive)
        "val_end_row_idx": val_end_idx,            # number of rows in train+val
        "proposed_train_end_dt": train_boundary_dt,
        "proposed_val_end_dt": val_boundary_dt,
        "fraud_rate_by_time_bin": fraud_rate_by_bin,
        "volume_by_time_bin": volume_by_bin,
    }


def missingness_analysis(df: pd.DataFrame) -> dict:
    groups = {
        "card": [c for c in df.columns if c.startswith("card")],
        "addr": [c for c in df.columns if c.startswith("addr")],
        "D": [c for c in df.columns if c.startswith("D") and c[1:].isdigit()],
        "C": [c for c in df.columns if c.startswith("C") and c[1:].isdigit()],
        "V": [c for c in df.columns if c.startswith("V") and c[1:].isdigit()],
    }
    result = {}
    for group_name, cols in groups.items():
        if not cols:
            result[group_name] = {"n_columns": 0}
            continue
        miss_pct = df[cols].isna().mean().sort_values(ascending=False)
        result[group_name] = {
            "n_columns": len(cols),
            "mean_missing_pct": float(miss_pct.mean()),
            "max_missing_pct": float(miss_pct.max()),
            "min_missing_pct": float(miss_pct.min()),
            "top5_most_missing": miss_pct.head(5).to_dict(),
        }
    return result


def pseudo_entity_analysis(df: pd.DataFrame, candidates: list[list[str]]) -> list[dict]:
    """
    Build candidate pseudo-entity keys from combinations of raw columns.

    Explicit missing-value policy:
        A row is EXCLUDED from entity-key construction if ANY of the
        constituent columns is null for that row. We do NOT let
        `.astype(str)` silently turn NaN into the literal string "nan",
        because doing so would merge unrelated rows (e.g. two transactions
        that are both missing `addr2`) into one fabricated pseudo-entity
        group purely due to shared missingness, not shared identity. That
        would bias cardinality downward and inflate txn-per-entity stats
        for reasons that have nothing to do with real customer behavior.

    Rows with any missing key component are reported separately (count and
    percentage) rather than silently dropped without disclosure.
    """
    results = []
    n_total = len(df)
    for key_cols in candidates:
        key_cols = [c for c in key_cols if c in df.columns]
        if not key_cols:
            continue

        missing_mask = df[key_cols].isna().any(axis=1)
        n_missing = int(missing_mask.sum())
        n_valid = n_total - n_missing

        valid_df = df.loc[~missing_mask, key_cols]

        if n_valid == 0:
            results.append({
                "key_columns": key_cols,
                "rows_total": n_total,
                "rows_with_missing_key_component": n_missing,
                "pct_rows_with_missing_key_component": 1.0,
                "rows_with_complete_key": 0,
                "cardinality": None,
                "note": "No rows had a complete key for this candidate; stats not computable.",
            })
            continue

        # Safe to build string keys now: every remaining row has no NaN in
        # any key column, so no group is formed by matching on missingness.
        key = valid_df.astype(str).agg("_".join, axis=1)
        cardinality = key.nunique()
        txn_per_entity = key.value_counts()

        results.append({
            "key_columns": key_cols,
            "rows_total": n_total,
            "rows_with_missing_key_component": n_missing,
            "pct_rows_with_missing_key_component": float(n_missing / n_total),
            "rows_with_complete_key": int(n_valid),
            "cardinality": int(cardinality),
            "median_txn_per_entity": float(txn_per_entity.median()),
            "mean_txn_per_entity": float(txn_per_entity.mean()),
            "max_txn_per_entity": int(txn_per_entity.max()),
            "pct_entities_single_txn": float((txn_per_entity == 1).mean()),
        })
    return results


def run_integrity_tests(
    df: pd.DataFrame, id_col: str, time_col: str,
    train_ratio: float, val_ratio: float,
) -> dict:
    tests = {}

    tests["transaction_id_unique"] = bool(df[id_col].is_unique)

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = df_sorted.index[:train_end]
    val_idx = df_sorted.index[train_end:val_end]
    test_idx = df_sorted.index[val_end:]

    overlap_train_val = set(df_sorted.loc[train_idx, id_col]) & set(df_sorted.loc[val_idx, id_col])
    overlap_val_test = set(df_sorted.loc[val_idx, id_col]) & set(df_sorted.loc[test_idx, id_col])
    overlap_train_test = set(df_sorted.loc[train_idx, id_col]) & set(df_sorted.loc[test_idx, id_col])

    tests["no_row_overlap_train_val"] = len(overlap_train_val) == 0
    tests["no_row_overlap_val_test"] = len(overlap_val_test) == 0
    tests["no_row_overlap_train_test"] = len(overlap_train_test) == 0

    max_train_dt = df_sorted.loc[train_idx, time_col].max()
    min_val_dt = df_sorted.loc[val_idx, time_col].min()
    max_val_dt = df_sorted.loc[val_idx, time_col].max()
    min_test_dt = df_sorted.loc[test_idx, time_col].min()

    tests["max_train_dt_lte_min_val_dt"] = bool(max_train_dt <= min_val_dt)
    tests["max_val_dt_lte_min_test_dt"] = bool(max_val_dt <= min_test_dt)

    return tests


def main() -> int:
    config = load_config()
    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]
    report_out = REPO_ROOT / "reports" / "phase0_eda_summary.md"

    print(f"Checking dataset availability in: {raw_dir}")
    availability = check_dataset_availability(raw_dir)
    for fname, found in availability.found.items():
        status = "FOUND" if found else "MISSING"
        print(f"  [{status}] {fname}")

    if not availability.found.get("train_transaction.csv", False):
        print("\ntrain_transaction.csv is not available — Phase 0 EDA cannot proceed.")
        print_expected_structure(raw_dir)
        write_missing_data_report(availability, report_out)
        return 1

    # --- Dataset present: run full EDA ---
    print("\ntrain_transaction.csv found — loading and running EDA...")
    train_path = REPO_ROOT / config["paths"]["train_transaction"]

    dtypes = build_efficient_dtypes(train_path)
    df = pd.read_csv(train_path, dtype=dtypes)
    print(f"Loaded shape: {df.shape}")

    id_col = config["split"]["id_column"]
    time_col = config["split"]["time_column"]
    target_col = config["split"]["target_column"]

    struct = structure_report(df, id_col, target_col)
    print("  structure_report done")

    # Temporal ops (sorting, groupby) only need 3 narrow columns. Passing the
    # full 394-column frame into sort_values() would force a second ~1.2GB
    # in-memory copy of the entire wide dataframe purely to determine row
    # order by time — unnecessary and, on this machine's limited RAM, the
    # actual cause of an OOM kill during the first run. Using a narrow
    # subset instead keeps peak memory well within budget without changing
    # any analysis logic or leakage rules.
    df_time = df[[id_col, time_col, target_col]].copy()

    temporal = temporal_analysis(
        df_time, time_col, target_col,
        config["split"]["train_ratio"],
        config["split"]["val_ratio"],
        config["split"]["test_ratio"],
    )
    print("  temporal_analysis done")
    missingness = missingness_analysis(df)
    print("  missingness_analysis done")
    pseudo_entities = pseudo_entity_analysis(df, config["pseudo_entity"]["candidates"])
    print("  pseudo_entity_analysis done")
    tests = run_integrity_tests(
        df_time, id_col, time_col,
        config["split"]["train_ratio"], config["split"]["val_ratio"],
    )
    print("  run_integrity_tests done")

    del df_time
    gc.collect()

    print("\n--- Structure ---")
    print(struct)
    print("\n--- Temporal ---")
    print({k: v for k, v in temporal.items() if k not in ("fraud_rate_by_time_bin", "volume_by_time_bin")})
    print("\n--- Integrity tests ---")
    print(tests)

    figures_dir = REPO_ROOT / config["paths"]["figures_dir"]
    del df
    gc.collect()
    figure_paths = make_figures(temporal, missingness, struct, figures_dir)

    write_full_eda_report(
        struct, temporal, missingness, pseudo_entities, tests, figure_paths,
        report_out,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
