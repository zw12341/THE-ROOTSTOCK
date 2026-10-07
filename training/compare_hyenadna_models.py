#!/usr/bin/env python3
"""Tabulate and plot validation/test metrics from HyenaDNA training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .hyenadna_models import HYENADNA_MODEL_REVISIONS
else:
    from hyenadna_models import HYENADNA_MODEL_REVISIONS

DEFAULT_RUNS_DIR = Path("training/runs")
DEFAULT_PLOT = Path("training/study/models/hyenadna_model_comparison.png")
DEFAULT_CSV = Path("training/study/models/hyenadna_model_metrics.csv")
DEFAULT_MARKDOWN = Path("training/study/models/hyenadna_model_metrics.md")
PREFERRED_GENES = ("ELF4", "ECT2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare structured metrics from HyenaDNA projection-head runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def model_label(model_name: str) -> str:
    model_id = model_name.rstrip("/").rsplit("/", 1)[-1]
    if model_id.startswith("hyenadna-"):
        model_id = model_id[len("hyenadna-") :]
    return model_id.replace("-seqlen-", "-").removesuffix("-hf")


def load_metrics(runs_dir: Path) -> list[dict]:
    paths = sorted(runs_dir.glob("*.metrics.json"))
    if not paths:
        raise FileNotFoundError(
            f"No *.metrics.json files found in {runs_dir}. Run "
            "training/train_hyenadna_models.sh first."
        )
    runs = []
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "model",
            "best_epoch",
            "initial_validation_ce",
            "best_validation_ce",
            "test_results",
        }
        missing = required - run.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
        run["source"] = str(path)
        run["label"] = model_label(run["model"])
        run["tests_by_gene"] = {
            result["gene"].upper(): result for result in run["test_results"]
        }
        run["validation_ce_improvement"] = (
            run["initial_validation_ce"] - run["best_validation_ce"]
        )
        elf4 = run["tests_by_gene"].get("ELF4")
        ect2 = run["tests_by_gene"].get("ECT2")
        run["circadian_specificity_margin"] = (
            elf4["ce_improvement"] - ect2["ce_improvement"]
            if elf4 is not None and ect2 is not None
            else None
        )
        runs.append(run)

    order = {
        model_label(model): index
        for index, model in enumerate(HYENADNA_MODEL_REVISIONS)
    }
    runs.sort(key=lambda run: (order.get(run["label"], len(order)), run["label"]))
    return runs


def genes_in(runs: list[dict]) -> list[str]:
    available = {
        gene for run in runs for gene in run["tests_by_gene"]
    }
    preferred = [gene for gene in PREFERRED_GENES if gene in available]
    return preferred + sorted(available - set(preferred))


def format_number(value: object, digits: int = 6) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_csv(runs: list[dict], genes: list[str], path: Path) -> None:
    base_fields = [
        "model",
        "model_revision",
        "best_epoch",
        "epochs_completed",
        "initial_validation_ce",
        "best_validation_ce",
        "validation_ce_improvement",
        "final_validation_ce",
        "circadian_specificity_margin",
    ]
    metric_fields = (
        "scored_bases",
        "pretrained_ce",
        "pretrained_perplexity",
        "fine_tuned_ce",
        "fine_tuned_perplexity",
        "ce_improvement",
    )
    fieldnames = base_fields + [
        f"{gene.lower()}_{metric}" for gene in genes for metric in metric_fields
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {field: run.get(field) for field in base_fields}
            for gene in genes:
                result = run["tests_by_gene"].get(gene, {})
                for metric in metric_fields:
                    row[f"{gene.lower()}_{metric}"] = result.get(metric)
            writer.writerow(row)


def markdown_table(runs: list[dict], genes: list[str]) -> str:
    headers = [
        "Model",
        "Best epoch",
        "Initial val CE",
        "Selected val CE",
        "Val ΔCE",
        "Final val CE",
    ]
    for gene in genes:
        headers.extend([f"{gene} base CE", f"{gene} tuned CE", f"{gene} ΔCE"])
    if "ELF4" in genes and "ECT2" in genes:
        headers.append("ELF4−ECT2 ΔCE")
    rows = []
    for run in runs:
        row = [
            run["label"],
            str(run["best_epoch"]),
            format_number(run["initial_validation_ce"]),
            format_number(run["best_validation_ce"]),
            format_number(run["validation_ce_improvement"]),
            format_number(run.get("final_validation_ce")),
        ]
        for gene in genes:
            result = run["tests_by_gene"].get(gene, {})
            row.extend(
                [
                    format_number(result.get("pretrained_ce")),
                    format_number(result.get("fine_tuned_ce")),
                    format_number(result.get("ce_improvement")),
                ]
            )
        if "ELF4" in genes and "ECT2" in genes:
            row.append(format_number(run["circadian_specificity_margin"]))
        rows.append(row)
    separator = ["---"] * len(headers)
    lines = [headers, separator, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in lines) + "\n"


def plot_metrics(runs: list[dict], genes: list[str], path: Path, dpi: int) -> None:
    labels = [run["label"] for run in runs]
    y = np.arange(len(runs))
    show_specificity = "ELF4" in genes and "ECT2" in genes
    panel_count = 1 + len(genes) + int(show_specificity)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(5.5 * panel_count, max(5.0, 0.65 * len(runs) + 2.0)),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    initial_validation = np.array(
        [run["initial_validation_ce"] for run in runs], dtype=float
    )
    selected_validation = np.array(
        [run["best_validation_ce"] for run in runs], dtype=float
    )
    for initial, selected, y_value in zip(
        initial_validation, selected_validation, y, strict=True
    ):
        improvement = initial - selected
        axes[0].plot([initial, selected], [y_value, y_value], color="#9aa0a6", lw=2)
        axes[0].scatter(initial, y_value, color="#777777", s=55, zorder=3)
        axes[0].scatter(selected, y_value, color="#315f9b", marker="D", s=55, zorder=3)
        axes[0].annotate(
            f"Δ {improvement:+.4f}", (max(initial, selected), y_value), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=9
        )
    axes[0].set_title("Validation: initial ○ → selected ◆")
    axes[0].set_xlabel("Cross-entropy (nats; lower is better)")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()

    for axis, gene in zip(axes[1 : 1 + len(genes)], genes, strict=True):
        for y_value, run in zip(y, runs, strict=True):
            result = run["tests_by_gene"].get(gene)
            if result is None:
                continue
            base = float(result["pretrained_ce"])
            tuned = float(result["fine_tuned_ce"])
            improvement = float(result["ce_improvement"])
            axis.plot([base, tuned], [y_value, y_value], color="#9aa0a6", lw=2)
            axis.scatter(base, y_value, color="#777777", s=55, zorder=3)
            axis.scatter(tuned, y_value, color="#d95f02", marker="D", s=55, zorder=3)
            axis.annotate(
                f"Δ {improvement:+.4f}",
                (max(base, tuned), y_value),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        axis.set_title(f"{gene}: pretrained ○ → fine-tuned ◆")
        axis.set_xlabel("Cross-entropy (nats; lower is better)")

    if show_specificity:
        specificity_axis = axes[-1]
        specificity = np.array(
            [run["circadian_specificity_margin"] for run in runs], dtype=float
        )
        specificity_axis.axvline(0.0, color="#777777", lw=1)
        specificity_axis.scatter(specificity, y, color="#2a8c6f", s=65, zorder=3)
        for value, y_value in zip(specificity, y, strict=True):
            specificity_axis.annotate(
                f"{value:+.4f}",
                (value, y_value),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=9,
            )
        specificity_axis.set_title("Circadian specificity")
        specificity_axis.set_xlabel("ELF4 ΔCE − ECT2 ΔCE (higher is better)")

    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        axis.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "HyenaDNA projection-head comparison\n"
        "Positive ΔCE means fine-tuning made the test sequence less surprising",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_metrics(args.runs_dir)
    genes = genes_in(runs)
    write_csv(runs, genes, args.csv)
    table = markdown_table(runs, genes)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(table, encoding="utf-8")
    plot_metrics(runs, genes, args.plot, args.dpi)

    print(table)
    print(f"Loaded {len(runs)} completed run(s) from {args.runs_dir}")
    print(f"Saved full metrics to {args.csv}")
    print(f"Saved readable table to {args.markdown}")
    print(f"Saved comparison plot to {args.plot}")


if __name__ == "__main__":
    main()
