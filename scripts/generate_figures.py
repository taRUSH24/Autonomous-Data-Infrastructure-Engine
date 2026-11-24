#!/usr/bin/env python3
"""Generates the Phase 5 visualization deliverables directly from the real
experiment output in experiments/results/ -- no synthetic data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def fig_confusion_matrices():
    df = pd.read_csv(RESULTS_DIR / "ml_results.csv")
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(model, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["clean", "fault"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["clean", "fault"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("Confusion matrices -- 4 anomaly-detection models on held-out TEST days (n=234 rows, 12 faulted)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    p1 = summary["phase1_anomaly_detection"]
    models = list(p1.keys())
    metrics = ["precision", "recall", "f1", "false_positive_rate"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(models))
    width = 0.2
    for i, metric in enumerate(metrics):
        vals = [p1[m][metric] for m in models]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=metric)
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("score")
    ax.set_title("Anomaly-detection model comparison (held-out TEST days)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_rca_comparison():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    p2 = summary["phase2_root_cause_analysis"]
    methods = [m for m in p2 if m not in ("n_fault_arrangements", "n_detection_misses")]
    top1 = [p2[m]["top1_accuracy"] for m in methods]
    top3 = [p2[m]["top3_accuracy"] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(methods))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], top1, width=width, label="Top-1 accuracy")
    ax.bar([xi + width / 2 for xi in x], top3, width=width, label="Top-3 accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Root-cause analysis: 3 methods compared (n={p2[methods[0]]['n_cases']} detected fault cases)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rca_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_repair_and_false_alarms():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    p4 = summary["phase4_repair"]
    fa = summary["phase4_false_alarms"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    labels = ["Detected", "Correctly\nrepaired", "Not detected /\nnot repaired"]
    detected_n = round(p4["detection_rate"] * p4["n_cases"])
    repaired_n = round(p4["repair_success_rate"] * p4["n_cases"])
    values = [detected_n, repaired_n, p4["n_cases"] - repaired_n]
    axes[0].bar(labels, [detected_n, repaired_n, p4["n_cases"]], color=["#4C72B0", "#55A868", "#CCCCCC"])
    axes[0].set_title(f"Agent repair outcomes (n={p4['n_cases']} fault cases)")
    axes[0].set_ylabel("count")
    lo, hi = p4["repair_success_ci95"]
    axes[0].errorbar([1], [repaired_n], yerr=[[repaired_n - lo * p4["n_cases"]], [hi * p4["n_cases"] - repaired_n]],
                      fmt="none", ecolor="black", capsize=5)

    axes[1].bar(["Clean days\nflagged (false alarm)", "Auto-repairs\napplied on clean data"],
                [fa["n_flagged_by_isolation_forest"], fa["n_auto_repairs_on_clean_data"]],
                color=["#C44E52", "#55A868"])
    axes[1].set_title(f"False-alarm check (n={fa['n_clean_days_checked']} clean days)")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "repair_and_false_alarms.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_confusion_matrices()
    fig_model_comparison()
    fig_rca_comparison()
    fig_repair_and_false_alarms()
    print(f"wrote figures -> {FIG_DIR}")
