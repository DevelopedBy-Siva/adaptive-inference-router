import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

FULL_TABLE  = "results/full_comparison.csv"
SWEEP_TABLE = "results/sweep_results.csv"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "yolov8n_pretrained": "#9ecae1",
    "yolov8l_pretrained": "#3182bd",
    "yolov8n_finetuned":  "#a1d99b",
    "yolov8l_finetuned":  "#31a354",
    "adaptive":           "#e6550d",
}
LABELS = {
    "yolov8n_pretrained": "YOLOv8n pretrained",
    "yolov8l_pretrained": "YOLOv8l pretrained",
    "yolov8n_finetuned":  "YOLOv8n fine-tuned",
    "yolov8l_finetuned":  "YOLOv8l fine-tuned",
    "adaptive":           "Adaptive Pipeline",
}


def load_data():
    df = pd.read_csv(FULL_TABLE)
    sweep = pd.read_csv(SWEEP_TABLE)
    return df, sweep


def plot_fps_vs_md100(df):
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in df.iterrows():
        key = row["model"].split("_T")[0] if "adaptive" in row["model"] else row["model"]
        color = COLORS.get(key, "#888888")
        label = LABELS.get(key, row["model"])
        ax.scatter(row["fps"], row["md_100"], color=color, s=160, zorder=5)
        ax.annotate(label, (row["fps"], row["md_100"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("FPS (higher = faster)", fontsize=12)
    ax.set_ylabel("MD@100 (lower = safer)", fontsize=12)
    ax.set_title("Speed vs Safety: FPS vs Missed Detections per 100 Frames\n(Caltech set00, 2500 frames, IoU ≥ 0.5)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.5, label="100 FPS threshold")
    ax.legend(fontsize=9)

    takeaway = "Takeaway: Adaptive pipeline delivers near-light-model FPS with better precision than YOLOv8n alone."
    fig.text(0.5, 0.01, takeaway, ha="center", fontsize=9, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = f"{FIGURES_DIR}/plot1_fps_vs_md100.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_recall_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    models = []
    recalls = []
    colors = []

    for _, row in df.iterrows():
        key = row["model"].split("_T")[0] if "adaptive" in row["model"] else row["model"]
        models.append(LABELS.get(key, row["model"]))
        recalls.append(row["recall"])
        colors.append(COLORS.get(key, "#888888"))

    x = np.arange(len(models))
    bars = ax.bar(x, recalls, color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Recall (IoU ≥ 0.5)", fontsize=12)
    ax.set_title("Recall Comparison: Pretrained vs Fine-tuned vs Adaptive Pipeline\n(Caltech set00, 2500 frames)", fontsize=12)
    ax.set_ylim(0, max(recalls) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    takeaway = "Takeaway: Fine-tuning consistently improves recall; adaptive pipeline matches fine-tuned YOLOv8n with higher precision."
    fig.text(0.5, 0.01, takeaway, ha="center", fontsize=9, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = f"{FIGURES_DIR}/plot2_recall_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_threshold_sweep(sweep):
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_md  = "#e6550d"
    color_fps = "#3182bd"
    color_hvy = "#31a354"

    x = sweep["threshold"].values

    ax1.plot(x, sweep["md_100"], "o-", color=color_md, linewidth=2.5,
             markersize=8, label="MD@100 (left axis)", zorder=5)
    ax1.set_xlabel("Confidence Threshold T", fontsize=12)
    ax1.set_ylabel("MD@100 (lower = safer)", fontsize=12, color=color_md)
    ax1.tick_params(axis="y", labelcolor=color_md)

    ax2 = ax1.twinx()
    ax2.plot(x, sweep["pct_heavy"], "s--", color=color_hvy, linewidth=2,
             markersize=8, label="Heavy triggers % (right)", zorder=4)
    ax2.plot(x, sweep["fps"], "^:", color=color_fps, linewidth=2,
             markersize=8, label="FPS (right)", zorder=4)
    ax2.set_ylabel("Heavy Triggers % / FPS", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax1.set_title("Threshold Sweep: Cost of Safety\nT=0.25–0.55 on Caltech set00", fontsize=12)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    takeaway = "Takeaway: T=0.25 gives best FPS (129.0) and recall (0.2464) with fewest heavy triggers (24.6%)."
    fig.text(0.5, 0.01, takeaway, ha="center", fontsize=9, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = f"{FIGURES_DIR}/plot3_threshold_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    df, sweep = load_data()
    print("Generating plots...")
    plot_fps_vs_md100(df)
    plot_recall_comparison(df)
    plot_threshold_sweep(sweep)
    print(f"\nAll 3 plots saved to {FIGURES_DIR}/")
    print("  plot1_fps_vs_md100.png")
    print("  plot2_recall_comparison.png")
    print("  plot3_threshold_sweep.png")