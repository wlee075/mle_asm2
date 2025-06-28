from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mlflow.tracking import MlflowClient

plt.rcParams.update({"figure.autolayout": True})
_FIELDS = ["run_id", "snapshot_date", "metric_name", "value"]


def dump_metrics(exp_name: str, outdir: Path) -> None:
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        sys.exit(f"[export_run_metrics] experiment {exp_name!r} not found")

    rows: list[dict] = []
    for run in client.search_runs([exp.experiment_id], "", max_results=50_000):
        snap = run.data.tags.get("snapshot_date", "")
        for m, v in run.data.metrics.items():
            rows.append({
                "run_id": run.info.run_id,
                "snapshot_date": snap,
                "metric_name": m,
                "value": v,
            })

    if not rows:
        print(f"[export_run_metrics] experiment {exp_name!r} has no metrics")
        return

    df = pd.DataFrame(rows, columns=_FIELDS)
    outdir.mkdir(parents=True, exist_ok=True)
    parquet_path = outdir / "metrics.parquet"
    csv_path     = outdir / "metrics.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[export_run_metrics] wrote {len(df):,} rows → {parquet_path} + {csv_path}")


def _plot_line(df: pd.DataFrame, ax, metric: str, title: str, ylabel: str):
    series = df[df.metric_name == metric].sort_values("snapshot_date")
    if series.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No {metric} metrics", ha="center", va="center")
        return
    series.plot(x="snapshot_date", y="value", marker="o", ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


def _plot_psi_heatmap(df: pd.DataFrame, ax) -> None:
    psi = df[df.metric_name.str.startswith("psi_")]
    if psi.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No PSI metrics", ha="center", va="center")
        return
    pivot = (
        psi
        .pivot_table(
            index="metric_name",
            columns="snapshot_date",
            values="value",
            aggfunc="mean",
        )
        .sort_index(axis=1)
        .sort_index(axis=0)
    )
    im = ax.imshow(pivot, aspect="auto", cmap="viridis", vmin=0, vmax=0.3)
    ax.set_title("Feature-level PSI (green→stable, yellow/red→drift)")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(
        range(len(pivot.index)),
        [idx.replace("psi_", "") for idx in pivot.index],
        fontsize=8,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)


def _plot_nulls(dq: pd.DataFrame, ax) -> None:
    if dq.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data-quality file found", ha="center", va="center")
        return
    dq.sort_values("snapshot_date").plot(
        kind="bar", x="snapshot_date", y="null_pct", legend=False, ax=ax
    )
    ax.set_ylabel("% nulls")
    ax.set_title("Data-quality – null % per run")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def build_report(metrics_fp: Path, outdir: Path) -> None:
    if not metrics_fp.exists():
        sys.exit(f"[generate_performance_report] metrics file not found: {metrics_fp}")
    metrics = pd.read_parquet(metrics_fp) if metrics_fp.suffix == ".parquet" \
              else pd.read_csv(metrics_fp, parse_dates=["snapshot_date"])
    metrics["snapshot_date"] = metrics["snapshot_date"].astype(str)

    dq_fp = Path("dashboards/data_quality.csv")
    dq = pd.read_csv(dq_fp, parse_dates=["snapshot_date"]) if dq_fp.exists() else pd.DataFrame()

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "report.pdf"
    html_path = outdir / "model_report.html"

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(5, 1, figsize=(8.5, 14))
        _plot_line(metrics,   axes[0], "auc",        "Model AUC Over Time",      "AUC")
        _plot_line(metrics,   axes[1], "f1_score",  "Model F1 Score Over Time", "F1 Score")
        _plot_line(metrics,   axes[2], "accuracy",  "Model Accuracy Over Time", "Accuracy")
        _plot_psi_heatmap(metrics, axes[3])
        _plot_nulls(dq,       axes[4])
        pdf.savefig(fig)
        plt.close(fig)

    buf = io.BytesIO()
    fig = plt.figure(figsize=(0.1, 0.1))
    plt.savefig(buf, format="png", dpi=10)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Model Performance Report</title></head>
<body>
  <h1>Model Performance Report</h1>
  <p>Generated from <code>{metrics_fp}</code></p>
  <embed src="report.pdf" type="application/pdf" width="100%" height="900px"/>
  <img src="data:image/png;base64,{b64}" alt="preview" style="display:none;">
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    print(f"[generate_performance_report] PDF  → {pdf_path}")
    print(f"[generate_performance_report] HTML → {html_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export metrics & build performance report")
    ap.add_argument(
        "--experiment",
        default="inference-monitoring",
        help="MLflow experiment name to pull metrics from",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Directory to write metrics.parquet, metrics.csv, report.pdf & report.html",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    dump_metrics(args.experiment, outdir)
    build_report(outdir / "metrics.parquet", outdir)

