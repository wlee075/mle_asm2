from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({"figure.autolayout": True})

def _plot_auc(df: pd.DataFrame, ax) -> None:
    auc = df[df.metric_name == "auc"].sort_values("snapshot_date")
    if auc.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No AUC metrics", ha="center", va="center")
        return
    auc.plot(x="snapshot_date", y="value", marker="o", ax=ax, legend=False)
    ax.set_title("Model AUC over time")
    ax.set_ylabel("AUC")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


def _plot_psi_heatmap(df: pd.DataFrame, ax) -> None:
    psi = df[df.metric_name.str.startswith("psi_")]
    if psi.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No PSI metrics", ha="center", va="center")
        return
    pivot = (
        psi.pivot(index="metric_name", columns="snapshot_date", values="value")
        .sort_index(axis=1)
        .sort_index(axis=0)
    )
    im = ax.imshow(pivot, aspect="auto", cmap="viridis", vmin=0, vmax=0.3)
    ax.set_title("Feature-level PSI (green → stable, yellow/red → drift)")
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
    if metrics_fp.suffix == ".parquet":
        metrics = pd.read_parquet(metrics_fp)
    else:
        metrics = pd.read_csv(metrics_fp, parse_dates=["snapshot_date"])
    metrics["snapshot_date"] = metrics["snapshot_date"].astype(str)

    dq_fp = Path("dashboards/data_quality.csv")
    dq = (
        pd.read_csv(dq_fp, parse_dates=["snapshot_date"])
        if dq_fp.exists()
        else pd.DataFrame()
    )

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "report.pdf"
    html_path = outdir / "model_report.html"

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
        _plot_auc(metrics, axes[0])
        _plot_psi_heatmap(metrics, axes[1])
        _plot_nulls(dq, axes[2])
        pdf.savefig(fig)
        plt.close(fig)

    buf = io.BytesIO()
    fig = plt.figure(figsize=(0.1, 0.1))
    plt.savefig(buf, format="png", dpi=10)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Model Performance Report</title></head>
<body>
<h1 style="font-family:Arial,Helvetica,sans-serif;">Model Performance Report</h1>
<p>Generated from <code>{metrics_fp}</code></p>
<embed src="report.pdf" type="application/pdf" width="100%" height="900px"/>
<!-- fallback if PDF embed fails -->
<img src="data:image/png;base64,{b64}" alt="report preview" style="display:none;">
</body></html>"""
    html_path.write_text(html, encoding="utf-8")

    print(f"[generate_performance_report] PDF  → {pdf_path}")
    print(f"[generate_performance_report] HTML → {html_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Parquet or CSV metrics file")
    ap.add_argument("--outdir", required=True, help="Directory to write PDF/HTML")
    args = ap.parse_args()

    build_report(Path(args.metrics), Path(args.outdir))