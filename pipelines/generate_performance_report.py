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

def _plot_metric(df: pd.DataFrame, metric: str, ax) -> None:
    sub = df[df.metric_name == metric].sort_values("snapshot_date")
    if sub.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No {metric} metrics", ha="center", va="center")
        return
    sub.plot(x="snapshot_date", y="value", marker="o", ax=ax, legend=False)
    ax.set_title(f"{metric.replace('_',' ').title()} Over Time")
    ax.set_ylabel(metric.replace('_',' ').title())
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    x_last = sub["snapshot_date"].iloc[-1]
    y_last = sub["value"].iloc[-1]
    ax.annotate(f"{y_last:.2f}", xy=(x_last, y_last), xytext=(5, 0), textcoords="offset points")


def build_report(metrics_fp: Path, outdir: Path) -> None:
    if metrics_fp.suffix == ".parquet":
        metrics = pd.read_parquet(metrics_fp)
    else:
        metrics = pd.read_csv(metrics_fp, parse_dates=["snapshot_date"])
    metrics["snapshot_date"] = pd.to_datetime(metrics["snapshot_date"])

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "model_performance_report.pdf"
    html_path = outdir / "model_performance_report.html"

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _plot_metric(metrics, "f1_score", axes[0])
        _plot_metric(metrics, "accuracy", axes[1])
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
<h1>Model Performance Report</h1>
<p>Generated from <code>{metrics_fp}</code></p>
<embed src="{pdf_path.name}" type="application/pdf" width="100%" height="800px"/>
<!-- fallback image preview -->
<img src="data:image/png;base64,{b64}" alt="report preview" style="display:none;"/>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")

    print(f"[generate_performance_report] PDF → {pdf_path}")
    print(f"[generate_performance_report] HTML → {html_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Parquet or CSV metrics file")
    ap.add_argument("--outdir", required=True, help="Directory to write report files")
    args = ap.parse_args()
    build_report(Path(args.metrics), Path(args.outdir))