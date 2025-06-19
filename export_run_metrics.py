from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

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
            rows.append(
                {"run_id": run.info.run_id,
                 "snapshot_date": snap,
                 "metric_name": m,
                 "value": v}
            )

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="model-training-full",
                    help="MLflow experiment name")
    ap.add_argument("--outdir", required=True,
                    help="Folder to write metrics.parquet / metrics.csv")
    args = ap.parse_args()

    dump_metrics(args.experiment, Path(args.outdir))