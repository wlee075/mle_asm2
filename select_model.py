import argparse
import os
import shutil
import logging
from mlflow.tracking import MlflowClient
from dateutil.relativedelta import relativedelta
from datetime import datetime
logging.basicConfig(level=logging.INFO)

def main(cand_dir: str, prod_dir: str, start: str, end: str, past_n: int = 3):
    client = MlflowClient()
    exp = client.get_experiment_by_name("model-training-full")
    if exp is None:
        logging.error("Experiment 'model-training-full' not found.")
        return
    all_runs = client.search_runs([exp.experiment_id], filter_string="", max_results=50000)
    logging.info(f"Fetched {len(all_runs)} runs.")
    best_per_snap = {}
    for run in all_runs:
        snap = run.data.tags.get("snapshot_date")
        auc  = run.data.metrics.get("auc")
        if not snap or auc is None or not (start <= snap <= end):
            continue
        if snap not in best_per_snap or auc > best_per_snap[snap][1]:
            best_per_snap[snap] = (run, auc)
    logging.info(f"Snapshots to promote: {list(best_per_snap.keys())}")

    os.makedirs(prod_dir, exist_ok=True)
    for snap, (run, auc) in best_per_snap.items():
        mtype = run.data.tags.get("model_type") or run.data.params.get("model_type")
        src = os.path.join(cand_dir, f"{mtype}_{snap}.pkl")
        if os.path.exists(src):
            dst = os.path.join(prod_dir, f"{mtype}_{snap}.pkl")
            shutil.copy(src, dst)
            logging.info(f"Promoted {src} to {dst} (AUC={auc})")
        else:
            snap_dt = datetime.strptime(snap, "%Y-%m-%d")
            cutoff  = snap_dt - relativedelta(years=past_n)
            candidates = []
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            for date_str in best_per_snap.keys():
                if cutoff_str <= date_str < snap:
                    candidates.append(date_str)
                if not candidates:
                    logging.error(f"No fallback snapshot within past {past_n} years for {snap}")
                    continue
                fallback = max(candidates)
                fb_src = os.path.join(cand_dir, f"{mtype}_{fallback}.pkl")
                fb_dst = os.path.join(prod_dir,    f"{mtype}_{fallback}.pkl")
                if os.path.exists(fb_src):
                    shutil.copy(fb_src, fb_dst)
                    logging.info(f"Fallback: Promoted {fb_src} as substitute for {snap} (AUC={best_per_snap[fallback][1]})")
                else:
                    logging.error(f"Fallback artifact missing: {fb_src}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Select best model per snapshot")
    parser.add_argument("--candidatedir",  required=True)
    parser.add_argument("--productiondir", required=True)
    parser.add_argument("--startdate",     required=True)
    parser.add_argument("--enddate",       required=True)
    parser.add_argument("--pastyears",     type=int, default=3)
    args = parser.parse_args()
    main(
        args.candidatedir,
        args.productiondir,
        args.startdate,
        args.enddate,
        args.pastyears
    )