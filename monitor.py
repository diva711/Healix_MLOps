
import pandas as pd
# NOTE: evidently renamed its public API in v0.5+.
# The original top-level imports (evidently.report, evidently.metric_preset)
# are now under evidently.legacy — which works in all versions >= 0.5.
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

def run_monitoring():
    ref = pd.read_csv("data.csv")[['wait_time_mins']]
    # Read production logs saved by serve.py
    curr = pd.read_csv("production_logs.csv", names=['wait_time_mins', 'satisfied'])

    # Create the Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr[['wait_time_mins']])
    report.save_html("drift_report.html")
    print("Report generated: drift_report.html")

if __name__ == "__main__":
    run_monitoring()