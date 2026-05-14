import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_monitoring():
    ref = pd.read_csv("data.csv")[['wait_time_mins']]
    # Simulate current data from the logs we saved
    curr = pd.read_csv("production_logs.csv", names=['wait_time_mins', 'satisfied'])

    # Create the Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr[['wait_time_mins']])
    report.save_html("drift_report.html")
    print("Report generated: drift_report.html")

if __name__ == "__main__":
    run_monitoring()