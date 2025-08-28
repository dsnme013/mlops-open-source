# src/monitor_evidently.py
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

os.makedirs("reports", exist_ok=True)

def main():
    ref = pd.read_csv("data/iris.csv")
    curr_path = "data/current.csv"
    if not os.path.exists(curr_path):
        ref.sample(frac=1, random_state=42).to_csv(curr_path, index=False)
        print("No current.csv found — made a simulated current.csv")

    curr = pd.read_csv(curr_path)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)
    report.save_html("reports/drift_report.html")
    print("Drift report saved!")

if __name__ == "__main__":
    main()
