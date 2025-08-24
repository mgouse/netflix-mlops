"""
Monitor retraining pipeline and show results
"""
import json
import os
from datetime import datetime
import time

def check_last_run():
    """Check the last training run results"""
    try:
        with open("/tmp/last_training_metrics.json", "r") as f:
            data = json.load(f)
            
        print(f"\n=== Last Training Run ===")
        print(f"Timestamp: {data['timestamp']}")
        print(f"Run Name: {data['run_name']}")
        print(f"Status: {data['status']}")
        print(f"Metrics:")
        for k, v in data['metrics'].items():
            print(f"  {k}: {v}")
    except FileNotFoundError:
        print("No training runs found yet.")

def monitor_jobs():
    """Monitor Kubernetes jobs"""
    os.system("kubectl get jobs -n netflix-ml --sort-by=.metadata.creationTimestamp")

if __name__ == "__main__":
    print("Monitoring Netflix Retraining Pipeline")
    print("Press Ctrl+C to stop")
    
    while True:
        os.system("clear")  # or "cls" for Windows
        check_last_run()
        print("\n=== Current Jobs ===")
        monitor_jobs()
        time.sleep(30)