import schedule
import time
import argparse
import os
import sys

# Import the main pipeline function from run_pipeline
from run_pipeline import run_the_pipeline

# Import configuration
try:
    import config
except ImportError:
    print("CRITICAL: config.py not found. Please create it from config.py.example.")
    sys.exit(1)


def job(args):
    """
    A wrapper function for the main pipeline logic that will be scheduled.
    """
    print(f"--- Running scheduled pipeline job at {time.ctime()} ---")
    try:
        # We pass the namespace object from argparse directly to the pipeline function
        run_the_pipeline(args)
    except Exception as e:
        print(f"An error occurred in the scheduled job: {e}")
    print(f"--- Scheduled job finished at {time.ctime()} ---")
    print(f"Next run is scheduled in {args.interval_hours} hour(s).")

if __name__ == "__main__":
    print("--- Starting Pipeline Scheduler ---")

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run the data pipeline on a schedule.")
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=1,
        help="Interval in hours to run the job."
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="If set, the pipeline will skip database operations."
    )

    args = parser.parse_args()

    # Check for API Key before starting
    if config.FINLIGHT_API_KEY == "YOUR_API_KEY_HERE":
        print("CRITICAL: FINLIGHT_API_KEY is not set in config.py. The scheduler will not run.")
        sys.exit(1)

    # Schedule the job to run at the specified interval
    print(f"Scheduling the pipeline to run every {args.interval_hours} hour(s).")
    schedule.every(args.interval_hours).hours.do(job, args=args)

    # Perform an initial run immediately upon starting the scheduler
    print("Performing initial pipeline run immediately...")
    job(args)

    # Main loop to run the scheduler
    print("Scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")
