import schedule
import time
import argparse
import os

# Import the main pipeline function and argument parser from the original script
from run_pipeline import run_the_pipeline, get_pipeline_args

def job(args):
    """
    A wrapper function for the main pipeline logic that will be scheduled.
    """
    print(f"--- Running scheduled pipeline job at {time.ctime()} ---")
    try:
        run_the_pipeline(args)
    except Exception as e:
        print(f"An error occurred in the scheduled job: {e}")
    print(f"--- Scheduled job finished at {time.ctime()} ---")
    print(f"Next run is scheduled in {args.interval_hours} hour(s).")

if __name__ == "__main__":
    print("--- Starting Pipeline Scheduler ---")

    # Use the same argument parser from the main pipeline script to ensure consistency
    parser = get_pipeline_args()

    # Add a new argument specific to the scheduler
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=1,
        help="Interval in hours to run the job."
    )
    args = parser.parse_args()

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
