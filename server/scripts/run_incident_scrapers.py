#!/usr/bin/env python3

import os
import sys
import logging
import traceback
from datetime import datetime
import importlib.util
import subprocess
import pandas as pd

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'incident_scrapers.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

def update_progress(progress, status_dict):
    """Update the progress in the shared status dictionary."""
    if status_dict:
        status_dict['progress'] = min(max(0, progress), 100)  # Ensure progress is between 0 and 100
        logging.info(f"Progress: {status_dict['progress']}%")

def run_stability_scraper(status_dict=None):
    """Run the StabilityAI scraper."""
    logging.info("Starting StabilityAI scraper...")
    update_progress(10, status_dict)
    
    try:
        stability_path = os.path.join(
            os.path.dirname(__file__),
            "data_gen_modules",
            "StabilityAI.py"
        )
        
        result = subprocess.run(
            [sys.executable, stability_path],
            capture_output=True,
            text=True
        )
        
        update_progress(40, status_dict)
        
        if result.returncode == 0:
            logging.info("StabilityAI scraper completed successfully")
            if result.stdout:
                logging.info("Output: " + result.stdout)
            return True
        else:
            logging.error("StabilityAI scraper failed")
            if result.stderr:
                logging.error("Error: " + result.stderr)
            return False
            
    except Exception as e:
        logging.error(f"Error running StabilityAI scraper: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def run_oac_scraper(status_dict=None):
    """Run the OpenAI/Anthropic/Character.AI scraper."""
    logging.info("Starting OAC scraper...")
    update_progress(50, status_dict)
    
    try:
        oac_path = os.path.join(
            os.path.dirname(__file__),
            "data_gen_modules",
            "incident_scraper_oac.py"
        )
        
        result = subprocess.run(
            [sys.executable, oac_path],
            capture_output=True,
            text=True
        )
        
        update_progress(90, status_dict)
        
        if result.returncode == 0:
            logging.info("OAC scraper completed successfully")
            if result.stdout:
                logging.info("Output: " + result.stdout)
            return True
        else:
            logging.error("OAC scraper failed")
            if result.stderr:
                logging.error("Error: " + result.stderr)
            return False
            
    except Exception as e:
        logging.error(f"Error running OAC scraper: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def remove_duplicate_errors(df):
    """
    Remove duplicate error incidents based on service and timestamp matching,
    keeping only one instance of each unique event.
    """
    # Convert timestamp columns to datetime if they're strings
    timestamp_cols = ['start_timestamp', 'close_timestamp', 'investigating_timestamp', 
                     'identified_timestamp', 'monitoring_timestamp', 'resolved_timestamp', 
                     'postmortem_timestamp']
    
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    
    # Group by service columns and timestamps to find duplicates
    service_cols = ['Playground', 'API', 'DALL-E', 'ChatGPT', 'api.anthropic.com', 
                   'claude.ai', 'console.anthropic.com', 'Character.AI', 'REST API', 
                   'gRPC API', 'Stable Assistant']
    
    # Create a more comprehensive composite key that includes all relevant fields
    df['duplicate_group'] = df.apply(
        lambda row: '_'.join([
            # Service flags (as a single string of 1s and 0s)
            ''.join(str(int(row[col])) for col in service_cols if col in df.columns),
            # Provider
            str(row['provider']),
            # Impact level
            str(row['incident_impact_level']),
            # First valid timestamp
            next((str(row[col]) for col in timestamp_cols 
                 if col in df.columns and pd.notna(row[col])), 'no_time'),
            # Title similarity (optional, remove if too strict)
            str(row['Incident_Title'])[:50]  # First 50 chars of title
        ]),
        axis=1
    )
    
    # Find all duplicates
    duplicates = df[df.duplicate_group.duplicated(keep=False)].sort_values('duplicate_group')
    
    if len(duplicates) > 0:
        # Count before deduplication
        total_before = len(df)
        unique_groups = duplicates.duplicate_group.nunique()
        duplicate_count = len(duplicates)
        
        logging.info(f"\nFound {duplicate_count} incidents in {unique_groups} duplicate groups:")
        
        # Log details of duplicate groups
        for group in duplicates.duplicate_group.unique():
            group_dupes = duplicates[duplicates.duplicate_group == group]
            logging.info(f"\nDuplicate group with {len(group_dupes)} incidents:")
            for _, row in group_dupes.iterrows():
                logging.info(f"- {row['incident_id']}: {row['Incident_Title']} ({row['provider']})")
                logging.info(f"  Start time: {row['start_timestamp']}")
                logging.info(f"  Impact level: {row['incident_impact_level']}")
                logging.info(f"  Services: {[col for col in service_cols if col in df.columns and row[col] == 1]}")
        
        # Remove all duplicates, keeping only the first occurrence
        df = df.drop_duplicates(subset=['duplicate_group'], keep='first')
        
        # Count after deduplication
        total_after = len(df)
        removed_count = total_before - total_after
        
        logging.info(f"\nDeduplication summary:")
        logging.info(f"- Total incidents before: {total_before}")
        logging.info(f"- Duplicate groups found: {unique_groups}")
        logging.info(f"- Total duplicates found: {duplicate_count}")
        logging.info(f"- Incidents removed: {removed_count}")
        logging.info(f"- Total incidents after: {total_after}")
    
    # Remove the temporary grouping column
    df = df.drop('duplicate_group', axis=1)
    
    # Verify no duplicates remain
    for group in service_cols:
        if group in df.columns:
            potential_dupes = df[df[group] == 1].groupby(['start_timestamp', 'incident_impact_level']).size()
            if (potential_dupes > 1).any():
                logging.warning(f"Warning: Possible remaining duplicates found for {group}")
    
    return df

def main(status_dict=None):
    start_time = datetime.now()
    logging.info(f"Starting incident scraper run at {start_time}")
    output_path = 'server/static/data/incident_stages_all.csv'
    
    try:
        if status_dict:
            status_dict['progress'] = 0
        
        update_progress(5, status_dict)
        
        # Run StabilityAI scraper first
        stability_success = run_stability_scraper(status_dict)
        
        if stability_success:
            logging.info("StabilityAI scraper completed successfully, proceeding with OAC scraper")
            
            # Run OAC scraper next
            oac_success = run_oac_scraper(status_dict)
            
            if oac_success:
                logging.info("All scrapers completed successfully")
                
                # Check for and remove duplicates
                if os.path.exists(output_path):
                    logging.info("Checking for duplicate incidents...")
                    df = pd.read_csv(output_path)
                    original_count = len(df)
                    df = remove_duplicate_errors(df)
                    
                    # Save the deduplicated data
                    df.to_csv(output_path, index=False)
                    logging.info(f"Removed {original_count - len(df)} duplicate incidents")
                    logging.info(f"Final incident count: {len(df)}")
                
                update_progress(100, status_dict)
            else:
                logging.error("OAC scraper failed")
                raise Exception("OAC scraper failed")
        else:
            logging.error("StabilityAI scraper failed, skipping OAC scraper")
            raise Exception("StabilityAI scraper failed")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error(traceback.format_exc())
        if status_dict:
            status_dict['error'] = str(e)
        raise
    finally:
        if status_dict:
            status_dict['is_running'] = False
    
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Scraper run completed at {end_time} (Duration: {duration})")

if __name__ == "__main__":
    main() 