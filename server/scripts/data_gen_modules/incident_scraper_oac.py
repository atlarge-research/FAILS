import time
import os
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import shutil

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.support.color import Color


from dateutil import parser
from dateutil.tz import gettz
from datetime import timedelta
import pandas as pd
from util_data import read_data, load_json_column, unnest_dict, get_incident_id, write_partitioned_data
from pathlib import Path

from incident_stages_stability import process_folder as process_stability_folder, get_paths as get_stability_paths


def calculate_start_date(end_date):
    start_date = end_date - relativedelta(months=2)
    start_date_str = start_date.strftime("%Y%m")
    end_date_str = end_date.strftime("%Y%m")
    return start_date_str, end_date_str


def get_archive_path(partition):
    start_date, end_date = calculate_start_date(partition)
    archive_folder = f"data/raw/incident/openai"
    os.makedirs(archive_folder, exist_ok=True)
    return f"{archive_folder}/incident_history_{start_date}_{end_date}.csv"


class MyIncidentPage:
    # Class variable for XPaths that do not change across instances
    UPDATE_XPATH = "//div[@class='row update-row']"
    SERVICE_XPATH = "//div[contains(@class, 'components-affected')]"
    INCIDENT_LIST_XPATH = "//a[contains(@class, 'incident-title')]"
    PAGE_XPATH = "(//h4[contains(@class, 'month-title')])[1]"
    PAGINATION_XPATH = "//div[@class='pagination']//i[@class='left-arrow']"
    SHOW_ALL_XPATH = "//div[contains(@class, 'expand-incidents') and @aria-expanded='false']"

    def __init__(self, driver, provider_url):
        self.driver = driver
        self.c_key = MAC_C_KEY
        self.provider_url = provider_url

    def get_incident_updates(self):
        updates = []
        update_rows = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, self.UPDATE_XPATH))
        )
        # get update by rows
        for update_row in update_rows:
            title = update_row.find_element(By.XPATH, ".//div[contains(@class, 'update-title')]").text
            body = update_row.find_element(By.XPATH, ".//div[contains(@class, 'update-body')]").text
            timestamp = update_row.find_element(By.XPATH, ".//div[contains(@class, 'update-timestamp')]").text

            updates.append({
                    "Update_Title": title,
                    "Update_Body": body,
                    "Update_Timestamp": timestamp
            })

        return json.dumps(updates)

    def get_incident_service(self):
        try:
            service = self.driver.find_element(By.XPATH, self.SERVICE_XPATH).text
        except NoSuchElementException:
            service = None
            print("Service element not found, setting service to None.")
        return service

    def switch_to_incident(self, incident, original_window):
        print("Switch to new window: ")
        title = incident.text
        link = incident.get_attribute('href')
        impact = incident.get_attribute('class').split(' ')[0]
        incident_color = Color.from_string(incident.value_of_css_property('color')).hex
        # switch to new tab to collect incident updates
        incident.send_keys(self.c_key + Keys.RETURN)
        WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
        new_window = [window for window in self.driver.window_handles if window != original_window][0]
        self.driver.switch_to.window(new_window)
        # collect incident updates
        # updates = self.get_incident_updates()
        # service = self.get_incident_service()
        record = pd.DataFrame({
            "Incident_Title": [title],
            "Incident_Link": [link],
            "Incident_color": [incident_color],
            "Incident_Impact": [impact],
            "Updates": [self.get_incident_updates()],
            "Service": [self.get_incident_service()]
        })
        # print(record[0]['Incident_Title'], ". ", json.loads(record[0]['Updates'][0])['Update_Timestamp'])
        print(record['Incident_Title'][0], ". ", json.loads(record['Updates'][0])[0]['Update_Timestamp'])
        # switch back
        self.driver.close()
        self.driver.switch_to.window(original_window)
        return record

    def get_incident_list(self):
        try:
            incident_list = WebDriverWait(self.driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, self.INCIDENT_LIST_XPATH))
            )
            print("Incidents found in this page: ", len(incident_list))
        except Exception as e:
            print("No incidents found.")
            incident_list = []
        return incident_list

    def archive_incidents(self, incident_df):
        date_str = self.driver.find_element(By.XPATH, self.PAGE_XPATH).text
        date_partition = datetime.strptime(date_str, "%B %Y")
        incident_df.to_csv(get_archive_path(date_partition), index=False)

    def loop_over_incidents(self):
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            incident_df = pd.DataFrame()
            flag_no_data = False
            try:
                # Collecting incident records in new tabs
                incident_list = self.get_incident_list()
                original_window = self.driver.current_window_handle
                if not incident_list:
                    flag_no_data = True
                else:
                    # loop over incidents when there are any
                    for incident_title in incident_list:
                        incident_record = self.switch_to_incident(incident_title, original_window)
                        incident_df = pd.concat([incident_df, pd.DataFrame(incident_record)])
                return incident_df, flag_no_data
            except StaleElementReferenceException:
                print("Stale element, restarting incidents looping process.")
                attempt += 1
                continue
            except Exception as e:
                print("Executing loop_over_incidents(). An error occurred: ", e)
                traceback.print_exc()

        print("\nWarning: this should not happen!!!\n")
        return [], True

    def show_all_incidents(self):
        show_all_buttons = self.driver.find_elements(By.XPATH, self.SHOW_ALL_XPATH)
        if show_all_buttons:
            for show_all in show_all_buttons:
                show_all.click()
            time.sleep(1)

    def go_to_previous_page(self):
        prev_page = self.driver.find_element(By.XPATH, self.PAGINATION_XPATH)
        if prev_page:
            prev_page.click()
            time.sleep(1)

    def collect_data_through_pagination(self, start_date=None, end_date=None):
        """
        Collect incident reports by incident history pages
        Args:
            start_date: timezone-aware datetime object - start date to collect incidents from
            end_date: timezone-aware datetime object - end date to collect incidents until
        """
        all_incidents_df = pd.DataFrame()
        try:
            self.driver.get(self.provider_url)
            stop_collection = False
            
            while not stop_collection:
                self.show_all_incidents()
                incident_df, flag_no_data = self.loop_over_incidents()
                
                if len(incident_df) > 0:
                    # Parse timestamps before filtering
                    incident_df['timestamp'] = incident_df['Updates'].apply(
                        lambda x: DataTransformer.parse_update_time(json.loads(x)[0]['Update_Timestamp'])
                    )
                    
                    # Sort by timestamp to ensure correct chronological order
                    incident_df = incident_df.sort_values('timestamp', ascending=False)
                    
                    if start_date:
                        # Check if all incidents on this page are older than start_date
                        if incident_df['timestamp'].max() < start_date:
                            print(f"All incidents on page are before {start_date}, stopping collection.")
                            break
                        
                        # Keep only incidents after start_date
                        incident_df = incident_df[incident_df['timestamp'] >= start_date]
                    
                    if end_date:
                        incident_df = incident_df[incident_df['timestamp'] <= end_date]
                    
                    if len(incident_df) > 0:
                        all_incidents_df = pd.concat([all_incidents_df, incident_df])
                
                if flag_no_data:
                    print("No more incidents found.")
                    break
                    
                self.go_to_previous_page()
            
            # Drop the temporary timestamp column
            if len(all_incidents_df) > 0 and 'timestamp' in all_incidents_df.columns:
                all_incidents_df = all_incidents_df.drop('timestamp', axis=1)
                
            return all_incidents_df
        except Exception as e:
            print("Error in collect_data_through_pagination(): ", e)
            traceback.print_exc()
            return all_incidents_df





class DataTransformer:
    @staticmethod
    def get_services(service_str):
        if pd.isna(service_str):
            return []
        else:        
            service_str = service_str.split(':')[-1].rstrip('.')
            service_str = service_str.replace("and", ",")
            services = service_str.split(',')
            return [service.strip() for service in services]
    
    @staticmethod    
    def parse_update_time(timestamp_str):
        tzinfos = {
            "PST": gettz("America/Los_Angeles"),
            "PDT": gettz("America/Los_Angeles")
        }
        timestamp_str = " ".join(timestamp_str.split(". ")[1].split(" - "))
        timestamp = parser.parse(timestamp_str, ignoretz=False, tzinfos=tzinfos)
        return timestamp.astimezone(gettz('UTC'))

    @staticmethod
    def parse_updates(updates_list, stages=None):
        if stages is None:
            stages = ["Investigating", "Identified", "Monitoring", "Resolved", "Postmortem"]
        updates_dict = {}
        
        for stage in stages:
            stage = stage.lower()
            updates_dict[f'{stage}_flag'] = int(0)
            updates_dict[f'{stage}_timestamp'] = None
            updates_dict[f'{stage}_description'] = None
        
        for update in updates_list:
            stage = update.get("Update_Title")
            if stage in stages:
                stage = stage.lower()
                updates_dict[f'{stage}_flag'] = int(1)
                updates_dict[f'{stage}_timestamp'] = DataTransformer.parse_update_time(update.get("Update_Timestamp"))
                updates_dict[f'{stage}_description'] = update.get("Update_Body")
        return updates_dict

    @staticmethod
    def get_incident_provider(incident_url):
        if pd.isna(incident_url):
            return incident_url
        else:
            return incident_url.split('/')[2].split('.')[1]

    @staticmethod
    def transform_incidents(df):
        try:
            # Drop duplicates on raw data first
            df = df.drop_duplicates(subset=['Incident_Link'])
            df = df.reset_index(drop=True)
            
            # Parse JSON in Updates column
            df = load_json_column(df, 'Updates')

            # Add provider column
            df['provider'] = df['Incident_Link'].apply(DataTransformer.get_incident_provider)
            
            # Get incident id
            df['incident_id'] = df['Incident_Link'].apply(get_incident_id)

            # Numerize incident impact level
            impact_mapping = {
                'impact-none': 0,
                'impact-minor': 1,
                'impact-major': 2,
                'impact-critical': 3,
                'impact-maintenance': 4
            }
            df['incident_impact_level'] = df['Incident_Impact'].map(impact_mapping)

            # Parse services
            df['services'] = df['Service'].apply(DataTransformer.get_services)
            all_services = ['Playground', 'API', 'DALL-E', 'ChatGPT', 'api.anthropic.com', 
                          'claude.ai', 'console.anthropic.com', 'Character.AI', 
                          'REST API', 'gRPC API', 'Stable Assistant']
            
            for service in all_services:
                df[service] = df['services'].apply(lambda x: 1 if service in x else 0)

            # Parse stages
            df_stages = df.copy()
            df_stages['parsed_updates'] = df_stages['Updates'].apply(DataTransformer.parse_updates)
            df_stages = unnest_dict(df_stages, 'parsed_updates')

            # Calculate time span
            cols_timestamp = [col for col in df_stages.columns if 'timestamp' in col]
            cols_timestamp.pop()  # remove the postmortem timestamp
            
            # Convert any string timestamps to datetime and handle NaT values
            for col in cols_timestamp:
                df_stages[col] = pd.to_datetime(df_stages[col], utc=True)
            
            # Calculate start and close timestamps, handling NaT values
            df_stages['start_timestamp'] = df_stages[cols_timestamp].apply(
                lambda x: min([ts for ts in x if pd.notna(ts)]) if any(pd.notna(x)) else pd.NaT,
                axis=1
            )
            
            df_stages['close_timestamp'] = df_stages[cols_timestamp].apply(
                lambda x: max([ts for ts in x if pd.notna(ts)]) if any(pd.notna(x)) else pd.NaT,
                axis=1
            )
            
            # Calculate time span only for rows with valid timestamps
            df_stages['time_span'] = pd.Timedelta('0 days')  # default value
            mask = pd.notna(df_stages['start_timestamp']) & pd.notna(df_stages['close_timestamp'])
            df_stages.loc[mask, 'time_span'] = df_stages.loc[mask, 'close_timestamp'] - df_stages.loc[mask, 'start_timestamp']
            df_stages['over_one_day'] = df_stages['time_span'] > timedelta(days=1)

            # Reorder columns
            cols_incident_info = ['incident_id', 'Incident_Title', 'incident_impact_level', 
                                'Incident_color', 'provider']
            cols_services = all_services
            cols_stages = [col for col in df_stages.columns 
                         if 'flag' in col or 'timestamp' in col or 'description' in col]
            cols_timespan = ['time_span', 'over_one_day']

            return df_stages[cols_incident_info + cols_services + cols_stages + cols_timespan]
            
        except Exception as e:
            print(f"Error transforming incidents: {str(e)}")
            traceback.print_exc()
            return None


def load_existing_incidents(output_path):
    """Load existing incidents from CSV if it exists."""
    if Path(output_path).exists():
        df = pd.read_csv(output_path)
        # Convert timestamp columns back to datetime
        timestamp_cols = [col for col in df.columns if 'timestamp' in col]
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col], utc=True)
        return df
    return pd.DataFrame()

def get_latest_incident_date(df):
    """Get the most recent incident date from existing data."""
    if len(df) > 0 and 'start_timestamp' in df.columns:
        return pd.to_datetime(df['start_timestamp'].max(), utc=True)
    return None

def merge_and_deduplicate(existing_df, new_df):
    """Merge existing and new incidents, removing duplicates."""
    if len(existing_df) == 0:
        return new_df
    if len(new_df) == 0:
        return existing_df
    
    # Define the expected column order
    expected_columns = [
        'incident_id', 'Incident_Title', 'incident_impact_level', 'Incident_color', 'provider',
        'Playground', 'API', 'DALL-E', 'ChatGPT', 'api.anthropic.com', 'claude.ai', 
        'console.anthropic.com', 'Character.AI', 'REST API', 'gRPC API', 'Stable Assistant',
        'investigating_flag', 'investigating_timestamp', 'investigating_description',
        'identified_flag', 'identified_timestamp', 'identified_description',
        'monitoring_flag', 'monitoring_timestamp', 'monitoring_description',
        'resolved_flag', 'resolved_timestamp', 'resolved_description',
        'postmortem_flag', 'postmortem_timestamp', 'postmortem_description',
        'start_timestamp', 'close_timestamp', 'time_span', 'over_one_day'
    ]
    
    # Create a dtype mapping from existing_df
    dtype_map = {col: existing_df[col].dtype for col in existing_df.columns}
    
    # Fill missing columns with appropriate empty values and dtypes
    for col in expected_columns:
        if col not in existing_df.columns:
            # Get dtype from new_df if available, otherwise use object
            dtype = new_df[col].dtype if col in new_df.columns else 'object'
            if col == 'StabilityAI':  # Initialize StabilityAI column with zeros
                existing_df[col] = 0
            else:
                existing_df[col] = pd.Series(dtype=dtype)
        if col not in new_df.columns:
            # Use dtype from existing_df
            dtype = dtype_map.get(col, 'object')
            if col == 'StabilityAI':  # Initialize StabilityAI column with zeros
                new_df[col] = 0
            else:
                new_df[col] = pd.Series(dtype=dtype)
    
    # Ensure columns are in the expected order
    existing_df = existing_df[expected_columns]
    new_df = new_df[expected_columns]
    
    # Concatenate with ignore_index to avoid index conflicts
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Sort by start_timestamp
    if 'start_timestamp' in combined_df.columns:
        combined_df = combined_df.sort_values('start_timestamp', ascending=False)
    
    # Remove duplicates based on incident_id, keeping the first (most recent) occurrence
    combined_df = combined_df.drop_duplicates(subset=['incident_id'], keep='first')
    
    # Set StabilityAI column to 1 for StabilityAI incidents
    combined_df.loc[combined_df['provider'] == 'StabilityAI', 'StabilityAI'] = 1
    
    # Instead of setting generic StabilityAI column
    stability_services = {
        'REST API': 'REST',
        'gRPC API': 'gRPC',
        'Stable Assistant': 'Assistant'
    }
    
    # Set specific service columns for StabilityAI incidents
    for service_col, service_name in stability_services.items():
        if service_col not in combined_df.columns:
            combined_df[service_col] = 0
        # Set service column to 1 for matching incidents
        combined_df.loc[
            (combined_df['provider'] == 'StabilityAI') & 
            (combined_df['Service'].str.contains(service_name, na=False)), 
            service_col
        ] = 1
    
    # Ensure final dataframe has the expected column order
    return combined_df[expected_columns]

def get_state_file_path():
    """Get path to state file."""
    state_dir = "server/static/data/state"
    os.makedirs(state_dir, exist_ok=True)
    return f"{state_dir}/incident_scraper_state.json"

def get_backup_path(original_path):
    """Generate backup path with timestamp."""
    backup_dir = "server/static/data/backups/incidents"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(original_path).name
    return f"{backup_dir}/{filename.replace('.csv', '')}_{timestamp}.csv"

def save_state(last_run_time, last_incident_date, success=True, error=None):
    """Save scraper state to file."""
    state = {
        'last_run_time': last_run_time.isoformat(),
        'last_incident_date': last_incident_date.isoformat() if last_incident_date else None,
        'last_run_success': success,
        'last_error': str(error) if error else None
    }
    with open(get_state_file_path(), 'w') as f:
        json.dump(state, f, indent=2)

def load_state():
    """Load scraper state from file."""
    try:
        with open(get_state_file_path(), 'r') as f:
            state = json.load(f)
            return {
                'last_run_time': pd.to_datetime(state['last_run_time'], utc=True),
                'last_incident_date': pd.to_datetime(state['last_incident_date'], utc=True) 
                    if state['last_incident_date'] else None,
                'last_run_success': state['last_run_success'],
                'last_error': state['last_error']
            }
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'last_run_time': None,
            'last_incident_date': None,
            'last_run_success': True,
            'last_error': None
        }

def validate_scraped_data(df, is_transformed=False):
    """Validate scraped data before merging.
    
    Args:
        df: DataFrame to validate
        is_transformed: Boolean indicating if this is transformed data
    """
    if df is None or len(df) == 0:
        return False, "No data found in scraped results"
    
    if is_transformed:
        # Validate transformed data
        required_columns = ['incident_id', 'Incident_Title', 'start_timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for null values in critical columns
        null_incidents = df[df['incident_id'].isnull()]
        if not null_incidents.empty:
            return False, f"Found {len(null_incidents)} incidents with null incident_id"
        
        # Validate timestamp format
        try:
            pd.to_datetime(df['start_timestamp'], utc=True)
        except Exception as e:
            return False, f"Invalid timestamp format: {str(e)}"
        
        # Check for duplicate incident IDs
        duplicates = df[df.duplicated(subset=['incident_id'], keep=False)]
        if not duplicates.empty:
            return False, f"Found {len(duplicates)} duplicate incident IDs"
    else:
        # Validate raw scraped data
        required_columns = ['Incident_Title', 'Incident_Link', 'Updates']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns in raw data: {missing_columns}"
        
        # Check for null values in critical columns
        null_titles = df[df['Incident_Title'].isnull()]
        if not null_titles.empty:
            return False, f"Found {len(null_titles)} incidents with null titles"
        
        # Validate Updates column contains valid JSON
        try:
            df['Updates'].apply(json.loads)
        except Exception as e:
            return False, f"Invalid JSON in Updates column: {str(e)}"
    
    return True, None

def backup_existing_data(output_path):
    """Create backup of existing data file."""
    if Path(output_path).exists():
        backup_path = get_backup_path(output_path)
        shutil.copy2(output_path, backup_path)
        print(f"Created backup at: {backup_path}")
        return backup_path
    return None

def restore_from_backup(backup_path, output_path):
    """Restore data from backup file."""
    if backup_path and Path(backup_path).exists():
        shutil.copy2(backup_path, output_path)
        print(f"Restored from backup: {backup_path}")
        return True
    return False

if __name__ == "__main__":
    MAC_C_KEY = Keys.COMMAND
    # WINDOWS_C_KEY = Keys.CONTROL

    # Define provider URLs
    provider_urls = [
        "https://status.openai.com/history/",
        "https://status.anthropic.com/history",
        "https://status.character.ai/history"
    ]

    # Output path
    output_path = 'server/static/data/incident_stages_all.csv'
    
    # Load state
    state = load_state()
    if not state['last_run_success']:
        print(f"Warning: Last run failed with error: {state['last_error']}")
    
    # Create backup before starting
    backup_path = backup_existing_data(output_path)
    
    try:
        # Load existing incidents
        existing_df = load_existing_incidents(output_path)
        print(f"Loaded {len(existing_df)} existing incidents")
        
        # Get the latest incident date from existing data
        latest_date = get_latest_incident_date(existing_df)
        
        # Set date range
        end_date = datetime.now(gettz('UTC'))
        
        if latest_date:
            start_date = latest_date - timedelta(days=1)
            print(f"Collecting incidents from {start_date} to {end_date}")
        else:
            start_date = None
            print("No existing data found. Collecting all historical incidents.")
        
        all_incidents_df = pd.DataFrame()
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        print("Starting Chrome in headless mode...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Collect data from each provider
        for url in provider_urls:
            print(f"\nCollecting data from: {url}")
            incident_page = MyIncidentPage(driver, url)
            provider_df = incident_page.collect_data_through_pagination(
                start_date=start_date,
                end_date=end_date
            )
            all_incidents_df = pd.concat([all_incidents_df, provider_df])
            print(f"Collected {len(provider_df)} incidents from {url}")

        driver.quit()
        print("\nBrowser session closed.")

        # After collecting all incidents, validate the raw data
        if len(all_incidents_df) > 0:
            print(f"\nValidating {len(all_incidents_df)} new incidents...")
            is_valid, error_message = validate_scraped_data(all_incidents_df, is_transformed=False)
            
            if not is_valid:
                raise ValueError(f"Raw data validation failed: {error_message}")
            
            print("Raw data validation successful")
            
            # Transform the data
            print(f"\nTransforming {len(all_incidents_df)} new incidents...")
            transformed_df = DataTransformer.transform_incidents(all_incidents_df)
            
            if transformed_df is not None:
                # Validate transformed data
                is_valid, error_message = validate_scraped_data(transformed_df, is_transformed=True)
                if not is_valid:
                    raise ValueError(f"Transformed data validation failed: {error_message}")
                
                print("Transformed data validation successful")
                
                # Merge with existing data and remove duplicates
                final_df = merge_and_deduplicate(existing_df, transformed_df)
                
                # Sort by start_timestamp
                final_df = final_df.sort_values('start_timestamp', ascending=False)
                
                # Write to csv
                final_df.to_csv(output_path, index=False)
                print(f"\nData successfully written to {output_path}")
                print(f"Total incidents in database: {len(final_df)}")
                print(f"New incidents added: {len(final_df) - len(existing_df)}")
                
                # Update state with success
                save_state(
                    last_run_time=end_date,
                    last_incident_date=get_latest_incident_date(final_df),
                    success=True
                )
            else:
                raise ValueError("Data transformation failed")
        else:
            print("No new incidents were collected")
            save_state(
                last_run_time=end_date,
                last_incident_date=latest_date,
                success=True
            )

        # After processing the main providers, handle StabilityAI data
        print("\nProcessing StabilityAI incidents...")
        stability_input_folder, stability_temp_output = get_stability_paths()
        
        # Process StabilityAI data
        process_stability_folder(stability_input_folder, stability_temp_output)
        
        # Read both datasets
        main_df = pd.read_csv(output_path)
        stability_df = pd.read_csv(stability_temp_output)
        
        # Add StabilityAI column to main_df if it doesn't exist
        if 'StabilityAI' not in main_df.columns:
            main_df['StabilityAI'] = 0
            
        # Function to safely convert timestamps
        def safe_convert_timestamp(ts):
            if pd.isna(ts):
                return pd.NaT
            try:
                return pd.to_datetime(ts, utc=True, format='mixed')
            except Exception as e:
                print(f"Error converting timestamp {ts}: {str(e)}")
                return pd.NaT
        
        # Convert timestamps in both dataframes
        timestamp_cols = ['start_timestamp', 'close_timestamp', 'investigating_timestamp', 
                         'identified_timestamp', 'monitoring_timestamp', 'resolved_timestamp', 
                         'postmortem_timestamp']
        
        for col in timestamp_cols:
            if col in main_df.columns:
                main_df[col] = main_df[col].apply(safe_convert_timestamp)
            if col in stability_df.columns:
                stability_df[col] = stability_df[col].apply(safe_convert_timestamp)
        
        # Check for existing StabilityAI incidents
        if len(main_df) > 0 and 'provider' in main_df.columns:
            existing_stability = main_df[main_df['provider'] == 'StabilityAI']
            if len(existing_stability) > 0:
                print(f"\nFound {len(existing_stability)} existing StabilityAI incidents")
                
                # Get the latest timestamp from existing StabilityAI incidents
                latest_existing = pd.NaT
                for col in timestamp_cols:
                    if col in existing_stability.columns:
                        col_max = existing_stability[col].max()
                        if pd.notna(col_max) and (pd.isna(latest_existing) or col_max > latest_existing):
                            latest_existing = col_max
                
                if pd.notna(latest_existing):
                    print(f"Latest existing StabilityAI incident timestamp: {latest_existing}")
                    
                    # Filter out older incidents from stability_df
                    new_incidents = []
                    for _, row in stability_df.iterrows():
                        incident_timestamps = [row[col] for col in timestamp_cols if col in row and pd.notna(row[col])]
                        if incident_timestamps:
                            latest_incident = max(incident_timestamps)
                            if latest_incident > latest_existing:
                                new_incidents.append(row)
                    
                    if new_incidents:
                        stability_df = pd.DataFrame(new_incidents)
                        print(f"Found {len(stability_df)} new StabilityAI incidents")
                    else:
                        stability_df = pd.DataFrame()
                        print("No new StabilityAI incidents found")
                else:
                    print("Could not determine latest existing incident timestamp")
            else:
                print("No existing StabilityAI incidents found in main database")
        else:
            print("No existing StabilityAI incidents found in main database")
        
        # Combine the datasets only if there are new incidents
        if not stability_df.empty:
            combined_df = pd.concat([main_df, stability_df], ignore_index=True)
            
            # Sort by start_timestamp, handling NaT values
            combined_df = combined_df.sort_values('start_timestamp', ascending=False, na_position='last')
            
            # Remove duplicates based on incident_id
            combined_df = combined_df.drop_duplicates(subset=['incident_id'], keep='first')
            
            # Save the combined dataset
            combined_df.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S%z')
            print(f"\nSuccessfully combined all provider data in {output_path}")
            print(f"Total incidents in database: {len(combined_df)}")
            print(f"New StabilityAI incidents added: {len(stability_df)}")
        else:
            print("\nNo new StabilityAI incidents to add")
        
        # Clean up temporary file
        if os.path.exists(stability_temp_output):
            os.remove(stability_temp_output)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
        
        # Restore from backup if something went wrong
        if backup_path:
            restore_from_backup(backup_path, output_path)
        
        # Update state with error
        save_state(
            last_run_time=datetime.now(gettz('UTC')),
            last_incident_date=latest_date,
            success=False,
            error=str(e)
        )
        
    finally:
        if 'driver' in locals():
            driver.quit()
            print("Browser session closed.")
