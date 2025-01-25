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
from util_data import read_data, load_json_column, unnest_dict, get_incident_id
from pathlib import Path

from incident_stages_stability import process_folder as process_stability_folder, get_paths as get_stability_paths
from incident_scraper_oac import DataTransformer, validate_scraped_data

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
        
        incident.send_keys(self.c_key + Keys.RETURN)
        WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
        new_window = [window for window in self.driver.window_handles if window != original_window][0]
        self.driver.switch_to.window(new_window)
        
        record = pd.DataFrame({
            "Incident_Title": [title],
            "Incident_Link": [link],
            "Incident_color": [incident_color],
            "Incident_Impact": [impact],
            "Updates": [self.get_incident_updates()],
            "Service": [self.get_incident_service()]
        })
        print(record['Incident_Title'][0], ". ", json.loads(record['Updates'][0])[0]['Update_Timestamp'])
        
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

    def loop_over_incidents(self):
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            incident_df = pd.DataFrame()
            flag_no_data = False
            try:
                incident_list = self.get_incident_list()
                original_window = self.driver.current_window_handle
                if not incident_list:
                    flag_no_data = True
                else:
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

    def collect_all_historical_data(self):
        """Collect all historical incident reports"""
        all_incidents_df = pd.DataFrame()
        try:
            self.driver.get(self.provider_url)
            stop_collection = False
            
            while not stop_collection:
                self.show_all_incidents()
                incident_df, flag_no_data = self.loop_over_incidents()
                
                if len(incident_df) > 0:
                    all_incidents_df = pd.concat([all_incidents_df, incident_df])
                
                if flag_no_data:
                    print("No more incidents found.")
                    break
                    
                self.go_to_previous_page()
                
            return all_incidents_df
        except Exception as e:
            print("Error in collect_all_historical_data(): ", e)
            traceback.print_exc()
            return all_incidents_df

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
    output_path = 'server/static/data/incident_stages_all_historical.csv'
    
    try:
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
            print(f"\nCollecting all historical data from: {url}")
            incident_page = MyIncidentPage(driver, url)
            provider_df = incident_page.collect_all_historical_data()
            all_incidents_df = pd.concat([all_incidents_df, provider_df])
            print(f"Collected {len(provider_df)} incidents from {url}")

        driver.quit()
        print("\nBrowser session closed.")

        # Transform and validate the data
        if len(all_incidents_df) > 0:
            print(f"\nValidating {len(all_incidents_df)} incidents...")
            is_valid, error_message = validate_scraped_data(all_incidents_df, is_transformed=False)
            
            if not is_valid:
                raise ValueError(f"Raw data validation failed: {error_message}")
            
            print("Raw data validation successful")
            
            print(f"\nTransforming {len(all_incidents_df)} incidents...")
            transformed_df = DataTransformer.transform_incidents(all_incidents_df)
            
            if transformed_df is not None:
                is_valid, error_message = validate_scraped_data(transformed_df, is_transformed=True)
                if not is_valid:
                    raise ValueError(f"Transformed data validation failed: {error_message}")
                
                print("Transformed data validation successful")
                
                transformed_df = transformed_df.sort_values('start_timestamp', ascending=False)
                transformed_df.to_csv(output_path, index=False)
                print(f"\nData successfully written to {output_path}")
                print(f"Total incidents in database: {len(transformed_df)}")
            else:
                raise ValueError("Data transformation failed")
        else:
            print("No incidents were collected")

        # Process StabilityAI data
        print("\nProcessing StabilityAI incidents...")
        stability_input_folder, stability_temp_output = get_stability_paths()
        process_stability_folder(stability_input_folder, stability_temp_output)
        
        # Read both datasets
        main_df = pd.read_csv(output_path)
        stability_df = pd.read_csv(stability_temp_output)
        
        # Add StabilityAI column to main_df if it doesn't exist
        if 'StabilityAI' not in main_df.columns:
            main_df['StabilityAI'] = 0
        
        # Convert timestamps in both dataframes
        timestamp_cols = ['start_timestamp', 'close_timestamp', 'investigating_timestamp', 
                         'identified_timestamp', 'monitoring_timestamp', 'resolved_timestamp', 
                         'postmortem_timestamp']
        
        for col in timestamp_cols:
            if col in main_df.columns:
                main_df[col] = pd.to_datetime(main_df[col], utc=True, format='mixed')
            if col in stability_df.columns:
                stability_df[col] = pd.to_datetime(stability_df[col], utc=True, format='mixed')
        
        # Combine the datasets
        combined_df = pd.concat([main_df, stability_df], ignore_index=True)
        
        # Sort by start_timestamp, handling NaT values
        combined_df = combined_df.sort_values('start_timestamp', ascending=False, na_position='last')
        
        # Remove duplicates based on incident_id
        combined_df = combined_df.drop_duplicates(subset=['incident_id'], keep='first')
        
        # Save the combined dataset
        combined_df.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S%z')
        print(f"\nSuccessfully combined all provider data in {output_path}")
        print(f"Total incidents in database: {len(combined_df)}")
        
        # Clean up temporary file
        if os.path.exists(stability_temp_output):
            os.remove(stability_temp_output)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
        
    finally:
        if 'driver' in locals():
            driver.quit()
            print("Browser session closed.") 