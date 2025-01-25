import os
import sys
import time
import traceback
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

# Add the server directory to Python path for imports
server_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if server_dir not in sys.path:
    sys.path.append(server_dir)

# Import the scraper classes using absolute imports
from scripts.data_gen_modules.incident_scraper_oac import MyIncidentPage as OpenAIIncidentPage
from scripts.data_gen_modules.StabilityAI import MyIncidentPage as StabilityAIIncidentPage
from scripts.data_gen_modules.incident_stages_stability import process_file

def setup_chrome_driver():
    """Configure and return a headless Chrome driver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    return webdriver.Chrome(options=chrome_options)

def scrape_provider_data(driver, url, IncidentPageClass, provider_name):
    """Scrape incident data for a specific provider"""
    print(f"Starting data collection for {provider_name}...")
    try:
        driver.get(url)
        incident_page = IncidentPageClass(driver)
        incident_page.collect_data_through_pagination()
        print(f"Completed data collection for {provider_name}")
    except Exception as e:
        print(f"Error collecting data for {provider_name}: {e}")
        traceback.print_exc()

def combine_incident_data():
    """Combine all incident data into a single CSV file"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_dir = os.path.join(base_dir, "static/data/raw/incident")
    output_dir = os.path.join(base_dir, "static/data")
    
    # Create directories if they don't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize webdriver
    driver = setup_chrome_driver()

    try:
        # Scrape data for each provider
        providers = {
            'openai': {
                'url': 'https://status.openai.com/history/',
                'class': OpenAIIncidentPage
            },
            'stabilityai': {
                'url': 'https://stabilityai.instatus.com/history/1',
                'class': StabilityAIIncidentPage
            },
            'anthropic': {
                'url': 'https://status.anthropic.com/history',
                'class': OpenAIIncidentPage  # Using same class as OpenAI since format is similar
            },
            'character': {
                'url': 'https://status.character.ai/history',
                'class': OpenAIIncidentPage  # Using same class as OpenAI since format is similar
            }
        }

        for provider_name, config in providers.items():
            scrape_provider_data(driver, config['url'], config['class'], provider_name)
            time.sleep(2)  # Brief pause between providers

    finally:
        driver.quit()

    # Process and combine the data
    try:
        import pandas as pd

        all_data = []
        for provider in providers.keys():
            provider_dir = os.path.join(raw_data_dir, provider)
            if os.path.exists(provider_dir):
                for file_name in os.listdir(provider_dir):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(provider_dir, file_name)
                        print(f"Processing {provider} file: {file_path}")
                        try:
                            df = process_file(file_path)
                            df['provider'] = provider  # Add provider column
                            all_data.append(df)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                            traceback.print_exc()

        if all_data:
            # Combine all dataframes
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Save combined data
            output_file = os.path.join(output_dir, "incident_stages_all.csv")
            final_df.to_csv(output_file, index=False)
            print(f"Successfully created combined incident data at: {output_file}")
        else:
            print("No data was processed. Check the individual provider logs for errors.")

    except Exception as e:
        print(f"Error combining incident data: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    combine_incident_data()