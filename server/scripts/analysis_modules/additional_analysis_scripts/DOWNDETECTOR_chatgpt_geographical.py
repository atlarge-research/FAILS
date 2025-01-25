from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import json

class DownDetectorScraper:
    def __init__(self):
        # List of Downdetector country domains
        self.domains = [
            'https://downdetector.com',
            'https://downdetector.com.au',
            'https://downdetector.co.uk',
            'https://downdetector.ca',
            'https://downdetector.in',
        ]
        
        # Initialize Chrome options
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--window-size=1920,1080')
        # self.options.add_argument('--headless')  # Uncomment for headless mode
        
        # Initialize the driver
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.maximize_window()

    def handle_popups(self):
        try:
            # Handle cookie consent
            cookie_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_button.click()
        except:
            print("No cookie popup found or already accepted")

        try:
            # Handle location popup - click Cancel
            cancel_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Cancel']"))
            )
            cancel_button.click()
            print("Clicked Cancel on location popup")
        except:
            print("No location popup found")

    def get_chart_data(self):
        try:
            # Wait for chart data to be loaded into window.DD object
            chart_data = self.driver.execute_script("""
                // Wait for Highcharts to be available
                if (typeof Highcharts === 'undefined' || !Highcharts.charts.length) {
                    return null;
                }
                
                // Get the first chart
                const chart = Highcharts.charts[0];
                if (!chart || !chart.series || !chart.series[0]) {
                    return null;
                }
                
                // Get the data points
                const series = chart.series[0];
                const points = series.points;
                if (!points || !points.length) {
                    return null;
                }
                
                // Get the last point
                const lastPoint = points[points.length - 1];
                return {
                    x: lastPoint.x,
                    y: lastPoint.y
                };
            """)
            
            if chart_data:
                return {
                    'reports': str(int(chart_data['y'])),
                    'timestamp': datetime.fromtimestamp(chart_data['x']/1000).strftime('%I:%M %p')
                }
            return None
        except Exception as e:
            print(f"Error getting chart data: {str(e)}")
            return None

    def scrape_openai_status(self):
        results = []
        
        for domain in self.domains:
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    print(f"\nTrying to scrape {domain}")
                    self.driver.get(f"{domain}/status/openai")
                    
                    # Handle popups
                    self.handle_popups()
                    
                    # Wait for chart to load
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "highcharts-container"))
                    )
                    
                    # Give extra time for chart data to load
                    time.sleep(2)
                    
                    # Get the chart data
                    data = self.get_chart_data()
                    
                    if data:
                        # Extract country from domain
                        country = domain.split('.')[-1] if len(domain.split('.')) > 2 else 'com'
                        
                        # Add to results
                        results.append({
                            'country': country,
                            'domain': domain,
                            'reports': data['reports'],
                            'timestamp': data['timestamp'],
                            'timezone': 'Local',
                            'scrape_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        print(f"Successfully scraped data from {domain}")
                        break
                    else:
                        print(f"Could not get chart data from {domain}")
                        retry_count += 1
                
                except Exception as e:
                    print(f"Error processing {domain}: {str(e)}")
                    retry_count += 1
                
                if retry_count < max_retries:
                    print(f"Retrying {domain} (attempt {retry_count + 1}/{max_retries})...")
                    time.sleep(3)
            
            # Add delay between domains
            time.sleep(3)
        
        return results

    def save_to_csv(self, results):
        df = pd.DataFrame(results)
        filename = f"downdetector_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def close(self):
        self.driver.quit()

def main():
    scraper = DownDetectorScraper()
    try:
        results = scraper.scrape_openai_status()
        if results:
            scraper.save_to_csv(results)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
