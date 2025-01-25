import time
import os
from datetime import date
import traceback
import argparse
import json

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException


class MyCalendarPage:
    # Class variable for XPaths that do not change across instances
    TOOLTIP_XPATH = "//div[@class='tooltip-content']"
    CALENDAR_XPATH = "//*[local-name()='svg' and @class ='day active']/*[local-name()='rect']"
    PAGINATION_XPATH = "//div[@class='pagination']//i[@class='left-arrow']"
    SERVICE_DROPDOWN_XPATH = "//div[contains(@class, 'select-input__dropdown-indicator')]//span"
    SERVICE_LIST_XPATH = "//div[contains(@class, 'select-input__menu-list')]"

    def __init__(self, driver, service):
        self.driver = driver
        self.service = service
        # self.uptime_df = pd.DataFrame(columns=['Date', 'Outages', 'Downtime (min)', 'Incidents', 'Service'])
        self.uptime_df = pd.DataFrame(columns=['Date', 'Outages', 'Outage_Color', 'Incidents', 'Service'])

    def get_tooltip_outages(self, tooltip):
        outage_list = tooltip.find_elements(By.XPATH, ".//div[contains(@class, 'outage-field')]")
        outages = []
        if not outage_list:
            return outages

        else:
            for outage in outage_list:
                # get outage_type/status from <span class=“label”>
                outage_type = outage.find_element(By.XPATH, ".//span[@class='label']").text
                # get downtime
                hours = outage.find_element(By.XPATH, ".//span[@class='value-hrs']").text.split()[0]
                minutes = outage.find_element(By.XPATH, ".//span[@class='value-mins']").text.split()[0]
                downtime = int(hours) * 60 + int(minutes)
                outages.append({
                        "Outage_Type": outage_type,
                        "Downtime (min)": downtime,
                })
            return json.dumps(outages)

    def get_tooltip_incidents(self, tooltip):
        incidents = []
        incident_list = tooltip.find_elements(By.XPATH,
                                                  ".//ul[@id='related-events-list']/li[@class='related-event']")
        for incident in incident_list:
            incident_element = incident.find_element(By.TAG_NAME, 'a')
            incidents.append({
                    "Incident_Title": incident_element.text,
                    "Incident_Link": incident_element.get_attribute('href')
            })
        return json.dumps(incidents)

    def send_no_data_warning(self, tooltip_date, tooltip):
        no_data_msg = tooltip.find_elements(By.XPATH, ".//div[@class='no-data-msg']")
        if no_data_msg:
            print({'Date': tooltip_date, 'Warning!': no_data_msg[0].text})
            return True

    def hover_over_rect(self, rect):
        """
        Hover over the rect to get the tooltip information.
        Tooltip is the popup when hovering over a calendar rect, containing the uptime information for that day.
        """
        ActionChains(self.driver).move_to_element(rect).perform()
        try:
            tooltip = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, self.TOOLTIP_XPATH))
            )

            color = rect.get_attribute('fill')
            tooltip_date = tooltip.find_element(By.XPATH, ".//p[@class='date']").text

            # check if data exists in the tooltip
            if self.send_no_data_warning(tooltip_date, tooltip):
                return None
            else:
                return pd.DataFrame({
                    "Date": [tooltip_date],
                    "Outages": [self.get_tooltip_outages(tooltip)],
                    "Outage_Color": [color],
                    "Incidents": [self.get_tooltip_incidents(tooltip)],
                    "Service": [self.service]
                })
        except Exception as e:
            print("Executing hover_over_rect(). An error occurred: ", e)
            traceback.print_exc()
            return None

    def get_calendar_rect_list(self):
        calendar_rect_list = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, self.CALENDAR_XPATH))
        )
        print("Active calendar rects found in this page: ", len(calendar_rect_list))
        return calendar_rect_list

    def loop_over_calendar(self):
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            flag_no_data = False
            try:
                # Collecting incident when hovering over calendar rect (a rect presents a day)
                calendar_rect_list = self.get_calendar_rect_list()
                for calendar_rect in calendar_rect_list:
                    rect_record = self.hover_over_rect(calendar_rect)
                    if rect_record is not None and not rect_record.empty:
                        print(rect_record)
                        self.uptime_df = pd.concat([self.uptime_df, rect_record], ignore_index=True)
                    else:
                        flag_no_data = True
                return flag_no_data
            except StaleElementReferenceException:
                print("Stale element, restarting incidents looping process.")
                attempt += 1
                continue
            except Exception as e:
                print("Executing loop_over_incidents(). An error occurred: ", e)
                traceback.print_exc()

        print("\nWarning: this should not happen!!!\n")
        return [], True

    def archive_uptime_by_service(self, uptime_df):
        # get now date as execution_date for archive
        execution_date = date.today().strftime("%Y-%m-%d")
        # archive_folder = f"data/raw/uptime/{self.service}/{execution_date}"
        archive_folder = f"data/raw/uptime/{execution_date}/{self.service}"
        os.makedirs(archive_folder, exist_ok=True)
        uptime_df.to_csv(f"{archive_folder}/uptime_history.csv", index=False)

    def go_to_previous_page(self):
        prev_page = self.driver.find_element(By.XPATH, self.PAGINATION_XPATH)
        if prev_page:
            prev_page.click()
            time.sleep(1)

    def change_service(self):
        # Dropdown
        service_dropdown = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, self.SERVICE_DROPDOWN_XPATH))
        )
        service_dropdown.click()

        # Select service from list
        service_list = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, self.SERVICE_LIST_XPATH))
        )
        options = service_list.find_elements(By.XPATH, ".//div[contains(@class, 'select-input__option')]")
        for option in options:
            if option.text.lower() == service:
                option.click()  # click operation
                break
        time.sleep(1)

    def collect_data_through_pagination(self):
        """Collect uptime data by uptime history pages"""
        try:
            if self.service != 'api':
                self.change_service()

            while True:
                # Get the uptime record by looping over the calendar rect list in the current page
                flag_no_data = self.loop_over_calendar()
                # Go to the previous page
                if not flag_no_data:
                    self.go_to_previous_page()
                else:
                    print("No more previous data. Ending uptime data collecting.")
                    break
            # Archive the uptime records for the selected service
            self.archive_uptime_by_service(self.uptime_df)
        except Exception as e:
            print("Executing collect_data_through_pagination(). An error occurred: ", e)
            traceback.print_exc()


def parse_arguments():
    valid_services = ['api', 'chatgpt', 'labs', 'playground'] # openai services
    # valid_services = ['claude.ai', 'api.anthropic.com', 'console.anthropic.com']  # anthropic services
    # valid_services = ['character.ai']  # character.ai service

    parser = argparse.ArgumentParser(description='Collect uptime data for a specified service. Default service is api.')
    parser.add_argument('-S', '--service', type=lambda s: str(s).lower(),
                        choices=valid_services, default='api',
                        help=f"Choose which service to collect data for.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    service = args.service
    print("Collecting uptime data for service: {}".format(service))

    driver = webdriver.Chrome()
    driver.get("https://status.openai.com/uptime/") # openai services
    # driver.get("https://status.anthropic.com/uptime")  # anthropic services
    # driver.get("https://status.character.ai/uptime")  # character.ai service

    try:
        calendar_page = MyCalendarPage(driver, service)
        calendar_page.collect_data_through_pagination()

    finally:
        # Close the browser
        # input("Press Enter to close the browser")
        driver.quit()
