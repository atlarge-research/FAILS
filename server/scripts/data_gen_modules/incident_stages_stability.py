import os
import pandas as pd
import uuid
from datetime import datetime
from dateutil import parser
from dateutil.tz import gettz
import json
import re

def generate_incident_id():
    return str(uuid.uuid4().hex[:12])  # 12-character unique ID

def get_services(service_str):
    if pd.isna(service_str):
        return []
    service_str = service_str.split(':')[-1].rstrip('.')
    service_str = service_str.replace("and", ",")
    services = service_str.split(',')
    return [service.strip() for service in services]


def parse_update_time(timestamp_str):
    tzinfos = {
        "PST": gettz("America/Los_Angeles"),
        "PDT": gettz("America/Los_Angeles"),
    }

    try:
        timestamp = parser.parse(timestamp_str, tzinfos=tzinfos)
        return timestamp.astimezone(gettz('UTC'))
    except Exception as e:
        print(f"Error parsing timestamp: {timestamp_str}, Error: {e}")
        return None


def clean_updates_str(updates_str):
    if isinstance(updates_str, str):
        try:
            # Convert single quotes around keys and values to double quotes
            updates_str = re.sub(r"'([^']+)'(?=\s*:)", r'"\1"', updates_str)  # Keys
            updates_str = re.sub(r"(?<=: )'([^']*)'", r'"\1"', updates_str)  # Values

            # Escape single quotes inside string values
            updates_str = re.sub(r'(?<!\\)"([^"]*?)\'([^"]*?)"(?!:)', r'"\1\\\'\2"', updates_str)

            return updates_str
        except Exception as e:
            print(f"Error cleaning updates_str: {e}")
            return updates_str  # Return original if cleaning fails
    return updates_str  # If updates_str is not a string, no cleaning needed


def parse_json_updates(updates):
    try:
        # Clean each update string before attempting to parse
        cleaned_updates = [clean_updates_str(str(update)) for update in updates]
        parsed_updates = [json.loads(update) for update in cleaned_updates]
        return parsed_updates
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def parse_update_time(timestamp_str):
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

    # Ensure the timestamp is in UTC
    timestamp = timestamp.replace(tzinfo=gettz('UTC'))
    return timestamp


def parse_updates(updates_str, stages=None):
    if stages is None:
        stages = ["Investigating", "Identified", "Monitoring", "Resolved", "Postmortem"]

    updates_dict = {f"{stage.lower()}_flag": 0 for stage in stages}
    updates_dict.update({f"{stage.lower()}_timestamp": None for stage in stages})
    updates_dict.update({f"{stage.lower()}_description": None for stage in stages})

    updates_str = clean_updates_str(updates_str)

    # checking the cleaned updates_str
    #print(f"Processed Updates (cleaned): {updates_str}")

    try:
        updates_list = json.loads(updates_str)  # Try parsing as JSON
        #print(f"Successfully parsed JSON updates: {updates_list}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        updates_list = []

    start_timestamp = None
    identified_timestamp = None
    close_timestamp = None

    for update in updates_list:
        if isinstance(update, dict):
            stage = update.get("Update_Title")
            if stage in stages:
                stage = stage.lower()
                updates_dict[f"{stage}_flag"] = 1
                updates_dict[f"{stage}_timestamp"] = parse_update_time(update.get("Update_Timestamp"))
                updates_dict[f"{stage}_description"] = update.get("Update_Body")

                if stage == "investigating" and start_timestamp is None:
                    start_timestamp = updates_dict[f"{stage}_timestamp"]
                if stage == "identified" and identified_timestamp is None:
                    identified_timestamp = updates_dict[f"{stage}_timestamp"]
                if stage == "resolved" or stage == "postmortem":
                    close_timestamp = updates_dict[f"{stage}_timestamp"]

    if identified_timestamp is None and start_timestamp is not None:
        identified_timestamp = start_timestamp

    return updates_dict, start_timestamp, identified_timestamp, close_timestamp

from datetime import timedelta

def format_time_span(start_timestamp, close_timestamp, tolerance=timedelta(seconds=1)):
    if start_timestamp and close_timestamp:
        # Ensure both timestamps are datetime objects
        if isinstance(start_timestamp, str):
            start_timestamp = parser.parse(start_timestamp)
        if isinstance(close_timestamp, str):
            close_timestamp = parser.parse(close_timestamp)

        # Ensure both timestamps are in the same timezone (e.g., UTC)
        if start_timestamp.tzinfo != close_timestamp.tzinfo:
            close_timestamp = close_timestamp.astimezone(start_timestamp.tzinfo)

        # Check if timestamps are nearly identical (within tolerance)
        time_diff = close_timestamp - start_timestamp
        if abs(time_diff) <= tolerance:
            return "0 days 00:00:00"

        days = time_diff.days
        hours, remainder = divmod(abs(time_diff).seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"

    return "Invalid timespan"  # If timestamps are missing

def identify_services(description):
    services = {
        "Playground": "playground",
        "API": "api",
        "Labs": "labs",
        "ChatGPT": "chatgpt",
        "api.anthropic.com": "api.anthropic.com",
        "claude.ai": "claude.ai",
        "console.anthropic.com": "console.anthropic.com",
        "Character.AI": "character.ai"
    }
    service_flags = {key: 0 for key in services.keys()}
    for service, keyword in services.items():
        if keyword in description.lower():
            service_flags[service] = 1
    return service_flags

def parse_service_string(service_str):
    """Parse the service string from StabilityAI scraper into service flags."""
    service_flags = {
        "REST API": 0,
        "gRPC API": 0,
        "Stable Assistant": 0
    }
    
    if pd.isna(service_str) or not isinstance(service_str, str):
        return service_flags
        
    # Remove any duplicate prefixes
    service_str = service_str.replace("This incident affected: This incident affected: ", "This incident affected: ")
    
    # Remove the prefix if it exists
    if "This incident affected: " in service_str:
        service_str = service_str.replace("This incident affected: ", "")
    
    # Split services if multiple are listed
    services = service_str.split(" and ")
    
    for service in services:
        service = service.strip()
        if service in service_flags:
            service_flags[service] = 1
            
    return service_flags

def process_file(file_path):
    raw_data = pd.read_csv(file_path)
    processed_data = []

    column_order = [
        "incident_id", "Incident_Title", "incident_impact_level", "Incident_color", "provider",
        "Playground", "API", "DALL-E", "ChatGPT", "api.anthropic.com", "claude.ai", 
        "console.anthropic.com", "Character.AI", "REST API", "gRPC API", "Stable Assistant",
        "investigating_flag", "investigating_timestamp", "investigating_description",
        "identified_flag", "identified_timestamp", "identified_description", "monitoring_flag",
        "monitoring_timestamp", "monitoring_description", "resolved_flag", "resolved_timestamp",
        "resolved_description", "postmortem_flag", "postmortem_timestamp", "postmortem_description",
        "start_timestamp", "close_timestamp", "time_span", "over_one_day"
    ]

    for idx, row in raw_data.iterrows():
        incident_id = generate_incident_id()
        incident_title = row["Incident_Title"]
        impact_level = row["Incident_Impact"]
        color = row.get("Incident_color", "")
        provider = "StabilityAI"
        description = row.get("Incident_Title", "") + " " + row.get("Updates", "")
        
        # Initialize service flags
        service_flags = {
            "Playground": 0,
            "API": 0,
            "DALL-E": 0,
            "ChatGPT": 0,
            "api.anthropic.com": 0,
            "claude.ai": 0,
            "console.anthropic.com": 0,
            "Character.AI": 0,
            "REST API": 0,
            "gRPC API": 0,
            "Stable Assistant": 0
        }

        # Clean and parse updates
        updates_field = row.get("Updates", "")
        if isinstance(updates_field, str):
            updates_field = clean_updates_str(updates_field)
            try:
                updates_list = json.loads(updates_field)
                
                # Process services first
                text_to_check = row.get("Incident_Title", "").lower()
                for update in updates_list:
                    text_to_check += f" {update.get('Update_Title', '')} {update.get('Update_Body', '')}".lower()

                # Parse the Service field
                if "Service" in row:
                    stability_services = parse_service_string(row["Service"])
                    service_flags.update(stability_services)

                # Check services in Updates if none were found
                if all(v == 0 for k, v in service_flags.items() if k in ["REST API", "gRPC API", "Stable Assistant"]):
                    if "rest" in text_to_check:
                        service_flags["REST API"] = 1
                    if "grpc" in text_to_check:
                        service_flags["gRPC API"] = 1
                    if any(term in text_to_check for term in ["stable assistant", "assistant", "stable-assistant"]):
                        service_flags["Stable Assistant"] = 1
                    
                    # If still no specific service found but API-related terms exist
                    if all(v == 0 for k, v in service_flags.items() if k in ["REST API", "gRPC API", "Stable Assistant"]):
                        if any(term in text_to_check for term in ["api", "latency", "error", "endpoint"]):
                            service_flags["REST API"] = 1
                            service_flags["gRPC API"] = 1

                # Now parse updates for timestamps
                updates, start_timestamp, identified_timestamp, close_timestamp = parse_updates(json.dumps(updates_list))

            except json.JSONDecodeError:
                print(f"Error parsing updates JSON for incident {idx}")
                updates, start_timestamp, identified_timestamp, close_timestamp = {}, None, None, None
        else:
            updates, start_timestamp, identified_timestamp, close_timestamp = {}, None, None, None

        if start_timestamp is None and close_timestamp is not None:
            start_timestamp = close_timestamp

        if identified_timestamp is None:
            identified_timestamp = start_timestamp

        if start_timestamp is None and identified_timestamp is not None:
            start_timestamp = identified_timestamp

        time_span = format_time_span(start_timestamp, close_timestamp)
        over_one_day = (close_timestamp - start_timestamp).days > 1 if start_timestamp and close_timestamp else False

        # Converting timestamps to string format
        start_timestamp = start_timestamp.strftime("%Y-%m-%d %H:%M:%S+00:00") if start_timestamp else None
        close_timestamp = close_timestamp.strftime("%Y-%m-%d %H:%M:%S+00:00") if close_timestamp else None
        identified_timestamp = identified_timestamp.strftime("%Y-%m-%d %H:%M:%S+00:00") if identified_timestamp else None

        processed_data.append({
            'incident_id': incident_id,
            'Incident_Title': incident_title,
            'incident_impact_level': impact_level,
            'Incident_color': color,
            'provider': provider,
            **service_flags,
            'investigating_flag': updates['investigating_flag'],
            'investigating_timestamp': updates['investigating_timestamp'],
            'investigating_description': updates['investigating_description'],
            'identified_flag': updates['identified_flag'],
            'identified_timestamp': updates['identified_timestamp'],
            'identified_description': updates['identified_description'],
            'monitoring_flag': updates['monitoring_flag'],
            'monitoring_timestamp': updates['monitoring_timestamp'],
            'monitoring_description': updates['monitoring_description'],
            'resolved_flag': updates['resolved_flag'],
            'resolved_timestamp': updates['resolved_timestamp'],
            'resolved_description': updates['resolved_description'],
            'postmortem_flag': updates['postmortem_flag'],
            'postmortem_timestamp': updates['postmortem_timestamp'],
            'postmortem_description': updates['postmortem_description'],
            'start_timestamp': start_timestamp,
            'close_timestamp': close_timestamp,
            'time_span': time_span,
            'over_one_day': over_one_day
        })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(processed_data)
    
    # Ensure columns are in the correct order
    df = df[column_order]
    return df

def process_folder(input_folder, output_file):
    all_data = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_path}")
            df = process_file(file_path)
            all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Processed incident stages saved to {output_file}.")

def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Gets the server directory
    input_folder = os.path.join(base_dir, "static/data/raw/incident/stabilityAI")
    output_file = os.path.join(base_dir, "static/data/incident_stages_stab.csv")
    return input_folder, output_file

# Inputs and Outputs
input_folder, output_file = get_paths()
process_folder(input_folder, output_file)
