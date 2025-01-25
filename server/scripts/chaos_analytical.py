import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from matplotlib.ticker import PercentFormatter
from zoneinfo import ZoneInfo

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Modularized Functions
def load_and_preprocess_data(path, start_date, end_date):
    df = pd.read_csv(path)

    # Convert the status columns to datetime
    status_lst = ['investigating', 'identified', 'monitoring', 'resolved', 'postmortem', 'start']
    for status in status_lst:
        df[f'{status}_timestamp'] = pd.to_datetime(df[f'{status}_timestamp'])

    # Replace NaT in 'postmortem_timestamp' with 'resolved_timestamp', 'monitoring_timestamp', etc.
    df['postmortem_timestamp'] = df['postmortem_timestamp'].fillna(df['resolved_timestamp'])
    df['postmortem_timestamp'] = df['postmortem_timestamp'].fillna(df['monitoring_timestamp'])
    df['postmortem_timestamp'] = df['postmortem_timestamp'].fillna(df['identified_timestamp'])
    df['postmortem_timestamp'] = df['postmortem_timestamp'].fillna(df['investigating_timestamp'])

    # Adjust the date filtering to include incidents where 'postmortem_timestamp' is missing
    df = df[(df['investigating_timestamp'] >= start_date) & (df['postmortem_timestamp'] <= end_date)]

    # Remove records with inconsistent timestamps
    inconsistent_records = df[df['investigating_timestamp'] > df['postmortem_timestamp']]
    df = df.drop(inconsistent_records.index).reset_index(drop=True)

    # Rename columns to match what we expect for the rest of the analysis
    df = df.rename(columns={
        'API': 'API-OpenAI',
        'ChatGPT': 'ChatGPT',
        'Labs': 'DALL·E',
        'Playground': 'Playground',
        'api.anthropic.com': 'API-Anthropic',
        'claude.ai': 'Claude',
        'console.anthropic.com': 'Console',
        'Character.AI': 'Character.AI'
    })

    return df


def filter_corner_cases(df):
    corner_cases_indices = []
    for index, record in df.iterrows():
        # Check for corner cases where timestamps are inconsistent
        if (pd.notnull(record['investigating_timestamp']) and pd.notnull(record['identified_timestamp']) and
            record['investigating_timestamp'] > record['identified_timestamp']):
            corner_cases_indices.append(index)
        # Repeat similar checks for other timestamp comparisons
        if (pd.notnull(record['identified_timestamp']) and pd.notnull(record['monitoring_timestamp']) and
            record['identified_timestamp'] > record['monitoring_timestamp']):
            corner_cases_indices.append(index)
        if (pd.notnull(record['monitoring_timestamp']) and pd.notnull(record['resolved_timestamp']) and
            record['monitoring_timestamp'] > record['resolved_timestamp']):
            corner_cases_indices.append(index)
        if (pd.notnull(record['resolved_timestamp']) and pd.notnull(record['postmortem_timestamp']) and
            record['resolved_timestamp'] > record['postmortem_timestamp']):
            corner_cases_indices.append(index)
        # Add any additional checks as necessary

    # Remove duplicate indices if any
    corner_cases_indices = list(set(corner_cases_indices))

    print(f"Corner cases identified: {len(corner_cases_indices)} records")
    df = df.drop(index=corner_cases_indices).reset_index(drop=True)
    print(f"Remaining records after dropping corner cases: {df.shape[0]}")
    return df


def generate_status_count_table(df):
    count_dict = {}
    # Let S1 = investigating, S2 = identified, S3 = monitoring, S4 = resolved, S5 = postmortem
    for index, record in df.iterrows():
        # Define all cases as per your original code
        # S1_S3_S4
        if (record['investigating_flag'] == 1 and record['monitoring_flag'] == 1 and
            record['resolved_flag'] == 1 and record['identified_flag'] == 0 and
            record['postmortem_flag'] == 0):
            count_dict['S1_S3_S4'] = count_dict.get('S1_S3_S4', 0) + 1
        # S1_S4
        elif (record['investigating_flag'] == 1 and record['resolved_flag'] == 1 and
              record['identified_flag'] == 0 and record['monitoring_flag'] == 0 and
              record['postmortem_flag'] == 0):
            count_dict['S1_S4'] = count_dict.get('S1_S4', 0) + 1
        # S1_S2_S3_S4
        elif (record['investigating_flag'] == 1 and record['identified_flag'] == 1 and
              record['monitoring_flag'] == 1 and record['resolved_flag'] == 1 and
              record['postmortem_flag'] == 0):
            count_dict['S1_S2_S3_S4'] = count_dict.get('S1_S2_S3_S4', 0) + 1
        # S2_S3_S4
        elif (record['identified_flag'] == 1 and record['monitoring_flag'] == 1 and
              record['resolved_flag'] == 1 and record['investigating_flag'] == 0 and
              record['postmortem_flag'] == 0):
            count_dict['S2_S3_S4'] = count_dict.get('S2_S3_S4', 0) + 1
        # S1_S2_S4
        elif (record['investigating_flag'] == 1 and record['identified_flag'] == 1 and
              record['resolved_flag'] == 1 and record['monitoring_flag'] == 0 and
              record['postmortem_flag'] == 0):
            count_dict['S1_S2_S4'] = count_dict.get('S1_S2_S4', 0) + 1
        # S2_S4
        elif (record['identified_flag'] == 1 and record['resolved_flag'] == 1 and
              record['investigating_flag'] == 0 and record['monitoring_flag'] == 0 and
              record['postmortem_flag'] == 0):
            count_dict['S2_S4'] = count_dict.get('S2_S4', 0) + 1
        # S4
        elif (record['resolved_flag'] == 1 and record['investigating_flag'] == 0 and
              record['identified_flag'] == 0 and record['monitoring_flag'] == 0 and
              record['postmortem_flag'] == 0):
            count_dict['S4'] = count_dict.get('S4', 0) + 1
        # S3_S4
        elif (record['monitoring_flag'] == 1 and record['resolved_flag'] == 1 and
              record['investigating_flag'] == 0 and record['identified_flag'] == 0 and
              record['postmortem_flag'] == 0):
            count_dict['S3_S4'] = count_dict.get('S3_S4', 0) + 1
        # All_with_S5 (sum of all cases where postmortem_flag == 1)
        elif record['postmortem_flag'] == 1:
            count_dict['All_with_S5'] = count_dict.get('All_with_S5', 0) + 1
        else:
            # Handle any other combinations not explicitly listed
            count_dict['Other'] = count_dict.get('Other', 0) + 1

    # Calculate total count
    sum_all = sum(count_dict.values())

    # Sort the count dictionary
    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))

    # Transform to DataFrame
    count_df = pd.DataFrame(count_dict.items(), columns=['case', 'count'])
    count_df['percentage'] = count_df['count'] / sum_all
    count_df['percentage'] = count_df['percentage'].map(lambda x: '{:.2%}'.format(x))

    # Add total row
    sum_all_row = pd.DataFrame({'case': 'sum_all', 'count': sum_all, 'percentage': '100.00%'}, index=[0])
    count_df = pd.concat([count_df, sum_all_row], axis=0).reset_index(drop=True)

    # Split 'case' into individual status flags if needed
    # (This step is optional based on your presentation needs)

    return count_df




def generate_service_status_table(df, service_lst):
    dfs = {service: df[df[service] == 1] for service in service_lst}
    count_dict_lst = {}

    for service in service_lst:
        count_dict = {}
        service_df = dfs[service]
        for index, record in service_df.iterrows():
            # Define all cases as per your original code
            # S1_S3_S4
            if (record['investigating_flag'] == 1 and record['monitoring_flag'] == 1 and
                record['resolved_flag'] == 1 and record['identified_flag'] == 0 and
                record['postmortem_flag'] == 0):
                count_dict['S1_S3_S4'] = count_dict.get('S1_S3_S4', 0) + 1
            # S1_S4
            elif (record['investigating_flag'] == 1 and record['resolved_flag'] == 1 and
                  record['identified_flag'] == 0 and record['monitoring_flag'] == 0 and
                  record['postmortem_flag'] == 0):
                count_dict['S1_S4'] = count_dict.get('S1_S4', 0) + 1
            # S1_S2_S3_S4
            elif (record['investigating_flag'] == 1 and record['identified_flag'] == 1 and
                  record['monitoring_flag'] == 1 and record['resolved_flag'] == 1 and
                  record['postmortem_flag'] == 0):
                count_dict['S1_S2_S3_S4'] = count_dict.get('S1_S2_S3_S4', 0) + 1
            # S2_S3_S4
            elif (record['identified_flag'] == 1 and record['monitoring_flag'] == 1 and
                  record['resolved_flag'] == 1 and record['investigating_flag'] == 0 and
                  record['postmortem_flag'] == 0):
                count_dict['S2_S3_S4'] = count_dict.get('S2_S3_S4', 0) + 1
            # S1_S2_S4
            elif (record['investigating_flag'] == 1 and record['identified_flag'] == 1 and
                  record['resolved_flag'] == 1 and record['monitoring_flag'] == 0 and
                  record['postmortem_flag'] == 0):
                count_dict['S1_S2_S4'] = count_dict.get('S1_S2_S4', 0) + 1
            # S2_S4
            elif (record['identified_flag'] == 1 and record['resolved_flag'] == 1 and
                  record['investigating_flag'] == 0 and record['monitoring_flag'] == 0 and
                  record['postmortem_flag'] == 0):
                count_dict['S2_S4'] = count_dict.get('S2_S4', 0) + 1
            # S4
            elif (record['resolved_flag'] == 1 and record['investigating_flag'] == 0 and
                  record['identified_flag'] == 0 and record['monitoring_flag'] == 0 and
                  record['postmortem_flag'] == 0):
                count_dict['S4'] = count_dict.get('S4', 0) + 1
            # S3_S4
            elif (record['monitoring_flag'] == 1 and record['resolved_flag'] == 1 and
                  record['investigating_flag'] == 0 and record['identified_flag'] == 0 and
                  record['postmortem_flag'] == 0):
                count_dict['S3_S4'] = count_dict.get('S3_S4', 0) + 1
            # All_with_S5 (sum of all cases where postmortem_flag == 1)
            elif record['postmortem_flag'] == 1:
                count_dict['All_with_S5'] = count_dict.get('All_with_S5', 0) + 1
            else:
                # Handle any other combinations not explicitly listed
                count_dict['Other'] = count_dict.get('Other', 0) + 1

        # Transform to DataFrame
        count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
        count_df = pd.DataFrame(count_dict.items(), columns=['case', f'{service}'])
        count_dict_lst[service] = count_df

    # Merge all service dataframes
    case_df = count_dict_lst[service_lst[0]]
    for service in service_lst[1:]:
        case_df = pd.merge(case_df, count_dict_lst[service], on='case', how='outer')

    # Replace NaN values with zeros and convert to integers
    case_df = case_df.fillna(0)
    for service in service_lst:
        case_df[service] = case_df[service].astype(int)

    # Calculate the sum for each case
    case_df['sum'] = case_df.iloc[:, 1:].sum(axis=1)

    # Sort the DataFrame by the 'sum' column in descending order
    case_df = case_df.sort_values(by='sum', ascending=False).reset_index(drop=True)

    # Transpose the DataFrame so that services are rows and cases are columns
    case_df = case_df.set_index('case').T.reset_index().rename(columns={'index': 'service'})

    # Reorder columns to match your original table if necessary
    desired_order = ['service', 'S1_S3_S4', 'S1_S4', 'S1_S2_S3_S4', 'S2_S3_S4',
                     'S1_S2_S4', 'S2_S4', 'S4', 'S3_S4', 'All_with_S5', 'Other']
    case_df = case_df.reindex(columns=[col for col in desired_order if col in case_df.columns])

    return case_df




def generate_service_percentage_table(case_df):
    # Calculate the sum for each service, excluding the 'service' column
    case_df['sum_all'] = case_df.drop(columns=['service']).sum(axis=1)
    
    # Convert counts to percentages
    for column in case_df.columns[1:]:
        if column != 'sum_all':
            case_df[column] = case_df[column] / case_df['sum_all']
    
    # Drop the 'sum_all' column
    case_df = case_df.drop(columns='sum_all')
    
    # Round the percentages for readability
    case_df = case_df.round(4)
    
    return case_df


def generate_stacked_bar_plot(case_df):
    fig, ax = plt.subplots(figsize=(16, 10))
    barWidth = 0.85
    bottom = [0] * len(case_df)
    colors = sns.color_palette('tab10', n_colors=len(case_df.columns[1:]))

    for column in case_df.columns[1:]:
        ax.bar(case_df['service'], case_df[column], bottom=bottom, label=column, color=colors.pop(0), width=barWidth)
        for index, value in enumerate(case_df[column]):
            if value > 0:
                ax.text(index, bottom[index] + value / 2, f'{value:.2%}', ha='center', va='center', color='white', fontsize=14)
        bottom = [sum(x) for x in zip(bottom, case_df[column])]

    labels = [
        'S1-S3-S4', 'S1-S4', 'S1-S2-S3-S4', 
        'S2-S3-S4', 'S1-S2-S4', 'S2-S4',
        'S4', 'S3-S4', 'All-with-S5'
    ]
    reversed_labels = labels[::-1]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed_labels, title='Status Cases', bbox_to_anchor=(1.01, 1.0), loc='upper left')
    ax.set_xlabel('Service', fontsize=20)
    ax.set_ylabel('Percentage', fontsize=20)

    y_tick_lst = ['0%', '20%', '40%', '60%', '80%', '100%']
    ax.set_yticklabels(y_tick_lst, fontsize=16)

    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.show()


# Single Function Wrapping All Functionality
def incident_status_count(start_date, end_date, service_lst):
    path = 'server/static/data/incident_stages.csv'

    # Load and preprocess data
    df = load_and_preprocess_data(path, start_date, end_date)

    # Filter corner cases
    df = filter_corner_cases(df)

    # Generate and print the status count table
    count_df = generate_status_count_table(df)
    print(count_df)

    # Generate incident status count by service
    case_df = generate_service_status_table(df, service_lst)
    print(case_df)

    # Generate percentage table
    case_percentage_df = generate_service_percentage_table(case_df)
    print(case_percentage_df)

    # Generate stacked bar plot
    generate_stacked_bar_plot(case_percentage_df)

def mean_model_params(start_date, end_date, service_list):
    # Step 1: Load and preprocess the data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)

    # Step 2: Calculate durations for each stage and add new columns
    dfs = {}
    for service in service_list:
        dfs[service] = df[df[service] == 1]
        # Order by start_timestamp and reset index
        dfs[service] = dfs[service].sort_values(by='start_timestamp').reset_index(drop=True)

    for service in service_list:
        dfs[service]['MTBF'] = dfs[service]['start_timestamp'].diff()
        dfs[service]['P_investigate'] = dfs[service]['identified_timestamp'] - dfs[service]['investigating_timestamp']
        dfs[service]['P_repair'] = dfs[service]['monitoring_timestamp'] - dfs[service]['identified_timestamp']
        dfs[service]['P_check'] = dfs[service]['resolved_timestamp'] - dfs[service]['monitoring_timestamp']
        dfs[service]['P_learn'] = dfs[service]['postmortem_timestamp'] - dfs[service]['resolved_timestamp']
        dfs[service]['MTTR'] = dfs[service]['resolved_timestamp'] - dfs[service][['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        dfs[service] = dfs[service].reset_index(drop=True)

    # Step 3: Calculate mean durations for model parameters by service
    duration_dict = {}
    for service in service_list:
        duration_dict[service] = {}
        for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn', 'MTTR', 'MTBF']:
            duration_dict[service][period] = dfs[service][period].mean()
    duration_df = pd.DataFrame(duration_dict).T.reset_index().rename(columns={'index': 'service'})

    # Transform durations to hours
    for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn', 'MTTR', 'MTBF']:
        duration_df[period] = duration_df[period].map(lambda x: x.total_seconds() / 60 / 60 if pd.notnull(x) else np.nan)

    # Step 4: Add rows of arithmetic mean and geometric mean
    df_all = pd.concat([dfs[service] for service in service_list], axis=0).reset_index(drop=True)
    arith_mean_row = {'service': 'Arith. Mean'}
    geom_mean_row = {'service': 'Geom. Mean'}

    for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn', 'MTTR', 'MTBF']:
        # Arithmetic Mean
        arith_mean = df_all[period].mean()
        arith_mean_row[period] = arith_mean.total_seconds() / 60 / 60 if pd.notnull(arith_mean) else np.nan

        # Geometric Mean
        df_all_period = df_all[period].dropna().map(lambda x: x.total_seconds() / 60 / 60)
        if len(df_all_period) > 0:
            geom_mean = gmean(df_all_period)
            geom_mean_row[period] = geom_mean
        else:
            geom_mean_row[period] = np.nan

    # Convert mean rows to DataFrame and concatenate
    arith_mean_df = pd.DataFrame([arith_mean_row])
    geom_mean_df = pd.DataFrame([geom_mean_row])
    duration_df = pd.concat([duration_df, arith_mean_df, geom_mean_df], ignore_index=True)

    # Transform P_learn and MTBF to days for reporting
    duration_df['P_learn'] = duration_df['P_learn'] / 24
    duration_df['MTBF'] = duration_df['MTBF'] / 24

    # Round the results for better readability
    duration_df = duration_df.round(2)

    # Step 5: Save the resulting DataFrame to a CSV file
    output_path = 'tables/table-mean-duration-of-model-parameters-by-service.csv'
    duration_df.to_csv(output_path, index=False)
    print(f"Table saved to: {output_path}")

import pandas as pd

def check_incidents_for_services(csv_path):
    """
    Reads the CSV file at the given path and checks for incidents across all services.
    Reports the number of incidents and the time range for each service.

    Parameters:
    - csv_path (str): The file path to the CSV file.

    Returns:
    - results (dict): A dictionary containing incident information for each service.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert timestamp columns to datetime
    status_lst = ['investigating', 'identified', 'monitoring', 'resolved', 'postmortem', 'start']
    for status in status_lst:
        timestamp_col = f'{status}_timestamp'
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

    # Map the service names to their column names in the CSV
    service_columns = {
        'API-OpenAI': 'API',
        'ChatGPT': 'ChatGPT',
        'DALL·E': 'Labs',
        'Playground': 'Playground',
        'API-Anthropic': 'api.anthropic.com',
        'Claude': 'claude.ai',
        'Console': 'console.anthropic.com',
        'Character.AI': 'Character.AI'
    }

    # Initialize a dictionary to hold results
    results = {}

    for service_name, column_name in service_columns.items():
        if column_name in df.columns:
            # Check if there are any incidents for this service
            service_df = df[df[column_name] == 1]
            if not service_df.empty:
                # There are incidents for this service
                min_date = service_df['investigating_timestamp'].min()
                max_date = service_df['postmortem_timestamp'].max()
                results[service_name] = {
                    'incidents': len(service_df),
                    'start_date': min_date,
                    'end_date': max_date
                }
            else:
                # No incidents for this service
                results[service_name] = {
                    'incidents': 0,
                    'start_date': None,
                    'end_date': None
                }
        else:
            # Service column not found in the CSV
            results[service_name] = {
                'incidents': 'Column not found',
                'start_date': None,
                'end_date': None
            }

    # Display the results
    for service, info in results.items():
        print(f"Service: {service}")
        if info['incidents'] == 'Column not found':
            print("  - Status: Column not found in the CSV file.")
        elif info['incidents'] == 0:
            print("  - Status: No incidents found.")
        else:
            print(f"  - Total Incidents: {info['incidents']}")
            print(f"  - Time Range: {info['start_date']} to {info['end_date']}")
        print()

    return results

def plot_mttr_percentages(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)
    
    # Filter corner cases
    df = filter_corner_cases(df)
    
    # Calculate the duration of each stage and add new columns
    df['P_investigate'] = df['identified_timestamp'] - df['investigating_timestamp']
    df['P_repair'] = df['monitoring_timestamp'] - df['identified_timestamp']
    df['P_check'] = df['resolved_timestamp'] - df['monitoring_timestamp']
    df['P_learn'] = df['postmortem_timestamp'] - df['resolved_timestamp']
    
    # Create a dictionary to hold DataFrames for each service
    dfs = {}
    for service in service_list:
        dfs[service] = df[df[service] == 1]
        # Order by start_timestamp and reset index
        dfs[service] = dfs[service].sort_values(by='start_timestamp').reset_index(drop=True)
        
    # Calculate the sum of each period for each service in hours
    sum_dict = {}
    for service in service_list:
        sum_dict[service] = {}
        for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn']:
            total_duration = dfs[service][period].sum()
            if pd.notnull(total_duration) and total_duration != pd.Timedelta(0):
                sum_dict[service][period] = total_duration.total_seconds() / 3600
            else:
                sum_dict[service][period] = 0.0  # Handle missing data

    # Create sum_df
    sum_df = pd.DataFrame(sum_dict).T.reset_index().rename(columns={'index': 'service'})
    # Format the periods to two decimal places
    for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn']:
        sum_df[period] = sum_df[period].map(lambda x: '{:.2f}'.format(float(x)))
    # Display sum_df
    print('Sum of each period for each service [hours]')
    print(sum_df)
    
    # Calculate the percentage of each period in the total incident duration for each service
    percent_dict = {}
    for service in service_list:
        percent_dict[service] = {}
        total_duration = sum([float(sum_df.loc[sum_df['service'] == service, period].values[0]) for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn']])
        if total_duration > 0:
            for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn']:
                period_duration = float(sum_df.loc[sum_df['service'] == service, period].values[0])
                percent_dict[service][period] = period_duration / total_duration
        else:
            for period in ['P_investigate', 'P_repair', 'P_check', 'P_learn']:
                percent_dict[service][period] = np.nan  # Handle missing data
    
    percent_df = pd.DataFrame(percent_dict).T.reset_index().rename(columns={'index': 'service'})
    percent_df = percent_df.round(4)
    print('Percentage of each period in the total incident duration for each service')
    print(percent_df)
    
    # Calculate the percentage of each period in MTTR duration for each service (excluding P_learn)
    percent_dict_MTTR = {}
    for service in service_list:
        percent_dict_MTTR[service] = {}
        mttr_duration = sum([float(sum_df.loc[sum_df['service'] == service, period].values[0]) for period in ['P_investigate', 'P_repair', 'P_check']])
        if mttr_duration > 0:
            for period in ['P_investigate', 'P_repair', 'P_check']:
                period_duration = float(sum_df.loc[sum_df['service'] == service, period].values[0])
                percent_dict_MTTR[service][period] = period_duration / mttr_duration
        else:
            for period in ['P_investigate', 'P_repair', 'P_check']:
                percent_dict_MTTR[service][period] = np.nan  # Handle missing data
    
    percent_df_MTTR = pd.DataFrame(percent_dict_MTTR).T.reset_index().rename(columns={'index': 'service'})
    percent_df_MTTR = percent_df_MTTR.round(4)
    print('Percentage of each period in MTTR duration for each service')
    print(percent_df_MTTR)
    
    # Plot the stacked bar plot
    print('Figure: Percent of MTTR for each service')
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    width = 0.7
    x = np.arange(len(service_list))
    bottom = np.zeros(len(service_list))
    colors = sns.color_palette('Set2', n_colors=3)
    
    for index, period in enumerate(['P_investigate', 'P_repair', 'P_check']):
        y_values = percent_df_MTTR[period].fillna(0).values.astype(float)
        ax.bar(x, y_values, width, label=period, bottom=bottom, color=colors[index])
        # Add percentage text in the middle of each bar
        for i, percent in enumerate(y_values):
            if not np.isnan(percent) and percent > 0:
                ax.text(i, bottom[i] + percent / 2, f'{percent:.2%}', ha='center', va='center', color='black', fontsize=12)
        bottom += y_values
    
    ax.set_ylabel('Percentage')
    ax.set_xticks(x)
    x_tick_service_lst = [service.replace('-', '\n') for service in service_list]
    ax.set_xticklabels(x_tick_service_lst, rotation=30, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Investigating', 'Repairing', 'Checking']
    ax.legend(labels=labels, bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=3)
    
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    y_tick_lst = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(y_tick_lst)
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=12)
    ax.set_ylabel('Percent of MTTR', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('stacked-bar-percent-of-MTTR.pdf')

def plot_mttr_ecdf(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)
    
    # Filter corner cases
    df = filter_corner_cases(df)
    
    # Create a dictionary to hold DataFrames for each service
    dfs = {}
    for service in service_list:
        dfs[service] = df[df[service] == 1]
        # Order by start_timestamp and reset index
        dfs[service] = dfs[service].sort_values(by='start_timestamp').reset_index(drop=True)
    
    # Calculate MTBF, MTTR, and durations of each stage for each service
    for service in service_list:
        service_df = dfs[service]
        service_df['MTBF'] = service_df['start_timestamp'].diff()
        service_df['P_investigate'] = service_df['identified_timestamp'] - service_df['investigating_timestamp']
        service_df['P_repair'] = service_df['monitoring_timestamp'] - service_df['identified_timestamp']
        service_df['P_check'] = service_df['resolved_timestamp'] - service_df['monitoring_timestamp']
        service_df['P_learn'] = service_df['postmortem_timestamp'] - service_df['resolved_timestamp']
        # Calculate MTTR as the time from the earliest known stage to 'resolved_timestamp'
        service_df['MTTR'] = service_df['resolved_timestamp'] - service_df[['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        dfs[service] = service_df.reset_index(drop=True)
    
    # Prepare data for plotting MTTR
    dfs_MTTR = {}
    for service in service_list:
        mttr_series = dfs[service]['MTTR'].dropna().dt.total_seconds() / 3600  # Convert to hours
        dfs_MTTR[service] = mttr_series
    
    # Plotting MTTR ECDF grouped by service
    print('MTTR grouped by service [hours], ECDF plot.')
    plt.figure(figsize=(16, 8))
    colors = sns.color_palette('Set2', n_colors=len(service_list))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', '*']
    
    for idx, service in enumerate(service_list):
        sns.ecdfplot(data=dfs_MTTR[service], label=service, color=colors[idx], linestyle='-', linewidth=2)
    
    # Add vertical lines at specific MTTR durations
    critical_points = [0.5, 1, 3, 10, 24]  # In hours
    for cp in critical_points:
        plt.axvline(x=cp, color='black', linestyle='--', linewidth=1)
        plt.text(cp, 1.01, f'{cp}h', color='black', fontsize=12, ha='center', va='bottom')
    
    plt.xlabel('MTTR [hours]', fontsize=14)
    plt.xscale('log')
    plt.ylabel('ECDF', fontsize=14)
    plt.legend(title='Service', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('MTTR_grouped_by_service_ecdf.png')
    
    # If you want to return the MTTR data, uncomment the following line
    # return dfs_MTTR

def plot_mtbf_ecdf_provider(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)
    
    # Filter corner cases
    df = filter_corner_cases(df)
    
    # Create a dictionary to hold DataFrames for each service
    dfs = {}
    for service in service_list:
        dfs[service] = df[df[service] == 1]
        # Order by start_timestamp and reset index
        dfs[service] = dfs[service].sort_values(by='start_timestamp').reset_index(drop=True)
    
    # Calculate MTBF for each service
    dfs_MTBF = {}
    for service in service_list:
        service_df = dfs[service]
        # Calculate MTBF as the difference between the start_timestamp of consecutive incidents
        service_df['MTBF'] = service_df['start_timestamp'].diff()
        # Convert MTBF to days
        mtbf_series = service_df['MTBF'].dropna().dt.total_seconds() / (3600 * 24)
        dfs_MTBF[service] = mtbf_series
    
    # Prepare data for percentages
    percent_dict = {}
    thresholds = [12/24, 1, 3, 7, 30]  # Thresholds in days
    for service in service_list:
        percent_dict[service] = {}
        mtbf_series = dfs_MTBF[service]
        for threshold in thresholds:
            percent_within_threshold = (mtbf_series <= threshold).sum() / len(mtbf_series)
            percent_dict[service][threshold] = percent_within_threshold
    percent_df = pd.DataFrame(percent_dict).T.reset_index().rename(columns={'index': 'service'})
    
    # Add percentage of MTBF over 1 month
    over_one_month_dict = {}
    for service in service_list:
        mtbf_series = dfs_MTBF[service]
        percent_over_one_month = 1 - (mtbf_series <= 30).sum() / len(mtbf_series)
        over_one_month_dict[service] = percent_over_one_month
    percent_df['>1 month'] = over_one_month_dict.values()
    percent_df = percent_df.round(4)
    print('Percentage of MTBF within thresholds for each service:')
    print(percent_df)
    
    # Group MTBF by provider
    provider_list = ['openai', 'anthropic', 'character']
    dfs_provider = {}
    dfs_provider_MTBF = {}
    for provider in provider_list:
        provider_df = df[df['provider'] == provider]
        # Order by start_timestamp and reset index
        provider_df = provider_df.sort_values(by='start_timestamp').reset_index(drop=True)
        # Calculate MTBF for the provider
        provider_df['MTBF'] = provider_df['start_timestamp'].diff()
        mtbf_series = provider_df['MTBF'].dropna().dt.total_seconds() / (3600 * 24)
        dfs_provider_MTBF[provider] = mtbf_series
        dfs_provider[provider] = provider_df
    
    # Plot MTBF ECDF grouped by provider
    print('MTBF grouped by provider [days], ECDF plot.')
    plt.figure(figsize=(10, 7))
    colors = sns.color_palette('bright', n_colors=len(provider_list))
    markers = ['^', 'o', 's']
    
    for idx, provider in enumerate(provider_list):
        sns.ecdfplot(
            data=dfs_provider_MTBF[provider],
            label=provider.capitalize(),
            color=colors[idx],
            linestyle='-',
            linewidth=2
        )
    
    # Add vertical lines at specific MTBF durations
    critical_points = [1, 7, 30]  # In days
    for cp in critical_points:
        plt.axvline(x=cp, color='black', linestyle='--', linewidth=1.5)
        plt.text(cp, 1.08, f'{int(cp)}d', color='black', fontsize=12, ha='center', va='top')
    
    plt.xlabel('MTBF [days]', fontsize=14)
    plt.xscale('log')
    plt.ylabel('ECDF', fontsize=14)
    plt.legend(title='Provider', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('MTBF_grouped_by_provider_ecdf.png')

def calculate_and_plot_mttr_percentages(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)
    
    # Filter corner cases
    df = filter_corner_cases(df)
    
    # Create a dictionary to hold DataFrames for each service
    dfs = {}
    for service in service_list:
        dfs[service] = df[df[service] == 1]
        # Order by start_timestamp and reset index
        dfs[service] = dfs[service].sort_values(by='start_timestamp').reset_index(drop=True)
    
    # Calculate MTTR for each service
    dfs_MTTR = {}
    for service in service_list:
        service_df = dfs[service]
        # Calculate MTTR as the time from the earliest known stage to 'resolved_timestamp'
        service_df['MTTR'] = service_df['resolved_timestamp'] - service_df[['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        # Convert MTTR to hours
        mttr_series = service_df['MTTR'].dropna().dt.total_seconds() / 3600
        dfs_MTTR[service] = mttr_series
    
    # Calculate percentage of MTTR occurrences within specified thresholds for each service
    print('MTTR percent at 30 mins, 1 hour, 3 hours, 10 hours, 1 day.')
    thresholds = [0.5, 1, 3, 10, 24]  # Thresholds in hours
    percent_dict = {}
    for service in service_list:
        mttr_series = dfs_MTTR[service]
        percent_dict[service] = {}
        for threshold in thresholds:
            percent_within_threshold = (mttr_series <= threshold).sum() / len(mttr_series)
            percent_dict[service][threshold] = percent_within_threshold
    percent_df = pd.DataFrame(percent_dict).T.reset_index().rename(columns={'index': 'service'})
    
    # Add percentage of MTTR over 1 day
    over_one_day_dict = {}
    for service in service_list:
        mttr_series = dfs_MTTR[service]
        percent_over_one_day = 1 - (mttr_series <= 24).sum() / len(mttr_series)
        over_one_day_dict[service] = percent_over_one_day
    percent_df['>1 day'] = over_one_day_dict.values()
    percent_df = percent_df.round(4)
    print('Percentage of MTTR within thresholds for each service:')
    print(percent_df)
    
    # Group MTTR by provider
    provider_list = ['openai', 'anthropic', 'character']
    dfs_provider = {}
    dfs_provider_MTTR = {}
    for provider in provider_list:
        provider_df = df[df['provider'] == provider]
        # Order by start_timestamp and reset index
        provider_df = provider_df.sort_values(by='start_timestamp').reset_index(drop=True)
        # Calculate MTTR
        provider_df['MTTR'] = provider_df['resolved_timestamp'] - provider_df[['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        mttr_series = provider_df['MTTR'].dropna().dt.total_seconds() / 3600  # Convert to hours
        dfs_provider_MTTR[provider] = mttr_series
        dfs_provider[provider] = provider_df
    
    # Plot MTTR ECDF grouped by provider
    print('MTTR grouped by provider [hours], ECDF plot.')
    plt.figure(figsize=(10, 7))
    colors = sns.color_palette('bright', n_colors=len(provider_list))
    markers = ['^', 'o', 's']
    
    for idx, provider in enumerate(provider_list):
        sns.ecdfplot(
            data=dfs_provider_MTTR[provider],
            label=provider.capitalize(),
            color=colors[idx],
            linestyle='-',
            linewidth=2
        )
    
    # Add vertical lines at specific MTTR durations
    critical_points = [10/60, 0.5, 3, 10, 24]  # Thresholds in hours
    labels_cp = ['10m', '0.5h', '3h', '10h', '24h']
    for cp, label in zip(critical_points, labels_cp):
        plt.axvline(x=cp, color='black', linestyle='--', linewidth=1.5)
        plt.text(cp, 1.08, label, color='black', fontsize=12, ha='center', va='top')
    
    plt.xlabel('MTTR [hours]', fontsize=14)
    plt.xscale('log')
    plt.ylabel('ECDF', fontsize=14)
    plt.legend(title='Provider', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('MTTR_grouped_by_provider_ecdf.png')
    
    # If you want to return the MTTR data, uncomment the following line
    # return percent_df, dfs_MTTR

def box_plot_mttr_by_service(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)
    
    # Filter corner cases
    df = filter_corner_cases(df)
    
    # Create DataFrames per service
    dfs = {}
    for service in service_list:
        service_df = df[df[service] == 1]
        service_df = service_df.sort_values(by='start_timestamp').reset_index(drop=True)
        dfs[service] = service_df
    
    # Calculate MTTR per service
    dfs_MTTR = {}
    for service in service_list:
        service_df = dfs[service]
        # Calculate MTTR as the time from the earliest known stage to 'resolved_timestamp'
        service_df['MTTR'] = service_df['resolved_timestamp'] - service_df[['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        # Convert MTTR to hours
        mttr_series = service_df['MTTR'].dropna().dt.total_seconds() / 3600  # Convert to hours
        dfs_MTTR[service] = mttr_series
        # Update dfs[service] with MTTR
        dfs[service] = service_df
    
    # Define provider list
    provider_list = ['openai', 'anthropic', 'character']
    
    # Create MTTR data per provider
    dfs_provider_MTTR = {}
    for provider in provider_list:
        provider_df = df[df['provider'] == provider]
        provider_df = provider_df.sort_values(by='start_timestamp').reset_index(drop=True)
        # Calculate MTTR
        provider_df['MTTR'] = provider_df['resolved_timestamp'] - provider_df[['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        mttr_series = provider_df['MTTR'].dropna().dt.total_seconds() / 3600  # Convert to hours
        dfs_provider_MTTR[provider] = mttr_series
    
    # Calculate MTTR percentages at thresholds per provider
    print('MTTR percent at 10 mins, 30 mins, 1 hour, 3 hours, 10 hours, 1 day.')
    thresholds = [10/60, 30/60, 1, 3, 10, 24]  # Thresholds in hours
    percent_dict = {}
    for provider in provider_list:
        percent_dict[provider] = {}
        mttr_series = dfs_provider_MTTR[provider]
        for threshold in thresholds:
            percent_within_threshold = (mttr_series <= threshold).sum() / len(mttr_series)
            percent_dict[provider][threshold] = percent_within_threshold
    # Add percentage of MTTR over 1 day
    over_one_day_dict = {}
    for provider in provider_list:
        mttr_series = dfs_provider_MTTR[provider]
        over_one_day_dict[provider] = 1 - (mttr_series <= 24).sum() / len(mttr_series)
    percent_df = pd.DataFrame(percent_dict).T.reset_index().rename(columns={'index': 'provider'})
    percent_df['>1 day'] = over_one_day_dict.values()
    percent_df = percent_df.round(4)
    print('Percentage of MTTR within thresholds for each provider:')
    print(percent_df)
    
    # Plotting horizontal box plot by service, MTTR
    print('Horizontal box plot by service, MTTR.')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = [service.replace('-', '\n') for service in service_list]
    flierprops = dict(marker='D', markersize=3)
    # Convert dfs_MTTR to DataFrame
    df_mttr = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dfs_MTTR.items()]))
    sns.boxplot(data=df_mttr, orient='h', ax=ax, color='white', linewidth=2, flierprops=flierprops)
    for patch in ax.artists:
        patch.set_edgecolor('black')   # Set the edge color to black
        patch.set_linewidth(2)         # Set the linewidth for the edges
    # Add text annotation for the median
    for i, service in enumerate(service_list):
        x = dfs_MTTR[service].median()
        ax.text(x, i, f'{x:.2f}', color='black', va='center', ha='center', fontsize=12)
    # Add lines at certain x positions
    thresholds_lines = [10/60, 30/60, 3, 10, 24]  # in hours
    labels_thresholds = ['10m', '0.5h', '3h', '10h', '24h']
    for cp, label in zip(thresholds_lines, labels_thresholds):
        ax.axvline(x=cp, color='black', linestyle='--', linewidth=1)
        ax.text(cp, -0.5, label, color='black', fontsize=12, ha='center', va='top')
    
    ax.set_yticks(range(len(service_list)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('MTTR [hours]', fontsize=14)
    ax.grid(axis='both', linestyle='--', alpha=0.6, which='both')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('horizontal-box-plot-by-service-MTTR.png')


def plot_mtbf_ecdf_by_service(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)
    
    # Filter corner cases
    df = filter_corner_cases(df)
    
    # Rename services in DataFrame
    df = df.rename(columns={
        'API': 'API-OpenAI',
        'ChatGPT': 'ChatGPT',
        'Labs': 'DALL·E',
        'Playground': 'Playground',
        'api.anthropic.com': 'API-Anthropic',
        'claude.ai': 'Claude',
        'console.anthropic.com': 'Console',
        'Character.AI': 'Character.AI'
    })
    
    # Create DataFrames per service
    dfs = {}
    for service in service_list:
        service_df = df[df[service] == 1]
        service_df = service_df.sort_values(by='start_timestamp').reset_index(drop=True)
        dfs[service] = service_df
    
    # Calculate MTBF and MTTR per service
    dfs_MTBF = {}
    for service in service_list:
        service_df = dfs[service]
        service_df['MTBF'] = service_df['start_timestamp'].diff()
        service_df['MTTR'] = service_df['resolved_timestamp'] - service_df[['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp']].min(axis=1)
        # Convert MTBF to days
        mtbf_series = service_df['MTBF'].dropna().dt.total_seconds() / (3600 * 24)
        dfs_MTBF[service] = mtbf_series
        # Update dfs[service] with MTBF and MTTR
        dfs[service] = service_df
    
    # Create DataFrames per provider
    provider_list = ['openai', 'anthropic', 'character']
    dfs_provider = {}
    for provider in provider_list:
        provider_df = pd.DataFrame()
        for service in service_list:
            service_df = dfs[service]
            if provider in service_df['provider'].unique():
                provider_service_df = service_df[service_df['provider'] == provider]
                provider_df = pd.concat([provider_df, provider_service_df], axis=0)
        provider_df = provider_df.sort_values(by='start_timestamp').reset_index(drop=True)
        dfs_provider[provider] = provider_df
    
    # Calculate MTBF per provider
    dfs_provider_MTBF = {}
    for provider in provider_list:
        provider_df = dfs_provider[provider]
        provider_df['MTBF'] = provider_df['start_timestamp'].diff()
        # Convert MTBF to days
        mtbf_series = provider_df['MTBF'].dropna().dt.total_seconds() / (3600 * 24)
        dfs_provider_MTBF[provider] = mtbf_series
        # Update dfs_provider[provider] with MTBF
        dfs_provider[provider] = provider_df
    
    # Calculate average MTBF for each service
    print('Average MTBF for each service [days]:')
    mtbf_dict = {}
    for service in service_list:
        mtbf_mean = dfs_MTBF[service].mean()
        mtbf_dict[service] = mtbf_mean
    mtbf_df = pd.DataFrame(mtbf_dict.items(), columns=['service', 'MTBF'])
    mtbf_df = mtbf_df.round(2)
    print(mtbf_df)
    
    # Calculate average MTBF for each provider
    print('Average MTBF for each provider [days]:')
    mtbf_provider_dict = {}
    for provider in provider_list:
        mtbf_mean = dfs_provider_MTBF[provider].mean()
        mtbf_provider_dict[provider] = mtbf_mean
    mtbf_provider_df = pd.DataFrame(mtbf_provider_dict.items(), columns=['provider', 'MTBF'])
    mtbf_provider_df = mtbf_provider_df.round(2)
    print(mtbf_provider_df)
    
    # Plot MTBF ECDF grouped by service
    print('MTBF grouped by service [days], ECDF plot.')
    plt.figure(figsize=(16, 8))
    colors = sns.color_palette('Set2', n_colors=len(service_list))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', '*']
    
    for idx, service in enumerate(service_list):
        sns.ecdfplot(
            data=dfs_MTBF[service],
            label=service.replace('-', ' '),
            color=colors[idx],
            linestyle='-',
            linewidth=3
        )
    
    # Add vertical lines at specific MTBF durations
    critical_points = [12/24, 1, 3, 7, 30]  # In days
    labels_cp = ['0.5d', '1d', '3d', '7d', '30d']
    for cp, label in zip(critical_points, labels_cp):
        plt.axvline(x=cp, color='black', linestyle='--', linewidth=1)
        plt.text(cp, 1.01, label, color='black', fontsize=12, ha='center', va='bottom')
    
    plt.xlabel('MTBF [days]', fontsize=14)
    plt.xscale('log')
    plt.ylabel('ECDF', fontsize=14)
    plt.legend(title='Service', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('MTBF_grouped_by_service_ecdf.png')
    
    # If you want to return the MTBF data, uncomment the following lines
    # return mtbf_df, mtbf_provider_df, dfs_MTBF, dfs_provider_MTBF

def plot_mtbf_boxplot_by_service(start_date, end_date, service_list):
    # Load and preprocess data
    path = 'server/static/data/incident_stages.csv'
    df = load_and_preprocess_data(path, start_date, end_date)

    # Filter corner cases
    df = filter_corner_cases(df)

    # Create DataFrames per service
    dfs = {}
    for service in service_list:
        service_df = df[df[service] == 1]
        service_df = service_df.sort_values(by='start_timestamp').reset_index(drop=True)
        dfs[service] = service_df

    # Calculate MTBF per service
    dfs_MTBF = {}
    for service in service_list:
        service_df = dfs[service]
        service_df['MTBF'] = service_df['start_timestamp'].diff()
        # Convert MTBF to days
        mtbf_series = service_df['MTBF'].dropna().dt.total_seconds() / (3600 * 24)
        dfs_MTBF[service] = mtbf_series
        # Update dfs[service] with MTBF
        dfs[service] = service_df

    # Prepare data for plotting
    print('Horizontal box plot by service, MTBF.')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = [service.replace('-', '\n') for service in service_list]
    flierprops = dict(marker='D', markersize=3)
    # Convert dfs_MTBF to DataFrame
    df_mtbf = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dfs_MTBF.items()]))

    sns.boxplot(data=df_mtbf, orient='h', ax=ax, color='white', linewidth=2, flierprops=flierprops)
    for patch in ax.artists:
        patch.set_edgecolor('black')   # Set the edge color to black
        patch.set_linewidth(2)         # Set the linewidth for the edges
    # Add text annotation for the median
    for i, service in enumerate(service_list):
        x = dfs_MTBF[service].median()
        ax.text(x, i, f'{x:.2f}', color='black', va='center', ha='center', fontsize=12)
    # Add vertical lines at certain x positions
    thresholds_lines = [1, 7, 30]  # in days
    labels_thresholds = ['1d', '7d', '30d']
    for cp, label in zip(thresholds_lines, labels_thresholds):
        ax.axvline(x=cp, color='black', linestyle='--', linewidth=1)
        ax.text(cp, -0.5, label, color='black', fontsize=12, ha='center', va='top')

    ax.set_yticks(range(len(service_list)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('MTBF [days]', fontsize=14)
    ax.grid(axis='both', linestyle='--', alpha=0.6, which='both')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show()
    # Optional: Save the figure
    # plt.savefig('horizontal-box-plot-by-service-MTBF.png')

def plot_incident_counts_by_day_of_week(start_date, end_date, service_list):
    # Load the data
    path = 'server/static/data/incident_stages.csv'
    df = pd.read_csv(path)

    # Convert timestamp columns to datetime and adjust timezones
    status_lst = ['investigating', 'identified', 'monitoring', 'resolved', 'postmortem', 'start']
    for status in status_lst:
        df[f'{status}_timestamp'] = pd.to_datetime(df[f'{status}_timestamp'], errors='coerce')
        # Convert to US/Pacific time
        df[f'{status}_timestamp'] = df[f'{status}_timestamp'].dt.tz_convert('US/Pacific')
        # Remove timezone information
        df[f'{status}_timestamp'] = df[f'{status}_timestamp']
    
    # Filter data based on the provided date range
    df = df[(df['investigating_timestamp'] >= start_date) & (df['postmortem_timestamp'] <= end_date)]

    # Rename columns to standardize service names
    df = df.rename(columns={
        'API': 'API-OpenAI',
        'ChatGPT': 'ChatGPT',
        'Labs': 'DALL·E',
        'Playground': 'Playground',
        'api.anthropic.com': 'API-Anthropic',
        'claude.ai': 'Claude',
        'console.anthropic.com': 'Console',
        'Character.AI': 'Character.AI'
    })

    # Create 'provider' column if not present
    if 'provider' not in df.columns:
        service_to_provider = {
            'API-OpenAI': 'openai',
            'ChatGPT': 'openai',
            'DALL·E': 'openai',
            'Playground': 'openai',
            'API-Anthropic': 'anthropic',
            'Claude': 'anthropic',
            'Console': 'anthropic',
            'Character.AI': 'character',
            # Add other services if necessary
        }
        df['provider'] = ''
        for service, provider in service_to_provider.items():
            if service in df.columns:
                df.loc[df[service] == 1, 'provider'] = provider

    # Filter out corner cases
    df = filter_corner_cases(df)

    # Prepare DataFrames per service
    dfs = {}
    for service in service_list:
        if service in df.columns:
            dfs[service] = df[df[service] == 1].reset_index(drop=True)
        else:
            print(f"Service {service} not found in data.")
            dfs[service] = pd.DataFrame()

    # Map providers to their services
    provider_services = {}
    for service in service_list:
        if not dfs[service].empty:
            provider = dfs[service]['provider'].iloc[0]
            provider_services.setdefault(provider, []).append(service)

    # Remove providers with no services
    provider_services = {k: v for k, v in provider_services.items() if v}

    # Calculate incident counts by day of the week for each service within each provider
    provider_day_of_week_counts = {}
    for provider, services in provider_services.items():
        day_of_week_df = pd.DataFrame(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        for service in services:
            df_service = dfs[service]
            df_service['Day'] = df_service['start_timestamp'].dt.day_name()
            df_service['Day'] = pd.Categorical(
                df_service['Day'],
                categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                ordered=True
            )
            day_of_week_counts = df_service.groupby('Day').size()
            day_of_week_df[service] = day_of_week_counts
        # Fill NaN with 0
        day_of_week_df = day_of_week_df.fillna(0)
        provider_day_of_week_counts[provider] = day_of_week_df

    # Plot the incident counts
    num_providers = len(provider_day_of_week_counts)
    fig_width = 10 * num_providers
    fig, axs = plt.subplots(1, num_providers, figsize=(fig_width, 6))
    if num_providers == 1:
        axs = [axs]
    fig.subplots_adjust(wspace=0.0)

    provider_color_palettes = {
        'openai': 'Blues',
        'anthropic': 'Oranges',
        'character': 'Greens',
        # Add other providers if necessary
    }
    provider_titles = {
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'character': 'Character.AI',
        # Add other providers if necessary
    }

    for ax, (provider, day_of_week_df) in zip(axs, provider_day_of_week_counts.items()):
        services = provider_services[provider]
        # Get the color palette
        palette_name = provider_color_palettes.get(provider, 'Set2')
        colors = sns.color_palette(palette_name, len(services))
        colors = colors[::-1]  # Reverse colors if needed
        x = np.arange(len(day_of_week_df.index))
        bottom = np.zeros(len(day_of_week_df))
        for i, service in enumerate(services):
            counts = day_of_week_df[service].values
            ax.bar(x, counts, width=0.6, color=colors[i], bottom=bottom, label=service)
            bottom += counts
        ax.set_title(provider_titles.get(provider, provider.title()), fontsize=20)
        ax.set_xlabel('')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylabel('Incident Count', fontsize=12)
        labels = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.8)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.subplots_adjust(wspace=0.0)
    plt.tight_layout()
    plt.show()
