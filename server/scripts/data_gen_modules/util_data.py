import os
import pandas as pd
import json
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(base_dir):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith('.csv'):
                print(f'loading: {file_path}')
                df = pd.concat([df, pd.read_csv(file_path)])
    return df


def write_partitioned_data(df, data_layer, data_type, file_name):
    partition = date.today().strftime('%Y-%m-%d')
    archive_folder = f"data/{data_layer}/{data_type}/{partition}"
    os.makedirs(archive_folder, exist_ok=True)
    path = os.path.join(archive_folder, file_name)
    print(f'writing data to: {path}')
    df.to_csv(path, index=False)
    return path


def get_incident_id(incident):
    if pd.isna(incident):
        return incident
    else:
        return incident.split('/')[-1]


def load_json_column(df, column):
    df[column] = df[column].apply(lambda x: json.loads(x))
    return df


def unnest_lst_json(df, column):
    df = df.explode(column).reset_index(drop=True)
    df = df.join(pd.json_normalize(df[column]))
    df.drop(columns=column, inplace=True)
    return df


def unnest_dict(df, column, inplace=True):
    df = df.join(pd.json_normalize(df[column]))
    df.drop(columns=column, inplace=inplace)
    return df


def load_stg_incident(execution_date):
    incident = pd.read_csv(f'data/stg/incident/{execution_date}/incident_stages.csv')
    # incident timestamp columns: to datetime (PDT/PST )
    cols_inci_timestamp = [col for col in incident.columns if 'timestamp' in col]
    incident[cols_inci_timestamp] = incident[cols_inci_timestamp].apply(
        lambda x: pd.to_datetime(x).dt.tz_convert('America/Los_Angeles'))
    incident['services'] = incident['services'].apply(lambda x: json.loads(x.replace("'", '"')))
    return incident


def load_stg_caledar_outage(execution_date,filename=None):
    if filename is None:
        filename = 'outage_history.csv'
    calendar_outage = pd.read_csv(f'data/stg/outage/{execution_date}/{filename}', parse_dates=['Date'])
    outage = calendar_outage.loc[calendar_outage['outage_flag'] == 1].copy().sort_values('Date', ignore_index=True)
    outage.rename(columns={'Date': 'date'}, inplace=True)
    return calendar_outage, outage


def load_clean_caledar_outage(execution_date,filename=None):
    if filename is None:
        filename = 'outage_history.csv'
    calendar_outage = pd.read_csv(f'data/clean/outage/{execution_date}/{filename}', parse_dates=['Date'])
    outage = calendar_outage.loc[calendar_outage['outage_flag'] == 1].copy().sort_values('Date', ignore_index=True)
    outage.rename(columns={'Date': 'date'}, inplace=True)
    return calendar_outage, outage


def generate_incident_dates(start, end):
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    num_days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=x) for x in range(num_days)]


def explode_incident_dates(df):
    df['date'] = df.apply(lambda x: generate_incident_dates(x['start_timestamp'], x['close_timestamp']), axis=1)
    df['date'] = df['date'].apply(pd.to_datetime)
    df = df.explode('date')
    return df


def explode_incident_services(df):
    df = df.explode('services')
    df.rename(columns={'services': 'service'}, inplace=True)
    df = df[df['service'].notna() & df['service'].ne('')]
    return df


def print_incident_with_service_stats(incident):
    tot_cnt = incident['incident_id'].nunique()
    no_service = incident[incident['services'].isnull() | incident['services'].apply(lambda x: len(x) == 0)]
    no_service_cnt = no_service['incident_id'].nunique()
    print("Incident with Service Statistics:")
    print(f'Total incidents: {tot_cnt}')
    print(f'Incidents with service: {tot_cnt - no_service_cnt} ({(tot_cnt - no_service_cnt)/len(incident):.2%})')
    print(f'Incidents with no service: {no_service_cnt} ({no_service_cnt/len(incident):.2%})')


impact_mapping_reverse ={
    0: 'Impact-none',
    1: 'Impact-minor',
    2: 'Impact-major',
    3: 'Impact-critical',
    4: 'Impact-maintenance'
}

# impact_colors = {'#27AE60', '#F39C12', '#D35400', '#b71439', '#1976D2'}
impact_colors = {
    'Impact-none': '#24A148',
    'Impact-minor': '#f1c21b',
    'Impact-major': '#ff832b',
    'Impact-critical': '#da1e28',
    'Impact-maintenance': '#0043ce'
}

def plot_distribution(df, category, metric, overall_mean=None, figsize=(14, 14), colors=None):
    if colors is None:
        colors = ['#1a80bb', '#2ca25f', '#ea801c', '#800074']

    # plt.style.use('default')
    sns.set(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    metric_cap = ' '.join([word.capitalize() for word in metric.split('_')])
    category_cap = ' '.join([word.capitalize() for word in category.split('_')])
    if overall_mean is None:
        overall_mean = df[metric].mean()

    # boxplot
    sns.boxplot(x=category, y=metric, data=df, ax=ax1, showmeans=True, meanprops={'color': 'black', 'marker': 'o', 'markersize': 15}, palette=colors)

    ax1.set_title(f'{metric_cap} Distribution by {category_cap} (Boxplot)', fontsize=30)
    ax1.set_xlabel('')
    ax1.set_ylabel('Duration (hours)', fontsize=30)
    ax1.tick_params(axis='x', labelsize=28, rotation=0, labelbottom=True)
    ax1.tick_params(axis='y', labelsize=28)
    # mean line
    ax1.axhline(y=overall_mean, color='r', linestyle='--', linewidth=2)
    ax1.text(ax1.get_xlim()[1], overall_mean, f'Mean: {overall_mean:.2f}',
             va='center', ha='left', backgroundcolor='w', fontsize=20)

    # violinplot
    sns.violinplot(x=category, y=metric, data=df, ax=ax2, palette=colors)
    ax2.set_title(f'{metric_cap} Distribution by {category_cap} (Violinplot)', fontsize=30)
    ax2.set_xlabel(f'{category_cap}', fontsize=30)
    ax2.set_ylabel('Duration (hours)', fontsize=30)
    ax2.tick_params(axis='x', labelsize=28, rotation=0)
    ax2.tick_params(axis='y', labelsize=28)
    # mean line
    ax2.axhline(y=overall_mean, color='r', linestyle='--', linewidth=2)
    ax2.text(ax2.get_xlim()[1], overall_mean, f'Mean: {overall_mean:.2f}',
             va='center', ha='left', backgroundcolor='w', fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    # save
    plt.savefig(f'plot/fig-{metric}-distribution-by-{category}.pdf')
    plt.show()

    description = df.groupby(category)[metric].describe()
    return description