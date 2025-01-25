import os
import pandas as pd
import warnings
import seaborn as sns
import ast
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def get_data_path_incident():
    """Get the path to the incident data CSV file"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       'static', 'data', 'incident_stages_all.csv')

def get_data_path_outage():
    """Get the path to the outage data CSV file"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       'static', 'data', 'outage_unexploded.csv')

def get_service_mapping():
    """Get the mapping of service IDs to CSV column names"""
    return {
        'OpenAI:API': 'API',
        'OpenAI:ChatGPT': 'ChatGPT',
        'OpenAI:DALL-E': 'DALL-E',
        'OpenAI:Playground': 'Playground',
        'Anthropic:API': 'api.anthropic.com',
        'Anthropic:Claude': 'claude.ai',
        'Anthropic:Console': 'console.anthropic.com',
        'Character.AI:Character.AI': 'Character.AI',
        'StabilityAI:REST': 'REST API',
        'StabilityAI:gRPC': 'gRPC API',
        'StabilityAI:Assistant': 'Stable Assistant'
    }

def get_service_mapping_outage():
    """Get the mapping of service IDs to CSV column names"""
    return {
        'OpenAI:API': 'api',
        'OpenAI:ChatGPT': 'chatgpt',
        'OpenAI:DALL-E': 'labs',
        'OpenAI:Playground': 'playground',
        'Anthropic:API': 'api.anthropic.com',
        'Anthropic:Claude': 'claude.ai',
        'Anthropic:Console': 'console.anthropic.com',
        'Character.AI:Character.AI': 'character.ai',
        'StabilityAI:REST': 'REST API',
        'StabilityAI:gRPC': 'gRPC API',
        'StabilityAI:Assistant': 'Stable Assistant'
    }


def load_and_prepare_data(start_date, end_date, timestamp_columns=None):
    """
    Load and prepare incident data with common preprocessing steps
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        timestamp_columns: List of timestamp columns to convert
    
    Returns:
        Preprocessed DataFrame
    """
    print(f"\nLoading data...")
    print(f"Requested date range: {start_date} to {end_date}")
    
    # Convert dates to UTC timezone
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    print(f"Converted to UTC: {start_date} to {end_date}")
    
    # Read data
    data_path = get_data_path_incident()
    print(f"Reading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Initial data loaded: {len(df)} rows")
    
    # Convert timestamp columns
    if timestamp_columns:
        df = safe_convert_timezone(df, timestamp_columns)
        print(f"Timestamps converted to UTC")
    
    # Filter by date range using investigating_timestamp
    if 'investigating_timestamp' in df.columns:
        before_filter = len(df)
        df = df[(df['investigating_timestamp'] >= start_date) & 
                (df['investigating_timestamp'] <= end_date)]
        print(f"Date filtering: {before_filter} -> {len(df)} rows")
    
    # Initialize service columns if they don't exist
    service_columns = list(get_service_mapping().values())
    for service in service_columns:
        if service not in df.columns:
            df[service] = 0
    
    # Add preprocessing for OpenAI DALL-E incidents
    openai_mask = df['provider'] == 'openai'
    if openai_mask.any():
        for idx, row in df[openai_mask].iterrows():
            title = str(row['Incident_Title']).lower()
            desc = str(row.get('investigating_description', '')).lower()
            
            # Check for DALL-E related keywords
            if any(keyword in title or keyword in desc for keyword in 
                  ['dall-e', 'dall e', 'dalle', 'image generation', 'image creation']):
                df.loc[idx, 'DALL-E'] = 1
    
    # No need for StabilityAI incident detection - the columns already exist
    # Just ensure the columns are numeric
    stability_columns = ['REST API', 'gRPC API', 'Stable Assistant']
    for col in stability_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
    
    # Convert any string 'True'/'False' to 1/0 in service columns
    for col in service_columns:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0)
            df[col] = df[col].astype(float)
    
    # Add debug info
    print("\nService columns after preprocessing:")
    for col in service_columns:
        if col in df.columns:
            incident_count = df[col].sum()
            print(f"{col}: {incident_count} incidents")
        else:
            print(f"{col}: column not found")
    
    return df


def load_and_prepare_data_outage(start_date, end_date):
    """
    Load and prepare outage data with common preprocessing steps

    Args:
        start_date (pd.Timestamp): Start date for filtering (UTC timezone)
        end_date (pd.Timestamp): End date for filtering (UTC timezone)

    Returns:
        pd.DataFrame: Preprocessed outage DataFrame
    """
    # Convert dates to UTC timezone
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Read data
    data_path = get_data_path_outage()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Convert 'Date' column to datetime and localize to UTC
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize('UTC')
    else:
        raise KeyError("'Date' column is missing in the outage data.")
    
    # Parse 'Incidents' column from string to list
    if 'Incidents' in df.columns:
        df['Incidents'] = df['Incidents'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    else:
        raise KeyError("'Incidents' column is missing in the outage data.")
    
    # Compute 'Incident Count' column
    df['Incident Count'] = df['Incidents'].apply(len)
    
    # Filter by date range using 'Date' column
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    return df


def get_services_to_analyze(selected_services):
    """
    Map selected service IDs to CSV column names
    
    Args:
        selected_services: List of service IDs
        
    Returns:
        List of mapped service names
    """
    service_mapping = get_service_mapping()
    services_to_analyze = []
    
    for service in selected_services:
        if service in service_mapping:
            mapped_service = service_mapping[service]
            services_to_analyze.append(mapped_service)
            
    if not services_to_analyze:
        raise ValueError("No valid services selected for analysis")
        
    return services_to_analyze

def get_services_to_analyze_outage(selected_services):
    """
    Map selected service IDs to CSV column names
    
    Args:
        selected_services: List of service IDs
        
    Returns:
        List of mapped service names
    """
    service_mapping = get_service_mapping_outage()
    services_to_analyze = []
    
    for service in selected_services:
        if service in service_mapping:
            mapped_service = service_mapping[service]
            services_to_analyze.append(mapped_service)
            
    if not services_to_analyze:
        raise ValueError("No valid services selected for analysis")
        
    return services_to_analyze

def setup_plotting_style():
    """Set up common plotting style settings"""
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
def safe_convert_timezone(df, timestamp_columns):
    """Safely convert timestamp columns to UTC timezone"""
    df = df.copy()
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize('UTC')
            else:
                df[col] = df[col].dt.tz_convert('UTC')
    return df

def safe_groupby(df, column, **kwargs):
    """Safely perform groupby operation with proper observed parameter"""
    return df.groupby(column, observed=True, **kwargs)

def safe_to_period(series, freq='M'):
    """Safely convert timestamp series to period, handling timezone"""
    return series.dt.tz_localize(None).dt.to_period(freq) 