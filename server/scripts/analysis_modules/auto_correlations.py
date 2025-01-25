import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from .utils import load_and_prepare_data

def analyze_autocorrelation(start_date, end_date, selected_services):
    """
    Analyze and visualize autocorrelations between incidents for each provider.
    Creates monthly, weekly and daily autocorrelation plots.

    Args:
        start_date (datetime): Start date for filtering incident data
        end_date (datetime): End date for filtering incident data
        selected_services (list): List of service identifiers to analyze

    Returns:
        dict: Dictionary containing Matplotlib Figure objects for each provider
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data(start_date, end_date, timestamp_columns=['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp', 'resolved_timestamp', 'postmortem_timestamp', 'start_timestamp', 'close_timestamp'])
        
        # Add at the start, after loading the data
        print("Available columns in data:")
        print(df.columns.tolist())
        
        # Convert timestamps to PDT and remove timezone info
        status_lst = ['investigating', 'identified', 'monitoring', 'resolved', 'postmortem', 'start']
        for status in status_lst:
            df[f'{status}_timestamp'] = pd.to_datetime(df[f'{status}_timestamp'], errors='coerce')
            df[f'{status}_timestamp'] = df[f'{status}_timestamp'].dt.tz_convert('US/Pacific')
            df[f'{status}_timestamp'] = pd.to_datetime(df[f'{status}_timestamp']).dt.tz_localize(None)

        # Filter out corner cases
        corner_cases = pd.DataFrame()
        for index, record in df.iterrows():
            if (record['investigating_timestamp'] > record['identified_timestamp'] or
                record['identified_timestamp'] > record['monitoring_timestamp'] or
                record['monitoring_timestamp'] > record['resolved_timestamp'] or
                record['resolved_timestamp'] > record['postmortem_timestamp']):
                corner_cases = pd.concat([corner_cases, record.to_frame().T])
        
        df = df.drop(corner_cases.index).reset_index(drop=True)

        # Rename columns
        df = df.rename(columns={
            'API': 'API-OpenAI',
            'ChatGPT': 'ChatGPT',
            'Labs': 'DALL-E',
            'Playground': 'Playground',
            'api.anthropic.com': 'API-Anthropic',
            'claude.ai': 'Claude',
            'console.anthropic.com': 'Console',
            'Character.AI': 'Character.AI',
            'REST API': 'REST API',
            'gRPC API': 'gRPC API',
            'Stable Assistant': 'Stable Assistant'
        })

        # Group data by provider
        service_lst = ['API-OpenAI', 'ChatGPT', 'DALL-E', 'Playground',
                      'API-Anthropic', 'Claude', 'Console', 'Character.AI',
                      'REST API', 'gRPC API', 'Stable Assistant']
        
        dfs = {}
        for service in service_lst:
            dfs[service] = df[df[service] == 1].reset_index(drop=True)

        # Update provider list and mapping to match actual data
        provider_lst = ['openai', 'anthropic', 'character', 'StabilityAI']  # Match exact provider names from data
        provider_mapping = {
            'openai': 'OpenAI:',
            'anthropic': 'Anthropic:',
            'character': 'Character.AI:',  # Changed from character.ai to character
            'StabilityAI': 'StabilityAI:'  # Exact match from data
        }
        
        # Add debug prints
        print("Available providers in data:")
        for service in service_lst:
            if service in df.columns:
                providers = df[df[service] == 1]['provider'].unique()
                print(f"{service}: {providers}")

        dfs_provider = {}
        for provider in provider_lst:
            prefix = provider_mapping[provider]
            provider_services = [s for s in selected_services if s.startswith(prefix)]
            if provider_services:
                dfs_provider[provider] = pd.DataFrame()
                for service in service_lst:
                    # Use exact provider name from data without any transformation
                    if provider in dfs[service]['provider'].unique():
                        print(f"- Found data for service {service}")
                        provider_data = dfs[service][dfs[service]['provider'] == provider]
                        dfs_provider[provider] = pd.concat([dfs_provider[provider], provider_data])
                    
                print(f"- Final dataframe size for {provider}: {len(dfs_provider[provider])}")

        # Create single figure instead of dict of subplots
        fig = plt.figure(figsize=(15, 10))
        
        for provider, provider_df in dfs_provider.items():
            if provider_df.empty:
                continue

            # Create a copy before modifications
            provider_df = provider_df.copy()

            # Calculate incident start times
            provider_df['incident_start_time'] = provider_df[[
                'investigating_timestamp',
                'identified_timestamp',
                'monitoring_timestamp',
                'resolved_timestamp'
            ]].min(axis=1)
            
            provider_df = provider_df[['resolved_flag', 'incident_start_time']]

            # Add time-based columns
            provider_df['year'] = provider_df['incident_start_time'].dt.year
            provider_df['month_of_year'] = provider_df['incident_start_time'].dt.month
            provider_df['week_of_year'] = provider_df['incident_start_time'].dt.isocalendar().week
            provider_df['day_of_year'] = provider_df['incident_start_time'].dt.dayofyear
            provider_df['hour_of_day'] = provider_df['incident_start_time'].dt.hour

            # Calculate numerical time values
            provider_df['year_num'] = provider_df['year'] - provider_df['year'].min()
            provider_df['month_num'] = provider_df['year_num'] * 12 + provider_df['month_of_year']
            provider_df['week_num'] = provider_df['year_num'] * 52 + provider_df['week_of_year']
            provider_df['day_num'] = provider_df['year_num'] * 365 + provider_df['day_of_year']
            provider_df['hour_num'] = provider_df['day_num'] * 24 + provider_df['hour_of_day']

            # Group data by different time periods
            df_month = provider_df[['month_num', 'resolved_flag']]
            group_by_month = df_month.groupby('month_num')['resolved_flag'].sum()
            df_month = pd.DataFrame({'month_num': range(max(group_by_month.index)+1), 'resolved_flag': 0})
            df_month = df_month.merge(group_by_month, how='left', on='month_num').fillna(0)
            df_month = df_month['resolved_flag_y']

            df_week = provider_df[['week_num', 'resolved_flag']]
            group_by_week = df_week.groupby('week_num')['resolved_flag'].sum()
            df_week = pd.DataFrame({'week_num': range(max(group_by_week.index)+1), 'resolved_flag': 0})
            df_week = df_week.merge(group_by_week, how='left', on='week_num').fillna(0)
            df_week = df_week['resolved_flag_y']

            df_day = provider_df[['day_num', 'resolved_flag']]
            group_by_day = df_day.groupby('day_num')['resolved_flag'].sum()
            df_day = pd.DataFrame({'day_num': range(max(group_by_day.index)+1), 'resolved_flag': 0})
            df_day = df_day.merge(group_by_day, how='left', on='day_num').fillna(0)
            df_day = df_day['resolved_flag_y']

            # Create autocorrelation plots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.3)

            # Monthly autocorrelation
            ax = axs[0]
            plot_acf(df_month, lags=12, ax=ax)
            ax.set_title('')
            ax.set_xlabel('Month (Lags=12)', fontsize=12)
            ax.set_ylim(bottom=0)

            # Weekly autocorrelation
            ax = axs[1]
            plot_acf(df_week, lags=50, ax=ax)
            ax.set_title('')
            ax.set_xlabel('Week (Lags=50)', fontsize=12)
            ax.set_ylim(bottom=0)

            # Daily autocorrelation
            ax = axs[2]
            plot_acf(df_day, lags=80, ax=ax)
            ax.set_title('')
            ax.set_xlabel('Day (Lags=80)', fontsize=12)
            ax.set_ylim(bottom=0)

            plt.suptitle(f"{provider.capitalize()} Incident Autocorrelations", fontsize=14, y=1.05)
            plt.tight_layout()

        # Return the figure
        return fig

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise