import pandas as pd
import matplotlib.pyplot as plt
from .utils import (
    load_and_prepare_data, 
    load_and_prepare_data_outage, 
    get_services_to_analyze,
    get_services_to_analyze_outage,
    setup_plotting_style
)
from datetime import datetime
import traceback

def timeline_incident_outage(start_date, end_date, selected_services):
    """
    Analyze and visualize outages and incidents for different services over time (monthly).

    Args:
        start_date (datetime): Start date for filtering data.
        end_date (datetime): End date for filtering data.
        selected_services (list): List of services to filter by.

    Returns:
        fig (matplotlib.figure.Figure): The figure object with the plots.
    """
    try:
        # Convert string dates to timestamps if they aren't already
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Map selected services to CSV column names
        services_to_analyze_outage = get_services_to_analyze_outage(selected_services)

        # Load outage data
        df_outage = load_and_prepare_data_outage(start_date, end_date)

        # Filter outage data by selected services
        if 'Service' in df_outage.columns:
            df_outage = df_outage[df_outage['Service'].isin(services_to_analyze_outage)]
        else:
            raise KeyError("'Service' column is missing in the outage data.")

        # Create 'Month' column for outages
        df_outage['Month'] = df_outage['Date'].dt.strftime('%Y-%m')

        # Aggregate outage counts by Service and Month, considering only outages
        monthly_outage = df_outage.groupby(['Service', 'Month'])['outage_flag'].sum().reset_index(name='Outage Count')


        # Load incident data
        df_incident = load_and_prepare_data(start_date, end_date, timestamp_columns=['start_timestamp', 'close_timestamp'])

        # Identify service columns in incident data
        incident_service_columns = get_services_to_analyze(selected_services)  # These should match the columns in incident CSV

        # Melt the incident DataFrame to have 'Service' and 'Incident Count'
        df_incident_melted = df_incident.melt(
            id_vars=['incident_id', 'Incident_Title', 'incident_impact_level', 'Incident_color', 'provider',
                     'investigating_flag', 'investigating_timestamp', 'investigating_description',
                     'identified_flag', 'identified_timestamp', 'identified_description',
                     'monitoring_flag', 'monitoring_timestamp', 'monitoring_description',
                     'resolved_flag', 'resolved_timestamp', 'resolved_description',
                     'postmortem_flag', 'postmortem_timestamp', 'postmortem_description',
                     'start_timestamp', 'close_timestamp', 'time_span', 'over_one_day'],
            value_vars=incident_service_columns,
            var_name='Service',
            value_name='Service_Flag'
        )

        # Filter rows where Service_Flag == 1 (service is involved in the incident)
        df_incident_filtered = df_incident_melted[df_incident_melted['Service_Flag'] == 1].copy()

        # Convert 'start_timestamp' to datetime and create 'Month' column
        df_incident_filtered['start_timestamp'] = pd.to_datetime(
            df_incident_filtered['start_timestamp'], errors='coerce'
        )
        df_incident_filtered['Month'] = df_incident_filtered['start_timestamp'].dt.strftime('%Y-%m')

        # Drop rows with NaT in 'start_timestamp'
        df_incident_filtered.dropna(subset=['start_timestamp'], inplace=True)

        # Aggregate incident counts by Service and Month
        monthly_incident = df_incident_filtered.groupby(['Service', 'Month']).size().reset_index(name='Incident Count')

        # Set up plotting style
        setup_plotting_style()

        # Create subplots: one for Incidents and one for Outages
        fig, (ax_incidents, ax_outages) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Define a color palette for consistency
        colors_incident = plt.cm.get_cmap('tab10', len(incident_service_columns))

        colors_outage = plt.cm.get_cmap('tab10', len(services_to_analyze_outage))

        # Plot Incidents
        for idx, service in enumerate(incident_service_columns):
            service_data = monthly_incident[monthly_incident['Service'] == service]
            ax_incidents.plot(
                service_data['Month'],
                service_data['Incident Count'],
                marker='o',
                linewidth=2,
                label=service,
                color=colors_incident(idx)
            )

        ax_incidents.set_ylabel("Incident Count", fontsize=12)
        ax_incidents.set_title("Monthly Incidents per Service", fontsize=14)
        ax_incidents.legend(title="Services", fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_incidents.grid(True, linestyle='--', color='gray', alpha=0.7)

        # Plot Outages
        for idx, service in enumerate(services_to_analyze_outage):
            service_data = monthly_outage[monthly_outage['Service'] == service]
            ax_outages.plot(
                service_data['Month'],
                service_data['Outage Count'],
                marker='^',
                linewidth=2,
                label=service,
                color=colors_outage(idx)
            )

        ax_outages.set_xlabel("Month", fontsize=12)
        ax_outages.set_ylabel("Outage Count", fontsize=12)
        ax_outages.set_title("Monthly Outages per Service", fontsize=14)
        ax_outages.legend(title="Services", fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_outages.grid(True, linestyle='--', color='gray', alpha=0.7)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Add overall title
        timeframe_start = start_date.strftime('%d %B %Y')
        timeframe_end = end_date.strftime('%d %B %Y')
        fig.suptitle(f"Incident and Outage Timeline ({timeframe_start} - {timeframe_end})", 
                    y=1.02, fontsize=16)

        return fig

    except Exception as e:
        print(f"An error occurred in timeline_incident_outage: {e}")
        traceback.print_exc()  # Add traceback for better error reporting
        return None
