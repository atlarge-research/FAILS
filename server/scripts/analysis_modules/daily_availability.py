import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style,
    get_service_mapping
)
import traceback

def analyze_daily_availability(start_date, end_date, services):
    """
    Analyze and visualize daily service availability patterns
    """
    try:
        print("\nStarting daily availability analysis...")
        print(f"Input parameters: start_date={start_date}, end_date={end_date}")
        print(f"Selected services: {services}")
        
        # Load and prepare data
        df = load_and_prepare_data(start_date, end_date)
        print("Data loaded successfully")
        
        # Get services to analyze using the utility function
        services_to_analyze = get_services_to_analyze(services)
        print(f"\nAnalyzing services: {services_to_analyze}")
        
        # Setup plotting style
        setup_plotting_style()
        
        # Convert timestamps to datetime
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        df['close_timestamp'] = pd.to_datetime(df['close_timestamp'])
        
        # Create date range for analysis
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Calculate daily availability for each service
        availability_data = {}
        for service in services_to_analyze:
            if service in df.columns:
                service_incidents = df[df[service] == 1].copy()
                if not service_incidents.empty:
                    print(f"Found {len(service_incidents)} incidents for {service}")
                    daily_availability = []
                    
                    for day in date_range:
                        day_start = day
                        day_end = day + pd.Timedelta(days=1)
                        
                        # Count minutes of downtime for this day
                        downtime_minutes = 0
                        for _, incident in service_incidents.iterrows():
                            incident_start = max(incident['start_timestamp'], day_start)
                            incident_end = min(incident['close_timestamp'], day_end)
                            if incident_end > incident_start:
                                downtime_minutes += (incident_end - incident_start).total_seconds() / 60
                        
                        # Calculate availability percentage
                        availability = 100 * (1 - downtime_minutes / (24 * 60))
                        daily_availability.append(availability)
                    
                    # Use display name for the service in the plot
                    display_name = service.replace('api.anthropic.com', 'Anthropic API')\
                                       .replace('claude.ai', 'Claude')\
                                       .replace('console.anthropic.com', 'Console')\
                                       .replace('Character.AI', 'Character.AI')\
                                       .replace('REST API', 'StabilityAI REST')\
                                       .replace('gRPC API', 'StabilityAI gRPC')
                    availability_data[display_name] = daily_availability
            else:
                print(f"Warning: Service {service} not found in data")
        
        if not availability_data:
            print("No availability data calculated")
            return None
            
        # Create figure with larger size for better readability
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot availability lines with distinct colors
        colors = sns.color_palette('husl', n_colors=len(availability_data))
        for idx, (service, avail) in enumerate(availability_data.items()):
            ax.plot(date_range, avail, 
                   label=service, 
                   color=colors[idx],
                   marker='o', 
                   markersize=4,
                   alpha=0.8,
                   linewidth=2)
        
        # Add reference lines for common SLA levels
        sla_levels = [99.9, 99.99, 99.999]
        sla_colors = ['red', 'orange', 'green']
        sla_labels = ['99.9%', '99.99%', '99.999%']
        
        for level, color, label in zip(sla_levels, sla_colors, sla_labels):
            ax.axhline(y=level, color=color, linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[1], level, f' {label}', 
                   va='center', ha='left', color=color)

        # Customize plot
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Availability (%)', fontsize=12)
        
        # Format title with LaTeX for better typography
        timeframe_start = pd.to_datetime(start_date).strftime('%d %B %Y').replace(' ', '\\;')
        timeframe_end = pd.to_datetime(end_date).strftime('%d %B %Y').replace(' ', '\\;')
        title = f"$\\mathrm{{Service\\;Availability\\;({timeframe_start}\\;-\\;{timeframe_end})}}$"
        plt.suptitle(title, y=1.02, fontsize=16)
        
        # Adjust legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Set y-axis limits to focus on high availability range
        ax.set_ylim(99, 100.1)
        
        # Add grid for better readability
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig

    except Exception as e:
        print(f"Error in daily availability analysis: {str(e)}")
        traceback.print_exc()
        return None 