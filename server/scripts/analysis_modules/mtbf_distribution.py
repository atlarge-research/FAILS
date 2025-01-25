import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)

def analyze_mtbf_distribution(start_date, end_date, selected_services):
    """
    Analyze and visualize MTBF distribution across services
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=['investigating_timestamp', 'start_timestamp', 'close_timestamp']
        )
        
        # Get services to analyze
        services_to_analyze = get_services_to_analyze(selected_services)
        
        # Setup plotting style
        setup_plotting_style()

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Calculate MTBF for each service
        mtbf_data = {}
        for service in services_to_analyze:
            service_df = df[df[service] == 1].copy()
            service_df = service_df.sort_values('investigating_timestamp')
            service_df.loc[:, 'MTBF'] = (service_df['investigating_timestamp'].diff()
                                        .dt.total_seconds() / (24 * 3600))  # Convert to days
            mtbf_data[service] = service_df['MTBF'].dropna()

        # Plot 1: ECDF Plot
        colors = sns.color_palette('Set2', n_colors=len(services_to_analyze))
        for idx, service in enumerate(services_to_analyze):
            sns.ecdfplot(
                data=mtbf_data[service],
                label=service.replace('_', ' '),
                color=colors[idx],
                linestyle='-',
                linewidth=2,
                ax=ax1
            )

        # Add vertical lines at key thresholds
        thresholds = [1, 7, 30]  # in days
        labels = ['1d', '7d', '30d']
        for threshold, label in zip(thresholds, labels):
            ax1.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            ax1.text(threshold, 1.01, label, rotation=0, ha='center', va='bottom')

        ax1.set_xlabel('MTBF [days]')
        ax1.set_ylabel('ECDF')
        ax1.set_xscale('log')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.legend(title='Service', bbox_to_anchor=(1.05, 1))

        # Plot 2: Percentage Distribution
        # Calculate percentages for each threshold
        percent_data = []
        for service in services_to_analyze:
            service_data = {'Service': service}
            mtbf_series = mtbf_data[service]
            for threshold in thresholds:
                percent = (mtbf_series <= threshold).mean() * 100
                service_data[f'≤{threshold}d'] = percent
            service_data['>30d'] = (mtbf_series > 30).mean() * 100
            percent_data.append(service_data)

        df_percent = pd.DataFrame(percent_data)
        
        # Create stacked bar plot
        df_percent.plot(
            x='Service',
            y=[f'≤{t}d' for t in thresholds] + ['>30d'],
            kind='bar',
            stacked=True,
            ax=ax2,
            colormap='Set3'
        )

        ax2.set_xlabel('Service')
        ax2.set_ylabel('Percentage of Incidents')
        ax2.legend(title='MTBF Duration', bbox_to_anchor=(1.05, 1))
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in MTBF distribution analysis: {str(e)}")
        raise 