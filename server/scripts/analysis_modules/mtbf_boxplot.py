import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)

def analyze_mtbf_boxplot(start_date, end_date, selected_services):
    """
    Create horizontal box plot showing MTBF distribution by service
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

        # Calculate MTBF for each service
        mtbf_data = {}
        for service in services_to_analyze:
            service_df = df[df[service] == 1].copy()
            service_df = service_df.sort_values('investigating_timestamp')
            service_df.loc[:, 'MTBF'] = (service_df['investigating_timestamp'].diff()
                                        .dt.total_seconds() / (24 * 3600))  # Convert to days
            mtbf_data[service] = service_df['MTBF'].dropna()

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert data to DataFrame for boxplot
        df_mtbf = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in mtbf_data.items()]))
        
        # Create boxplot
        flierprops = dict(marker='D', markersize=3)
        sns.boxplot(data=df_mtbf, orient='h', ax=ax, color='white', 
                   linewidth=2, flierprops=flierprops)
        
        # Customize boxplot appearance
        for patch in ax.artists:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Add median values
        for i, service in enumerate(services_to_analyze):
            x = mtbf_data[service].median()
            ax.text(x, i, f'{x:.2f}', color='black', va='center', ha='center', fontsize=12)

        # Add reference lines
        thresholds = [1, 7, 30]  # in days
        labels = ['1d', '7d', '30d']
        for threshold, label in zip(thresholds, labels):
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1)
            ax.text(threshold, -0.5, label, color='black', fontsize=12, ha='center', va='top')

        # Customize plot
        ax.set_yticks(range(len(services_to_analyze)))
        ax.set_yticklabels([s.replace('_', ' ') for s in services_to_analyze], fontsize=12)
        ax.set_xlabel('MTBF [days]', fontsize=14)
        ax.grid(axis='both', linestyle='--', alpha=0.6, which='both')
        ax.set_xscale('log')
        plt.title('MTBF Distribution by Service', pad=20)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in MTBF boxplot analysis: {str(e)}")
        raise 