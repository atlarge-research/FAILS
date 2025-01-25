import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)

def analyze_mtbf_provider(start_date, end_date, selected_services):
    """
    Analyze and visualize MTBF distribution across providers
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=['investigating_timestamp', 'start_timestamp', 'close_timestamp']
        )
        
        # Setup plotting style
        setup_plotting_style()

        # Group by provider
        provider_list = ['openai', 'anthropic', 'character', 'StabilityAI']
        dfs_provider = {}
        dfs_provider_mtbf = {}

        for provider in provider_list:
            provider_df = df[df['provider'] == provider].copy()
            provider_df = provider_df.sort_values('investigating_timestamp')
            provider_df.loc[:, 'MTBF'] = (provider_df['investigating_timestamp'].diff()
                                        .dt.total_seconds() / (24 * 3600))  # Convert to days
            dfs_provider_mtbf[provider] = provider_df['MTBF'].dropna()
            dfs_provider[provider] = provider_df

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size for better visibility

        # Enhanced font sizes
        plt.rcParams.update({
            'font.size': 26,
            'axes.titlesize': 26,
            'axes.labelsize': 26,
            'xtick.labelsize': 26,
            'ytick.labelsize': 26,
            'legend.fontsize': 26
        })

        # Plot ECDF for each provider
        # Define display names for providers
        display_names = {
            'openai': 'OpenAI',
            'StabilityAI': 'StabilityAI'
        }
        colors = ['#6495ED', '#F08080', '#3CB371', '#DAA520', '#708090']
        
        for idx, provider in enumerate(provider_list):
            if not dfs_provider_mtbf[provider].empty:
                sns.ecdfplot(
                    data=dfs_provider_mtbf[provider],
                    label=display_names.get(provider, provider.capitalize()),
                    color=colors[idx],
                    linestyle='-',
                    linewidth=5
                )

        # Add reference lines
        thresholds = [1, 7, 30]  # in days
        labels = ['1d', '7d', '30d']
        for threshold, label in zip(thresholds, labels):
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.9, linewidth=1.5)
            ax.text(threshold, 0.95, label, color='black', fontsize=26, ha='center', va='center', transform=ax.get_xaxis_transform())

        ax.set_xlabel('MTBF [days]', fontsize=26)
        ax.set_ylabel('ECDF', fontsize=26)
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', borderaxespad=0.5)

        plt.tick_params(axis='both', which='major', labelsize=26)
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in MTBF provider analysis: {str(e)}")
        raise 