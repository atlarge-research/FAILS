import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)

def analyze_mttr_provider(start_date, end_date, selected_services):
    """
    Analyze and visualize MTTR distribution across providers
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=['investigating_timestamp', 'resolved_timestamp', 'start_timestamp', 'close_timestamp']
        )
        
        # Setup plotting style
        setup_plotting_style()

        # Group by provider
        provider_list = ['openai', 'anthropic', 'character', 'StabilityAI']
        dfs_provider = {}
        dfs_provider_mttr = {}

        for provider in provider_list:
            provider_df = df[df['provider'] == provider].copy()
            provider_df = provider_df.sort_values('investigating_timestamp')
            provider_df.loc[:, 'MTTR'] = (provider_df['resolved_timestamp'] - 
                                        provider_df['investigating_timestamp']).dt.total_seconds() / 3600
            dfs_provider_mttr[provider] = provider_df['MTTR'].dropna()
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
            if not dfs_provider_mttr[provider].empty:
                sns.ecdfplot(
                    data=dfs_provider_mttr[provider],
                    label=display_names.get(provider, provider.capitalize()),
                    color=colors[idx],
                    linestyle='-',
                    linewidth=5,
                )

        # Add reference lines
        thresholds = [10/60, 0.5, 3, 10, 24]  # in hours
        labels = ['10m', '30m', '3h', '10h', '24h']
        for threshold, label in zip(thresholds, labels):
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.9, linewidth=1.5)
            ax.text(threshold, 0.95, label, color='black', fontsize=26, ha='center', va='center', transform=ax.get_xaxis_transform())

        ax.set_xlabel('MTTR [hours]', fontsize=26)
        ax.set_ylabel('ECDF', fontsize=26)
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', borderaxespad=0.5)

        plt.tick_params(axis='both', which='major', labelsize=26)
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in MTTR provider analysis: {str(e)}")
        raise 