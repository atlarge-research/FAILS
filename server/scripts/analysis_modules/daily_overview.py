import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style,
    safe_groupby,
    safe_to_period,
    get_service_mapping
)

def analyze_daily_overview(start_date, end_date, selected_services):
    """
    Generate daily overview analysis showing incident counts by hour of the day.

    Args:
        start_date (datetime): Start date for filtering data.
        end_date (datetime): End date for filtering data.
        selected_services (list): List of services to analyze.

    Returns:
        fig (matplotlib.figure.Figure): The figure object with the plots.
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=[f'{status}_timestamp' for status in 
                             ['investigating', 'identified', 'monitoring', 
                              'resolved', 'postmortem']] + ['start_timestamp', 'close_timestamp']
        )

        # Get services to analyze and create provider groupings
        services_to_analyze = get_services_to_analyze(selected_services)
        provider_services = {}
        service_mapping = get_service_mapping()
        
        # Group selected services by provider
        for service_id in selected_services:
            provider, _ = service_id.split(':')
            if service_id in service_mapping:
                if provider not in provider_services:
                    provider_services[provider] = []
                provider_services[provider].append(service_mapping[service_id])

        # Remove providers with no services
        provider_services = {k: v for k, v in provider_services.items() if v}
        
        if not provider_services:
            raise ValueError("No valid services selected for analysis")

        # Setup enhanced plotting style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })

        # Calculate figure size based on number of providers
        num_providers = len(provider_services)
        fig_width = max(8, min(20, 7 * num_providers))
        fig, axs = plt.subplots(1, num_providers, figsize=(fig_width, 12))
        
        # Ensure axs is always a list
        if num_providers == 1:
            axs = [axs]

        provider_colors = {
            'OpenAI': sns.color_palette("Blues_r", n_colors=6),
            'Anthropic': sns.color_palette("Oranges_r", n_colors=6),
            'Character.AI': sns.color_palette("Greens_r", n_colors=6),
            'StabilityAI': sns.color_palette("Reds_r", n_colors=6)
        }

        # Find max value across all providers for consistent y-axis scaling
        max_incidents = 0
        for provider, services in provider_services.items():
            if provider == 'Character.AI':
                provider_df = df[df['provider'] == 'character']
            else:
                provider_df = df[df[services].any(axis=1)]
            
            if not provider_df.empty:
                provider_df['Hour'] = provider_df['investigating_timestamp'].dt.hour
                # Calculate total stacked height for each hour
                if provider == 'Character.AI':
                    hourly_total = provider_df.groupby('Hour').size().max()
                else:
                    # Sum all services for each hour
                    service_totals = []
                    for hour in range(24):
                        hour_df = provider_df[provider_df['Hour'] == hour]
                        total = sum(hour_df[service].sum() for service in services)
                        service_totals.append(total)
                    hourly_total = max(service_totals)
                max_incidents = max(max_incidents, hourly_total)

        # Calculate y-axis limit with padding
        y_max = max(1, max_incidents)  # No padding, exact fit
        y_ticks = min(y_max + 1, 10)  # Maximum 10 tick marks
        
        for idx, (provider, services) in enumerate(provider_services.items()):
            ax = axs[idx]
            
            if provider == 'Character.AI':
                # Special handling for Character.AI - use provider column
                provider_df = df[df['provider'] == 'character'].copy()
            else:
                # Original logic for other providers
                provider_df = df[df[services].any(axis=1)].copy() if not df.empty else pd.DataFrame()

            if provider_df.empty:
                ax.text(0.5, 0.5, 'No Data Available', 
                       horizontalalignment='center',
                       verticalalignment='center', 
                       fontsize=14, 
                       color='gray',
                       transform=ax.transAxes)
                continue

            # Extract hour and count incidents
            provider_df['Hour'] = provider_df['investigating_timestamp'].dt.hour
            
            if provider == 'Character.AI':
                # Single service for Character.AI
                hour_counts = pd.DataFrame(index=range(24))
                counts = provider_df.groupby('Hour').size()
                hour_counts['Character.AI'] = counts.reindex(range(24)).fillna(0)
            else:
                # Count by service
                hour_counts = pd.DataFrame(index=range(24))
                for service in services:
                    service_df = provider_df[provider_df[service] == 1]
                    counts = service_df.groupby('Hour').size()
                    hour_counts[service] = counts.reindex(range(24)).fillna(0)

            # Plot stacked bar chart
            colors = provider_colors.get(provider, sns.color_palette("husl", len(services)))
            hour_counts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)

            # Enhanced styling
            ax.set_title(f"$\\mathrm{{{provider}}}$", fontsize=20, pad=20)
            ax.set_xlabel(r'$\mathrm{Hour\;of\;Day}$', fontsize=16)
            if idx == 0:
                ax.set_ylabel(r'$\mathrm{Incident\;Count}$', fontsize=16, labelpad=15)

            # Customize grid
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            ax.set_axisbelow(True)

            # Format x-axis labels
            ax.set_xticks(range(0, 24, 3))
            ax.set_xticklabels([f"$\\mathrm{{{str(h).zfill(2)}}}$" for h in range(0, 24, 3)])

            # Add value labels on bars
            bottoms = np.zeros(24)
            for service in hour_counts.columns:
                values = hour_counts[service].values
                for i, v in enumerate(values):
                    if v > 0:
                        ax.text(i, bottoms[i] + v/2, f'{int(v)}',
                               ha='center', va='center',
                               color='white', fontweight='bold',
                               fontsize=12)
                    bottoms[i] += values[i]

            # Customize legend
            if num_providers == 1:
                ax.legend(title=r'$\mathrm{Services}$',
                         bbox_to_anchor=(1.05, 1),
                         frameon=True,
                         edgecolor='black',
                         fancybox=False)
            else:
                ax.legend(title=r'$\mathrm{Services}$',
                         loc='upper right',
                         frameon=True,
                         edgecolor='black',
                         fancybox=False)

            # Add box around plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_edgecolor('black')

            # Set y-axis limits and ticks consistently across all subplots
            ax.set_ylim(0, y_max + 0.5)  # Just a little padding at top
            ax.yaxis.set_major_locator(plt.MaxNLocator(y_ticks))
            
            # Remove top margin
            ax.margins(y=0)

        # Adjust layout with minimal spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Slightly more space at top for taller figure
        
        # Add more space between subplots if there are multiple providers
        if num_providers > 1:
            plt.subplots_adjust(wspace=0.3)

        # Add overall title
        timeframe_start = pd.to_datetime(start_date).strftime('%d %B %Y').replace(' ', '\\;')
        timeframe_end = pd.to_datetime(end_date).strftime('%d %B %Y').replace(' ', '\\;')
        title = f"$\\mathrm{{Hourly\\;Incident\\;Distribution\\;by\\;Provider\\;({timeframe_start}\\;-\\;{timeframe_end})\\;[UTC]}}$"
        if df.empty:
            title += "\n$\\mathrm{(No\\;incidents\\;recorded\\;in\\;this\\;period)}$"
        fig.suptitle(title, y=0.98, fontsize=22)  # Title closer to plot

        return fig

    except Exception as e:
        print(f"Error in daily overview analysis: {str(e)}")
        raise
