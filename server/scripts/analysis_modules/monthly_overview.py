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

def analyze_monthly_overview(start_date, end_date, selected_services):
    """
    Generate monthly overview analysis showing incident counts by day of week
    """
    try:
        print(f"\nStarting monthly overview analysis...")
        print(f"Input parameters: start_date={start_date}, end_date={end_date}")
        print(f"Selected services: {selected_services}")
        
        # Add debug logging
        print("\nDebugging service mapping:")
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=[f'{status}_timestamp' for status in 
                             ['investigating', 'identified', 'monitoring', 
                              'resolved', 'postmortem']] + ['start_timestamp', 'close_timestamp']
        )
        
        # Debug available columns
        print("\nAvailable columns in DataFrame:")
        print(df.columns.tolist())
        
        # Debug service mapping
        services_to_analyze = get_services_to_analyze(selected_services)
        print("\nService mapping:")
        for service_id in selected_services:
            if service_id in get_service_mapping():
                mapped_service = get_service_mapping()[service_id]
                print(f"{service_id} -> {mapped_service}")
                # Check if the mapped column exists and has any non-zero values
                if mapped_service in df.columns:
                    count = df[mapped_service].sum()
                    print(f"Found {count} incidents for {mapped_service}")
                else:
                    print(f"Warning: Column {mapped_service} not found in DataFrame")
            else:
                print(f"Warning: No mapping found for {service_id}")
                
        print(f"\nData loaded: {len(df)} rows")
        print(f"Date range requested: {start_date} to {end_date}")
        print(f"Available columns: {df.columns.tolist()}")  # Check column names
        
        # Get services to analyze and create provider groupings
        services_to_analyze = get_services_to_analyze(selected_services)
        print(f"\nServices to analyze: {services_to_analyze}")  # Check mapped services
        
        provider_services = {}
        service_mapping = get_service_mapping()
        
        # Group selected services by provider
        for service_id in selected_services:
            provider, _ = service_id.split(':')
            if service_id in service_mapping:
                if provider not in provider_services:
                    provider_services[provider] = []
                mapped_service = service_mapping[service_id]
                provider_services[provider].append(mapped_service)
                print(f"Mapped {service_id} to {mapped_service}")  # Check service mapping
            else:
                print(f"Warning: Service ID not found in mapping: {service_id}")

        # Debug provider services
        print("\nProvider services after mapping:")
        for provider, services in provider_services.items():
            print(f"{provider}: {services}")
            if provider == 'Character.AI':
                # Extra debug for Character.AI data
                if not df.empty:
                    char_ai_data = df[df[services].any(axis=1)]
                    print(f"Character.AI data points: {len(char_ai_data)}")
                    if not char_ai_data.empty:
                        print(f"Date range for Character.AI: {char_ai_data['investigating_timestamp'].min()} to {char_ai_data['investigating_timestamp'].max()}")

        # Remove providers with no services
        provider_services = {k: v for k, v in provider_services.items() if v}
        
        if not provider_services:
            raise ValueError("No valid services selected for analysis")

        # Setup plotting
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
        fig, axs = plt.subplots(1, num_providers, figsize=(fig_width, 10))
        
        # Ensure axs is always a list
        if num_providers == 1:
            axs = [axs]

        provider_colors = {
            'OpenAI': sns.color_palette("Blues_r", n_colors=6),
            'Anthropic': sns.color_palette("Oranges_r", n_colors=6),
            'Character.AI': sns.color_palette("Greens_r", n_colors=6),
            'StabilityAI': sns.color_palette("Reds_r", n_colors=6)
        }

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for idx, (provider, services) in enumerate(provider_services.items()):
            ax = axs[idx]
            
            if provider == 'Character.AI':
                # Special handling for Character.AI - use provider column
                provider_df = df[df['provider'] == 'character'].copy()
                
                # Count incidents by day for Character.AI
                provider_df['Day'] = provider_df['investigating_timestamp'].dt.day_name()
                provider_df['Day'] = pd.Categorical(provider_df['Day'], categories=weekdays, ordered=True)
                
                # Create day counts with a single "Character.AI" service column
                day_counts = pd.DataFrame()
                service_counts = safe_groupby(provider_df, 'Day').size()
                day_counts['Character.AI'] = service_counts
            else:
                # Original logic for other providers
                provider_df = df[df[services].any(axis=1)].copy() if not df.empty else pd.DataFrame()
                
                if not provider_df.empty:
                    provider_df['Day'] = provider_df['investigating_timestamp'].dt.day_name()
                    provider_df['Day'] = pd.Categorical(provider_df['Day'], categories=weekdays, ordered=True)
                    
                    # Count incidents by day and service
                    day_counts = pd.DataFrame()
                    for service in services:
                        service_counts = safe_groupby(provider_df[provider_df[service] == 1], 'Day').size()
                        day_counts[service] = service_counts

            if provider_df.empty:
                # Handle empty data case
                ax.text(0.5, 0.5, 'No Data Available', 
                       horizontalalignment='center',
                       verticalalignment='center', 
                       fontsize=14, 
                       color='gray',
                       transform=ax.transAxes)
                day_counts = pd.DataFrame(0, index=weekdays, columns=services)
            else:
                day_counts = day_counts.reindex(weekdays).fillna(0)

            # Plot stacked bar chart with enhanced styling
            colors = provider_colors.get(provider, sns.color_palette("husl", len(services)))
            day_counts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
            
            # Enhanced title and labels
            ax.set_title(f"$\\mathrm{{{provider}}}$", fontsize=20, pad=20)
            ax.set_xlabel('')
            if idx == 0:
                ax.set_ylabel(r'$\mathrm{Incident\;Count}$', fontsize=16, labelpad=15)
            
            # Customize grid
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            ax.set_axisbelow(True)
            
            # Enhance tick labels
            ax.tick_params(axis='x', rotation=45, length=6, width=1)
            ax.tick_params(axis='y', length=6, width=1)
            
            # Format x-axis labels
            ax.set_xticklabels([f"$\\mathrm{{{day[:3]}}}$" for day in weekdays])
            
            # Add value labels on bars (only for non-zero values)
            bottoms = np.zeros(len(weekdays))
            for service in day_counts.columns:
                values = day_counts[service].values
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

            # Set y-axis to start at 0 and have reasonable upper limit
            ax.set_ylim(0, max(1, ax.get_ylim()[1]))

        # Adjust layout
        plt.tight_layout()
        
        # Add more space between subplots if there are multiple providers
        if num_providers > 1:
            plt.subplots_adjust(wspace=0.3)

        # Add overall title with note if no data
        timeframe_start = pd.to_datetime(start_date).strftime('%d %B %Y').replace(' ', '\\;')
        timeframe_end = pd.to_datetime(end_date).strftime('%d %B %Y').replace(' ', '\\;')
        title = f"$\\mathrm{{Daily\\;Incident\\;Distribution\\;by\\;Provider\\;({timeframe_start}\\;-\\;{timeframe_end})}}$"
        if df.empty:
            title += "\n$\\mathrm{(No\\;incidents\\;recorded\\;in\\;this\\;period)}$"
        fig.suptitle(title, y=1.05, fontsize=22)

        return fig

    except Exception as e:
        print(f"Error in monthly overview analysis: {str(e)}")
        raise 
    