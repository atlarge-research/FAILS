import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)
import traceback

IMPACT_LEVELS = {
    0: "None - Post-resolved notification or documentation update",
    1: "Minor - Brief disruption or elevated error rates with limited impact",
    2: "Major - Significant service degradation affecting core functionality",
    3: "Critical - Complete service outage or severe widespread impact",
    4: "Maintenance - Planned maintenance or scheduled updates"
}

def get_provider_display_name(provider):
    """Map provider names to their display format"""
    provider_names = {
        'anthropic': 'Anthropic',
        'openai': 'OpenAI',
        'google': 'Google',
        'microsoft': 'Microsoft',
        'meta': 'Meta',
        'character': 'Character.AI',
        'characterai': 'Character.AI',
        'stabilityai': 'StabilityAI',
        'stability': 'StabilityAI',
        'StabilityAI': 'StabilityAI',
        'Character.AI': 'Character.AI',
        'OpenAI': 'OpenAI',
        'Anthropic': 'Anthropic'
    }
    # Handle case variations by converting to lowercase for lookup
    lookup_key = provider.lower().replace('.', '').replace(' ', '')
    return provider_names.get(lookup_key, provider.title())

def analyze_incident_distribution(start_date, end_date, services):
    """
    Analyze and visualize the distribution of incident impact levels across providers
    """
    try:
        print(f"Starting incident distribution analysis with dates: {start_date} to {end_date}")
        
        # Load data with timestamp columns specified
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=['investigating_timestamp', 'start_timestamp', 'close_timestamp']
        )
        print(f"Loaded data shape: {df.shape}")
        
        # Convert impact levels to numeric if they aren't already
        df['incident_impact_level'] = pd.to_numeric(df['incident_impact_level'], errors='coerce')
        
        # Setup plotting
        setup_plotting_style()
        plt.rcParams.update({
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        })
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create box plot first (underneath)
        sns.boxplot(
            data=df,
            x='provider',
            y='incident_impact_level',
            width=0.3,
            color='lightgray',
            showfliers=False,
            zorder=2,
            ax=ax
        )
        
        # Add violin plot
        sns.violinplot(
            data=df,
            x='provider',
            y='incident_impact_level',
            bw_method=0.5,
            cut=0.8,
            density_norm='width',
            inner=None,
            alpha=0.3,
            zorder=1,
            ax=ax
        )
        
        # Get providers and their display names
        providers = df['provider'].unique()
        provider_labels = [get_provider_display_name(p) for p in providers]
        
        # Update x-axis labels with proper provider names
        ax.set_xticks(range(len(providers)))
        ax.set_xticklabels(provider_labels, rotation=0, fontsize=18)
        
        # Add mean and median lines to each violin
        for idx, provider in enumerate(providers):
            provider_data = df[df['provider'] == provider]
            if not provider_data.empty:
                mean = provider_data['incident_impact_level'].mean()
                median = provider_data['incident_impact_level'].median()
                
                # Add mean line
                ax.hlines(y=mean, xmin=idx-0.2, xmax=idx+0.2, 
                         color='blue', linestyle='--', linewidth=2, alpha=0.5)
                
                # Add median line
                ax.hlines(y=median, xmin=idx-0.2, xmax=idx+0.2, 
                         color='red', linestyle='-', linewidth=2.5, alpha=0.7)
        
        # Add individual points
        for idx, provider in enumerate(providers):
            provider_data = df[df['provider'] == provider]
            
            for impact_level in range(5):
                level_data = provider_data[provider_data['incident_impact_level'] == impact_level]
                
                if len(level_data) > 0:
                    n_points = len(level_data)
                    
                    if n_points == 1:
                        x_positions = [idx]
                    else:
                        std = min(0.2, 0.4 / np.sqrt(n_points))
                        x_positions = np.random.normal(idx, std, n_points)
                        x_positions = np.clip(x_positions, idx - 0.4, idx + 0.4)
                    
                    ax.scatter(
                        x_positions,
                        [impact_level] * n_points,
                        color='darkred',
                        alpha=0.5,
                        s=40,
                        zorder=3
                    )
        
        # Add legend with all elements
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', alpha=0.7,
                  label='Quartile Range'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2,
                   label='Median (Box)'),
            Patch(facecolor='gray', alpha=0.3,
                  label='Distribution'),
            Line2D([0], [0], marker='o', color='darkred',
                   markersize=8, alpha=0.5,
                   label='Individual Incidents'),
            Line2D([0], [0], color='red', linestyle='-', linewidth=2.5, alpha=0.7,
                   label='Median (Overall)'),
            Line2D([0], [0], color='blue', linestyle='--', linewidth=2, alpha=0.5,
                   label='Mean')
        ]
        
        ax.legend(handles=legend_elements,
                 loc='center left',
                 bbox_to_anchor=(1.02, 0.5),
                 title='Plot Elements',
                 title_fontsize=18,
                 framealpha=0.9,
                 edgecolor='black',
                 fontsize=16)
        
        # Get impact colors from data
        impact_colors = {}
        for level in range(5):
            color = df[df['incident_impact_level'] == level]['Incident_color'].iloc[0] \
                    if len(df[df['incident_impact_level'] == level]) > 0 else '#333333'
            impact_colors[level] = color
        
        # Set y-axis ticks and colored labels
        severity_labels = [IMPACT_LEVELS[i].split(' - ')[0] for i in range(5)]
        ax.set_yticks(range(5))
        ax.set_yticklabels([])  # Remove default labels
        
        # Add colored impact level labels
        for i, label in enumerate(severity_labels):
            ax.text(-0.1, i, f"{i} ({label})", 
                    transform=ax.get_yaxis_transform(),
                    ha='right',
                    va='center',
                    color=impact_colors[i],
                    fontweight='bold',
                    fontsize=18)
        
        # Customize plot
        ax.set_xlabel(r'$\mathrm{Provider}$', fontsize=20, labelpad=15)
        ax.set_ylabel(r'$\mathrm{Impact\;Level}$', fontsize=20, labelpad=15)
        ax.set_ylim(-0.5, 4.5)
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')
        
        # Add box around plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')
        
        # Add timeframe to title
        timeframe_start = pd.to_datetime(start_date).strftime('%d %B %Y').replace(' ', '\\;')
        timeframe_end = pd.to_datetime(end_date).strftime('%d %B %Y').replace(' ', '\\;')
        title = f"$\\mathrm{{Distribution\\;of\\;Incident\\;Impact\\;Levels\\;({timeframe_start}\\;-\\;{timeframe_end})}}$"
        if df.empty:
            title += "\n$\\mathrm{(No\\;incidents\\;recorded\\;in\\;this\\;period)}$"
        fig.suptitle(title, y=0.95, fontsize=22)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, top=0.88, bottom=0.12, left=0.15)
        
        return fig

    except Exception as e:
        print(f"Error in incident distribution analysis: {str(e)}")
        traceback.print_exc()
        return None
