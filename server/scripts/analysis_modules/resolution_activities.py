import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)

def analyze_resolution_activities(start_date, end_date, selected_services):
    """
    Analyze and visualize resolution activities and patterns
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data(
            start_date, 
            end_date,
            timestamp_columns=[
                'investigating_timestamp', 'identified_timestamp',
                'monitoring_timestamp', 'resolved_timestamp',
                'postmortem_timestamp', 'start_timestamp', 'close_timestamp'
            ]
        )
        
        # Get services to analyze
        services_to_analyze = get_services_to_analyze(selected_services)
        
        # Setup plotting style
        setup_plotting_style()

        # Calculate durations for each stage
        dfs = {}
        for service in services_to_analyze:
            service_df = df[df[service] == 1].copy()
            service_df = service_df.sort_values('start_timestamp').reset_index(drop=True)
            
            # Calculate stage durations
            service_df['P_investigate'] = (service_df['identified_timestamp'] - 
                                         service_df['investigating_timestamp']).dt.total_seconds() / 3600
            service_df['P_repair'] = (service_df['monitoring_timestamp'] - 
                                    service_df['identified_timestamp']).dt.total_seconds() / 3600
            service_df['P_check'] = (service_df['resolved_timestamp'] - 
                                   service_df['monitoring_timestamp']).dt.total_seconds() / 3600
            service_df['P_learn'] = (service_df['postmortem_timestamp'] - 
                                   service_df['resolved_timestamp']).dt.total_seconds() / 3600
            
            dfs[service] = service_df

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Plot 1: Average stage durations
        stage_names = ['Investigation', 'Repair', 'Monitoring']
        stage_cols = ['P_investigate', 'P_repair', 'P_check']
        colors = sns.color_palette('husl', n_colors=len(stage_names))
        
        x = np.arange(len(services_to_analyze))
        width = 0.8
        bottom = np.zeros(len(services_to_analyze))

        for i, (stage, col) in enumerate(zip(stage_names, stage_cols)):
            values = [dfs[service][col].mean() for service in services_to_analyze]
            ax1.bar(x, values, width, label=stage, bottom=bottom, color=colors[i])
            
            # Add percentage labels
            for j, v in enumerate(values):
                if v > 0:
                    ax1.text(j, bottom[j] + v/2, f'{v:.1f}h', 
                            ha='center', va='center', color='white')
                bottom[j] += v

        ax1.set_title('Average Resolution Stage Durations')
        ax1.set_xlabel('Service')
        ax1.set_ylabel('Duration (hours)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace('_', '\n') for s in services_to_analyze], rotation=45)
        ax1.legend()

        # Plot 2: Stage duration percentages
        bottom = np.zeros(len(services_to_analyze))
        
        for i, (stage, col) in enumerate(zip(stage_names, stage_cols)):
            values = [dfs[service][col].sum() for service in services_to_analyze]
            total = np.array([dfs[service][stage_cols].sum().sum() for service in services_to_analyze])
            percentages = values / total * 100
            
            ax2.bar(x, percentages, width, label=stage, bottom=bottom, color=colors[i])
            
            # Add percentage labels
            for j, v in enumerate(percentages):
                if v > 0:
                    ax2.text(j, bottom[j] + v/2, f'{v:.1f}%', 
                            ha='center', va='center', color='white')
                bottom[j] += v

        ax2.set_title('Resolution Stage Distribution')
        ax2.set_xlabel('Service')
        ax2.set_ylabel('Percentage of Total Resolution Time')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', '\n') for s in services_to_analyze], rotation=45)
        ax2.legend()

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in resolution activities analysis: {str(e)}")
        raise 