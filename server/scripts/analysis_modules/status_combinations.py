import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import (
    load_and_prepare_data,
    get_services_to_analyze,
    setup_plotting_style
)

def analyze_status_combinations(start_date, end_date, selected_services):
    """
    Analyze failure recovery patterns and generate stacked bar plot
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
        
        # Get services to analyze
        services_to_analyze = get_services_to_analyze(selected_services)
        
        # Create separate DataFrames for each service
        dfs = {}
        valid_services = []  # Keep track of services that have incidents
        for service in services_to_analyze:
            if service in df.columns:
                service_data = df[df[service] == 1]
                if not service_data.empty:
                    dfs[service] = service_data.reset_index(drop=True)
                    valid_services.append(service)
                    print(f"\nFound {len(service_data)} incidents for {service}")
                else:
                    print(f"\nNo incidents found for {service}")
            else:
                print(f"\nWarning: Service column {service} not found in data")

        # Use only valid services for analysis
        services_to_analyze = valid_services

        # Setup plotting style
        setup_plotting_style()

        # Initialize count dictionary for each service
        service_counts = {}
        
        # Get all unique status combinations from the data
        status_flags = ['investigating_flag', 'identified_flag', 'monitoring_flag', 
                       'resolved_flag', 'postmortem_flag']
        
        all_combinations = set()
        for service in services_to_analyze:
            for _, record in dfs[service].iterrows():
                flags = [record[flag] for flag in status_flags]
                status_str = '_'.join([f'S{i+1}' for i, flag in enumerate(flags) if flag == 1])
                if status_str:  # Only add non-empty combinations
                    all_combinations.add(status_str)
        
        status_combinations = sorted(list(all_combinations))

        for service in services_to_analyze:
            counts = {combo: 0 for combo in status_combinations}
            for _, record in dfs[service].iterrows():
                flags = [record[flag] for flag in status_flags]
                status_str = '_'.join([f'S{i+1}' for i, flag in enumerate(flags) if flag == 1])
                if status_str in counts:
                    counts[status_str] += 1
            
            service_counts[service] = counts

        # Create DataFrame for plotting
        plot_data = pd.DataFrame(service_counts).T
        plot_data = plot_data.fillna(0)

        # Calculate percentages
        row_sums = plot_data.sum(axis=1)
        plot_data_pct = plot_data.div(row_sums, axis=0)

        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot stacked bars
        bottom = np.zeros(len(services_to_analyze))
        colors = sns.color_palette('tab10', n_colors=len(status_combinations))
        
        # Service labels mapping
        service_labels = {
            'API': 'O1',
            'ChatGPT': 'O2',
            'DALL-E': 'O3',
            'Playground': 'O4',
            'api.anthropic.com': 'A1',
            'claude.ai': 'A2',
            'console.anthropic.com': 'A3',
            'Character.AI': 'C1',
            'REST API': 'S1',
            'gRPC API': 'S2',
            'Stable Assistant': 'S3'
        }

        # Create x-tick labels
        x_tick_labels = [service_labels.get(service, service) for service in services_to_analyze]
        
        # Plot each status combination
        for i, status in enumerate(status_combinations):
            values = plot_data_pct[status]
            ax.bar(range(len(services_to_analyze)), values, bottom=bottom, 
                  label=status, color=colors[i], width=0.85)
            
            # Add percentage labels
            for j, v in enumerate(values):
                if v > 0.01:  # Only show labels for values > 1%
                    ax.text(j, bottom[j] + v/2, f'{v:.1%}', 
                           ha='center', va='center', color='white',
                           fontsize=12, fontweight='bold')
            bottom += values

        # Customize plot
        ax.set_xticks(range(len(services_to_analyze)))
        ax.set_xticklabels(x_tick_labels, fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
        
        # Add legend with status descriptions
        status_descriptions = {
            'S1': 'Investigating',
            'S2': 'Identified',
            'S3': 'Monitoring',
            'S4': 'Resolved',
            'S5': 'Postmortem'
        }
        
        # Add service label mapping as text
        label_text = '\n'.join([
            f"{short}: {long}" for short, long in service_labels.items()
            if short in services_to_analyze
        ])
        plt.text(
            1.15, 0.02, label_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.8)
        )

        plt.legend(title='Status Combinations', 
                  bbox_to_anchor=(1.05, 1.0),
                  loc='upper left')
        
        plt.tight_layout()
        
        return fig

    except Exception as e:
        print(f"Error in failure recovery analysis: {str(e)}")
        raise