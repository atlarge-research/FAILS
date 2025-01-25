import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from .utils import (
    load_and_prepare_data_outage,
    get_services_to_analyze_outage,
    setup_plotting_style,
    get_service_mapping_outage
)
import traceback

def analyze_outage_cooccurrence_matrix(start_date, end_date, selected_services):
    """
    Analyze and visualize service outage co-occurrence patterns.

    Args:
        start_date (datetime): Start date for filtering outage data.
        end_date (datetime): End date for filtering outage data.
        selected_services (list): List of service identifiers to analyze.

    Returns:
        fig (matplotlib.figure.Figure): The figure object with the co-occurrence heatmap.
    """
    try:
        # Step 1: Load and Prepare Outage Data
        service_mapping_outage = get_service_mapping_outage()
        # Map selected_services to their corresponding service names in the outage CSV
        services_to_analyze_mapped = [service_mapping_outage.get(service, service) for service in selected_services]

        print(f"Selected services to analyze: {services_to_analyze_mapped}")
        
        df_outage = load_and_prepare_data_outage(start_date, end_date)

        print(f"The services in the outage data are: {df_outage['Service'].unique()}")
        
        # Convert 'outage_flag' to numeric (0 or 1)
        df_outage['outage_flag'] = pd.to_numeric(df_outage['outage_flag'], errors='coerce').fillna(0).astype(int)
        
        # Filter by selected services
        if 'Service' not in df_outage.columns:
            raise KeyError("'Service' column is missing in the outage data.")
        
        df_outage = df_outage[df_outage['Service'].isin(services_to_analyze_mapped)]
        
        # Rename services to standardized names if necessary
        # Assuming 'get_service_mapping_outage' maps selected_services to standardized names
        # If not, adjust accordingly based on your mapping function
        service_mapping_reverse = {v: k for k, v in service_mapping_outage.items()}
        df_outage['Service'] = df_outage['Service'].map(lambda x: service_mapping_reverse.get(x, x))
        
        # Step 2: Ensure Complete Date Range
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates_df = pd.DataFrame({'Date': all_dates})
        
        # Pivot outage data to have services as columns and dates as rows
        pivot_df = df_outage.pivot_table(index='Date', columns='Service', values='outage_flag', fill_value=0)
        
        # Ensure all selected services are present as columns
        for service in selected_services:
            standardized_service = service_mapping_outage.get(service, service)
            if standardized_service not in pivot_df.columns:
                pivot_df[standardized_service] = 0

        # Don't remove StabilityAI services
        if not selected_services:
            raise ValueError("No services selected for analysis")

        # Ensure pivot_df has all selected services
        for service in selected_services:
            if service not in pivot_df.columns:
                pivot_df[service] = 0

        pivot_df = pivot_df[selected_services]
        
        # Step 3: Calculate Co-occurrence Matrix
        cooccurrence = np.zeros((len(selected_services), len(selected_services)), dtype=int)
        
        for i, service1 in enumerate(selected_services):
            for j, service2 in enumerate(selected_services):
                if i <= j:
                    # Count days where both services have outage_flag == 1
                    count = ((pivot_df[service1] == 1) & (pivot_df[service2] == 1)).sum()
                    cooccurrence[i, j] = count
                    if i != j:
                        cooccurrence[j, i] = count  # Mirror for lower triangle
        
        # Create DataFrame for heatmap
        cooccurrence_df = pd.DataFrame(cooccurrence, index=selected_services, columns=selected_services)
        
        # Step 4: Calculate Percentages for Annotations
        total_outages = pivot_df.sum()
        
        # Initialize annotation DataFrame
        annotation_df = cooccurrence_df.copy().astype(str)
        
        for i in range(len(selected_services)):
            for j in range(len(selected_services)):
                if i != j and cooccurrence_df.iloc[i, j] > 0:
                    pct_i = (cooccurrence_df.iloc[i, j] / total_outages[selected_services[i]]) * 100 if total_outages[selected_services[i]] > 0 else 0
                    pct_j = (cooccurrence_df.iloc[i, j] / total_outages[selected_services[j]]) * 100 if total_outages[selected_services[j]] > 0 else 0
                    annotation_df.iloc[i, j] = f'{cooccurrence_df.iloc[i, j]:.0f}\n({pct_i:.1f}%/{pct_j:.1f}%)'
                else:
                    annotation_df.iloc[i, j] = f'{cooccurrence_df.iloc[i, j]:.0f}'
        
        # Step 5: Create Heatmap Visualization
        setup_plotting_style()
        fig, ax = plt.subplots(figsize=(14, 12))  # Larger figure for better readability

        # Define color palette and normalization
        custom_colors = sns.color_palette('YlOrRd', as_cmap=True)
        max_count = cooccurrence_df.values.max()
        custom_bins = np.linspace(0, max_count, num=6)  # 5 bins
        norm = BoundaryNorm(custom_bins, custom_colors.N)

        # Create the heatmap
        sns.heatmap(
            cooccurrence_df,
            annot=annotation_df,
            fmt='',
            cmap=custom_colors,
            norm=norm,
            cbar_kws={'label': 'Number of Co-occurring Outages'},
            square=True,
            linewidths=0.5,
            ax=ax
        )

        # Customize heatmap
        ax.set_title('Service Outage Co-occurrence Matrix', fontsize=18, weight='bold', pad=20)
        ax.set_xlabel('Service', fontsize=14, labelpad=10)
        ax.set_ylabel('Service', fontsize=14, labelpad=10)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12, rotation=0)

        # Adjust layout for better display
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Add padding at the top

        return fig

    except Exception as e:
        print(f"Error in outage co-occurrence matrix analysis: {str(e)}")
        traceback.print_exc()
        return None
