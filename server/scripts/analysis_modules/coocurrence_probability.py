# Filename: cooccurrence_probability.py

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

def analyze_cooccurrence_probability(start_date, end_date, selected_services):
    """
    Analyze and visualize service outage co-occurrence probabilities.

    Args:
        start_date (datetime): Start date for filtering outage data.
        end_date (datetime): End date for filtering outage data.
        selected_services (list): List of service identifiers to analyze.

    Returns:
        fig (matplotlib.figure.Figure): The figure object with the co-occurrence probability heatmap.
    """
    try:
        # Step 1: Load and Prepare Outage Data
        service_mapping_outage = get_service_mapping_outage()
        
        # Map selected_services to their corresponding standardized service names in the outage CSV
        # If 'selected_services' are in 'Provider:Service' format, ensure the mapping is correct
        # For example, 'OpenAI:API' should map to 'api' or a standardized name
        # Adjust the mapping based on your actual data
        services_to_analyze_mapped = []
        for service in selected_services:
            # Assuming 'service_mapping_outage' maps 'Provider:Service' to standardized names
            mapped_service = service_mapping_outage.get(service, service)
            services_to_analyze_mapped.append(mapped_service)
        
        # Debug: Print mapped services
        print("Selected Services Mapped:", services_to_analyze_mapped)
        
        df_outage = load_and_prepare_data_outage(start_date, end_date)
        
        # Convert 'outage_flag' to numeric (0 or 1)
        df_outage['outage_flag'] = pd.to_numeric(df_outage['outage_flag'], errors='coerce').fillna(0).astype(int)
        
        # Step 2: Filter by Selected Services
        if 'Service' not in df_outage.columns:
            raise KeyError("'Service' column is missing in the outage data.")
        
        # Filter DataFrame to include only the selected services
        df_outage = df_outage[df_outage['Service'].isin(services_to_analyze_mapped)]
        
        # No reverse mapping needed; keep 'Service' as standardized names
        # Remove or comment out the reverse mapping to maintain consistency
        # service_mapping_reverse = {v: k for k, v in service_mapping_outage.items()}
        # df_outage['Service'] = df_outage['Service'].map(lambda x: service_mapping_reverse.get(x, x))
        
        # Step 3: Ensure Complete Date Range and Pivot Data
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates_df = pd.DataFrame({'Date': all_dates})
        
        # Pivot outage data to have services as columns and dates as rows
        pivot_df = df_outage.pivot_table(index='Date', columns='Service', values='outage_flag', fill_value=0)
        
        # Ensure all selected services are present as columns
        for service in services_to_analyze_mapped:
            if service not in pivot_df.columns:
                pivot_df[service] = 0
        
        # Debug: Print pivot DataFrame shape and columns before selection
        print("Pivot shape before selecting columns:", pivot_df.shape)
        print("Column names before selecting:", pivot_df.columns.tolist())
        
        # Reorder columns to match the mapped services list
        # This ensures consistency in the heatmap
        pivot_df = pivot_df[services_to_analyze_mapped]
        
        # Debug: Print pivot DataFrame shape and columns after selection
        print("Pivot shape after selecting columns:", pivot_df.shape)
        print("Column names after selecting:", pivot_df.columns.tolist())
        
        # Check if there are nonzero values in the pivot table
        if pivot_df.empty:
            raise ValueError("No data to analyze after filtering services.")
        else:
            nonzero_count = pivot_df.astype(bool).sum().sum()
            zero_count = (pivot_df == 0).sum().sum()
            print("Amount of nonzero values in the pivot table:", nonzero_count)
            print("Amount of zero values in the pivot table:", zero_count)
        
        # Step 4: Calculate Co-occurrence Probabilities
        outage_probability = {}
        for service_ref in services_to_analyze_mapped:
            proportions = []
            for service_cmp in services_to_analyze_mapped:
                total_cmp_outages = pivot_df[service_cmp].sum()
                if total_cmp_outages > 0:
                    co_outages = ((pivot_df[service_cmp] == 1) & (pivot_df[service_ref] == 1)).sum()
                    p = (co_outages / total_cmp_outages) * 100
                else:
                    p = 0.0
                proportions.append(round(p, 2))
            outage_probability[service_ref] = proportions
        
        # Create DataFrame for heatmap
        df_outage_probability = pd.DataFrame(outage_probability, index=services_to_analyze_mapped)
        df_outage_probability = df_outage_probability.T  # Transpose to have services as both rows and columns
        
        # Debug: Print outage probability DataFrame
        print("Outage Probability DataFrame:\n", df_outage_probability)
        
        # Step 5: Save the Probability Table to CSV
        # Reset index to include 'Service' as a column
        df_outage_probability.reset_index(inplace=True)
        df_outage_probability.rename(columns={'index': 'Service'}, inplace=True)
        
        # Step 6: Create Heatmap Visualization
        setup_plotting_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define color palette and normalization
        custom_colors = sns.color_palette('Greys', 5)
        cmap = ListedColormap(custom_colors)
        custom_bins = [0, 20, 40, 60, 80, 100]
        norm = BoundaryNorm(custom_bins, cmap.N)
        
        # Create the heatmap
        # Set 'Service' as index for proper heatmap plotting
        heatmap_data = df_outage_probability.set_index('Service')
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=".2f",
            cbar_kws={'label': 'Co-occurrence Outage Probability (%)'},
            linewidths=.5,
            linecolor='gray',
            ax=ax
        )
        
        # Customize heatmap
        ax.set_title('Service Outage Co-occurrence Probability Matrix', fontsize=16, pad=20)
        ax.set_xlabel('Service', fontsize=14)
        ax.set_ylabel('Service', fontsize=14)
        
        # Rotate x-axis labels and set horizontal alignment
        ax.tick_params(axis='x', rotation=30, labelsize=14)
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_fontweight('bold')  # Make labels bold
        
        # Rotate y-axis labels and set font size
        ax.tick_params(axis='y', rotation=0, labelsize=14)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')  # Make labels bold
        
        # Remove y-axis label as per sample code
        ax.set_ylabel('')
        
        plt.tight_layout()
        
        # Display the heatmap
        plt.show()
        
        return fig
    except Exception as e:
        print(f"Error in service outage co-occurrence probability analysis: {str(e)}")
        raise