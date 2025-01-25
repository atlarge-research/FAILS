import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import utility functions
# Ensure that these utility functions are correctly implemented and accessible.
from .utils import (
    load_and_prepare_data_outage,
    setup_plotting_style
)

def analyze_service_outages(start_date, end_date, services):
    """
    Analyze and visualize service outage details, showing combinations of services
    that were impacted together during outages.

    Args:
        start_date (datetime): Start date for filtering outage data.
        end_date (datetime): End date for filtering outage data.
        services (list): List of service identifiers to analyze.

    Returns:
        dict: Dictionary containing Matplotlib Figure objects for each provider.
    """
    try:
        # Load and prepare data
        df = load_and_prepare_data_outage(start_date, end_date)
        
        # Group services by provider
        provider_mapping = {
            'openai': {
                'services': ['api', 'chatgpt', 'labs', 'playground'],
                'labels': {
                    'api': 'O1: API-OpenAI',
                    'chatgpt': 'O2: ChatGPT',
                    'labs': 'O3: DALL·E',
                    'playground': 'O4: Playground'
                }
            },
            'anthropic': {
                'services': ['api.anthropic.com', 'claude.ai', 'console.anthropic.com'],
                'labels': {
                    'api.anthropic.com': 'A1: API-Anthropic',
                    'claude.ai': 'A2: Claude',
                    'console.anthropic.com': 'A3: Console'
                }
            }
        }

        figures = {}
        for provider, config in provider_mapping.items():
            # Get data for this provider
            provider_data = df[df['provider'] == provider].copy()
            if provider_data.empty:
                print(f"No data available for provider: {provider}")
                continue

            # Create binary columns for each service
            for service in config['services']:
                provider_data[service] = (provider_data['Service'] == service).astype(int)
            
            # Create combinations of impacted services
            def get_impacted_services(row):
                impacted = []
                for service in config['services']:
                    if row[service] == 1:
                        short_label = config['labels'][service].split(':')[0]
                        impacted.append(short_label)
                return ', '.join(sorted(impacted)) if impacted else 'None'
            
            provider_data['impacted_services'] = provider_data.apply(get_impacted_services, axis=1)
            
            # Count combinations
            combinations_count = provider_data['impacted_services'].value_counts()
            combinations_count = combinations_count[combinations_count.index != 'None']
            
            if combinations_count.empty:
                print(f"No service combinations found for provider: {provider}")
                continue

            # Plot
            plt.figure(figsize=(10, 8))
            ax = combinations_count[::-1].plot(
                kind='barh',
                color='grey',
                edgecolor='black'
            )
            
            # Add count labels
            for i, count in enumerate(combinations_count[::-1]):
                plt.text(count + 0.5, i, str(count), ha='left', va='center', fontsize=12)
            
            plt.title(f"{provider.capitalize()}", fontsize=16, pad=20)
            plt.ylabel("Impacted Services", fontsize=14)
            plt.xlabel("Number of Incidents", fontsize=14)
            
            # Add service label mapping
            label_text = '\n'.join([f"{k}: {v.split(':')[1].strip()}" 
                                  for k, v in config['labels'].items()])
            plt.text(
                0.99, 0.01, label_text,
                ha='right', va='bottom',
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5)
            )
            
            plt.tight_layout()
            figures[provider] = plt.gcf()
        
        return figures

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
