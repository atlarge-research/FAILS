import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

IMPACT_LEVELS = {
    0: "None - Post-resolved notification or documentation update",
    1: "Minor - Brief disruption or elevated error rates with limited impact",
    2: "Major - Significant service degradation affecting core functionality",
    3: "Critical - Complete service outage or severe widespread impact",
    4: "Maintenance - Planned maintenance or scheduled updates"
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_provider_display_name(provider):
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

def analyze_impact_levels(df, output_dir):
    df['incident_impact_level'] = pd.to_numeric(df['incident_impact_level'])
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    timeframe_start = df['start_timestamp'].min()
    timeframe_end = df['start_timestamp'].max()
    
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.boxplot(
        data=df,
        x='provider',
        y='incident_impact_level',
        width=0.3,
        color='lightgray',
        showfliers=False,
        zorder=2
    )
    
    sns.violinplot(
        data=df,
        x='provider',
        y='incident_impact_level',
        bw_method=0.5,
        cut=0.8,
        density_norm='width',
        inner=None,
        alpha=0.5,
        zorder=1,
        color='cornflowerblue'
    )
    
    providers = df['provider'].unique()
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
                
                plt.scatter(
                    x_positions,
                    [impact_level] * n_points,
                    color='darkred',
                    marker='o',
                    alpha=0.6,
                    edgecolor='black',
                    linewidth=0.5
                )
    
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', alpha=0.7,
              label='Quartile Range'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2,
               label='Median'),
        Patch(facecolor='gray', alpha=0.3,
              label='Distribution'),
        Line2D([0], [0], marker='o', color='darkred',
               markersize=8, alpha=0.5,
               label='Individual Incidents'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=2.5, alpha=0.7,
               label='Median'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, alpha=0.5,
               label='Mean')
    ]
    
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5),
              title='Plot Elements',
              title_fontsize=26,
              framealpha=0.9,
              edgecolor='black',
              fontsize=26)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.88, bottom=0.12, left=0.15)
    
    plt.xticks(range(len(providers)), 
               [get_provider_display_name(p) for p in providers],
               rotation=0,
               fontsize=26)
    
    timeframe_str = f"{timeframe_start.strftime('%B %Y')} - {timeframe_end.strftime('%B %Y')}"
    
    # fig.suptitle(r'$\mathrm{Distribution\;of\;Incident\;Impact\;Levels\;and\;Analysis\;Period:\; ' + timeframe_str + '}$', y=0.95, x=0.5, fontsize=24, ha='center')
    
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    
    impact_colors = {}
    for level in range(5):
        color = df[df['incident_impact_level'] == level]['Incident_color'].iloc[0] \
                if len(df[df['incident_impact_level'] == level]) > 0 else '#333333'
        impact_colors[level] = color
    
    severity_labels = [IMPACT_LEVELS[i].split(' - ')[0] for i in range(5)]
    plt.yticks(range(5))
    
    for i, label in enumerate(severity_labels):
        ax.text(-0.1, i, f"{i} ({label})", 
                transform=ax.get_yaxis_transform(),
                ha='right',
                va='center',
                color='black',
                fontweight='bold',
                fontsize=26)
    
    ax.set_yticklabels([])
    
    plt.ylim(-0.5, 4.5)
    
    plt.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    plt.savefig(
        os.path.join(output_dir, 'impact_level_distribution_'+timeframe_str+'.png'), 
        dpi=300, 
        bbox_inches='tight', 
        facecolor='white',
        pad_inches=0.2
    )
    plt.close()
    
    stats_dict = {}
    for provider in providers:
        provider_data = df[df['provider'] == provider]
        
        stats_dict[provider] = {
            'count': provider_data.shape[0],
            'mean': provider_data['incident_impact_level'].mean(),
            'median': provider_data['incident_impact_level'].median(),
            'std': provider_data['incident_impact_level'].std(),
            'mode': provider_data['incident_impact_level'].mode()[0],
            'skewness': provider_data['incident_impact_level'].skew(),
            'kurtosis': provider_data['incident_impact_level'].kurt(),
            'distribution': provider_data['incident_impact_level'].value_counts().to_dict()
        }
    
    return stats_dict, timeframe_start, timeframe_end

def format_stats_report(stats_dict, timeframe_start, timeframe_end):
    report = "Incident Impact Level Distribution Analysis\n"
    report += "=" * 50 + "\n\n"
    
    report += f"Analysis Period: {timeframe_start.strftime('%B %Y')} - {timeframe_end.strftime('%B %Y')}\n"
    report += "=" * 50 + "\n\n"
    
    report += "Impact Level Definitions:\n"
    report += "-" * 30 + "\n"
    for level, description in IMPACT_LEVELS.items():
        severity = description.split(' - ')[0]
        details = description.split(' - ')[1]
        report += f"Level {level} ({severity}): {details}\n"
    report += "\n" + "=" * 50 + "\n\n"
    
    report += "Cross-Provider Impact Level Comparison:\n"
    report += "-" * 30 + "\n"
    report += ("Note: Impact levels are standardized across providers to enable comparison.\n"
              "However, each provider may have slightly different internal criteria for\n"
              "assigning impact levels. The standardization is based on:\n"
              "- Scope of impact (number of users/services affected)\n"
              "- Duration of incident\n"
              "- Severity of service degradation\n"
              "- Business impact\n\n")
    report += "=" * 50 + "\n\n"
    
    for provider, stats in stats_dict.items():
        report += f"Provider: {get_provider_display_name(provider)}\n"
        report += "-" * 30 + "\n"
        report += f"Total Incidents: {stats['count']}\n"
        report += f"Mean Impact Level: {stats['mean']:.2f}\n"
        report += f"Median Impact Level: {stats['median']:.2f}\n"
        report += f"Standard Deviation: {stats['std']:.2f}\n"
        report += f"Mode: {stats['mode']}\n"
        report += f"Skewness: {stats['skewness']:.2f}\n"
        report += f"Kurtosis: {stats['kurtosis']:.2f}\n"
        
        report += "\nDistribution of Impact Levels:\n"
        for level, count in stats['distribution'].items():
            percentage = (count / stats['count']) * 100
            severity = IMPACT_LEVELS[level].split(' - ')[0]
            details = IMPACT_LEVELS[level].split(' - ')[1]
            report += f"Level {level} ({severity}): {details}\n"
            report += f"  {count} incidents ({percentage:.1f}%)\n"
        
        report += "\n" + "=" * 50 + "\n\n"
    
    return report

def main():
    output_dir = 'server/static/additional_analysis/incident_impact_levels'
    ensure_dir(output_dir)
    
    df = pd.read_csv('server/static/data/incident_stages_all.csv')
    
    stats_dict, timeframe_start, timeframe_end = analyze_impact_levels(df, output_dir)
    
    report = format_stats_report(stats_dict, timeframe_start, timeframe_end)
    
    report_path = os.path.join(output_dir, 'impact_level_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    return stats_dict, report

if __name__ == "__main__":
    main()
