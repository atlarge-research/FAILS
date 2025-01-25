import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from datetime import datetime
import pandas as pd
from .analysis_modules.monthly_overview import analyze_monthly_overview
from .analysis_modules.daily_overview import analyze_daily_overview
from .analysis_modules.status_combinations import analyze_status_combinations
from .analysis_modules.mttr_distribution import analyze_mttr_distribution
from .analysis_modules.mtbf_distribution import analyze_mtbf_distribution
from .analysis_modules.resolution_activities import analyze_resolution_activities
from .analysis_modules.daily_availability import analyze_daily_availability
from .analysis_modules.cooccurrence_matrix import analyze_outage_cooccurrence_matrix
from .analysis_modules.auto_correlations import analyze_autocorrelation
from .analysis_modules.incident_outage import timeline_incident_outage
from .analysis_modules.coocurrence_probability import analyze_cooccurrence_probability
from .analysis_modules.service_incidents import analyze_service_incidents
from .analysis_modules.mttr_boxplot import analyze_mttr_boxplot
from .analysis_modules.mtbf_boxplot import analyze_mtbf_boxplot
from .analysis_modules.mttr_provider import analyze_mttr_provider
from .analysis_modules.mtbf_provider import analyze_mtbf_provider
from .analysis_modules.incident_distribution import analyze_incident_distribution
import base64
from openai import OpenAI
from dotenv import load_dotenv
import io
import traceback

# Get the absolute path to the .env file
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env at: {ENV_PATH}")

# Load environment variables from the correct location
load_dotenv(ENV_PATH)

# print(f"API Key available: {'OPENAI_API_KEY' in os.environ}")
# print(f"API Key value: {os.getenv('OPENAI_API_KEY')[:5]}..." if os.getenv('OPENAI_API_KEY') else "No API key found")

# Clear any existing proxy settings that might interfere
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']

try:
    # First attempt: Full configuration
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1",
        timeout=30.0
    )
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")
    try:
        # Second attempt: Minimal configuration
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e2:
        print(f"Warning: Second attempt failed: {e2}")
        try:
            # Third attempt: Legacy initialization
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            client = openai
        except Exception as e3:
            print(f"Warning: All initialization attempts failed: {e3}")
            client = None

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'plots')

def save_plot(fig, plot_type, start_date, end_date, services):
    """Helper function to save plots with consistent formatting and unique names"""
    start_str = pd.to_datetime(start_date).strftime('%Y%m%d')
    end_str = pd.to_datetime(end_date).strftime('%Y%m%d')
    
    service_names = []
    for service in services:
        provider, name = service.split(':')
        if provider == 'OpenAI':
            service_names.append(f'OpenAI_{name}')
        elif provider == 'Anthropic':
            if name == 'API':
                service_names.append('Anthropic_API')
            elif name == 'Claude':
                service_names.append('Anthropic_Claude')
            elif name == 'Console':
                service_names.append('Anthropic_Console')
        elif provider == 'Character.AI':
            service_names.append('Character.AI')
        elif provider == 'StabilityAI':
            if name == 'REST':
                service_names.append('StabilityAI_REST')
            elif name == 'gRPC':
                service_names.append('StabilityAI_gRPC')
            elif name == 'Assistant':
                service_names.append('StabilityAI_Assistant')
            else:
                service_names.append(f'StabilityAI_{name}')
        elif provider == 'Google':
            service_names.append(f'Google_{name}')
    
    service_str = '-'.join(service_names)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%H%M%S')
    filename = f'{plot_type}_{start_str}_{end_str}__{service_str}__{timestamp}.png'
    
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)
    return f'/static/plots/{filename}'

def cleanup_old_plots():
    """Remove plots older than 1 hour"""
    current_time = datetime.now()
    for filename in os.listdir(PLOTS_DIR):
        if filename.endswith('.png'):
            file_path = os.path.join(PLOTS_DIR, filename)
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if (current_time - file_time).total_seconds() > 3600:  
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to remove old plot {filename}: {e}")

def generate_monthly_overview(start_date, end_date, services):
    """Generate monthly overview of incidents"""
    try:
        cleanup_old_plots()
        fig = analyze_monthly_overview(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'monthly_overview', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating monthly overview: {e}")
        return None
    
def generate_daily_overview(start_date, end_date, services):
    """Generate daily overview of incidents"""
    try:
        cleanup_old_plots()
        fig = analyze_daily_overview(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'daily_overview', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating daily overview: {e}")
        return None
    
def generate_cooccurrence_probability(start_date, end_date, services):
    """Generate co-occurrence probability analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_cooccurrence_probability(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'cooccurrence_probability', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating co-occurrence probability: {e}")
        return None
    
def generate_service_incidents(start_date, end_date, services):
    """Generate service incidents analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_service_incidents(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'service_incidents', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating service incidents: {e}")
        return None
    
    
def generate_incident_outage(start_date, end_date, services):
    """Generate incident outage timeline analysis"""
    try:
        cleanup_old_plots()
        fig = timeline_incident_outage(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'incident_outage', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating incident outage: {e}")
        return None
    

def generate_status_combinations(start_date, end_date, services):
    """Generate status combinations analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_status_combinations(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'status_combinations', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating status combinations: {e}")
        return None

def generate_resolution_activities(start_date, end_date, services):
    """Generate resolution activities analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_resolution_activities(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'resolution_activities', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating resolution activities: {e}")
        return None

def generate_mttr_distribution(start_date, end_date, services):
    """Generate MTTR distribution analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_mttr_distribution(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'mttr_distribution', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating MTTR distribution: {e}")
        return None

def generate_mtbf_distribution(start_date, end_date, services):
    """Generate MTBF distribution analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_mtbf_distribution(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'mtbf_distribution', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating MTBF distribution: {e}")
        return None

def generate_mttr_provider(start_date, end_date, services):
    """Generate MTTR per provider analysis"""
    fig, ax = plt.subplots(figsize=(8, 6))

    return save_plot(fig, 'figure7.png', start_date, end_date, services)

def generate_mtbf_provider(start_date, end_date, services):
    """Generate MTBF per provider analysis"""
    fig, ax = plt.subplots(figsize=(8, 6))

    return save_plot(fig, 'figure8.png', start_date, end_date, services)


def generate_autocorrelations(start_date, end_date, services):
    """Generate autocorrelations analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_autocorrelation(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'autocorrelations', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating autocorrelations: {e}")
        return None

def generate_daily_availability(start_date, end_date, services):
    """Generate daily availability analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_daily_availability(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'daily_availability', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating daily availability: {e}")
        return None

def generate_cooccurrence_matrix(start_date, end_date, services):
    """Generate co-occurrence matrix analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_outage_cooccurrence_matrix(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'cooccurrence_matrix', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating co-occurrence matrix: {e}")
        return None

# def generate_all_plots(start_date, end_date, services):
#     plots = {}
#     errors = []
    
#     try:
#         # Daily Availability
#         try:
#             daily_availability_path = generate_daily_availability(start_date, end_date, services)
#             if daily_availability_path:
#                 plots['figure11'] = daily_availability_path
#         except Exception as e:
#             print(f"Error generating daily availability: {str(e)}")
#             errors.append(f"Daily availability failed: {str(e)}")
            
#         # Service Co-occurrence
#         try:
#             cooccurrence_matrix_path = generate_cooccurrence_matrix(start_date, end_date, services)
#             if cooccurrence_matrix_path:
#                 plots['figure12'] = cooccurrence_matrix_path
#         except Exception as e:
#             print(f"Error generating co-occurrence matrix: {str(e)}")
#             errors.append(f"Co-occurrence matrix failed: {str(e)}")
            
        
#     except Exception as e:
#         print(f"Error in generate_all_plots: {str(e)}")
#         errors.append(f"Overall plot generation failed: {str(e)}")
        
#     return plots, errors

# Add individual generate functions for each plot type
def generate_mttr_boxplot(start_date, end_date, services):
    """Generate MTTR boxplot analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_mttr_boxplot(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'mttr_boxplot', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating MTTR boxplot: {e}")
        return None

def generate_mtbf_boxplot(start_date, end_date, services):
    """Generate MTBF boxplot analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_mtbf_boxplot(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'mtbf_boxplot', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating MTBF boxplot: {e}")
        return None

def generate_mttr_provider(start_date, end_date, services):
    """Generate MTTR provider analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_mttr_provider(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'mttr_provider', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating MTTR provider analysis: {e}")
        return None

def generate_mtbf_provider(start_date, end_date, services):
    """Generate MTBF provider analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_mtbf_provider(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'mtbf_provider', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating MTBF provider analysis: {e}")
        return None
    
def generate_incident_distribution(start_date, end_date, services):
    """Generate incident distribution analysis"""
    try:
        cleanup_old_plots()
        fig = analyze_incident_distribution(start_date, end_date, services)
        if fig:
            return save_plot(fig, 'incident_distribution', start_date, end_date, services)
        return None
    except Exception as e:
        print(f"Error generating incident distribution: {e}")
        traceback.print_exc()
        return None

def encode_image_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64

def get_plot_specific_prompt(plot_type, start_date=None, end_date=None, services=None):
    """Get analysis prompt specific to each plot type with date and service context"""
    
    # Format dates for better readability
    date_context = ""
    if start_date and end_date:
        start = pd.to_datetime(start_date).strftime('%B %d, %Y')
        end = pd.to_datetime(end_date).strftime('%B %d, %Y')
        date_context = f" for the period {start} to {end}"
    
    # Format services for better readability
    service_context = ""
    if services:
        service_names = [s.split(':')[1] if ':' in s else s for s in services]
        if len(service_names) == 1:
            service_context = f" for {service_names[0]}"
        else:
            service_list = ", ".join(service_names[:-1]) + f" and {service_names[-1]}"
            service_context = f" for {service_list}"

    prompts = {
        'figure1': f"Analyze this monthly incident distribution{date_context}{service_context}. Focus on days with highest incident counts and identify any weekly patterns or trends.",
        
        'figure2': f"Review this daily incident pattern analysis{date_context}{service_context}. Look for hours of the day with elevated incident rates and identify any common patterns.",
        
        'figure3': f"Examine this MTTR (Mean Time To Recovery) distribution{date_context}{service_context}. The top plot shows the cumulative distribution functions per service and the bottom plot shows the percentage of incidents by service. Focus on the differences betwen services, the faster (left shifted CDF) and slower (right shifted CDF) services, and any patterns.",
        
        'figure4': f"Analyze this provider-level Mean Time To Recovery comparison{date_context}{service_context}. Compare recovery times across different providers and identify which providers have faster (left shifted CDF) and slower (right shifted CDF) recovery times.",
        
        'figure5': f"Review this Mean Time To Recovery boxplot distribution{date_context}{service_context}. Look for services with high median recovery times or large spreads in their recovery times.",
        
        'figure6': f"Examine this MTBF (Mean Time Between Failures) distribution{date_context}{service_context}. The top plot shows the cumulative distribution functions per service and the bottom plot shows the percentage of incidents by service. Focus on the differences betwen services, the faster (left shifted CDF) and slower (right shifted CDF) services, and any patterns.",
        
        'figure7': f"Analyze this provider-based Mean Time Between Failures comparison{date_context}{service_context}. Compare times between failures across different providers and identify which providers have faster (left shifted CDF) and slower (right shifted CDF) times.",
        
        'figure8': f"Review this Mean Time Between Failures boxplot distribution{date_context}{service_context}. Look for services with short intervals between failures or high variability.",
        
        'figure9': f"Analyze these resolution stages {date_context}{service_context}. Identify any bottlenecks in the resolution process and outliers.",
        
        'figure10': f"Examine these status combinations {date_context}{service_context}. Look for which status combinations are most often concurrent and identify any unusual outliers per service.",
        
        'figure11': f"Review this daily availability analysis{date_context}{service_context}. Focus on periods of low availability and identify any concerning availability patterns.",
        
        'figure12': f"Analyze this service co-occurrence matrix{date_context}{service_context}. Look for services that frequently fail together and identify strong correlations between service failures.",
        
        'figure13': f"Examine this co-occurrence probability analysis{date_context}{service_context}. Focus on the probability of simultaneous failures and identify the most interdependent services.",
        
        'figure14': f"Review these service-specific failure co-occurrence patterns{date_context}{service_context}. Look for services that fail together and identify any concerning trends.",
        
        'figure15': f"Analyze this incident outage timeline{date_context}{service_context}. Focus on periods with multiple concurrent outages and identify any temporal patterns in outages.",
        
        'figure16': f"Examine these temporal autocorrelations{date_context}{service_context}. Look for significant lag correlations that might indicate systemic issues or periodic patterns.",
        
        'figure17': f"Analyze this incident impact distribution{date_context}{service_context}. Focus on the distribution of incident severity levels across services. Identify which services have the highest proportion of high-impact incidents. Do not mention median values."
    }
    
    base_prompt = f"Analyze this plot{date_context}{service_context} and identify significant patterns or trends."
    return prompts.get(plot_type, base_prompt)

def analyze_plot(image_base64, plot_type=None):
    """Analyze plot using GPT-4o-mini with plot-specific prompts"""
    if client is None:
        return {
            "success": False,
            "analysis": "AI analysis currently unavailable. Please try again later.",
            "error": "OpenAI client not initialized"
        }
        
    try:
        prompt = get_plot_specific_prompt(plot_type) if plot_type else "Analyze this plot and identify significant patterns."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt} Provide a concise analysis in about 50-70 words, focusing on the most significant findings. Explain any statistical or data analysis concepts. Format the response as a clear statement without any prefixes or numbering."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=250
        )
        
        analysis = response.choices[0].message.content.strip()
        

        analysis = analysis.lstrip('0123456789.- *')
        analysis = analysis.strip()
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        print(f"Error in analyze_plot: {str(e)}")
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }

def summarize_analyses(analyses, start_date=None, end_date=None, services=None):
    """Summarize multiple plot analyses into a cohesive summary with context"""
    try:
        # Format date range for context
        date_context = ""
        if start_date and end_date:
            start = pd.to_datetime(start_date).strftime('%B %d, %Y')
            end = pd.to_datetime(end_date).strftime('%B %d, %Y')
            date_context = f"Time Period: {start} to {end}\n"
        
        # Format services for context
        service_context = ""
        if services:
            service_names = [s.split(':')[1] if ':' in s else s for s in services]
            service_context = f"Services Analyzed: {', '.join(service_names)}\n"

        # Ensure analyses is a list and contains valid data
        if not isinstance(analyses, list):
            raise ValueError("Analyses must be a list")

        # Format the analyses text properly, with error checking
        analyses_text = []
        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue
            title = analysis.get('title', '')
            content = analysis.get('analysis', '')
            if title and content:
                analyses_text.append(f"{title}: {content}")

        if not analyses_text:
            raise ValueError("No valid analyses found")

        analyses_str = "\n".join(analyses_text)

        summary_prompt = f"""
        You are analyzing a set of LLM service metrics with the following context:
        {date_context}{service_context}
        Based on the following individual plot analyses, provide a structured executive summary 
        (200-250 words) that highlights the most critical insights requiring attention.
        
        Format the response as follows:
        - Start with a brief overview paragraph
        - Use bold markdown (**text**) for key metrics and findings
        - Group related insights together into clear sections
        - Focus on patterns and their implications
        
        Focus on:
        1. Major reliability issues
        2. Performance bottlenecks
        3. Concerning patterns

        Individual analyses:
        {analyses_str}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ],
            max_tokens=400
        )
        
        return {
            "success": True,
            "analysis": response.choices[0].message.content.strip()
        }
        
    except Exception as e:
        print(f"Error in summarize_analyses: {str(e)}")  # Add detailed logging
        print(f"Analyses data received: {analyses}")  # Debug print
        return {
            "success": False,
            "error": f"Summary failed: {str(e)}"
        }



os.makedirs(PLOTS_DIR, exist_ok=True) 