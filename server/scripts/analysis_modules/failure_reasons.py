from openai import OpenAI
import os
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_failure_reasons(df, query=None, history=None):
    """Analyze failure reasons based on incident data and user query with enhanced distributed systems context"""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        # Define all expected columns based on the complete dataset structure
        required_columns = [
            # Incident identifiers and basic info
            'incident_id', 'Incident_Title', 'incident_impact_level', 'Incident_color',
            'provider',
            
            # Service impact flags
            'Playground', 'API', 'Labs', 'ChatGPT', 
            'api.anthropic.com', 'claude.ai', 'console.anthropic.com',
            'Character.AI', 'StabilityAI',
            
            # Incident stage flags
            'investigating_flag', 'identified_flag', 'monitoring_flag', 
            'resolved_flag', 'postmortem_flag',
            
            # Timestamp columns
            'investigating_timestamp', 'identified_timestamp', 
            'monitoring_timestamp', 'resolved_timestamp', 
            'postmortem_timestamp', 'start_timestamp', 'close_timestamp',
            
            # Description columns
            'investigating_description', 'identified_description',
            'monitoring_description', 'resolved_description',
            'postmortem_description',
            
            # Duration related columns
            'time_span', 'over_one_day'
        ]
        
        # Calculate dataset timeframe
        time_cols = ['investigating_timestamp', 'identified_timestamp', 'monitoring_timestamp', 
                    'resolved_timestamp', 'postmortem_timestamp', 'start_timestamp', 'close_timestamp']
        timeframe = {}
        for col in time_cols:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col])
                    timeframe[col] = {
                        'earliest': dates.min(),
                        'latest': dates.max(),
                        'range_days': (dates.max() - dates.min()).days
                    }
                except Exception as e:
                    print(f"Warning: Could not process dates for {col}: {e}")

        # Calculate service-specific metrics
        service_columns = ['Playground', 'API', 'Labs', 'ChatGPT', 'api.anthropic.com',
                         'claude.ai', 'console.anthropic.com', 'Character.AI', 'StabilityAI']
        service_impacts = {col: df[col].sum() for col in service_columns if col in df.columns}

        # Calculate MTTR (Mean Time To Resolution)
        df['duration_minutes'] = pd.to_timedelta(df['time_span']).dt.total_seconds() / 60
        mttr_by_provider = df.groupby('provider')['duration_minutes'].agg(['mean', 'median', 'count']).round(2)

        # Analyze incident progression
        progression_metrics = {
            'identified_rate': (df['identified_flag'].sum() / len(df) * 100).round(2),
            'monitoring_rate': (df['monitoring_flag'].sum() / len(df) * 100).round(2),
            'resolution_rate': (df['resolved_flag'].sum() / len(df) * 100).round(2),
            'postmortem_rate': (df['postmortem_flag'].sum() / len(df) * 100).round(2)
        }

        # Analyze severity distribution
        severity_distribution = df['incident_impact_level'].value_counts().sort_index()
        
        # Calculate incident timing patterns
        df['hour'] = pd.to_datetime(df['start_timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['start_timestamp']).dt.day_name()
        timing_patterns = {
            'peak_hours': df.groupby('hour').size().nlargest(3).index.tolist(),
            'busiest_days': df.groupby('day_of_week').size().nlargest(3).index.tolist()
        }

        # Create enhanced data context
        data_context = f"""
        Dataset Overview:
        1. Timeframe:
           - Analysis period: {min(t['earliest'] for t in timeframe.values() if 'earliest' in t)} to {max(t['latest'] for t in timeframe.values() if 'latest' in t)}
           - Total days covered: {max(t['range_days'] for t in timeframe.values() if 'range_days' in t)}
        
        2. Incident Volume:
           - Total incidents: {len(df)}
           - Long-running incidents (>24h): {df['over_one_day'].sum()}
           - Average incident duration: {df['duration_minutes'].mean():.2f} minutes
        
        3. Provider Distribution:
           {df['provider'].value_counts().to_dict()}
        
        4. Service Impact Analysis:
           {service_impacts}
        
        5. Incident Severity Distribution:
           {severity_distribution.to_dict()}
        
        6. Resolution Metrics:
           - Mean Time To Resolution by Provider:
           {mttr_by_provider.to_dict('index')}
           
        7. Incident Management:
           - Identification rate: {progression_metrics['identified_rate']}%
           - Monitoring rate: {progression_metrics['monitoring_rate']}%
           - Resolution rate: {progression_metrics['resolution_rate']}%
           - Postmortem rate: {progression_metrics['postmortem_rate']}%
        
        8. Timing Patterns:
           - Peak incident hours: {timing_patterns['peak_hours']}
           - Most incident-prone days: {timing_patterns['busiest_days']}
        """

        # Enhanced system prompt with distributed systems context
        system_prompt = f"""You are an expert in analyzing technical incidents and failures in distributed LLM systems. 
        Your expertise covers:
        1. Distributed Systems Architecture
           - API Gateway patterns and failure modes
           - Load balancing and rate limiting strategies
           - Service mesh architectures and resilience
           - State management and consistency patterns
           - Cross-service dependencies and failure propagation
        
        2. LLM Service Failure Modes
           - Rate limiting and quota exhaustion patterns
           - Token context window issues and handling
           - Model availability and versioning strategies
           - Request timeout and latency spike management
           - Resource allocation and scaling challenges
        
        3. Failure Analysis Patterns
           - Cascading failures and their prevention
           - Retry storms and backoff strategy optimization
           - Connection pooling and resource management
           - Resource exhaustion detection and mitigation
           - Error budget management and SLO tracking
        
        4. Recovery Strategies
           - Circuit breaking implementation patterns
           - Fallback mechanism design and testing
           - Graceful degradation approaches
           - Service redundancy and failover strategies
           - Disaster recovery planning and execution
        
        You have access to a comprehensive dataset of LLM service incidents with the following context:
        
        {data_context}
        
        Guidelines for Analysis:
        1. Focus on distributed systems implications of each failure
        2. Identify patterns that suggest architectural improvements
        3. Consider both direct and indirect failure causes
        4. Analyze potential cascading effects
        5. Suggest preventive measures based on observed patterns
        
        Format your responses using markdown:
        - Use headers (##) for main sections
        - Use bold (**) for key findings
        - Use lists (-) for recommendations
        - Include specific metrics when relevant
        """

        # Prepare chat messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if history:
            messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in history])

        # If no specific query, provide a default analysis
        if not query:
            query = """Analyze the most significant failure patterns in this dataset and provide:
            1. Key failure modes identified
            2. System-level implications
            3. Recommended architectural improvements
            4. Specific preventive measures"""

        messages.append({"role": "user", "content": query})

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return f"Error in analysis: {str(e)}"

    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return f"Error in data preprocessing: {str(e)}"
