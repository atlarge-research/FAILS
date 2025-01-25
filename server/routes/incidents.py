from flask import Blueprint, jsonify
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from scripts.analysis_modules.additional_analysis_scripts.PRED_failure import FailurePredictionModel

incidents_bp = Blueprint('incidents', __name__)

@incidents_bp.route('/api/incidents', methods=['GET'])
def get_incidents():
    try:
        # You can change this to point to any CSV file
        csv_path = os.path.join('static', 'data', 'incident_stages_all.csv')
        df = pd.read_csv(csv_path)
        df = df.replace({np.nan: None})
        
        processed_data = []
        for _, row in df.iterrows():
            # Create status info string
            status_info = []
            if 'identified_flag' in df.columns and row.get('identified_flag'):
                status_info.append('Identified')
            if 'monitoring_flag' in df.columns and row.get('monitoring_flag'):
                status_info.append('Monitoring')
            if 'resolved_flag' in df.columns and row.get('resolved_flag'):
                status_info.append('Resolved')

            # Get all service columns (any boolean column that's not a flag)
            service_columns = [col for col in df.columns 
                             if col not in ['identified_flag', 'monitoring_flag', 'resolved_flag', 'investigating_flag', 'postmortem_flag'] 
                             and str(row[col]).lower() in ['true', 'false', '1', '0']
                             and col not in ['incident_id', 'Incident_Title', 'incident_impact_level', 'provider', 'over_one_day']]

            incident = {
                'incident_id': row.get('incident_id'),
                'provider': row.get('provider'),
                'Incident_Title': row.get('Incident_Title'),
                'incident_impact_level': int(row['incident_impact_level']) if pd.notna(row.get('incident_impact_level')) else None,
                'incident_color': row.get('Incident_color') if pd.notna(row.get('Incident_color')) else '#333333',
                'start_timestamp': row.get('start_timestamp'),
                'close_timestamp': row.get('close_timestamp'),
                'duration': row.get('time_span'),
                'status': ' → '.join(status_info) if status_info else 'Investigating',
            }

            # Add service flags dynamically
            for service in service_columns:
                incident[service] = bool(row[service]) if pd.notna(row[service]) else False

            processed_data.append(incident)
        
        return jsonify({
            'success': True,
            'data': processed_data
        })
    except Exception as e:
        print(f"Error loading incidents: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 

@incidents_bp.route('/api/predictive-analysis', methods=['GET'])
def get_predictive_analysis():
    try:
        # Initialize the model
        model = FailurePredictionModel()
        
        csv_path = os.path.join('static', 'data', 'incident_stages_all.csv')
        df = pd.read_csv(csv_path)
        

        daily_incidents, history = model.train_models(df)
        
        lookback_data = model.get_recent_incidents(df, days=model.lookback_days)
        
        # Make predictions for next 14 days
        current_date = datetime.now()
        predictions = []
        
        for i in range(model.forecast_days):
            pred_date = current_date + timedelta(days=i)
            pred = model.predict_failures(pred_date, lookback_data)
            predictions.append(pred)
        
        # Calculate some additional metrics for the response
        recent_average = float(lookback_data[-7:].mean())  # Last 7 days average
        total_predicted = sum(p['predicted_incidents'] for p in predictions)
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': predictions,
                'recent_average': recent_average,
                'total_predicted': total_predicted,
                'lookback_data': lookback_data.tolist(),
                'model_metrics': {
                    'training_loss': history.history['loss'][-1],
                    'validation_loss': history.history['val_loss'][-1]
                }
            }
        })
        
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500