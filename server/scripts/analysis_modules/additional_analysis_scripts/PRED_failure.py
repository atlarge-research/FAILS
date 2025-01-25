import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

class FailurePredictionModel:
    def __init__(self):
        self.time_series_model = None
        self.classification_model = None
        self.label_encoders = {}
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.lookback_days = 30
        self.forecast_days = 14
        
    def preprocess_data(self, df):
        print("Starting data preprocessing...")
        df = df.copy()
        
        # Convert timestamps and sort by date
        df['timestamp'] = pd.to_datetime(df['start_timestamp'])
        df = df.sort_values('timestamp')
        
        # Print date range
        print(f"Data spans from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Calculate incident duration
        df['duration'] = pd.to_datetime(df['close_timestamp']) - pd.to_datetime(df['start_timestamp'])
        df['duration_hours'] = df['duration'].dt.total_seconds() / 3600
        
        print(f"Average incident duration: {df['duration_hours'].mean():.2f} hours")
        
        # Encode categorical variables
        categorical_cols = ['incident_impact_level', 'Incident_color', 'provider']
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                print(f"Encoded {col} with {len(self.label_encoders[col].classes_)} unique values")
        
        # Count affected services
        service_cols = ['Playground', 'API', 'Labs', 'ChatGPT', 'claude.ai', 'api.anthropic.com', 'console.anthropic.com', 'Character.AI']
        df['services_affected'] = df[service_cols].sum(axis=1)
        print(f"Average services affected per incident: {df['services_affected'].mean():.2f}")
        
        return df
    
    def prepare_time_series_features(self, df, lookback=None):
        if lookback is None:
            lookback = self.lookback_days
            
        print("\nPreparing time series features...")
        print(f"Using lookback period of {lookback} days")
        
        # Create daily incident counts with expanded window
        dates = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
        daily_incidents = pd.DataFrame(index=dates)
        daily_incidents['incident_count'] = df.groupby(df['timestamp'].dt.date).size()
        daily_incidents = daily_incidents.fillna(0)
        
        # Store raw counts and dates for reference
        self.raw_incident_counts = daily_incidents['incident_count'].values
        self.incident_dates = daily_incidents.index
        
        # Scale target values separately
        target_values = daily_incidents['incident_count'].values.reshape(-1, 1)
        daily_incidents['incident_count_scaled'] = self.target_scaler.fit_transform(target_values)
        
        # Add rolling statistics as features
        daily_incidents['rolling_mean_7d'] = daily_incidents['incident_count'].rolling(window=7, min_periods=1).mean()
        daily_incidents['rolling_mean_30d'] = daily_incidents['incident_count'].rolling(window=30, min_periods=1).mean()
        daily_incidents['rolling_std_7d'] = daily_incidents['incident_count'].rolling(window=7, min_periods=1).std().fillna(0)
        
        print(f"Total days in dataset: {len(daily_incidents)}")
        print(f"Days with incidents: {(daily_incidents['incident_count'] > 0).sum()}")
        print(f"Average daily incidents: {daily_incidents['incident_count'].mean():.2f}")
        print(f"Max daily incidents: {daily_incidents['incident_count'].max()}")
        print(f"7-day rolling average range: {daily_incidents['rolling_mean_7d'].min():.2f} - {daily_incidents['rolling_mean_7d'].max():.2f}")
        
        # Scale features separately from target
        feature_columns = ['rolling_mean_7d', 'rolling_mean_30d', 'rolling_std_7d']
        feature_values = daily_incidents[feature_columns].values
        scaled_features = self.feature_scaler.fit_transform(feature_values)
        
        for i, col in enumerate(feature_columns):
            daily_incidents[f'{col}_scaled'] = scaled_features[:, i]
        
        sequences = []
        targets = []
        
        # Create sequences with scaled features
        for i in range(len(daily_incidents) - lookback):
            # Get scaled features
            feature_sequence = []
            for col in feature_columns:
                feature_sequence.append(daily_incidents[f'{col}_scaled'].iloc[i:i+lookback].values)
            sequences.append(np.column_stack(feature_sequence))
            
            # Get scaled target
            targets.append(daily_incidents['incident_count_scaled'].iloc[i+lookback])
        
        print(f"Created {len(sequences)} training sequences")
        print(f"Sequence shape: {sequences[0].shape} (days × features)")
        
        return np.array(sequences), np.array(targets), daily_incidents
    
    def build_time_series_model(self, input_shape):
        print(f"Building model with input shape: {input_shape}")
        
        model = keras.Sequential([
            # Input shape should be (lookback_days, n_features)
            keras.layers.Input(shape=input_shape),
            
            # First LSTM layer - match input features
            keras.layers.LSTM(128, return_sequences=True, 
                            input_shape=input_shape,
                            activation='tanh',
                            recurrent_activation='sigmoid'),
            keras.layers.Dropout(0.3),
            
            # Second LSTM layer
            keras.layers.LSTM(64, activation='tanh',
                            recurrent_activation='sigmoid'),
            keras.layers.Dropout(0.3),
            
            # Dense layers
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)  # Single output for incident count
        ])
        
        # Print model summary
        model.summary()
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train_models(self, df):
        print("\nStarting model training...")
        processed_df = self.preprocess_data(df)
        
        # Prepare time series data
        X_seq, y_seq, daily_incidents = self.prepare_time_series_features(processed_df)
        
        # Print shapes for debugging
        print(f"\nInput sequence shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")
        
        # Split data ensuring time order is preserved
        train_size = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:train_size]
        y_train_seq = y_seq[:train_size]
        X_test_seq = X_seq[train_size:]
        y_test_seq = y_seq[train_size:]
        
        print(f"\nTraining set shape: {X_train_seq.shape}")
        print(f"Test set shape: {X_test_seq.shape}")
        
        # Build and train time series model with correct input shape
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])  # (lookback_days, n_features)
        self.time_series_model = self.build_time_series_model(input_shape)
        
        # Add callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.time_series_model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss = self.time_series_model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print(f"\nTest Loss: {test_loss[0]:.4f}")
        print(f"Test MAE: {test_loss[1]:.4f}")
        
        # Make some test predictions to verify model behavior
        print("\nVerifying model predictions...")
        for i in range(min(5, len(X_test_seq))):
            true_value = self.target_scaler.inverse_transform([[y_test_seq[i]]])[0][0]
            pred_scaled = self.time_series_model.predict(X_test_seq[i:i+1], verbose=0)[0][0]
            pred_value = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
            print(f"True: {true_value:.2f}, Predicted: {pred_value:.2f}")
        
        # Train classification model
        print("\nTraining classification model...")
        feature_cols = [
            'hour', 'day', 'day_of_week', 'month', 'services_affected',
            'incident_impact_level_encoded'
        ]
        
        X = processed_df[feature_cols]
        y = processed_df['incident_impact_level_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.classification_model.fit(X_train, y_train)
        
        # Print classification metrics
        train_score = self.classification_model.score(X_train, y_train)
        test_score = self.classification_model.score(X_test, y_test)
        print(f"Classification train accuracy: {train_score:.4f}")
        print(f"Classification test accuracy: {test_score:.4f}")
        
        return daily_incidents, history
    
    def get_recent_incidents(self, df, days=7):
        """Get actual incident counts for the last n days"""
        # Ensure we have a timestamp column
        if 'timestamp' not in df.columns and 'start_timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['start_timestamp'])
        
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found in data")
        
        end_date = df['timestamp'].max()
        start_date = end_date - pd.Timedelta(days=days-1)
        
        print(f"Getting incidents from {start_date.date()} to {end_date.date()}")
        
        recent_counts = df[df['timestamp'].dt.date >= start_date.date()]
        daily_counts = recent_counts.groupby(recent_counts['timestamp'].dt.date).size()
        
        # Ensure we have exactly n days by filling missing dates with 0
        date_range = pd.date_range(start_date.date(), end_date.date(), freq='D')
        daily_counts = daily_counts.reindex(date_range, fill_value=0)
        
        print(f"Daily incident counts: {daily_counts.values}")
        return daily_counts.values
    
    def predict_failures(self, current_date, lookback_data):
        pd_date = pd.Timestamp(current_date)
        
        # Create feature matrix for lookback period
        lookback_series = pd.Series(lookback_data)
        feature_matrix = np.column_stack([
            lookback_series.rolling(window=7, min_periods=1).mean(),
            lookback_series.rolling(window=30, min_periods=1).mean(),
            lookback_series.rolling(window=7, min_periods=1).std().fillna(0)
        ])
        
        # Scale features
        scaled_features = self.feature_scaler.transform(feature_matrix)
        
        # Reshape for LSTM input (samples, timesteps, features)
        scaled_features = scaled_features.reshape(1, scaled_features.shape[0], scaled_features.shape[1])
        
        # Make prediction
        scaled_pred = self.time_series_model.predict(scaled_features, verbose=0)[0][0]
        
        # Inverse transform the prediction
        prediction = self.target_scaler.inverse_transform([[scaled_pred]])[0][0]
        prediction = max(0, float(prediction))  # Ensure non-negative
        
        # Prepare features for classification
        features = pd.DataFrame({
            'hour': [pd_date.hour],
            'day': [pd_date.day],
            'day_of_week': [pd_date.dayofweek],
            'month': [pd_date.month],
            'services_affected': [1],
            'incident_impact_level_encoded': [1]
        })
        
        severity_pred = self.classification_model.predict_proba(features)
        
        return {
            'predicted_incidents': prediction,
            'severity_probabilities': severity_pred[0].tolist(),
            'prediction_date': current_date.strftime('%Y-%m-%d')
        }

def plot_incidents_and_predictions(daily_incidents, predictions, history):
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Historical incidents
    plt.subplot(3, 1, 1)
    sns.lineplot(data=daily_incidents, x=daily_incidents.index, y='incident_count')
    plt.title('Historical Incidents Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Incidents')
    
    # Plot 2: Training history
    plt.subplot(3, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot 3: Predictions
    plt.subplot(3, 1, 3)
    pred_dates = [pred['prediction_date'] for pred in predictions]
    pred_values = [pred['predicted_incidents'] for pred in predictions]
    plt.plot(pred_dates, pred_values, marker='o')
    plt.title('Predicted Incidents for Next 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Predicted Number of Incidents')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('incident_analysis.png')
    plt.close()

def main():
    # Load data and verify
    print("Loading data...")
    df = pd.read_csv('server/static/data/incident_stages.csv')
    print(f"Loaded {len(df)} incidents")
    print(f"Columns available: {df.columns.tolist()}")
    
    # Initialize and train model
    print("\nInitializing model...")
    model = FailurePredictionModel()
    
    # Preprocess data first
    print("Preprocessing data...")
    processed_df = model.preprocess_data(df)
    print(f"Processed data shape: {processed_df.shape}")
    
    # Get recent incidents using processed data
    print("\nGetting recent incidents...")
    lookback_data = model.get_recent_incidents(processed_df, days=model.lookback_days)
    print(f"Lookback data shape: {lookback_data.shape}")
    
    # Train models
    print("\nTraining models...")
    daily_incidents, history = model.train_models(df)
    
    # Make predictions for next n days
    print("\nMaking predictions...")
    current_date = datetime.now()
    predictions = []
    
    for i in range(model.forecast_days):
        pred_date = current_date + timedelta(days=i)
        pred = model.predict_failures(pred_date, lookback_data)
        predictions.append(pred)
        print(f"Predictions for {pred['prediction_date']}:")
        print(f"Expected incidents: {pred['predicted_incidents']:.2f}")
    
    # Plot results
    plot_incidents_and_predictions(daily_incidents, predictions, history)
    print("\nVisualization saved as 'incident_analysis.png'")

if __name__ == "__main__":
    main() 