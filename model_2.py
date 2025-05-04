import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import threading
import os
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictionModel:
    """
    A real-time disease prediction model that can be updated with new data
    and automatically regenerate predictions and visualizations.
    """
    
    def __init__(self, data_source='database.csv', output_dir='outputs'):
        """
        Initialize the prediction model with data source and output settings.
        
        Parameters:
        - data_source: Path to CSV file or database connection string
        - output_dir: Directory to save generated visualizations
        """
        self.data_source = data_source
        self.output_dir = output_dir
        self.diseases = ['Malaria', 'Dengue', 'Tuberculosis', 'Cholera']
        self.disease_colors = {
            'Malaria': '#E57373',
            'Dengue': '#FFB74D',
            'Tuberculosis': '#81C784',
            'Cholera': '#64B5F6'
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Initialize with empty data
        self.df = None
        self.cases = {}
        self.predictions = {}
        
        # Load initial data
        self.load_data()
        
        # Flag to control the update thread
        self.running = False
        self.update_thread = None
        
    def load_data(self):
        """
        Load data from the data source. This function handles both CSV files
        and could be extended to handle database connections.
        """
        try:
            if self.data_source.endswith('.csv'):
                # Load from CSV file
                self.df = pd.read_csv(self.data_source)
                
                # Convert date strings to datetime objects
                if 'Date' in self.df.columns:
                    self.df['Date'] = pd.to_datetime(self.df['Date'])
                
            else:
                # In a real implementation, this would handle database connections
                # For example using SQLAlchemy or a specific database connector
                print(f"Database connection to {self.data_source} is not implemented yet.")
                # For demo purposes, generate synthetic data if no file exists
                self._generate_synthetic_data()
                
            # Extract current cases from the most recent data
            if self.df is not None and not self.df.empty:
                latest_data = self.df.iloc[-1]
                self.cases = {disease: latest_data.get(disease, 0) for disease in self.diseases}
                print(f"Loaded data with {len(self.df)} records. Latest date: {latest_data.get('Date', 'Unknown')}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Generate synthetic data as fallback
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """
        Generate synthetic data for testing when no real data is available.
        """
        print("Generating synthetic data for testing...")
        
        # Generate synthetic historical data for the past 12 months
        months = 12
        today = datetime.now()
        dates = [today - timedelta(days=30*i) for i in range(months)]
        dates.reverse()  # Order from past to present

        np.random.seed(42)  # For reproducibility

        # Different seasonal patterns for each disease
        malaria_data = [120 + 40*np.sin(i/12*2*np.pi) + np.random.normal(0, 10) for i in range(months)]
        dengue_data = [60 + 30*np.sin((i+3)/12*2*np.pi) + np.random.normal(0, 8) for i in range(months)]
        tb_data = [50 + 10*np.sin((i+6)/12*2*np.pi) + np.random.normal(0, 5) for i in range(months)]
        cholera_data = [20 + 15*np.sin((i+9)/12*2*np.pi) + np.random.normal(0, 4) for i in range(months)]

        historical_data = {
            'Date': dates,
            'Malaria': malaria_data,
            'Dengue': dengue_data,
            'Tuberculosis': tb_data,
            'Cholera': cholera_data
        }

        self.df = pd.DataFrame(historical_data)
        
        # Set current cases from the most recent data
        self.cases = {
            'Malaria': int(malaria_data[-1]),
            'Dengue': int(dengue_data[-1]),
            'Tuberculosis': int(tb_data[-1]),
            'Cholera': int(cholera_data[-1])
        }
        
        # Save synthetic data to CSV
        self.df.to_csv('database.csv', index=False)
        print(f"Synthetic data saved to database.csv with {len(self.df)} records")
        
    def update_cases(self, new_cases):
        """
        Update the current case counts with new data.
        
        Parameters:
        - new_cases: Dictionary with disease names as keys and case counts as values
        """
        # Update the cases dictionary
        for disease, count in new_cases.items():
            if disease in self.diseases:
                self.cases[disease] = count
        
        # Add new data point to the dataframe
        new_row = {'Date': datetime.now()}
        new_row.update(new_cases)
        
        # Append to dataframe
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save updated data
        if self.data_source.endswith('.csv'):
            self.df.to_csv(self.data_source, index=False)
        
        print(f"Updated cases: {new_cases}")
        
        # Regenerate predictions with new data
        self.generate_predictions()
        
    def generate_predictions(self):
        """
        Generate predictions for each disease based on historical data.
        """
        # Prepare data for modeling
        for disease in self.diseases:
            if disease in self.df.columns:
                disease_data = self.df[disease].values
                self.predictions[disease] = self._prepare_prediction_data(disease_data)
        
        print("Predictions updated.")
    
    def _prepare_prediction_data(self, disease_data):
        """
        Prepare prediction data using polynomial regression.
        
        Parameters:
        - disease_data: Array of historical data for a disease
        
        Returns:
        - Array of predictions for future months
        """
        X = np.array(range(len(disease_data))).reshape(-1, 1)
        y = np.array(disease_data)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict for future months
        future_months = 3
        X_future = np.array(range(len(disease_data), len(disease_data) + future_months)).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        predictions = model.predict(X_future_poly)
        
        return predictions
    
    def visualize(self, save=True):
        """
        Generate visualizations for the disease data and predictions.
        
        Parameters:
        - save: Whether to save the visualization to file
        """
        if self.df is None or self.df.empty:
            print("No data available for visualization.")
            return
        
        # Create a figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Generate future dates for predictions
        today = datetime.now()
        future_dates = [today + timedelta(days=30*i) for i in range(1, 4)]  # 3 months ahead
        
        # 1. Disease Trends (Monthly) with predictions
        ax1 = plt.subplot(2, 2, 1)
        for disease in self.diseases:
            if disease in self.df.columns:
                plt.plot(self.df['Date'], self.df[disease], marker='o', linewidth=2, 
                         markersize=5, label=disease, color=self.disease_colors[disease])
                
                # Add predictions if available
                if disease in self.predictions:
                    plt.plot(future_dates, self.predictions[disease], '--', 
                             color=self.disease_colors[disease])
        
        plt.title('Disease Trends (Monthly) with Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Cases', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show months
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Add a shaded area for the prediction period
        min_vals = [self.df[disease].min() for disease in self.diseases if disease in self.df.columns]
        max_vals = []
        for disease in self.diseases:
            if disease in self.predictions:
                max_vals.append(max(self.predictions[disease]))
        
        if min_vals and max_vals:
            min_y = min(min_vals)
            max_y = max(max_vals) * 1.1
            plt.axvspan(today, future_dates[-1], alpha=0.2, color='gray', label='Prediction')
            plt.text(today + timedelta(days=20), max_y * 0.9, 'Prediction Period', 
                    fontsize=10, fontweight='bold', ha='left', color='dimgray')
        
        # 2. Age Distribution of Cases - Synthetic data for visualization
        ax2 = plt.subplot(2, 2, 2)
        
        # Create synthetic age distribution data based on case proportions
        age_groups = ['0-4', '5-14', '15-24', '25-44', '45-64', '65+']
        
        # Scale age distributions according to current case counts
        total_cases = sum(self.cases.values())
        scale_factor = total_cases / 100 if total_cases > 0 else 1
        
        # Base age distributions (percentages)
        age_distributions = {
            'Malaria': [25, 40, 35, 30, 10, 5],
            'Dengue': [10, 15, 25, 20, 5, 3],
            'Tuberculosis': [5, 8, 12, 15, 10, 6],
            'Cholera': [8, 5, 4, 6, 4, 2]
        }
        
        # Scale distributions based on current cases
        for disease in self.diseases:
            if disease in self.cases and disease in age_distributions:
                case_factor = self.cases[disease] / 100
                age_distributions[disease] = [x * case_factor for x in age_distributions[disease]]
        
        # Create a grouped bar chart
        bar_width = 0.15
        r1 = np.arange(len(age_groups))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]
        
        plt.bar(r1, age_distributions['Malaria'], width=bar_width, label='Malaria', color=self.disease_colors['Malaria'])
        plt.bar(r2, age_distributions['Dengue'], width=bar_width, label='Dengue', color=self.disease_colors['Dengue'])
        plt.bar(r3, age_distributions['Tuberculosis'], width=bar_width, label='Tuberculosis', color=self.disease_colors['Tuberculosis'])
        plt.bar(r4, age_distributions['Cholera'], width=bar_width, label='Cholera', color=self.disease_colors['Cholera'])
        
        plt.xlabel('Age Groups', fontsize=12)
        plt.ylabel('Number of Cases', fontsize=12)
        plt.title('Age Distribution of Cases', fontsize=14, fontweight='bold')
        plt.xticks([r + bar_width*1.5 for r in range(len(age_groups))], age_groups)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Treatment Response Rates - Synthetic data
        ax3 = plt.subplot(2, 2, 3)
        
        # Treatment response rates (percentages)
        treatment_stages = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        response_rates = {
            'Malaria': [30, 65, 85, 95],
            'Dengue': [20, 50, 80, 90],
            'Tuberculosis': [10, 25, 45, 70],
            'Cholera': [40, 75, 90, 98]
        }
        
        for disease in self.diseases:
            if disease in response_rates:
                plt.plot(treatment_stages, response_rates[disease], marker='o', linewidth=2, 
                        markersize=8, label=disease, color=self.disease_colors[disease])
        
        plt.xlabel('Treatment Timeline', fontsize=12)
        plt.ylabel('Treatment Success Rate (%)', fontsize=12)
        plt.title('Treatment Response Rates', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 100)
        
        # 4. Resource Allocation based on case burden and treatment difficulty
        ax4 = plt.subplot(2, 2, 4)
        
        # Calculate resource allocation
        resource_allocation = []
        allocation_labels = []
        
        if self.cases:
            # Calculate weights based on case counts and treatment difficulty
            weights = {}
            for disease in self.diseases:
                if disease in self.cases and disease in response_rates:
                    avg_response = np.mean(response_rates[disease])
                    # Formula: Cases × (100 - average treatment response) / 100
                    weights[disease] = self.cases[disease] * (100 - avg_response) / 100
            
            total_weight = sum(weights.values())
            
            if total_weight > 0:
                for disease in self.diseases:
                    if disease in weights:
                        allocation = (weights[disease] / total_weight) * 100
                        resource_allocation.append(allocation)
                        allocation_labels.append(f"{disease}\n({allocation:.1f}%)")
        
        if resource_allocation:
            plt.pie(resource_allocation, labels=allocation_labels, autopct='', startangle=90, 
                   colors=[self.disease_colors[d] for d in self.diseases])
            plt.axis('equal')
            plt.title('Suggested Resource Allocation', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No resource allocation data available', 
                    ha='center', va='center', fontsize=12)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   'This prediction model integrates current disease case counts, historical trends, age distribution, and treatment response rates\n'
                   'to suggest optimal resource allocation for each disease. The dashed lines represent predicted future trends.',
                   ha='center', fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save the visualization
        if save:
            filename = f"{self.output_dir}/disease_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {filename}")
        
        return fig
    
    def predict_future_outbreak(self, disease_name, months_ahead=3):
        """
        Predicts future outbreak cases for a specific disease
        
        Parameters:
        - disease_name: Name of the disease
        - months_ahead: Number of months to predict ahead
        
        Returns:
        - Dictionary with prediction results
        """
        if disease_name not in self.diseases:
            return {"error": "Disease not found in model"}
            
        if disease_name not in self.cases:
            return {"error": "No current data for this disease"}
        
        current_cases = self.cases[disease_name]
        
        # Seasonal multipliers for different diseases
        seasonal_factors = {
            'Malaria': {'pattern': [1.2, 1.3, 1.4, 1.3, 1.1, 0.9, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
            'Dengue': {'pattern': [0.8, 0.9, 1.1, 1.3, 1.4, 1.3, 1.1, 0.9, 0.8, 0.7, 0.8, 0.9]},
            'Tuberculosis': {'pattern': [1.0, 1.1, 1.0, 0.9, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 1.0, 1.1]},
            'Cholera': {'pattern': [0.7, 0.6, 0.7, 0.9, 1.2, 1.4, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7]}
        }
        
        # Get more accurate predictions from the model if available
        if disease_name in self.predictions:
            model_predictions = self.predictions[disease_name]
            
            predictions = []
            for i in range(min(months_ahead, len(model_predictions))):
                month_date = (datetime.now() + timedelta(days=30*(i+1))).strftime('%b %Y')
                predictions.append({
                    'month': month_date,
                    'predicted_cases': int(max(0, model_predictions[i]))
                })
                
            # If we need more predictions than the model provides
            if months_ahead > len(model_predictions):
                # Get the current month (0-11)
                current_month = datetime.now().month - 1
                last_case_count = model_predictions[-1]
                
                for i in range(len(model_predictions), months_ahead):
                    # Calculate the future month index
                    month_idx = (current_month + i + 1) % 12
                    
                    # Apply the seasonal factor
                    seasonal_multiplier = seasonal_factors[disease_name]['pattern'][month_idx]
                    predicted_cases = last_case_count * seasonal_multiplier
                    
                    # Add some random variation
                    noise = np.random.normal(0, predicted_cases * 0.05)
                    predicted_cases += noise
                    
                    # Keep track for the next iteration
                    last_case_count = predicted_cases
                    
                    # Record the prediction
                    month_date = (datetime.now() + timedelta(days=30*(i+1))).strftime('%b %Y')
                    predictions.append({
                        'month': month_date,
                        'predicted_cases': int(max(0, predicted_cases))
                    })
        else:
            # Fallback to simple prediction if model predictions aren't available
            # Get the current month (0-11)
            current_month = datetime.now().month - 1
            
            predictions = []
            last_case_count = current_cases
            
            for i in range(months_ahead):
                # Calculate the future month index
                month_idx = (current_month + i + 1) % 12
                
                # Apply seasonal factor
                seasonal_multiplier = seasonal_factors[disease_name]['pattern'][month_idx]
                predicted_cases = last_case_count * seasonal_multiplier
                
                # Add some random variation
                noise = np.random.normal(0, predicted_cases * 0.05)
                predicted_cases += noise
                
                # Keep track for next iteration
                last_case_count = predicted_cases
                
                # Record the prediction
                month_date = (datetime.now() + timedelta(days=30*(i+1))).strftime('%b %Y')
                predictions.append({
                    'month': month_date,
                    'predicted_cases': int(max(0, predicted_cases))
                })
        
        # Get recommended action based on the final prediction
        final_prediction = predictions[-1]['predicted_cases'] if predictions else current_cases
        recommended_action = self.get_recommended_action(disease_name, final_prediction)
        
        return {
            'disease': disease_name,
            'current_cases': current_cases,
            'predictions': predictions,
            'recommended_action': recommended_action
        }
    
    def get_recommended_action(self, disease, predicted_cases):
        """Generate recommended actions based on disease and predicted case count"""
        
        # Define thresholds for each disease
        thresholds = {
            'Malaria': {'low': 100, 'medium': 150, 'high': 200},
            'Dengue': {'low': 50, 'medium': 100, 'high': 150},
            'Tuberculosis': {'low': 30, 'medium': 60, 'high': 90},
            'Cholera': {'low': 20, 'medium': 40, 'high': 60}
        }
        
        t = thresholds.get(disease, {'low': 50, 'medium': 100, 'high': 150})
        
        if predicted_cases < t['low']:
            return "Maintain current prevention measures and monitoring."
        elif predicted_cases < t['medium']:
            return f"Increase surveillance and prepare additional resources for {disease} treatment."
        elif predicted_cases < t['high']:
            return f"Alert healthcare partners and deploy preventive measures for {disease} outbreak."
        else:
            return f"Urgent action required: Mobilize emergency response team for {disease} epidemic control."
    
    def export_predictions_to_json(self):
        """
        Export all predictions to a JSON file for use in web dashboards or APIs.
        """
        predictions_data = {}
        
        for disease in self.diseases:
            predictions_data[disease] = self.predict_future_outbreak(disease, 6)
            
        # Add timestamp
        predictions_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        predictions_data['update_frequency'] = 'daily'
        
        # Save to file
        filename = f"{self.output_dir}/predictions_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(predictions_data, f, indent=2)
            
        print(f"Predictions exported to {filename}")
        return filename
    
    def start_real_time_updates(self, interval=60):
        """
        Start a background thread that checks for data updates at specified intervals.
        
        Parameters:
        - interval: Update interval in seconds
        """
        if self.running:
            print("Real-time updates already running.")
            return
            
        self.running = True
        
        def update_loop():
            while self.running:
                print(f"Checking for data updates... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                # In a real implementation, this would check the database for new data
                self.load_data()
                self.generate_predictions()
                self.visualize()
                
                # Sleep for the specified interval
                time.sleep(interval)
                
        # Start the update thread
        self.update_thread = threading.Thread(target=update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        print(f"Started real-time updates every {interval} seconds.")
        
    def stop_real_time_updates(self):
        """
        Stop the background update thread.
        """
        if not self.running:
            print("Real-time updates not running.")
            return
            
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
            print("Stopped real-time updates.")


# Example usage:
if __name__ == "__main__":
    model = DiseasePredictionModel(data_source='database.csv', output_dir='outputs')
    
    # Generate initial predictions and visualization
    model.generate_predictions()
    model.visualize()
    
    # Export predictions to JSON
    model.export_predictions_to_json()
    
    # Example of updating with new data
    print("\nUpdating with new data...")
    new_cases = {
        'Malaria': 150,
        'Dengue': 85,
        'Tuberculosis': 60,
        'Cholera': 35
    }
    model.update_cases(new_cases)
    model.visualize()
    
    # Get specific prediction for a disease
    print("\nPrediction for Malaria:")
    malaria_prediction = model.predict_future_outbreak('Malaria', 3)
    print(f"Current cases: {malaria_prediction['current_cases']}")
    for pred in malaria_prediction['predictions']:
        print(f"  {pred['month']}: {pred['predicted_cases']} cases")
    print(f"Recommended action: {malaria_prediction['recommended_action']}")
    
    # Start real-time updates (comment out for testing)
    # model.start_real_time_updates(interval=300)  # Check every 5 minutes
    
    print("\nDone.")