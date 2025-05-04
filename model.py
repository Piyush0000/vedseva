import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Create sample data based on the dashboard information
diseases = ['Malaria', 'Dengue', 'Tuberculosis', 'Cholera']
cases = [145, 78, 56, 29]  # Current case counts

# Generate synthetic historical data for the past 12 months
months = 12
today = datetime.now()
dates = [today - timedelta(days=30*i) for i in range(months)]
dates.reverse()  # Order from past to present

# Create synthetic monthly data with seasonal patterns for each disease
np.random.seed(42)  # For reproducibility

# Different seasonal patterns for each disease
malaria_data = [120 + 40*np.sin(i/12*2*np.pi) + np.random.normal(0, 10) for i in range(months)]
dengue_data = [60 + 30*np.sin((i+3)/12*2*np.pi) + np.random.normal(0, 8) for i in range(months)]
tb_data = [50 + 10*np.sin((i+6)/12*2*np.pi) + np.random.normal(0, 5) for i in range(months)]
cholera_data = [20 + 15*np.sin((i+9)/12*2*np.pi) + np.random.normal(0, 4) for i in range(months)]

# Make sure the last values match the current numbers
malaria_data[-1] = cases[0]
dengue_data[-1] = cases[1]
tb_data[-1] = cases[2]
cholera_data[-1] = cases[3]

historical_data = {
    'Date': dates,
    'Malaria': malaria_data,
    'Dengue': dengue_data,
    'Tuberculosis': tb_data,
    'Cholera': cholera_data
}

df = pd.DataFrame(historical_data)

# Create prediction model using polynomial regression
# Prepare data for modeling
def prepare_prediction_data(disease_data):
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

# Get predictions for each disease
malaria_pred = prepare_prediction_data(malaria_data)
dengue_pred = prepare_prediction_data(dengue_data)
tb_pred = prepare_prediction_data(tb_data)
cholera_pred = prepare_prediction_data(cholera_data)

# Generate future dates for predictions
future_dates = [today + timedelta(days=30*i) for i in range(1, 4)]  # 3 months ahead

# Create a figure with 2x2 subplots
fig = plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Custom colors
disease_colors = {
    'Malaria': '#E57373',
    'Dengue': '#FFB74D',
    'Tuberculosis': '#81C784',
    'Cholera': '#64B5F6'
}

# 1. Disease Trends (Monthly) with predictions
ax1 = plt.subplot(2, 2, 1)
for disease in diseases:
    plt.plot(df['Date'], df[disease], marker='o', linewidth=2, markersize=5, label=disease, color=disease_colors[disease])

# Add predictions
all_dates = list(df['Date']) + future_dates
plt.plot(future_dates, malaria_pred, '--', color=disease_colors['Malaria'])
plt.plot(future_dates, dengue_pred, '--', color=disease_colors['Dengue'])
plt.plot(future_dates, tb_pred, '--', color=disease_colors['Tuberculosis'])
plt.plot(future_dates, cholera_pred, '--', color=disease_colors['Cholera'])

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
min_y = min([min(malaria_data), min(dengue_data), min(tb_data), min(cholera_data)])
max_y = max([max(malaria_pred), max(dengue_pred), max(tb_pred), max(cholera_pred)]) * 1.1
plt.axvspan(today, future_dates[-1], alpha=0.2, color='gray', label='Prediction')
plt.text(today + timedelta(days=20), max_y * 0.9, 'Prediction Period', 
         fontsize=10, fontweight='bold', ha='left', color='dimgray')

# 2. Age Distribution of Cases
ax2 = plt.subplot(2, 2, 2)

# Create synthetic age distribution data
age_groups = ['0-4', '5-14', '15-24', '25-44', '45-64', '65+']
malaria_age = [25, 40, 35, 30, 10, 5]  # More common in children and young adults
dengue_age = [10, 15, 25, 20, 5, 3]    # Affects all age groups
tb_age = [5, 8, 12, 15, 10, 6]         # More in adults
cholera_age = [8, 5, 4, 6, 4, 2]       # Often affects children most

# Create a grouped bar chart
bar_width = 0.15
r1 = np.arange(len(age_groups))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

plt.bar(r1, malaria_age, width=bar_width, label='Malaria', color=disease_colors['Malaria'])
plt.bar(r2, dengue_age, width=bar_width, label='Dengue', color=disease_colors['Dengue'])
plt.bar(r3, tb_age, width=bar_width, label='Tuberculosis', color=disease_colors['Tuberculosis'])
plt.bar(r4, cholera_age, width=bar_width, label='Cholera', color=disease_colors['Cholera'])

plt.xlabel('Age Groups', fontsize=12)
plt.ylabel('Number of Cases', fontsize=12)
plt.title('Age Distribution of Cases', fontsize=14, fontweight='bold')
plt.xticks([r + bar_width*1.5 for r in range(len(age_groups))], age_groups)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Treatment Response Rates
ax3 = plt.subplot(2, 2, 3)

# Create synthetic treatment success rate data (in percentage)
treatment_stages = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
malaria_response = [30, 65, 85, 95]
dengue_response = [20, 50, 80, 90]
tb_response = [10, 25, 45, 70]  # TB takes longer to treat
cholera_response = [40, 75, 90, 98]

plt.plot(treatment_stages, malaria_response, marker='o', linewidth=2, markersize=8, label='Malaria', color=disease_colors['Malaria'])
plt.plot(treatment_stages, dengue_response, marker='s', linewidth=2, markersize=8, label='Dengue', color=disease_colors['Dengue'])
plt.plot(treatment_stages, tb_response, marker='^', linewidth=2, markersize=8, label='Tuberculosis', color=disease_colors['Tuberculosis'])
plt.plot(treatment_stages, cholera_response, marker='d', linewidth=2, markersize=8, label='Cholera', color=disease_colors['Cholera'])

plt.xlabel('Treatment Timeline', fontsize=12)
plt.ylabel('Treatment Success Rate (%)', fontsize=12)
plt.title('Treatment Response Rates', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 100)

# 4. Resource Allocation
ax4 = plt.subplot(2, 2, 4)

# Calculate suggested resource allocation based on case numbers and treatment response
# Formula: Cases × (100 - average treatment response) / 100
total_cases = sum(cases)
malaria_weight = cases[0] * (100 - np.mean(malaria_response)) / 100
dengue_weight = cases[1] * (100 - np.mean(dengue_response)) / 100
tb_weight = cases[2] * (100 - np.mean(tb_response)) / 100
cholera_weight = cases[3] * (100 - np.mean(cholera_response)) / 100

total_weight = malaria_weight + dengue_weight + tb_weight + cholera_weight
resource_allocation = [
    (malaria_weight / total_weight) * 100,
    (dengue_weight / total_weight) * 100,
    (tb_weight / total_weight) * 100,
    (cholera_weight / total_weight) * 100
]

plt.pie(resource_allocation, labels=diseases, autopct='%1.1f%%', startangle=90, colors=[disease_colors[d] for d in diseases])
plt.axis('equal')
plt.title('Suggested Resource Allocation Based on Case Burden and Treatment Difficulty', fontsize=14, fontweight='bold')

# Add explanatory text
plt.figtext(0.5, 0.01, 
            'This prediction model integrates current disease case counts, historical trends, age distribution, and treatment response rates\n'
            'to suggest optimal resource allocation for each disease. The dashed lines represent predicted future trends.',
            ha='center', fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Show the plot
plt.savefig('disease_prediction_model.png', dpi=300, bbox_inches='tight')
plt.show()

# Example of how to use this model for specific predictions
def predict_future_outbreak(disease_name, current_cases, months_ahead=3):
    """
    Predicts future outbreak cases for a specific disease
    
    Parameters:
    - disease_name: Name of the disease
    - current_cases: Current number of cases
    - months_ahead: Number of months to predict ahead
    
    Returns:
    - Dictionary with prediction results
    """
    # This is a simplified example - in a real scenario, you would use more complex models
    # Here we're using historical seasonal patterns and current trends
    
    # Seasonal multipliers for different diseases
    seasonal_factors = {
        'Malaria': {'pattern': [1.2, 1.3, 1.4, 1.3, 1.1, 0.9, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        'Dengue': {'pattern': [0.8, 0.9, 1.1, 1.3, 1.4, 1.3, 1.1, 0.9, 0.8, 0.7, 0.8, 0.9]},
        'Tuberculosis': {'pattern': [1.0, 1.1, 1.0, 0.9, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 1.0, 1.1]},
        'Cholera': {'pattern': [0.7, 0.6, 0.7, 0.9, 1.2, 1.4, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7]}
    }
    
    if disease_name not in seasonal_factors:
        return {"error": "Disease not found in model"}
    
    # Get the current month (0-11)
    current_month = datetime.now().month - 1
    
    predictions = []
    last_case_count = current_cases
    
    for i in range(months_ahead):
        # Calculate the future month index (wrapping around if needed)
        month_idx = (current_month + i + 1) % 12
        
        # Apply the seasonal factor
        seasonal_multiplier = seasonal_factors[disease_name]['pattern'][month_idx]
        
        # For simplicity, we'll use a growth/decline model based on the seasonal pattern
        # In a real model, you'd incorporate more factors like interventions, population movements, etc.
        predicted_cases = last_case_count * seasonal_multiplier
        
        # Add some random variation (to simulate real-world unpredictability)
        noise = np.random.normal(0, predicted_cases * 0.05)  # 5% noise
        predicted_cases += noise
        
        # Keep track for the next iteration
        last_case_count = predicted_cases
        
        # Record the prediction
        month_date = (datetime.now() + timedelta(days=30*(i+1))).strftime('%b %Y')
        predictions.append({
            'month': month_date,
            'predicted_cases': int(max(0, predicted_cases))
        })
    
    return {
        'disease': disease_name,
        'current_cases': current_cases,
        'predictions': predictions,
        'recommended_action': get_recommended_action(disease_name, predictions[-1]['predicted_cases'])
    }

def get_recommended_action(disease, predicted_cases):
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

# Example usage of the prediction function
malaria_prediction = predict_future_outbreak('Malaria', 145, 3)
print(f"Malaria Prediction Results:")
print(f"Current cases: {malaria_prediction['current_cases']}")
for pred in malaria_prediction['predictions']:
    print(f"  {pred['month']}: {pred['predicted_cases']} cases")
print(f"Recommended action: {malaria_prediction['recommended_action']}")

# To use this model in production, you would need to:
# 1. Replace synthetic data with real historical data
# 2. Train more sophisticated models (ARIMA, Prophet, or machine learning models)
# 3. Integrate with the actual dashboard data source
# 4. Set up automated retraining as new data becomes available