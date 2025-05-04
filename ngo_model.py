import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import base64
from io import BytesIO
import json

# Generate synthetic data for our visualizations
np.random.seed(42)

# 1. Disease Cases Data for Pie Chart
diseases = ['Malaria', 'Dengue', 'Tuberculosis', 'Cholera']
cases = [145, 78, 56, 29]
total_population = 10000

# 2. Time Series Data for Response Rate
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
response_rates = [65, 68, 72, 75, 78, 82]  # Percentages

# Add some noise and create training data
x_time = np.array(range(len(months))).reshape(-1, 1)
y_response = np.array(response_rates)

# Create and train a simple linear regression model
model_response = LinearRegression()
model_response.fit(x_time, y_response)

# Predict next 3 months
future_months = ['Jul', 'Aug', 'Sep']
x_future = np.array(range(len(months), len(months) + len(future_months))).reshape(-1, 1)
future_predictions = model_response.predict(x_future)

# Combined data for plotting
all_months = months + future_months
all_response_rates = np.concatenate([response_rates, future_predictions])

# 3. Outcomes Data for Pie Chart
outcomes = {
    'Cured': 78,
    'Deaths': 12, 
    'Ongoing Treatment': 10
}

# 4. Risk Prediction Model
# Generate synthetic data for regions
regions = ['North', 'South', 'East', 'West', 'Central']
features = np.random.rand(100, 4)  # Random features like population density, sanitation, etc.
risk_scores = 0.3 + 0.4 * features[:, 0] + 0.2 * features[:, 1] + 0.1 * np.random.rand(100)
risk_scores = np.clip(risk_scores, 0, 1)  # Ensure values between 0 and 1

# Associate regions with data points
region_indices = np.random.choice(len(regions), size=100)
region_labels = [regions[i] for i in region_indices]

# Create a dataframe for analysis
risk_df = pd.DataFrame({
    'region': region_labels,
    'feature1': features[:, 0],
    'feature2': features[:, 1],
    'feature3': features[:, 2],
    'feature4': features[:, 3],
    'risk_score': risk_scores
})

# Group by region to get average risk scores
region_risk = risk_df.groupby('region')['risk_score'].mean().reset_index()
region_risk = region_risk.sort_values('risk_score', ascending=False)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes as base64
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# Generate the visualizations
def generate_charts():
    charts = {}
    
    # 1. Pie Chart - Disease Cases
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(cases, labels=diseases, autopct='%1.1f%%', startangle=90, shadow=True)
    ax1.axis('equal')
    plt.title('Disease Distribution')
    charts['disease_pie'] = fig_to_base64(fig1)
    plt.close(fig1)
    
    # 2. Line Chart - Response Rates Over Time with ML Predictions
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Actual data
    ax2.plot(months, response_rates, 'o-', color='blue', label='Actual Response Rate')
    # Predicted data
    ax2.plot(future_months, future_predictions, 'o--', color='red', label='ML Predicted Rate')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Response Rate (%)')
    ax2.set_title('Treatment Response Rate Over Time with ML Predictions')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    charts['response_line'] = fig_to_base64(fig2)
    plt.close(fig2)
    
    # 3. Pie Chart - Treatment Outcomes
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    outcome_values = list(outcomes.values())
    outcome_labels = list(outcomes.keys())
    colors = ['#4CAF50', '#F44336', '#2196F3']  # Green, Red, Blue
    ax3.pie(outcome_values, labels=outcome_labels, autopct='%1.1f%%', 
            startangle=90, shadow=True, colors=colors)
    ax3.axis('equal')
    plt.title('Treatment Outcomes')
    charts['outcome_pie'] = fig_to_base64(fig3)
    plt.close(fig3)
    
    # 4. ML Risk Prediction Bar Chart
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='region', y='risk_score', data=region_risk, palette='YlOrRd')
    ax4.set_xlabel('Region')
    ax4.set_ylabel('Risk Score (0-1)')
    ax4.set_title('ML-Predicted Disease Risk by Region')
    
    # Add threshold line for critical alert
    ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Critical Risk Threshold')
    ax4.legend()
    
    for i, row in enumerate(region_risk.itertuples()):
        if row.risk_score > 0.4:
            ax4.text(i, row.risk_score + 0.02, 'Alert!', ha='center', color='darkred', fontweight='bold')
    
    charts['risk_bar'] = fig_to_base64(fig4)
    plt.close(fig4)
    
    return charts

# Generate all charts
chart_data = generate_charts()

# Save the chart data to a JSON file for use in JavaScript
chart_metadata = {
    'disease_data': {
        'labels': diseases,
        'values': cases,
        'total_population': total_population
    },
    'response_data': {
        'months': all_months,
        'actual': response_rates.tolist(),
        'predicted': future_predictions.tolist()
    },
    'outcome_data': outcomes,
    'risk_data': region_risk.to_dict(orient='records')
}

# Combine image data and metadata
output_data = {
    'images': chart_data,
    'metadata': chart_metadata
}

# In a real application, you would save this to a file
with open('chart_data.json', 'w') as f:
    json.dump(output_data, f)

print("Chart generation complete. The visualizations are ready for integration into the web page.")