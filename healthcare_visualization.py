from flask import Flask, render_template, request, jsonify
import pandas as pd
from io import StringIO
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

app = Flask(__name__)

# Load the CSV data (replace with your actual file path)
DATA = pd.read_csv('district_health_data.csv')

@app.route('/')
def index():
    states = DATA['state'].unique().tolist()
    return render_template('index.html', states=states)

@app.route('/get_districts')
def get_districts():
    state = request.args.get('state')
    districts = DATA[DATA['state'] == state]['district'].unique().tolist()
    return jsonify(districts)

@app.route('/get_graph_data')
def get_graph_data():
    district = request.args.get('district')
    if not district:
        return jsonify({'error': 'No district selected'})

    district_data = DATA[DATA['district'] == district]
    graph_data = {}

    # 1. Pie Chart - Disease Rate
    disease_counts = district_data['disease'].value_counts().to_dict()
    graph_data['disease_pie'] = {'labels': list(disease_counts.keys()), 'data': list(disease_counts.values())}

    # 2. Line Chart - Monthly Spike
    monthly_cases = district_data.groupby('month')['cases'].sum().sort_index().to_dict()
    graph_data['monthly_spike'] = {'labels': list(monthly_cases.keys()), 'data': list(monthly_cases.values())}

    # 3. Medical Camp vs Time (Bar Chart)
    monthly_camps = district_data.groupby('month')['medical_camps'].sum().sort_index().to_dict()
    graph_data['medical_camps'] = {'labels': list(monthly_camps.keys()), 'data': list(monthly_camps.values())}

    # 4. Age vs Disease (Bar Chart - grouped by age)
    age_disease = district_data.groupby(['age_group', 'disease'])['cases'].sum().unstack(fill_value=0).to_dict('index')
    graph_data['age_disease'] = age_disease

    return jsonify(graph_data)

if __name__ == '__main__':
    app.run(debug=True)