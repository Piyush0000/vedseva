from flask import Flask, render_template, request, jsonify
import pandas as pd
from io import StringIO
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

app = Flask(__name__)

# Load the CSV data (replace with your actual file path or data)
csv_data = """state,district,month,year,disease,cases,age_group,medical_camps
Karnataka,Bengaluru Urban,January,2025,Respiratory Infection,145,0-15,3
Karnataka,Bengaluru Urban,February,2025,Respiratory Infection,132,0-15,4
Karnataka,Bengaluru Urban,March,2025,Respiratory Infection,98,0-15,5
Karnataka,Bengaluru Urban,April,2025,Respiratory Infection,67,0-15,6
Karnataka,Bengaluru Urban,January,2025,Dengue,23,0-15,3
Karnataka,Bengaluru Urban,February,2025,Dengue,45,0-15,4
Karnataka,Bengaluru Urban,March,2025,Dengue,78,0-15,5
Karnataka,Bengaluru Urban,April,2025,Dengue,120,0-15,6
Karnataka,Bengaluru Urban,January,2025,Malaria,12,0-15,3
Karnataka,Bengaluru Urban,February,2025,Malaria,15,0-15,4
Karnataka,Bengaluru Urban,March,2025,Malaria,18,0-15,5
Karnataka,Bengaluru Urban,April,2025,Malaria,22,0-15,6
Karnataka,Bengaluru Urban,January,2025,Diabetes,5,0-15,3
Karnataka,Bengaluru Urban,February,2025,Diabetes,6,0-15,4
Karnataka,Bengaluru Urban,March,2025,Diabetes,7,0-15,5
Karnataka,Bengaluru Urban,April,2025,Diabetes,8,0-15,6
Karnataka,Bengaluru Rural,January,2025,Respiratory Infection,110,0-15,2
Karnataka,Bengaluru Rural,February,2025,Respiratory Infection,100,0-15,3
Maharashtra,Mumbai City,January,2025,Respiratory Infection,178,0-15,5
Maharashtra,Mumbai City,February,2025,Respiratory Infection,152,0-15,6
Maharashtra,Pune,January,2025,Dengue,30,0-15,2
Maharashtra,Pune,February,2025,Dengue,55,0-15,3
"""
DATA = pd.read_csv("healthcare_data.csv")

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

    # 3. Medical Camp vs Time
    monthly_camps = district_data.groupby('month')['medical_camps'].sum().sort_index().to_dict()
    graph_data['medical_camps'] = {'labels': list(monthly_camps.keys()), 'data': list(monthly_camps.values())}

    # 4. Age vs Disease
    age_disease = district_data.groupby(['age_group', 'disease'])['cases'].sum().unstack(fill_value=0).to_dict('index')
    graph_data['age_disease'] = age_disease

    return jsonify(graph_data)

if __name__ == '__main__':
    app.run(debug=True)