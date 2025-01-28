from flask import Flask, jsonify, render_template_string
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from datetime import datetime

def main():
    config = CalibratedConfig()
    evaluator = CalibratedEvaluator(config)
    metrics = evaluator.measure_age()
    evaluator.print_results(metrics)

# Flask application setup
app = Flask(__name__)

# Initialize model globally
config = CalibratedConfig()
model = CalibratedTimeModel(config)
try:
    model.load_state_dict(torch.load('UniversalClock.pth'))
    model.eval()
except FileNotFoundError:
    print("Warning: Model weights file 'UniversalClock.pth' not found. Using default initialization.")

# HTML template for the clock display
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Universal Clock</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .clock-container {
            text-align: center;
            padding: 2em;
            background-color: #2a2a2a;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        }
        .time {
            font-size: 3em;
            margin: 0.5em 0;
        }
        .model-time {
            font-size: 2em;
            color: #4CAF50;
            margin: 0.5em 0;
        }
        .refresh {
            color: #888;
            font-size: 0.8em;
        }
    </style>
    <script>
        function updateClock() {
            fetch('/predict')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('model-time').textContent = 
                        Math.floor(data.age_seconds).toLocaleString() + ' seconds';
                });
            
            const now = new Date();
            document.getElementById('actual-time').textContent = 
                now.toLocaleTimeString();
            
            setTimeout(updateClock, 1000);
        }
    </script>
</head>
<body onload="updateClock()">
    <div class="clock-container">
        <h2>Universal Clock</h2>
        <div class="time">
            Current Time: <span id="actual-time"></span>
        </div>
        <div class="model-time">
            Model Prediction: <span id="model-time"></span>
        </div>
        <p class="refresh">Updates automatically every second</p>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['GET'])
def predict():
    with torch.no_grad():
        prediction, metrics = model()
    
    return jsonify({
        'age_seconds': prediction.item(),
        'metrics': metrics
    })

if __name__ == '__main__':
    app.run(debug=True) 