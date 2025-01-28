import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass
from typing import Dict
from datetime import datetime
import time
import streamlit as st  # type: ignore # Add streamlit import

def main():
    # Page config
    st.set_page_config(
        page_title="Universal Clock",
        page_icon="⏰",
        layout="centered"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .clock-container {
            background-color: #2a2a2a;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            text-align: center;
            margin: 2rem 0;
        }
        .time {
            font-size: 3rem;
            margin: 1rem 0;
        }
        .model-time {
            font-size: 2rem;
            color: #4CAF50;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize model
    config = CalibratedConfig() # type: ignore
    model = CalibratedTimeModel(config) # type: ignore
    try:
        model.load_state_dict(torch.load('UniversalClock.pth'))
        model.eval()
    except FileNotFoundError:
        st.warning("Warning: Model weights file 'UniversalClock.pth' not found. Using default initialization.")

    # Create placeholder for clock display
    clock_container = st.empty()

    while True:
        with torch.no_grad():
            prediction, metrics = model()

        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Update clock display
        clock_container.markdown(f"""
            <div class="clock-container">
                <h2>Universal Clock</h2>
                <div class="time">
                    Current Time: {current_time}
                </div>
                <div class="model-time">
                    Model Prediction: {int(prediction.item()):,} seconds
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)

if __name__ == '__main__':
    main() 