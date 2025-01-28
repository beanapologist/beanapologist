import streamlit as st
from datetime import datetime
import time

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
        </style>
    """, unsafe_allow_html=True)

    # Create placeholder for clock display
    clock_container = st.empty()

    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Update clock display
        clock_container.markdown(f"""
            <div class="clock-container">
                <h2>Universal Clock</h2>
                <div class="time">
                    Current Time: {current_time}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)

if __name__ == '__main__':
    main() 