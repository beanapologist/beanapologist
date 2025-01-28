import streamlit as st
from datetime import datetime
import time

def calculate_universe_age():
    # Universe age constants
    UNIVERSE_AGE_YEARS = 13.799e9  # 13.799 billion years
    SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
    
    # Calculate base age in seconds
    base_age_seconds = UNIVERSE_AGE_YEARS * SECONDS_PER_YEAR
    
    # Add seconds elapsed since Jan 1, 1970 (Unix epoch)
    current_seconds = time.time()
    
    # Total age in seconds
    total_age_seconds = base_age_seconds + current_seconds
    
    return total_age_seconds

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
        .universe-age {
            font-size: 2rem;
            color: #4CAF50;
            margin: 1rem 0;
            word-wrap: break-word;
        }
        .note {
            font-size: 0.8rem;
            color: #888;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create placeholder for clock display
    clock_container = st.empty()

    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        universe_age = calculate_universe_age()
        
        # Update clock display
        clock_container.markdown(f"""
            <div class="clock-container">
                <h2>Universal Clock</h2>
                <div class="time">
                    Current Time: {current_time}
                </div>
                <div class="universe-age">
                    Universe Age: {universe_age:,.0f} seconds
                </div>
                <div class="note">
                    Based on the current scientific estimate of 13.799 billion years
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)

if __name__ == '__main__':
    main() 