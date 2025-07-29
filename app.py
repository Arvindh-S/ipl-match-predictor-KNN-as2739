import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
try:
    with open('best_knn_model.pkl', 'rb') as file:
        cricket_model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'best_knn_model.pkl' is in the same directory.")
    st.stop()

# Page setup
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

def setup_cricket_style():
    st.markdown("""
    <style>
    /* Dark cricket theme background */
    .stApp {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 50%, #0f4c75 100%);
        color: white;
    }
    
    /* Main content - semi-transparent overlay for readability */
    .main .block-container {
        background: rgba(0, 0, 0, 0.3);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    .header-section {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    }
    
    /* Ensure all text is white and visible */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }
    
    /* Input labels */
    .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        background: rgba(0, 0, 0, 0.4) !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
    }
    
    /* Section headers */
    .section-title {
        background: rgba(76, 175, 80, 0.8);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(255, 107, 53, 0.6);
    }
    
    /* Input fields */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 6px !important;
        color: white !important;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        backdrop-filter: blur(5px) !important;
    }
    
    [data-testid="metric-container"] label {
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }
    
    [data-testid="metric-container"] div {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
    }
    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
        text-align: center;
    }
    
    /* Info and warning boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.3) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

setup_cricket_style()

# Header
st.markdown("""
<div class="header-section">
    <h1>🏏 IPL Match Predictor</h1>
    <p style="font-size: 1.2rem; margin: 0;">Predict the winner during second innings chase</p>
</div>
""", unsafe_allow_html=True)

# Description
st.markdown("""
### 🎯 How It Works
Enter the current match details and get instant win probability predictions using advanced machine learning!
""")

# Team data
ipl_teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
    'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
]

cricket_venues = [
    'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 'Delhi', 'Hyderabad',
    'Jaipur', 'Chandigarh', 'Ahmedabad', 'Pune', 'Indore', 'Lucknow',
    'Dharamsala', 'Visakhapatnam', 'Abu Dhabi', 'Dubai', 'Sharjah'
]

# Team selection
st.markdown('<div class="section-title">🏟️ Team Selection</div>', unsafe_allow_html=True)

team_col1, team_col2 = st.columns(2)

with team_col1:
    batting_team = st.selectbox("🏏 Batting Team (Chasing)", ipl_teams)
    bowling_team = st.selectbox("🥎 Bowling Team (Defending)", ipl_teams)

with team_col2:
    toss_winner = st.selectbox("🪙 Toss Winner", [batting_team, bowling_team])
    venue_city = st.selectbox("🏟️ Match Venue", sorted(cricket_venues))

# Match situation
st.markdown('<div class="section-title">📊 Current Match Situation</div>', unsafe_allow_html=True)

match_col1, match_col2 = st.columns(2)

with match_col1:
    target_total = st.number_input("🎯 Target Score", min_value=1, value=180, step=1)
    current_total = st.number_input("📊 Current Score", min_value=0, value=45, step=1)

with match_col2:
    overs_done = st.number_input("⏱️ Overs Completed", min_value=0.0, max_value=20.0, value=8.0, step=0.1, format="%.1f")
    wickets_down = st.number_input("🏏 Wickets Lost", min_value=0, max_value=10, value=2, step=1)

# Live stats
if overs_done > 0:
    runs_remaining = target_total - current_total
    balls_remaining = 120 - (int(overs_done) * 6 + round((overs_done - int(overs_done)) * 10))
    wickets_left = 10 - wickets_down
    current_rate = current_total / overs_done
    required_rate = (runs_remaining * 6) / balls_remaining if balls_remaining > 0 else 0

    st.markdown('<div class="section-title">📈 Live Match Statistics</div>', unsafe_allow_html=True)
    
    stat1, stat2, stat3, stat4 = st.columns(4)
    
    with stat1:
        st.metric("Runs Needed", runs_remaining)
    with stat2:
        st.metric("Balls Left", balls_remaining)
    with stat3:
        st.metric("Current Rate", f"{current_rate:.2f}")
    with stat4:
        st.metric("Required Rate", f"{required_rate:.2f}")

# Prediction
st.markdown("---")
if st.button("🎯 Predict Match Winner", use_container_width=True):
    
    if batting_team == bowling_team:
        st.error("❌ Please select different teams for batting and bowling!")
    elif current_total >= target_total:
        st.balloons()
        st.success(f"🎉 **{batting_team}** has already won!")
    elif overs_done <= 0:
        st.warning("⚠️ Please enter overs completed for prediction.")
    else:
        try:
            # Calculate features
            runs_remaining = target_total - current_total
            balls_remaining = 120 - (int(overs_done) * 6 + round((overs_done - int(overs_done)) * 10))
            wickets_left = 10 - wickets_down
            current_rate = current_total / overs_done
            required_rate = (runs_remaining * 6) / balls_remaining if balls_remaining > 0 else float('inf')

            # Prepare data
            game_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [venue_city],
                'runs_left': [runs_remaining],
                'balls_left': [balls_remaining],
                'wickets_left': [wickets_left],
                'first_innings_score': [target_total - 1],
                'current_run_rate': [current_rate],
                'required_run_rate': [required_rate],
                'toss_winner': [toss_winner]
            })

            # Predict
            match_prediction = cricket_model.predict_proba(game_data)
            batting_probability = match_prediction[0][1] * 100
            bowling_probability = match_prediction[0][0] * 100

            # Show results
            st.markdown("### 🏆 Match Prediction Results")
            st.markdown("*Based on current match situation and historical data*")
          
            result1, result2 = st.columns(2)

            with result1:
                st.markdown(f"""
                <div class="result-card">
                    <h3>🏏 {batting_team}</h3>
                    <h2 style="color: #ff6b35;">{batting_probability:.1f}%</h2>
                    <p>Win Probability</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(batting_probability))

            with result2:
                st.markdown(f"""
                <div class="result-card">
                    <h3>🥎 {bowling_team}</h3>
                    <h2 style="color: #00bcd4;">{bowling_probability:.1f}%</h2>
                    <p>Win Probability</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(int(bowling_probability))
          
            # Final verdict
            st.markdown("---")
            
            if batting_probability > 60:
                st.success(f"🎯 **{batting_team}** is strongly favored to win!")
            elif bowling_probability > 60:
                st.info(f"🛡️ **{bowling_team}** has the upper hand!")
            else:
                st.warning("⚖️ Too close to call - exciting finish ahead!")

        except Exception as error:
            st.error(f"❌ Prediction failed: {error}")

# Instructions
with st.expander("📚 How to Use This Predictor"):
    st.markdown("""
    **Step-by-Step Guide:**
    
    1. **Team Selection** - Choose batting and bowling teams
    2. **Toss Winner** - Select who won the toss  
    3. **Venue** - Pick the match venue
    4. **Match Details** - Enter target, current score, overs, and wickets
    5. **Predict** - Click the button to get win probabilities
    
    **Note:** The model works best after at least 3-4 overs have been bowled.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p>🏏 <strong>IPL Match Predictor</strong> | Powered by Machine Learning</p>
    <p><em>May the best team win! 🏆</em></p>
</div>
""", unsafe_allow_html=True)