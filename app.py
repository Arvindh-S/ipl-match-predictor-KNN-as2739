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

def smooth_probabilities(raw_probs, smoothing_factor=0.15):
    """
    Apply smoothing to extreme probabilities to make them more realistic
    """
    # Get the probabilities
    prob_0, prob_1 = raw_probs[0], raw_probs[1]
    
    # Apply smoothing - pull extreme values towards center
    if prob_1 > 0.85:  # If batting team probability is very high
        prob_1 = 0.85 - (prob_1 - 0.85) * smoothing_factor
        prob_0 = 1 - prob_1
    elif prob_1 < 0.15:  # If batting team probability is very low
        prob_1 = 0.15 + (0.15 - prob_1) * smoothing_factor
        prob_0 = 1 - prob_1
    
    return [prob_0, prob_1]

def calculate_contextual_adjustment(runs_remaining, balls_remaining, wickets_left, required_rate, current_rate):
    """
    Calculate contextual adjustments based on match situation
    """
    adjustment = 0
    
    # Pressure situations
    if required_rate > 12:
        adjustment -= 0.1  # Favor bowling team
    elif required_rate < 6:
        adjustment += 0.1  # Favor batting team
    
    # Wickets in hand
    if wickets_left <= 3:
        adjustment -= 0.05  # Pressure on batting team
    elif wickets_left >= 7:
        adjustment += 0.05  # Batting team has depth
    
    # Death overs
    if balls_remaining <= 30:  # Last 5 overs
        if required_rate > 10:
            adjustment -= 0.08
        else:
            adjustment += 0.03
    
    # Run rate comparison
    rate_diff = required_rate - current_rate
    if rate_diff > 3:
        adjustment -= 0.05
    elif rate_diff < -1:
        adjustment += 0.05
    
    return max(-0.25, min(0.25, adjustment))  # Cap adjustment between -25% and +25%

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

# Prediction methodology
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

            # Get raw prediction
            raw_prediction = cricket_model.predict_proba(game_data)
            
            smoothed_probs = smooth_probabilities(raw_prediction[0])
            
            contextual_adj = calculate_contextual_adjustment(
                runs_remaining, balls_remaining, wickets_left, required_rate, current_rate
            )
            
            batting_probability = max(5, min(95, (smoothed_probs[1] + contextual_adj) * 100))
            bowling_probability = 100 - batting_probability

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
          
            # Show match insights
            st.markdown("---")
            st.markdown("### 📊 Match Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                if required_rate > current_rate + 2:
                    st.warning("⚠️ Required run rate significantly higher than current rate")
                elif required_rate < current_rate - 1:
                    st.success("✅ Batting team ahead of required rate")
                else:
                    st.info("ℹ️ Match evenly poised")
            
            with insight_col2:
                if wickets_left <= 3:
                    st.error("🚨 Few wickets remaining - pressure situation")
                elif wickets_left >= 7:
                    st.success("💪 Plenty of wickets in hand")
                else:
                    st.info("⚖️ Moderate wickets remaining")
            
            # Final verdict
            st.markdown("---")
            
            if batting_probability > 65:
                st.success(f"🎯 **{batting_team}** is strongly favored to win!")
            elif bowling_probability > 65:
                st.info(f"🛡️ **{bowling_team}** has the upper hand!")
            else:
                st.warning("⚖️ Too close to call - exciting finish ahead!")
                
            # Show key factors
            st.markdown("### 🔍 Key Factors")
            factors = []
            
            if required_rate > 12:
                factors.append("🔥 High required run rate favors bowling team")
            elif required_rate < 6:
                factors.append("🎯 Low required run rate favors batting team")
                
            if balls_remaining <= 30:
                factors.append("⏰ Death overs - crucial phase")
                
            if wickets_left <= 3:
                factors.append("🎯 Limited wickets increase pressure")
                
            if abs(required_rate - current_rate) > 3:
                factors.append("📊 Significant difference in run rates")
            
            if factors:
                for factor in factors:
                    st.write(f"• {factor}")
            else:
                st.write("• Match is evenly balanced with no major advantage")

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
    
    **Probability Calculation:**
    - The model considers historical match data and current situation
    - Probabilities are adjusted based on match context (run rates, wickets, pressure situations)
    - Extreme predictions are smoothed to provide more realistic probabilities
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p>🏏 <strong>IPL Match Predictor</strong> | Powered by Machine Learning</p>
    <p><em>May the best team win! 🏆</em></p>
</div>
""", unsafe_allow_html=True)
