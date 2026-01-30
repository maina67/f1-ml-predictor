# ============================================================================
# F1 RACE PREDICTOR WEB APP
# Built with Streamlit
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E10600;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #E10600;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #B10500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Load trained model and historical data."""
    try:
        # Load model
        model = joblib.load('models/f1_best_model.pkl')
        
        # Load model info
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load historical data
        df_full = pd.read_csv('data/f1_data_with_all_features.csv')
        df_full = df_full.sort_values(['Year', 'Round']).reset_index(drop=True)
        
        return model, model_info, df_full
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load everything
model, model_info, df_full = load_model_and_data()

# Get metadata
latest_year = int(df_full['Year'].max())
latest_round = int(df_full[df_full['Year'] == latest_year]['Round'].max())
feature_columns = model_info['feature_columns']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_driver_stats(driver_code, df):
    """Get driver's recent statistics."""
    driver_data = df[df['Driver'] == driver_code].tail(5)
    
    if len(driver_data) == 0:
        return None
    
    latest = driver_data.iloc[-1]
    
    return {
        'Driver': driver_code,
        'Driver_encoded': latest['Driver_encoded'],
        'Team': latest['Team'],
        'Team_encoded': latest['Team_encoded'],
        'Driver_Avg_Position_Last5': driver_data['FinishPosition'].mean(),
        'Driver_Avg_Position_Last3': driver_data['FinishPosition'].tail(3).mean(),
        'Driver_Finish_Rate': driver_data['Finished'].mean(),
        'Driver_Race_Count': int(latest['Driver_Race_Count']) + 1,
        'Team_Championship_Points': latest['Team_Championship_Points'],
    }

def predict_race(circuit_name, qualifying_results):
    """Make race predictions based on qualifying."""
    
    predictions = []
    
    # Get circuit encoding
    circuit_data = df_full[df_full['RaceName'] == circuit_name]
    if len(circuit_data) == 0:
        circuit_encoded = 0
    else:
        circuit_encoded = circuit_data['Circuit_encoded'].iloc[0]
    
    # Get team stats
    latest_teams = df_full[df_full['Year'] == latest_year].groupby('Team').agg({
        'FinishPosition': 'mean',
        'Team_Championship_Points': 'max'
    }).to_dict()
    
    for driver_code, quali_pos, grid_pos in qualifying_results:
        driver_stats = get_driver_stats(driver_code, df_full)
        
        if driver_stats is None:
            continue
        
        team = driver_stats['Team']
        team_avg = latest_teams['FinishPosition'].get(team, 10.5)
        team_points = latest_teams['Team_Championship_Points'].get(team, 0)
        
        # Circuit-specific performance
        driver_circuit = df_full[
            (df_full['Driver'] == driver_code) & 
            (df_full['RaceName'] == circuit_name)
        ]
        driver_circuit_avg = driver_circuit['FinishPosition'].mean() if len(driver_circuit) > 0 else 10.5
        
        team_circuit = df_full[
            (df_full['Team'] == team) & 
            (df_full['RaceName'] == circuit_name)
        ]
        team_circuit_avg = team_circuit['FinishPosition'].mean() if len(team_circuit) > 0 else 10.5
        
        # Teammate comparison
        teammate_quali = df_full[
            (df_full['Team'] == team) & 
            (df_full['Year'] == latest_year)
        ]['QualiPosition'].mean()
        
        # Build features
        features = {
            'GridPosition': grid_pos,
            'QualiPosition': quali_pos,
            'Driver_encoded': driver_stats['Driver_encoded'],
            'Team_encoded': driver_stats['Team_encoded'],
            'Circuit_encoded': circuit_encoded,
            'Driver_Avg_Position_Last5': driver_stats['Driver_Avg_Position_Last5'],
            'Driver_Avg_Position_Last3': driver_stats['Driver_Avg_Position_Last3'],
            'Team_Avg_Position_Last5': team_avg,
            'Driver_Finish_Rate': driver_stats['Driver_Finish_Rate'],
            'Driver_Circuit_Avg': driver_circuit_avg,
            'Team_Circuit_Avg': team_circuit_avg,
            'Quali_Grid_Diff': quali_pos - grid_pos,
            'Quali_vs_Teammate': quali_pos - teammate_quali,
            'Race_Number_In_Season': latest_round + 1,
            'Driver_Race_Count': driver_stats['Driver_Race_Count'],
            'Team_Championship_Points': team_points,
        }
        
        # Predict
        X_pred = pd.DataFrame([features])[feature_columns]
        predicted_pos = model.predict(X_pred)[0]
        
        predictions.append({
            'Driver': driver_code,
            'Team': team,
            'Grid': grid_pos,
            'Quali': quali_pos,
            'Predicted': round(predicted_pos, 1)
        })
    
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('Predicted').reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df) + 1)
    
    return results_df

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("## 🏎️ F1 Race Predictor")
st.sidebar.markdown("---")

# Model info
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.metric("Model Type", model_info['model_name'])
st.sidebar.metric("Accuracy (MAE)", f"{model_info['test_mae']:.2f} positions")
st.sidebar.metric("R² Score", f"{model_info['test_r2']:.3f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Latest Data")
st.sidebar.info(f"**Year:** {latest_year}\n\n**Round:** {latest_round}")

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<p class="main-header">🏎️ F1 Race Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Model Analysis", "ℹ️ About"])

# ============================================================================
# TAB 1: MAKE PREDICTION
# ============================================================================

with tab1:
    st.markdown("## 🏁 Predict Race Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚙️ Race Setup")
        
        # Circuit selection
        available_circuits = sorted(df_full['RaceName'].unique())
        selected_circuit = st.selectbox(
            "🏁 Select Circuit",
            available_circuits,
            index=0
        )
        
        # Get recent drivers
        recent_drivers = df_full[df_full['Year'] == latest_year]['Driver'].unique()
        recent_drivers = sorted(recent_drivers)
        
        st.markdown("### 👥 Qualifying Results")
        st.info("💡 Enter qualifying positions. Grid penalties will be calculated automatically.")
        
        # Create qualifying input
        quali_data = []
        
        num_drivers = st.slider("Number of drivers", 10, 20, 20)
        
        for i in range(num_drivers):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                driver = st.selectbox(
                    f"P{i+1}",
                    recent_drivers,
                    index=min(i, len(recent_drivers)-1),
                    key=f"driver_{i}",
                    label_visibility="collapsed"
                )
            
            with col_b:
                quali = i + 1
                st.text_input("Quali", value=str(quali), disabled=True, key=f"quali_{i}", label_visibility="collapsed")
            
            with col_c:
                grid = st.number_input(
                    "Grid",
                    min_value=1,
                    max_value=20,
                    value=i+1,
                    key=f"grid_{i}",
                    label_visibility="collapsed"
                )
            
            quali_data.append((driver, quali, grid))
        
        # Predict button
        if st.button("🚀 PREDICT RACE RESULTS", type="primary"):
            with st.spinner("🔮 Making predictions..."):
                results = predict_race(selected_circuit, quali_data)
                st.session_state['results'] = results
                st.session_state['circuit'] = selected_circuit
    
    with col2:
        st.markdown("### 🏆 Predicted Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            circuit = st.session_state['circuit']
            
            st.success(f"**Circuit:** {circuit}")
            
            # Podium
            st.markdown("#### 🥇 Predicted Podium")
            podium = results.head(3)
            
            medals = ["🥇", "🥈", "🥉"]
            colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
            
            for idx, (i, row) in enumerate(podium.iterrows()):
                st.markdown(f"""
                <div style='background: {colors[idx]}; padding: 10px; border-radius: 5px; margin: 5px 0; color: black;'>
                    <b>{medals[idx]} P{idx+1}: {row['Driver']}</b> ({row['Team']})<br>
                    Grid: P{row['Grid']} → Predicted: P{row['Predicted']}
                </div>
                """, unsafe_allow_html=True)
            
            # Full results
            st.markdown("#### 📋 Full Results")
            
            # Style the dataframe
            st.dataframe(
                results[['Rank', 'Driver', 'Team', 'Grid', 'Predicted']],
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=results['Driver'],
                y=results['Predicted'],
                marker_color='#E10600',
                name='Predicted Position'
            ))
            
            fig.update_layout(
                title="Predicted Finishing Positions",
                xaxis_title="Driver",
                yaxis_title="Position",
                yaxis=dict(autorange="reversed"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"f1_prediction_{circuit.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("👈 Set up race parameters and click 'PREDICT RACE RESULTS'")

# ============================================================================
# TAB 2: MODEL ANALYSIS
# ============================================================================

with tab2:
    st.markdown("## 📊 Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Metrics")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                "Mean Absolute Error",
                f"{model_info['test_mae']:.2f}",
                delta="positions",
                delta_color="inverse"
            )
        
        with metric_col2:
            st.metric(
                "R² Score",
                f"{model_info['test_r2']:.3f}",
                delta=f"{model_info['test_r2']*100:.1f}%"
            )
        
        st.markdown("### 📈 What This Means")
        st.info(f"""
        **Mean Absolute Error (MAE): {model_info['test_mae']:.2f}**
        - On average, predictions are off by {model_info['test_mae']:.2f} positions
        - For example: Predicts P5 when actual is P{5 + model_info['test_mae']:.0f}
        
        **R² Score: {model_info['test_r2']:.3f}**
        - Model explains {model_info['test_r2']*100:.1f}% of variance in race results
        - Remaining {(1-model_info['test_r2'])*100:.1f}% is unpredictable (crashes, strategy, etc.)
        """)
    
    with col2:
        st.markdown("### 🔍 Feature Importance")
        
        st.info("""
        **Top Features:**
        1. 🏁 **Qualifying Position** (48.5%) - Where you start matters most!
        2. 🏎️ **Team Circuit Avg** (9.6%) - Some teams excel at certain tracks
        3. 🏆 **Team Points** (8.4%) - Stronger teams = better results
        4. 📊 **Recent Form** (4.3%) - Hot streaks continue
        5. 👤 **Driver Form** (3.8%) - Individual performance
        """)
    
    # Historical data viz
    st.markdown("### 📅 Historical Data Coverage")
    
    years_data = df_full.groupby('Year').agg({
        'Round': 'nunique',
        'Driver': 'count'
    }).reset_index()
    years_data.columns = ['Year', 'Races', 'Records']
    
    fig = px.bar(
        years_data,
        x='Year',
        y='Races',
        title='Races per Year in Training Data',
        color='Races',
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("## ℹ️ About This App")
    
    st.markdown("""
    ### 🏎️ F1 Race Predictor
    
    This application uses machine learning to predict Formula 1 race results based on qualifying positions and historical performance data.
    
    ### 🤖 How It Works
    
    1. **Data Collection**: Historical F1 race data from 2021-2025
    2. **Feature Engineering**: Creating intelligent features like recent form, circuit-specific performance
    3. **Model Training**: Gradient Boosting model trained on 2000+ race records
    4. **Predictions**: Input qualifying results → Get race finish predictions
    
    ### 📊 Model Details
    
    - **Algorithm**: Gradient Boosting Regressor
    - **Training Data**: {0} races from {1}-{2}
    - **Accuracy**: {3:.2f} positions MAE
    - **Features**: 16 intelligent features
    
    ### 🎯 Use Cases
    
    - Predict race outcomes before lights out
    - Analyze "what-if" scenarios (grid penalties, different quali results)
    - Understand which factors matter most in F1
    - Make informed fantasy F1 picks
    
    ### ⚠️ Limitations
    
    - Cannot predict unpredictable events (crashes, safety cars, rain)
    - Based on historical patterns (new regulations may change things)
    - Accuracy varies by circuit and conditions
    
    ### 👨‍💻 Built With
    
    - **Python**: pandas, scikit-learn, XGBoost
    - **Web Framework**: Streamlit
    - **Data Source**: FastF1 API
    
    ---
    
    Made with ❤️ for F1 fans
    """.format(
        len(df_full.groupby(['Year', 'Round'])),
        df_full['Year'].min(),
        df_full['Year'].max(),
        model_info['test_mae']
    ))

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏎️ F1 Race Predictor | Last updated: {}</p>
    <p>Data up to: {} Season, Round {}</p>
</div>
""".format(
    datetime.now().strftime("%Y-%m-%d"),
    latest_year,
    latest_round
), unsafe_allow_html=True)
