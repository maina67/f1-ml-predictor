# ============================================================================
# PREDICT FUTURE F1 RACES
# Use the trained model to predict upcoming race results
# ============================================================================

import pandas as pd
import numpy as np
import joblib
import json

print("="*70)
print("🔮 F1 RACE PREDICTION SYSTEM")
print("="*70)

# ============================================================================
# 1. LOAD THE TRAINED MODEL
# ============================================================================

print("\n1️⃣ Loading trained model...")

# Load model
model = joblib.load('/content/data/f1_best_model.pkl')

# Load feature info
with open('/content/data/model_info.json', 'r') as f:
    model_info = json.load(f)

feature_columns = model_info['feature_columns']

print(f"✓ Loaded: {model_info['model_name']}")
print(f"✓ Test MAE: {model_info['test_mae']:.3f} positions")
print(f"✓ Features: {len(feature_columns)}")

# ============================================================================
# 2. LOAD HISTORICAL DATA (for context)
# ============================================================================

print("\n2️⃣ Loading historical data...")

df_full = pd.read_csv('/content/data/f1_data_with_all_features.csv')
df_full = df_full.sort_values(['Year', 'Round']).reset_index(drop=True)

latest_year = df_full['Year'].max()
latest_round = df_full[df_full['Year'] == latest_year]['Round'].max()

print(f"✓ Latest data: {latest_year} Round {latest_round}")
print(f"✓ Total records: {len(df_full)}")

# ============================================================================
# 3. CREATE PREDICTION TEMPLATE
# ============================================================================

print("\n3️⃣ Creating prediction template...")

def get_driver_latest_stats(driver_code, df):
    """Get a driver's latest statistics."""
    driver_data = df[df['Driver'] == driver_code].tail(5)  # Last 5 races
    
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
        'Driver_Race_Count': latest['Driver_Race_Count'] + 1,
        'Team_Championship_Points': latest['Team_Championship_Points'],
    }

# Get active drivers from latest season
active_drivers = df_full[df_full['Year'] == latest_year]['Driver'].unique()
print(f"✓ Found {len(active_drivers)} active drivers")

# ============================================================================
# 4. PREDICT NEXT RACE
# ============================================================================

print("\n4️⃣ Making predictions for next race...")

def predict_race(circuit_name, qualifying_results, df_historical):
    """
    Predict race results based on qualifying.
    
    Parameters:
    - circuit_name: Name of the circuit (e.g., "Monaco Grand Prix")
    - qualifying_results: List of (driver_code, quali_position, grid_position)
    - df_historical: Historical data
    
    Returns:
    - DataFrame with predictions
    """
    
    predictions = []
    
    # Get circuit encoding
    circuit_encoded = df_historical[df_historical['RaceName'] == circuit_name]['Circuit_encoded'].iloc[0] if circuit_name in df_historical['RaceName'].values else 0
    
    # Get team stats
    latest_teams = df_historical[df_historical['Year'] == latest_year].groupby('Team').agg({
        'FinishPosition': 'mean',
        'Team_Championship_Points': 'max'
    }).to_dict()
    
    for driver_code, quali_pos, grid_pos in qualifying_results:
        # Get driver stats
        driver_stats = get_driver_latest_stats(driver_code, df_historical)
        
        if driver_stats is None:
            continue
        
        # Get team stats
        team = driver_stats['Team']
        team_avg = latest_teams['FinishPosition'].get(team, 10.5)
        team_points = latest_teams['Team_Championship_Points'].get(team, 0)
        
        # Get circuit-specific performance
        driver_circuit_history = df_historical[
            (df_historical['Driver'] == driver_code) & 
            (df_historical['RaceName'] == circuit_name)
        ]
        driver_circuit_avg = driver_circuit_history['FinishPosition'].mean() if len(driver_circuit_history) > 0 else 10.5
        
        team_circuit_history = df_historical[
            (df_historical['Team'] == team) & 
            (df_historical['RaceName'] == circuit_name)
        ]
        team_circuit_avg = team_circuit_history['FinishPosition'].mean() if len(team_circuit_history) > 0 else 10.5
        
        # Calculate teammate comparison (simplified)
        teammate_quali_avg = df_historical[
            (df_historical['Team'] == team) & 
            (df_historical['Year'] == latest_year)
        ]['QualiPosition'].mean()
        
        # Build feature vector
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
            'Quali_vs_Teammate': quali_pos - teammate_quali_avg,
            'Race_Number_In_Season': latest_round + 1,
            'Driver_Race_Count': driver_stats['Driver_Race_Count'],
            'Team_Championship_Points': team_points,
        }
        
        predictions.append({
            'Driver': driver_code,
            'Team': team,
            'GridPosition': grid_pos,
            'QualiPosition': quali_pos,
            'Features': features
        })
    
    # Make predictions
    for pred in predictions:
        X_pred = pd.DataFrame([pred['Features']])[feature_columns]
        predicted_position = model.predict(X_pred)[0]
        pred['Predicted_Position'] = predicted_position
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'Driver': p['Driver'],
            'Team': p['Team'],
            'Grid': p['GridPosition'],
            'Quali': p['QualiPosition'],
            'Predicted_Finish': round(p['Predicted_Position'], 1)
        }
        for p in predictions
    ])
    
    # Sort by predicted position
    results_df = results_df.sort_values('Predicted_Finish')
    results_df['Predicted_Rank'] = range(1, len(results_df) + 1)
    
    return results_df

# ============================================================================
# 5. EXAMPLE: PREDICT A RACE
# ============================================================================

print("\n5️⃣ Example Prediction...")
print("\nLet's predict a race! I'll use recent driver data.")
print("You can customize the qualifying results below.")

# Example qualifying results (Driver, Quali Position, Grid Position)
# Format: [('VER', 1, 1), ('HAM', 2, 2), ...]

# Get top drivers from latest data
recent_drivers = df_full[df_full['Year'] == latest_year].groupby('Driver').size().sort_values(ascending=False).head(20).index

example_quali = []
for i, driver in enumerate(recent_drivers[:20], 1):
    example_quali.append((driver, i, i))  # Assuming no grid penalties

# Predict for a circuit (use one from your data)
available_circuits = df_full['RaceName'].unique()
example_circuit = available_circuits[0]  # First circuit in your data

print(f"\n📍 Circuit: {example_circuit}")
print(f"👥 Drivers: {len(example_quali)}")

prediction_results = predict_race(example_circuit, example_quali, df_full)

print("\n" + "="*70)
print("🏁 PREDICTED RACE RESULTS")
print("="*70)
print(prediction_results.to_string(index=False))

print("\n💡 How to read this:")
print("  • Grid = Starting position")
print("  • Quali = Qualifying position")
print("  • Predicted_Finish = Where model thinks they'll finish")
print("  • Lower number = Better position")

# ============================================================================
# 6. SAVE PREDICTIONS
# ============================================================================

print("\n6️⃣ Saving predictions...")

prediction_results.to_csv('/content/data/race_prediction.csv', index=False)
print("✓ Saved: race_prediction.csv")

# ============================================================================
# 7. CUSTOMIZABLE PREDICTION FUNCTION
# ============================================================================

print("\n" + "="*70)
print("✅ PREDICTION SYSTEM READY!")
print("="*70)

print("\n🎯 How to make YOUR OWN predictions:")
print("""
# Define qualifying results (customize this!)
my_qualifying = [
    ('VER', 1, 1),   # Verstappen: Quali P1, Grid P1
    ('HAM', 2, 2),   # Hamilton: Quali P2, Grid P2
    ('LEC', 3, 3),   # Leclerc: Quali P3, Grid P3
    # ... add all 20 drivers
]

# Pick a circuit
my_circuit = "Monaco Grand Prix"  # Or any circuit from your data

# Make prediction
my_results = predict_race(my_circuit, my_qualifying, df_full)
print(my_results)
""")

print("\n📋 Available circuits in your data:")
for i, circuit in enumerate(available_circuits[:10], 1):
    print(f"  {i}. {circuit}")
print(f"  ... and {len(available_circuits) - 10} more")

print("\n" + "="*70)