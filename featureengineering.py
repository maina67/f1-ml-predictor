# ============================================================================
# STEP 2: FEATURE ENGINEERING
# We're creating "smart" features that help predict race results
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

# Load combined data
df = pd.read_csv('/content/data/f1_data_combined_all_years.csv')

print(f"\n📥 Loaded: {len(df)} records from {df['Year'].nunique()} years")
print(f"Years: {sorted(df['Year'].unique())}")

# ============================================================================
# 1. ENCODE CATEGORICAL VARIABLES
# ============================================================================
# 🤔 WHY: ML models need numbers, not text like "Hamilton" or "Mercedes"
# 📊 WHAT: Convert driver names, teams, circuits into numbers
# 💡 EXAMPLE: "VER" → 0, "HAM" → 1, "LEC" → 2, etc.

print("\n" + "="*70)
print("1️⃣ ENCODING CATEGORICAL VARIABLES")
print("="*70)
print("Converting text (drivers, teams, circuits) into numbers...")
print("Why? ML models only understand numbers, not names!")

# Encode Driver (VER=0, HAM=1, etc.)
le_driver = LabelEncoder()
df['Driver_encoded'] = le_driver.fit_transform(df['Driver'])

# Encode Team (Red Bull=0, Mercedes=1, etc.)
le_team = LabelEncoder()
df['Team_encoded'] = le_team.fit_transform(df['Team'])

# Encode Circuit (Monaco=0, Silverstone=1, etc.)
le_circuit = LabelEncoder()
df['Circuit_encoded'] = le_circuit.fit_transform(df['RaceName'])

print(f"\n✓ Encoded {df['Driver'].nunique()} drivers")
print(f"✓ Encoded {df['Team'].nunique()} teams")
print(f"✓ Encoded {df['RaceName'].nunique()} circuits")

print(f"\nExample: {df['Driver'].iloc[0]} → {df['Driver_encoded'].iloc[0]}")
print(f"Example: {df['Team'].iloc[0]} → {df['Team_encoded'].iloc[0]}")

# ============================================================================
# 2. HISTORICAL PERFORMANCE FEATURES
# ============================================================================
# 🤔 WHY: Recent form matters! A driver who finished P1, P2, P3 recently 
#         is more likely to finish well than one who finished P15, P18, P20
# 📊 WHAT: Calculate rolling averages of recent race results
# 💡 EXAMPLE: If last 5 races were [1, 2, 3, 2, 1], average = 1.8

print("\n" + "="*70)
print("2️⃣ HISTORICAL PERFORMANCE FEATURES")
print("="*70)
print("Creating 'recent form' features...")
print("Why? A driver on a hot streak is likely to continue performing well!")

# Sort by year and round (chronological order)
df = df.sort_values(['Year', 'Round']).reset_index(drop=True)

# Driver's average finish in LAST 5 RACES
# shift(1) = don't include current race (that's what we're predicting!)
df['Driver_Avg_Position_Last5'] = df.groupby('Driver')['FinishPosition'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)

print("\n✓ Driver_Avg_Position_Last5")
print("  What: Average finish position in last 5 races")
print("  Example: If VER finished [1, 1, 2, 1, 3], this = 1.6")
print("  Lower = Better (P1 is best)")

# Driver's VERY recent form (last 3 races)
df['Driver_Avg_Position_Last3'] = df.groupby('Driver')['FinishPosition'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

print("\n✓ Driver_Avg_Position_Last3")
print("  What: Even more recent form (last 3 races)")
print("  Why: Recent races matter more than older ones")

# Team's average position (last 5 races)
df['Team_Avg_Position_Last5'] = df.groupby('Team')['FinishPosition'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)

print("\n✓ Team_Avg_Position_Last5")
print("  What: Team's average finish (both drivers combined)")
print("  Why: Strong teams = better results")

# Driver's FINISH RATE (reliability)
# How often does driver actually finish races? (vs DNF - Did Not Finish)
df['Driver_Finish_Rate'] = df.groupby('Driver')['Finished'].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
)

print("\n✓ Driver_Finish_Rate")
print("  What: % of races finished in last 10 races")
print("  Example: 0.9 = finished 9 out of 10 races")
print("  Why: Finishing matters! Can't score points if you DNF")

# ============================================================================
# 3. CIRCUIT-SPECIFIC PERFORMANCE
# ============================================================================
# 🤔 WHY: Some drivers are AMAZING at certain tracks (e.g., Hamilton at 
#         Silverstone) but average at others
# 📊 WHAT: Track how each driver/team performs at each specific circuit
# 💡 EXAMPLE: VER averages P1.3 at Spa but P2.5 at Monaco

print("\n" + "="*70)
print("3️⃣ CIRCUIT-SPECIFIC PERFORMANCE")
print("="*70)
print("Some drivers dominate certain tracks...")
print("Why? Each circuit is unique (Monaco vs Monza are completely different!)")

# Driver's historical average at THIS circuit
df['Driver_Circuit_Avg'] = df.groupby(['Driver', 'RaceName'])['FinishPosition'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)

print("\n✓ Driver_Circuit_Avg")
print("  What: Driver's avg finish at this specific circuit")
print("  Example: HAM at Silverstone = 1.8, HAM at Monaco = 3.2")
print("  Why: Track-specific strengths matter!")

# Team's historical average at THIS circuit
df['Team_Circuit_Avg'] = df.groupby(['Team', 'RaceName'])['FinishPosition'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)

print("\n✓ Team_Circuit_Avg")
print("  What: Team's avg finish at this circuit")
print("  Why: Some cars suit certain tracks better")

# ============================================================================
# 4. QUALIFYING ADVANTAGE
# ============================================================================
# 🤔 WHY: Starting position is HUGE in F1! Hard to overtake
# 📊 WHAT: Features related to qualifying performance
# 💡 EXAMPLE: Started P1 but grid P3 = 2-place penalty

print("\n" + "="*70)
print("4️⃣ QUALIFYING ADVANTAGE")
print("="*70)
print("'Quali is everything' - in F1, starting position matters A LOT!")
print("Why? Overtaking is hard, starting ahead gives huge advantage")

# Difference between quali position and grid position (penalties)
df['Quali_Grid_Diff'] = df['QualiPosition'] - df['GridPosition']

print("\n✓ Quali_Grid_Diff")
print("  What: Difference between where you qualified and where you start")
print("  Example: Qualified P1 but start P5 = got a 4-place penalty")
print("  Negative = penalty, Positive = gained positions")

# How driver compared to teammate in qualifying
df['Quali_vs_Teammate'] = df.groupby(['Year', 'Round', 'Team'])['QualiPosition'].transform(
    lambda x: x - x.mean()
)

print("\n✓ Quali_vs_Teammate")
print("  What: How far ahead/behind your teammate in quali")
print("  Example: -2 = you were 2 positions ahead of teammate")
print("  Why: Beating your teammate = you're performing well")

# ============================================================================
# 5. SEASON PROGRESSION
# ============================================================================
# 🤔 WHY: Early season vs late season matters (car development, championship pressure)
# 📊 WHAT: Track where we are in the season
# 💡 EXAMPLE: Race 1 vs Race 23 - teams might be focusing on next year by R23

print("\n" + "="*70)
print("5️⃣ SEASON PROGRESSION")
print("="*70)
print("Early season vs late season can affect results...")
print("Why? Car development, championship pressure, team strategies")

# Which race number in the season
df['Race_Number_In_Season'] = df['Round']

print("\n✓ Race_Number_In_Season")
print("  What: Is this race 1, 10, or 23 of the season?")
print("  Why: Teams develop cars throughout season")

# Driver's experience (total career races)
df['Driver_Race_Count'] = df.groupby('Driver').cumcount() + 1

print("\n✓ Driver_Race_Count")
print("  What: How many F1 races has this driver done in total?")
print("  Example: VER might have 150+, a rookie might have 5")
print("  Why: Experience matters!")

# ============================================================================
# 6. TEAM STRENGTH INDICATORS
# ============================================================================
# 🤔 WHY: Championship-leading teams have momentum and resources
# 📊 WHAT: Track team's championship points
# 💡 EXAMPLE: Red Bull with 600 points vs Williams with 5 points

print("\n" + "="*70)
print("6️⃣ TEAM STRENGTH INDICATORS")
print("="*70)
print("Championship-leading teams have momentum...")
print("Why? More points = better car = better results (usually)")

# Calculate cumulative team championship points
team_points = df.groupby(['Year', 'Round', 'Team'])['Points'].sum().reset_index()
team_points['Team_Championship_Points'] = team_points.groupby(['Year', 'Team'])['Points'].cumsum()

# Merge back to main dataframe
df = df.merge(
    team_points[['Year', 'Round', 'Team', 'Team_Championship_Points']], 
    on=['Year', 'Round', 'Team'], 
    how='left'
)

print("\n✓ Team_Championship_Points")
print("  What: Team's total points so far this season")
print("  Example: Red Bull = 650, Williams = 5")
print("  Why: More points = stronger team = better results")

# ============================================================================
# 7. HANDLE MISSING VALUES
# ============================================================================
# 🤔 WHY: Some data might be missing (first race, data errors)
# 📊 WHAT: Fill in gaps intelligently
# 💡 EXAMPLE: First race of season has no "last 5 races" data

print("\n" + "="*70)
print("7️⃣ HANDLING MISSING VALUES")
print("="*70)
print("Some features might have gaps (first race, new drivers, etc.)")
print("We need to fill these intelligently...")

missing_before = df.isnull().sum().sum()

# If quali position missing, use grid position
df['QualiPosition'] = df['QualiPosition'].fillna(df['GridPosition'])

# Fill missing lap times with race median
df['AvgLapTime'] = df.groupby(['Year', 'Round'])['AvgLapTime'].transform(
    lambda x: x.fillna(x.median())
)

# For historical features (first races), fill with overall average
historical_cols = [
    'Driver_Avg_Position_Last5', 
    'Driver_Avg_Position_Last3', 
    'Team_Avg_Position_Last5', 
    'Driver_Circuit_Avg', 
    'Team_Circuit_Avg'
]

for col in historical_cols:
    overall_avg = df['FinishPosition'].mean()
    df[col] = df[col].fillna(overall_avg)

# Default finish rate for new drivers
df['Driver_Finish_Rate'] = df['Driver_Finish_Rate'].fillna(0.85)

# Fill other features
df['Quali_Grid_Diff'] = df['Quali_Grid_Diff'].fillna(0)
df['Quali_vs_Teammate'] = df['Quali_vs_Teammate'].fillna(0)
df['Team_Championship_Points'] = df['Team_Championship_Points'].fillna(0)

missing_after = df.isnull().sum().sum()

print(f"\n✓ Missing values before: {missing_before}")
print(f"✓ Missing values after: {missing_after}")
print(f"✓ Filled {missing_before - missing_after} missing values")

# ============================================================================
# 8. SELECT FINAL FEATURES
# ============================================================================

print("\n" + "="*70)
print("8️⃣ SELECTING FEATURES FOR MODEL")
print("="*70)

# These are the features the model will use to make predictions
feature_columns = [
    # Starting position features
    'GridPosition',           # Where they start (most important!)
    'QualiPosition',          # Qualifying position
    
    # Who they are
    'Driver_encoded',         # Which driver
    'Team_encoded',           # Which team
    'Circuit_encoded',        # Which circuit
    
    # Recent form
    'Driver_Avg_Position_Last5',   # Recent performance
    'Driver_Avg_Position_Last3',   # Very recent performance
    'Team_Avg_Position_Last5',     # Team form
    'Driver_Finish_Rate',          # Reliability
    
    # Track-specific
    'Driver_Circuit_Avg',     # Performance at this track
    'Team_Circuit_Avg',       # Team performance at this track
    
    # Qualifying strength
    'Quali_Grid_Diff',        # Penalties
    'Quali_vs_Teammate',      # vs teammate
    
    # Context
    'Race_Number_In_Season',  # Early/late season
    'Driver_Race_Count',      # Experience
    
    # Team strength
    'Team_Championship_Points',  # Championship position
]

# This is what we're trying to predict
target = 'FinishPosition'

print(f"\n✓ Selected {len(feature_columns)} features")
print(f"✓ Target: {target} (what we're predicting)")

print("\n📋 Feature List:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 9. CREATE CLEAN DATASET
# ============================================================================

print("\n" + "="*70)
print("9️⃣ CREATING CLEAN DATASET")
print("="*70)

# Select only the features we need
df_ml = df[feature_columns + [target]].copy()

# Remove any remaining rows with missing data
rows_before = len(df_ml)
df_ml = df_ml.dropna()
rows_after = len(df_ml)

print(f"\n✓ Rows before cleaning: {rows_before}")
print(f"✓ Rows after cleaning: {rows_after}")
print(f"✓ Removed: {rows_before - rows_after} rows")

# ============================================================================
# 10. SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*70)
print("🔟 SAVING PROCESSED DATA")
print("="*70)

# Save the ML-ready dataset
df_ml.to_csv('/content/data/f1_data_ml_ready.csv', index=False)
print("✓ Saved: f1_data_ml_ready.csv (for training)")

# Save the full dataset with all features (for analysis)
df.to_csv('/content/data/f1_data_with_all_features.csv', index=False)
print("✓ Saved: f1_data_with_all_features.csv (full data)")

# ============================================================================
# 11. SUMMARY & STATISTICS
# ============================================================================

print("\n" + "="*70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*70)

print(f"\n📊 Dataset Summary:")
print(f"  Total records: {len(df_ml)}")
print(f"  Features: {len(feature_columns)}")
print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
print(f"  Races: {df.groupby(['Year', 'Round']).ngroups}")

print(f"\n🎯 What We Created:")
print(f"  • {len(feature_columns)} intelligent features")
print(f"  • Historical performance tracking")
print(f"  • Circuit-specific analysis")
print(f"  • Qualifying advantages")
print(f"  • Team strength indicators")

print(f"\n📈 Feature Statistics (sample):")
sample_features = ['GridPosition', 'Driver_Avg_Position_Last5', 'Team_Championship_Points']
for feat in sample_features:
    print(f"  {feat}:")
    print(f"    Min: {df_ml[feat].min():.2f}")
    print(f"    Max: {df_ml[feat].max():.2f}")
    print(f"    Avg: {df_ml[feat].mean():.2f}")

print("\n" + "="*70)
print("🚀 READY FOR STEP 3: MODEL TRAINING!")
print("="*70)
print("\nWhat we'll do next:")
print("  1. Split data into training & testing sets")
print("  2. Train multiple ML models")
print("  3. Evaluate which model works best")
print("  4. Make predictions!")
print("="*70)
```

---

## **🎓 KEY CONCEPTS EXPLAINED:**

### **Why GridPosition is So Important:**
- In F1, starting P1 has ~40% chance of winning
- Starting P10 has ~1% chance of winning
- Overtaking is HARD!

### **Why Recent Form Matters:**
- A driver on a hot streak (P1, P1, P2) vs struggling (P15, P18, P20)
- Momentum and confidence matter

### **Why Circuit-Specific Features:**
- Monaco: tight, technical, slow → favors certain drivers
- Monza: fast, long straights → different drivers shine
- Hamilton dominates Silverstone, Verstappen dominates Austria

---

## **🎯 RUN THIS NOW!**

Copy the entire code above and run it in Colab.

**You should see output like:**
```
✅ FEATURE ENGINEERING COMPLETE!
📊 Dataset Summary:
  Total records: 2000+
  Features: 16
  Years: 2021 - 2025
  
🚀 READY FOR STEP 3: MODEL TRAINING!