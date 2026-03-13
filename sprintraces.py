#sprint data collection
import fastf1
import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

cache_dir = '/content/f1_cache'
os.makedirs(cache_dir, exist_ok=True)
os.makedirs('/content/data', exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def get_sprint_rounds(year):
    """Auto-detect sprint rounds by checking the event schedule."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        sprint_rounds = []
        
        for _, event in schedule.iterrows():
            round_num = event['RoundNumber']
            if pd.isna(round_num):
                continue
            
            # Check if this event has a sprint session
            try:
                event_obj = fastf1.get_event(year, int(round_num))
                # Sprint events have session names containing 'Sprint'
                session_names = [
                    event_obj.get_session_name(i) 
                    for i in range(1, 6)
                ]
                if any('Sprint' in str(s) and 'Qualifying' not in str(s) 
                       for s in session_names):
                    sprint_rounds.append(int(round_num))
                    print(f"  ✓ Round {round_num}: {event['EventName']} — has Sprint")
            except Exception:
                continue
        
        return sprint_rounds
    except Exception as e:
        print(f"  ✗ Could not get schedule for {year}: {e}")
        return []

print("="*70)
print("AUTO-DETECTING SPRINT ROUNDS")
print("="*70)

SPRINT_ROUNDS = {}
for year in [2021, 2022, 2023, 2024, 2025, 2026]:
    print(f"\n── {year} ──")
    rounds = get_sprint_rounds(year)
    SPRINT_ROUNDS[year] = rounds
    print(f"  Found {len(rounds)} sprint rounds: {rounds}")

print(f"\n✓ Total sprint weekends: {sum(len(v) for v in SPRINT_ROUNDS.values())}")

#confirm data we already have 
# See what we already have vs what we need
already_saved = pd.read_csv('/content/data/f1_sprint_data_raw.csv')
print("Already collected:")
for year in sorted(already_saved['Year'].unique()):
    rounds = sorted(already_saved[already_saved['Year']==year]['Round'].unique())
    print(f"  {year}: rounds {rounds}")

print("\nNeed to collect:")
for year, rounds in SPRINT_ROUNDS.items():
    saved_rounds = sorted(already_saved[
        already_saved['Year']==year
    ]['Round'].unique()) if year in already_saved['Year'].values else []
    missing = [r for r in rounds if r not in saved_rounds]
    if missing:
        print(f"  {year}: rounds {missing}")


#collect functions
def collect_sprint_race(year, round_num):
    """
    Collect a single sprint race + sprint shootout data.
    Sessions: 'S' = Sprint race, 'SQ' = Sprint Shootout (qualifying)
    """
    try:
        print(f"  📍 {year} Round {round_num} Sprint...", end=" ", flush=True)

        # Load sprint race
        sprint = fastf1.get_session(year, round_num, 'S')
        sprint.load()

        # Load sprint shootout (called 'Q' in 2021-2022, 'SQ' from 2023)
        try:
            shootout = fastf1.get_session(year, round_num, 'SQ')
            shootout.load()
            shootout_type = 'SQ'
        except Exception:
            try:
                shootout = fastf1.get_session(year, round_num, 'Q')
                shootout.load()
                shootout_type = 'Q'
            except Exception:
                shootout = None
                shootout_type = None

        # Also load the main race qualifying for pace comparison
        try:
            main_quali = fastf1.get_session(year, round_num, 'Q')
            main_quali.load()
        except Exception:
            main_quali = None

        results  = sprint.results
        race_data = []

        for idx, driver in results.iterrows():
            driver_code = driver['Abbreviation']

            features = {
                'Year':            year,
                'Round':           round_num,
                'RaceName':        sprint.event['EventName'],
                'Driver':          driver_code,
                'Team':            driver['TeamName'],
                'SprintPosition':  driver['Position'],
                'SprintGrid':      driver['GridPosition'],
                'SprintPoints':    driver['Points'],
                'Status':          driver['Status'],
                'Laps':            driver.get('Laps', 0),
                'Finished':        1 if str(driver['Status']).strip() == 'Finished' else 0,
            }

            # Sprint Shootout position
            if shootout is not None:
                sq_result = shootout.results[
                    shootout.results['Abbreviation'] == driver_code
                ]
                features['ShootoutPosition'] = (
                    sq_result.iloc[0]['Position'] if not sq_result.empty else np.nan
                )
            else:
                features['ShootoutPosition'] = np.nan

            # Sprint average lap time
            try:
                s_laps = sprint.laps.pick_driver(driver_code)
                valid  = s_laps[s_laps['LapTime'].notna()]
                if len(valid) > 0:
                    times  = valid['LapTime'].dt.total_seconds()
                    median = times.median()
                    clean  = times[times < median * 1.15]
                    features['SprintAvgLapTime'] = clean.mean() if len(clean) > 0 else np.nan
                else:
                    features['SprintAvgLapTime'] = np.nan
            except Exception:
                features['SprintAvgLapTime'] = np.nan

            # Main race qualifying lap time (for pace comparison)
            try:
                if main_quali is not None:
                    q_result = main_quali.results[
                        main_quali.results['Abbreviation'] == driver_code
                    ]
                    if not q_result.empty and 'Q3' in q_result.columns:
                        q3 = q_result.iloc[0]['Q3']
                        features['MainQualiTime'] = (
                            q3.total_seconds() if pd.notna(q3) else np.nan
                        )
                    else:
                        features['MainQualiTime'] = np.nan
                else:
                    features['MainQualiTime'] = np.nan
            except Exception:
                features['MainQualiTime'] = np.nan

            race_data.append(features)

        df = pd.DataFrame(race_data)

        # Sprint vs main race pace difference (per race, relative to field median)
        if df['SprintAvgLapTime'].notna().any() and df['MainQualiTime'].notna().any():
            sprint_median = df['SprintAvgLapTime'].median()
            quali_median  = df['MainQualiTime'].median()
            df['SprintPaceDelta'] = (
                (df['SprintAvgLapTime'] / sprint_median) -
                (df['MainQualiTime'] / quali_median)
            )
        else:
            df['SprintPaceDelta'] = 0.0

        print(f"✓ {len(df)} drivers")
        time.sleep(4)  # Rate limit buffer
        return df

    except Exception as e:
        error = str(e)
        if "RateLimitExceededError" in error:
            print("⏸️ RATE LIMIT")
            raise Exception("RATE_LIMIT")
        print(f"✗ {error[:60]}")
        return None


def collect_all_sprints(sprint_rounds_dict):
    """Collect all sprint weekends, save progress per year."""
    print("="*70)
    print("COLLECTING ALL SPRINT RACE DATA")
    print("="*70)

    all_data   = []
    total_done = 0

    for year, rounds in sprint_rounds_dict.items():
        if not rounds:
            continue

        print(f"\n── {year} ({len(rounds)} sprints) ──")
        year_data = []

        for round_num in rounds:
            # Skip if already collected
            save_path = f'/content/data/sprint_{year}.csv'
            if os.path.exists(save_path):
                existing = pd.read_csv(save_path)
                if round_num in existing['Round'].values:
                    print(f"  ⏭️  Round {round_num} — already collected")
                    year_data.append(
                        existing[existing['Round'] == round_num]
                    )
                    continue

            try:
                df = collect_sprint_race(year, round_num)
                if df is not None:
                    year_data.append(df)
                    total_done += 1

                    # Save year progress
                    year_df = pd.concat(year_data, ignore_index=True)
                    year_df.to_csv(save_path, index=False)

            except Exception as e:
                if "RATE_LIMIT" in str(e):
                    print(f"\n⏸️ Rate limit hit — save progress and wait 1 hour")
                    print(f"   Re-run this cell to resume automatically")
                    break

        if year_data:
            all_data.append(pd.concat(year_data, ignore_index=True))

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv('/content/data/f1_sprint_data_raw.csv', index=False)

        print(f"\n{'='*70}")
        print(f"✅ SPRINT COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total sprint races: {combined.groupby(['Year','Round']).ngroups}")
        print(f"Total records:      {len(combined)}")

        for year in sorted(combined['Year'].unique()):
            y = combined[combined['Year'] == year]
            print(f"  {year}: {y.groupby('Round').ngroups} sprints, {len(y)} records")

        return combined
    return None


# Run collection
sprint_df = collect_all_sprints(SPRINT_ROUNDS)

#feature enginnering for sprint
from sklearn.preprocessing import LabelEncoder

def engineer_sprint_features(df, df_main):
    """
    Build sprint-specific features by combining sprint data
    with historical main race data for context.
    df      = raw sprint data
    df_main = your existing f1_data_with_all_features.csv
    """
    print("Engineering sprint features...")

    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)

    # ── Encode categoricals (reuse same encoding as main model) ──────────────
    le_driver  = LabelEncoder()
    le_team    = LabelEncoder()
    le_circuit = LabelEncoder()

    # Fit on ALL drivers/teams across both datasets so encodings are consistent
    all_drivers  = pd.concat([df['Driver'],  df_main['Driver']]).unique()
    all_teams    = pd.concat([df['Team'],    df_main['Team']]).unique()
    all_circuits = pd.concat([df['RaceName'],df_main['RaceName']]).unique()

    le_driver.fit(all_drivers)
    le_team.fit(all_teams)
    le_circuit.fit(all_circuits)

    df['Driver_encoded']  = le_driver.transform(df['Driver'])
    df['Team_encoded']    = le_team.transform(df['Team'])
    df['Circuit_encoded'] = le_circuit.transform(df['RaceName'])

    # ── Rolling sprint performance (last 3 sprints — only ~3-6 per season) ──
    df['Driver_Sprint_Avg_Last3'] = df.groupby('Driver')['SprintPosition'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['Team_Sprint_Avg_Last3'] = df.groupby('Team')['SprintPosition'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # ── Driver's main race form (from main dataset) ──────────────────────────
    main_form = df_main.groupby('Driver').apply(
        lambda x: x.tail(5)['FinishPosition'].mean()
    ).reset_index()
    main_form.columns = ['Driver', 'MainRace_Avg_Last5']
    df = df.merge(main_form, on='Driver', how='left')

    # ── Team championship points (from main dataset latest round) ────────────
    team_pts = df_main.groupby('Team')['Team_Championship_Points'].max().reset_index()
    df = df.merge(team_pts, on='Team', how='left')

    # ── Fill missing values ───────────────────────────────────────────────────
    sprint_avg = df['SprintPosition'].mean()

    df['ShootoutPosition']      = df['ShootoutPosition'].fillna(df['SprintGrid'])
    df['SprintAvgLapTime']      = df.groupby(['Year','Round'])['SprintAvgLapTime'].transform(
        lambda x: x.fillna(x.median())
    )
    df['SprintPaceDelta']       = df['SprintPaceDelta'].fillna(0.0)
    df['Driver_Sprint_Avg_Last3'] = df['Driver_Sprint_Avg_Last3'].fillna(sprint_avg)
    df['Team_Sprint_Avg_Last3']   = df['Team_Sprint_Avg_Last3'].fillna(sprint_avg)
    df['MainRace_Avg_Last5']    = df['MainRace_Avg_Last5'].fillna(sprint_avg)
    df['Team_Championship_Points'] = df['Team_Championship_Points'].fillna(0)

    print(f"✓ Features engineered: {len(df)} records")
    return df


# Load main race data for context features
df_main = pd.read_csv('/content/data/f1_data_with_all_features.csv')

sprint_featured = engineer_sprint_features(sprint_df, df_main)
sprint_featured.to_csv('/content/data/f1_sprint_with_features.csv', index=False)
print("✓ Saved: f1_sprint_with_features.csv")

#train sprintmodel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

SPRINT_FEATURES = [
    'ShootoutPosition',          # Sprint shootout grid (most important)
    'SprintGrid',                # Starting grid (accounts for penalties)
    'Driver_encoded',
    'Team_encoded',
    'Circuit_encoded',
    'Driver_Sprint_Avg_Last3',   # Historical sprint form
    'Team_Sprint_Avg_Last3',     # Team sprint form
    'MainRace_Avg_Last5',        # Main race pace proxy
    'SprintPaceDelta',           # Sprint vs main race pace gap
    'Team_Championship_Points',
]

df_sprint_ml = sprint_featured[SPRINT_FEATURES + ['SprintPosition']].dropna()

print(f"Sprint ML dataset: {len(df_sprint_ml)} records")
print(f"Features: {len(SPRINT_FEATURES)}")

X = df_sprint_ml[SPRINT_FEATURES]
y = df_sprint_ml['SprintPosition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sprint_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

sprint_model.fit(X_train, y_train)

y_pred = sprint_model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"SPRINT MODEL RESULTS")
print(f"{'='*50}")
print(f"  Test MAE : {mae:.3f} positions")
print(f"  Test R²  : {r2:.3f}")

# Feature importance
print(f"\nTop features:")
imp = pd.DataFrame({
    'Feature': SPRINT_FEATURES,
    'Importance': sprint_model.feature_importances_
}).sort_values('Importance', ascending=False)
for _, row in imp.iterrows():
    print(f"  {row['Feature']:<35} {row['Importance']:.1%}")

#save model
import os

joblib.dump(sprint_model, '/content/data/f1_sprint_model.pkl')
print("✓ Saved: f1_sprint_model.pkl")

sprint_info = {
    'sprint_feature_columns': SPRINT_FEATURES,
    'test_mae': float(mae),
    'test_r2':  float(r2),
    'model':    'Gradient Boosting',
}

# Merge into existing model_info.json
with open('/content/data/model_info.json') as f:
    model_info = json.load(f)

model_info['sprint_model'] = sprint_info

with open('/content/data/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Updated: model_info.json")

from google.colab import files
for path in [
    '/content/data/f1_sprint_model.pkl',
    '/content/data/f1_sprint_with_features.csv',
    '/content/data/model_info.json',
]:
    if os.path.exists(path):
        files.download(path)
        print(f"✓ Downloaded: {os.path.basename(path)}")

