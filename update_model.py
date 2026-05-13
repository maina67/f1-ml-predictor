# ============================================================================
# F1 MODEL UPDATER
# Run after each race weekend:  python update_model.py
# ============================================================================

import subprocess
import sys
import os
import argparse
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────

MODELS_DIR    = "data"
RACE_DATA_DIR = "race data"         # local data folder
GITHUB_BRANCH  = "main"
COMMIT_PREFIX  = "🏁 Auto-update"

# ── ARGUMENT PARSER ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="F1 Model Updater")
parser.add_argument("--year",  type=int, default=datetime.now().year,
                    help="Season year (default: current year)")
parser.add_argument("--round", type=int, required=True,
                    help="Race round number just completed")
parser.add_argument("--name",  type=str, default="",
                    help="Race name (e.g. 'Monaco Grand Prix')")
parser.add_argument("--sprint", action="store_true",
                    help="Include sprint race update for this round")
parser.add_argument("--skip-collect",  action="store_true",
                    help="Skip data collection (use existing CSVs)")
parser.add_argument("--skip-retrain",  action="store_true",
                    help="Skip model retraining (just push existing files)")
parser.add_argument("--skip-push",     action="store_true",
                    help="Skip GitHub push (local update only)")
args = parser.parse_args()

# ── HELPERS ──────────────────────────────────────────────────────────────────

def run(cmd, desc):
    print(f"\n{'─'*60}")
    print(f"▶  {desc}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {cmd}")
        sys.exit(1)
    return result


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── MAIN ─────────────────────────────────────────────────────────────────────

print(f"""
╔══════════════════════════════════════════════╗
║        F1 MODEL UPDATER                      ║
╚══════════════════════════════════════════════╝
  Year  : {args.year}
  Round : {args.round}  {args.name}
  Sprint: {'Yes' if args.sprint else 'No'}
  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")

# ── STEP 1: COLLECT NEW RACE DATA ────────────────────────────────────────────

if not args.skip_collect:
    section("STEP 1: Collecting race data")

    collect_script = f"""
import fastf1
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('./f1_cache')
os.makedirs('f1_cache', exist_ok=True)

year  = {args.year}
round = {args.round}

def collect_race(year, round_num):
    print(f"Loading {{year}} Round {{round_num}} race...")
    race  = fastf1.get_session(year, round_num, 'R')
    race.load()
    quali = fastf1.get_session(year, round_num, 'Q')
    quali.load()

    results   = race.results
    race_data = []

    for idx, driver in results.iterrows():
        driver_code = driver['Abbreviation']
        features = {{
            'Year': year, 'Round': round_num,
            'RaceName': race.event['EventName'],
            'Driver': driver_code, 'Team': driver['TeamName'],
            'FinishPosition': driver['Position'],
            'GridPosition': driver['GridPosition'],
            'Points': driver['Points'], 'Status': driver['Status'],
            'Laps': driver.get('Laps', 0),
        }}
        qr = quali.results[quali.results['Abbreviation'] == driver_code]
        features['QualiPosition'] = qr.iloc[0]['Position'] if not qr.empty else np.nan

        laps = race.laps.pick_driver(driver_code)
        if len(laps) > 0:
            valid = laps[laps['LapTime'].notna()]
            if len(valid) > 0:
                times  = valid['LapTime'].dt.total_seconds()
                median = times.median()
                clean  = times[times < median * 1.2]
                features['AvgLapTime'] = clean.mean() if len(clean) > 0 else np.nan
            else:
                features['AvgLapTime'] = np.nan
        else:
            features['AvgLapTime'] = np.nan

        features['Finished'] = 1 if str(driver['Status']).strip() == 'Finished' else 0
        race_data.append(features)

    return pd.DataFrame(race_data)

# Collect the race
new_df = collect_race(year, round)
print(f"Collected {{len(new_df)}} driver records")

# Load existing combined data
combined_path = '{COLAB_DATA_DIR}/f1_data_combined_all_years.csv'
existing      = pd.read_csv(combined_path)

# Remove existing entry for this round if present (re-run safety)
existing = existing[~((existing['Year'] == year) & (existing['Round'] == round))]

# Append and save
updated = pd.concat([existing, new_df], ignore_index=True)
updated = updated.sort_values(['Year', 'Round']).reset_index(drop=True)
updated.to_csv(combined_path, index=False)
print(f"Updated combined CSV: {{len(updated)}} total records")
"""

    # Write and run the collection script
    with open("_collect_tmp.py", "w") as f:
        f.write(collect_script)

    run("python _collect_tmp.py", f"Collecting {args.year} Round {args.round} data")
    os.remove("_collect_tmp.py")

    # Sprint collection if needed
    if args.sprint:
        sprint_collect = f"""
import fastf1
import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('./f1_cache')

year  = {args.year}
round = {args.round}

print(f"Loading sprint data for {{year}} Round {{round}}...")

race = fastf1.get_session(year, round, 'S')
race.load()

try:
    shootout = fastf1.get_session(year, round, 'SQ')
    shootout.load()
except:
    try:
        shootout = fastf1.get_session(year, round, 'Q')
        shootout.load()
    except:
        shootout = None

try:
    main_quali = fastf1.get_session(year, round, 'Q')
    main_quali.load()
except:
    main_quali = None

results   = race.results
race_data = []

for idx, driver in results.iterrows():
    driver_code = driver['Abbreviation']
    features = {{
        'Year': year, 'Round': round,
        'RaceName': race.event['EventName'],
        'Driver': driver_code, 'Team': driver['TeamName'],
        'SprintPosition': driver['Position'],
        'SprintGrid': driver['GridPosition'],
        'SprintPoints': driver['Points'], 'Status': driver['Status'],
        'Laps': driver.get('Laps', 0),
        'Finished': 1 if str(driver['Status']).strip() == 'Finished' else 0,
    }}

    if shootout is not None:
        sq = shootout.results[shootout.results['Abbreviation'] == driver_code]
        features['ShootoutPosition'] = sq.iloc[0]['Position'] if not sq.empty else np.nan
    else:
        features['ShootoutPosition'] = np.nan

    try:
        laps  = race.laps.pick_driver(driver_code)
        valid = laps[laps['LapTime'].notna()]
        if len(valid) > 0:
            times  = valid['LapTime'].dt.total_seconds()
            median = times.median()
            clean  = times[times < median * 1.15]
            features['SprintAvgLapTime'] = clean.mean() if len(clean) > 0 else np.nan
        else:
            features['SprintAvgLapTime'] = np.nan
    except:
        features['SprintAvgLapTime'] = np.nan

    features['MainQualiTime']  = np.nan
    features['SprintPaceDelta'] = 0.0
    race_data.append(features)

new_sprint = pd.DataFrame(race_data)

sprint_path = '{COLAB_DATA_DIR}/f1_sprint_data_raw.csv'
existing    = pd.read_csv(sprint_path)
existing    = existing[~((existing['Year'] == year) & (existing['Round'] == round))]
updated     = pd.concat([existing, new_sprint], ignore_index=True)
updated     = updated.sort_values(['Year', 'Round']).reset_index(drop=True)
updated.to_csv(sprint_path, index=False)
print(f"Sprint data updated: {{len(updated)}} total records")
"""
        with open("_sprint_collect_tmp.py", "w") as f:
            f.write(sprint_collect)
        run("python _sprint_collect_tmp.py", "Collecting sprint data")
        os.remove("_sprint_collect_tmp.py")

else:
    print("\n⏭️  Skipping data collection (--skip-collect)")

# ── STEP 2: RE-RUN FEATURE ENGINEERING ───────────────────────────────────────

if not args.skip_retrain:
    section("STEP 2: Feature engineering")

    feature_script = f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("Running feature engineering...")

df = pd.read_csv('{COLAB_DATA_DIR}/f1_data_combined_all_years.csv')
df = df.sort_values(['Year', 'Round']).reset_index(drop=True)

le_driver  = LabelEncoder()
le_team    = LabelEncoder()
le_circuit = LabelEncoder()

df['Driver_encoded']  = le_driver.fit_transform(df['Driver'])
df['Team_encoded']    = le_team.fit_transform(df['Team'])
df['Circuit_encoded'] = le_circuit.fit_transform(df['RaceName'])

df['Driver_Avg_Position_Last5'] = df.groupby('Driver')['FinishPosition'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
df['Driver_Avg_Position_Last3'] = df.groupby('Driver')['FinishPosition'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
df['Team_Avg_Position_Last5'] = df.groupby('Team')['FinishPosition'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
df['Driver_Finish_Rate'] = df.groupby('Driver')['Finished'].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
df['Driver_Circuit_Avg'] = df.groupby(['Driver','RaceName'])['FinishPosition'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean())
df['Team_Circuit_Avg'] = df.groupby(['Team','RaceName'])['FinishPosition'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean())

df['QualiPosition']     = df['QualiPosition'].fillna(df['GridPosition'])
df['Quali_Grid_Diff']   = df['QualiPosition'] - df['GridPosition']
df['Quali_vs_Teammate'] = df.groupby(['Year','Round','Team'])['QualiPosition'].transform(
    lambda x: x - x.mean())
df['Race_Number_In_Season'] = df['Round']
df['Driver_Race_Count']     = df.groupby('Driver').cumcount() + 1

team_pts = df.groupby(['Year','Round','Team'])['Points'].sum().reset_index()
team_pts['Team_Championship_Points'] = team_pts.groupby(['Year','Team'])['Points'].cumsum()
df = df.merge(team_pts[['Year','Round','Team','Team_Championship_Points']],
              on=['Year','Round','Team'], how='left')

overall_avg = df['FinishPosition'].mean()
for col in ['Driver_Avg_Position_Last5','Driver_Avg_Position_Last3',
            'Team_Avg_Position_Last5','Driver_Circuit_Avg','Team_Circuit_Avg']:
    df[col] = df[col].fillna(overall_avg)

df['Driver_Finish_Rate']       = df['Driver_Finish_Rate'].fillna(0.85)
df['Quali_Grid_Diff']          = df['Quali_Grid_Diff'].fillna(0)
df['Quali_vs_Teammate']        = df['Quali_vs_Teammate'].fillna(0)
df['Team_Championship_Points'] = df['Team_Championship_Points'].fillna(0)
df['AvgLapTime'] = df.groupby(['Year','Round'])['AvgLapTime'].transform(
    lambda x: x.fillna(x.median()))

df.to_csv('{COLAB_DATA_DIR}/f1_data_with_all_features.csv', index=False)
print(f"Features saved: {{len(df)}} records")

FEATURES = [
    'GridPosition','QualiPosition','Driver_encoded','Team_encoded','Circuit_encoded',
    'Driver_Avg_Position_Last5','Driver_Avg_Position_Last3','Team_Avg_Position_Last5',
    'Driver_Finish_Rate','Driver_Circuit_Avg','Team_Circuit_Avg','Quali_Grid_Diff',
    'Quali_vs_Teammate','Race_Number_In_Season','Driver_Race_Count','Team_Championship_Points',
]
df_ml = df[FEATURES + ['FinishPosition']].dropna()
df_ml.to_csv('{COLAB_DATA_DIR}/f1_data_ml_ready.csv', index=False)
print(f"ML-ready CSV saved: {{len(df_ml)}} records")
"""

    with open("_features_tmp.py", "w") as f:
        f.write(feature_script)
    run("python _features_tmp.py", "Running feature engineering")
    os.remove("_features_tmp.py")

    # ── STEP 3: RETRAIN MODEL ─────────────────────────────────────────────

    section("STEP 3: Retraining models")

    retrain_script = f"""
import pandas as pd
import numpy as np
import joblib, json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Retraining main race model...")

df = pd.read_csv('{COLAB_DATA_DIR}/f1_data_ml_ready.csv')

FEATURES = [
    'GridPosition','QualiPosition','Driver_encoded','Team_encoded','Circuit_encoded',
    'Driver_Avg_Position_Last5','Driver_Avg_Position_Last3','Team_Avg_Position_Last5',
    'Driver_Finish_Rate','Driver_Circuit_Avg','Team_Circuit_Avg','Quali_Grid_Diff',
    'Quali_vs_Teammate','Race_Number_In_Season','Driver_Race_Count','Team_Championship_Points',
]

X = df[FEATURES]
y = df['FinishPosition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2  = r2_score(y_test, model.predict(X_test))

print(f"  MAE: {{mae:.3f}} positions")
print(f"  R²:  {{r2:.3f}}")

joblib.dump(model, '{COLAB_DATA_DIR}/f1_best_model.pkl')
print("  Saved: f1_best_model.pkl")

# Update model_info.json
with open('{COLAB_DATA_DIR}/model_info.json') as f:
    info = json.load(f)

info['feature_columns'] = FEATURES
info['gbm_model'] = {{'name':'Gradient Boosting','file':'f1_best_model.pkl',
                      'test_mae': float(mae), 'test_r2': float(r2)}}
info['last_updated'] = '{datetime.now().strftime("%Y-%m-%d %H:%M")}'
info['last_round']   = {args.round}
info['last_year']    = {args.year}

with open('{COLAB_DATA_DIR}/model_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print("  Updated: model_info.json")
print(f"Training complete — MAE: {{mae:.3f}}")
"""

    with open("_retrain_tmp.py", "w") as f:
        f.write(retrain_script)
    run("python _retrain_tmp.py", "Retraining GBM model")
    os.remove("_retrain_tmp.py")

    # Sprint retrain if needed
    if args.sprint:
        section("STEP 3b: Retraining sprint model")

        sprint_retrain = f"""
import pandas as pd
import numpy as np
import joblib, json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Retraining sprint model...")

df_sprint = pd.read_csv('{COLAB_DATA_DIR}/f1_sprint_data_raw.csv')
df_main   = pd.read_csv('{COLAB_DATA_DIR}/f1_data_with_all_features.csv')

df_sprint = df_sprint.sort_values(['Year','Round']).reset_index(drop=True)

le_driver  = LabelEncoder()
le_team    = LabelEncoder()
le_circuit = LabelEncoder()

all_drivers  = pd.concat([df_sprint['Driver'],  df_main['Driver']]).unique()
all_teams    = pd.concat([df_sprint['Team'],    df_main['Team']]).unique()
all_circuits = pd.concat([df_sprint['RaceName'],df_main['RaceName']]).unique()

le_driver.fit(all_drivers)
le_team.fit(all_teams)
le_circuit.fit(all_circuits)

df_sprint['Driver_encoded']  = le_driver.transform(df_sprint['Driver'])
df_sprint['Team_encoded']    = le_team.transform(df_sprint['Team'])
df_sprint['Circuit_encoded'] = le_circuit.transform(df_sprint['RaceName'])

df_sprint['Driver_Sprint_Avg_Last3'] = df_sprint.groupby('Driver')['SprintPosition'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
df_sprint['Team_Sprint_Avg_Last3'] = df_sprint.groupby('Team')['SprintPosition'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())

main_form = df_main.groupby('Driver').apply(
    lambda x: x.tail(5)['FinishPosition'].mean()).reset_index()
main_form.columns = ['Driver','MainRace_Avg_Last5']
df_sprint = df_sprint.merge(main_form, on='Driver', how='left')

team_pts = df_main.groupby('Team')['Team_Championship_Points'].max().reset_index()
df_sprint = df_sprint.merge(team_pts, on='Team', how='left')

sprint_avg = df_sprint['SprintPosition'].mean()
df_sprint['ShootoutPosition']        = df_sprint['ShootoutPosition'].fillna(df_sprint['SprintGrid'])
df_sprint['Driver_Sprint_Avg_Last3'] = df_sprint['Driver_Sprint_Avg_Last3'].fillna(sprint_avg)
df_sprint['Team_Sprint_Avg_Last3']   = df_sprint['Team_Sprint_Avg_Last3'].fillna(sprint_avg)
df_sprint['MainRace_Avg_Last5']      = df_sprint['MainRace_Avg_Last5'].fillna(sprint_avg)
df_sprint['Team_Championship_Points']= df_sprint['Team_Championship_Points'].fillna(0)
df_sprint['SprintPaceDelta']         = df_sprint.get('SprintPaceDelta', pd.Series(0.0, index=df_sprint.index)).fillna(0.0)

df_sprint.to_csv('{COLAB_DATA_DIR}/f1_sprint_with_features.csv', index=False)

SPRINT_FEATURES = [
    'ShootoutPosition','SprintGrid','Driver_encoded','Team_encoded','Circuit_encoded',
    'Driver_Sprint_Avg_Last3','Team_Sprint_Avg_Last3','MainRace_Avg_Last5',
    'SprintPaceDelta','Team_Championship_Points',
]

df_ml = df_sprint[SPRINT_FEATURES + ['SprintPosition']].dropna()
X = df_ml[SPRINT_FEATURES]
y = df_ml['SprintPosition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2  = r2_score(y_test, model.predict(X_test))

joblib.dump(model, '{COLAB_DATA_DIR}/f1_sprint_model.pkl')
print(f"Sprint model saved — MAE: {{mae:.3f}}")

with open('{COLAB_DATA_DIR}/model_info.json') as f:
    info = json.load(f)
info['sprint_model'] = {{'sprint_feature_columns': SPRINT_FEATURES,
                          'test_mae': float(mae), 'test_r2': float(r2)}}
with open('{COLAB_DATA_DIR}/model_info.json', 'w') as f:
    json.dump(info, f, indent=2)
"""

        with open("_sprint_retrain_tmp.py", "w") as f:
            f.write(sprint_retrain)
        run("python _sprint_retrain_tmp.py", "Retraining sprint model")
        os.remove("_sprint_retrain_tmp.py")

else:
    print("\n⏭️  Skipping retrain (--skip-retrain)")

# ── STEP 4: PUSH TO GITHUB ────────────────────────────────────────────────────

if not args.skip_push:
    section("STEP 4: Pushing to GitHub")

    race_label = args.name if args.name else f"Round {args.round}"
    commit_msg = f"{COMMIT_PREFIX}: {args.year} {race_label}"

    run("git add data/", "Staging data files")
    run(f'git commit -m "{commit_msg}"', "Committing changes")
    run(f"git push origin {GITHUB_BRANCH}", "Pushing to GitHub")

    print(f"""
✅ DONE! Changes pushed to GitHub.
   Streamlit Community Cloud will redeploy automatically in ~1 minute.
   Commit: {commit_msg}
""")
else:
    print("\n⏭️  Skipping push (--skip-push)")

# ── SUMMARY ──────────────────────────────────────────────────────────────────

section("COMPLETE")
print(f"""
  Year   : {args.year}
  Round  : {args.round}  {args.name}
  Sprint : {'Yes' if args.sprint else 'No'}
  Pushed : {'Yes' if not args.skip_push else 'No (--skip-push)'}

  Next steps:
  • Check your Streamlit app URL — it will redeploy automatically
  • Verify predictions look correct for the next race
""")