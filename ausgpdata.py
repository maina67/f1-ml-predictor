!pip install fastf1 --upgrade -q
import fastf1
print(fastf1.__version__)  # Should be 3.8.1+

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

print("✓ Setup complete")

def collect_race_2026_australia():
    """
    Collect 2026 Australian GP data.
    Round 1 of the 2026 season.
    """
    print("="*70)
    print("COLLECTING 2026 AUSTRALIAN GP")
    print("="*70)

    try:
        print("\n📍 Loading Race session...")
        race = fastf1.get_session(2026, 1, 'R')
        race.load()
        print("✓ Race loaded")

        print("📍 Loading Qualifying session...")
        quali = fastf1.get_session(2026, 1, 'Q')
        quali.load()
        print("✓ Qualifying loaded")

    except Exception as e:
        print(f"✗ Session load error: {e}")
        return None

    results = race.results
    race_data = []

    for idx, driver in results.iterrows():
        driver_code = driver['Abbreviation']

        features = {
            'Year': 2026,
            'Round': 1,
            'RaceName': race.event['EventName'],
            'Driver': driver_code,
            'Team': driver['TeamName'],
            'FinishPosition': driver['Position'],
            'GridPosition': driver['GridPosition'],
            'Points': driver['Points'],
            'Status': driver['Status'],
            'Laps': driver.get('Laps', 0),
        }

        # Qualifying position
        try:
            quali_result = quali.results[quali.results['Abbreviation'] == driver_code]
            if not quali_result.empty:
                features['QualiPosition'] = quali_result.iloc[0]['Position']
            else:
                features['QualiPosition'] = np.nan
        except:
            features['QualiPosition'] = np.nan

        # Average lap time
        try:
            driver_laps = race.laps.pick_driver(driver_code)
            if len(driver_laps) > 0:
                valid_laps = driver_laps[driver_laps['LapTime'].notna()]
                if len(valid_laps) > 0:
                    lap_times = valid_laps['LapTime'].dt.total_seconds()
                    median_time = lap_times.median()
                    clean_laps = lap_times[lap_times < median_time * 1.2]
                    features['AvgLapTime'] = clean_laps.mean() if len(clean_laps) > 0 else np.nan
                else:
                    features['AvgLapTime'] = np.nan
            else:
                features['AvgLapTime'] = np.nan
        except:
            features['AvgLapTime'] = np.nan

        features['Finished'] = 1 if str(driver['Status']).strip() == 'Finished' else 0
        race_data.append(features)

    df = pd.DataFrame(race_data)
    print(f"\n✓ Collected {len(df)} drivers")
    print(df[['Driver', 'Team', 'GridPosition', 'QualiPosition', 'FinishPosition', 'Points']].to_string())
    return df


australia_df = collect_race_2026_australia()

def update_dataset_with_2026(australia_df):
    """
    Load existing dataset, append 2026 Australia, save updated file.
    """
    print("="*70)
    print("UPDATING DATASET WITH 2026 DATA")
    print("="*70)

    if australia_df is None:
        print("✗ No Australia data to add")
        return None

    # Load existing combined dataset
    try:
        existing_df = pd.read_csv('/content/data/f1_data_combined_all_years.csv')
        print(f"✓ Loaded existing data: {len(existing_df)} records")
        print(f"  Years: {sorted(existing_df['Year'].unique())}")
    except Exception as e:
        print(f"✗ Could not load existing data: {e}")
        print("  Make sure your combined CSV is uploaded to /content/data/")
        return None

    # Check if 2026 Round 1 already exists
    already_exists = (
        (existing_df['Year'] == 2026) & 
        (existing_df['Round'] == 1)
    ).any()

    if already_exists:
        print("⚠️  2026 Round 1 already in dataset - removing old version")
        existing_df = existing_df[~((existing_df['Year'] == 2026) & (existing_df['Round'] == 1))]

    # Append new data
    updated_df = pd.concat([existing_df, australia_df], ignore_index=True)
    updated_df = updated_df.sort_values(['Year', 'Round']).reset_index(drop=True)

    # Save
    updated_df.to_csv('/content/data/f1_data_combined_all_years.csv', index=False)

    print(f"\n✓ Updated dataset saved")
    print(f"  Total records: {len(updated_df)}")
    print(f"\n📊 Breakdown by year:")
    for year in sorted(updated_df['Year'].unique()):
        year_data = updated_df[updated_df['Year'] == year]
        races = year_data['Round'].nunique()
        print(f"  {year}: {races} races, {len(year_data)} records")

    return updated_df


updated_df = update_dataset_with_2026(australia_df)

from google.colab import files

if updated_df is not None:
    files.download('/content/data/f1_data_combined_all_years.csv')
    print("✓ Downloaded updated dataset - save this to your Windsurf project!")