"""
F1 Smart Data Collector
- Automatically resumes from where it stopped
- Handles rate limits gracefully
- Saves progress continuously
- Never loses data
"""

import fastf1
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Edit these as needed
# ============================================================================

CONFIG = {
    'years': [2021, 2022, 2023, 2024],  # Which years to collect
    'cache_dir': './f1_cache',           # Where to cache data
    'output_dir': './data',              # Where to save CSVs
    'delay_between_races': 3,            # Seconds between races
    'delay_between_seasons': 300,        # Seconds between seasons (5 min)
}

# ============================================================================
# SETUP
# ============================================================================

os.makedirs(CONFIG['cache_dir'], exist_ok=True)
os.makedirs(CONFIG['output_dir'], exist_ok=True)
fastf1.Cache.enable_cache(CONFIG['cache_dir'])

# ============================================================================
# COLLECTION FUNCTIONS
# ============================================================================

def collect_race_data(year, race_round):
    """Collect data for a single race with error handling."""
    try:
        print(f"  📍 {year} R{race_round:02d}...", end=" ", flush=True)

        race = fastf1.get_session(year, race_round, 'R')
        race.load()

        quali = fastf1.get_session(year, race_round, 'Q')
        quali.load()

        results = race.results
        race_data = []

        for idx, driver in results.iterrows():
            driver_code = driver['Abbreviation']

            features = {
                'Year': year,
                'Round': race_round,
                'RaceName': race.event['EventName'],
                'Driver': driver_code,
                'Team': driver['TeamName'],
                'FinishPosition': driver['Position'],
                'GridPosition': driver['GridPosition'],
                'Points': driver['Points'],
                'Status': driver['Status'],
                'Laps': driver.get('Laps', 0),
            }

            quali_result = quali.results[quali.results['Abbreviation'] == driver_code]
            if not quali_result.empty:
                features['QualiPosition'] = quali_result.iloc[0]['Position']
            else:
                features['QualiPosition'] = np.nan

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

            features['Finished'] = 1 if driver['Status'] == 'Finished' else 0
            race_data.append(features)

        df = pd.DataFrame(race_data)
        print(f"✓ {len(df)} drivers")

        time.sleep(CONFIG['delay_between_races'])
        return df

    except Exception as e:
        error_msg = str(e)
        if "RateLimitExceededError" in error_msg:
            print(f"⏸️ RATE LIMIT")
            raise Exception("RATE_LIMIT")
        else:
            print(f"✗ {error_msg[:40]}")
            return None


def load_existing_data(year):
    """Load existing data for a year if it exists."""
    filepath = os.path.join(CONFIG['output_dir'], f'f1_data_{year}.csv')

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        collected_rounds = set(df['Round'].unique())
        print(f"  📂 Found existing: {len(collected_rounds)} races")
        return df, collected_rounds
    else:
        print(f"  📂 No existing data")
        return None, set()


def save_year_data(year, df):
    """Save data for a specific year."""
    filepath = os.path.join(CONFIG['output_dir'], f'f1_data_{year}.csv')
    df.to_csv(filepath, index=False)
    print(f"  💾 Saved: {filepath}")


def collect_year_smart(year):
    """
    Smart collection for a year:
    - Loads existing data
    - Only collects missing rounds
    - Saves incrementally
    """
    print(f"\n{'='*70}")
    print(f"YEAR {year}")
    print(f"{'='*70}")

    # Load what we already have
    existing_df, collected_rounds = load_existing_data(year)
    all_data = [existing_df] if existing_df is not None else []

    # Try to collect rounds 1-25
    new_races = 0

    for round_num in range(1, 26):
        # Skip if already collected
        if round_num in collected_rounds:
            print(f"  ⏭️  {year} R{round_num:02d} - Already have")
            continue

        try:
            race_df = collect_race_data(year, round_num)

            if race_df is not None and not race_df.empty:
                all_data.append(race_df)
                new_races += 1

                # Save every 5 new races
                if new_races % 5 == 0:
                    combined = pd.concat(all_data, ignore_index=True)
                    save_year_data(year, combined)
                    print(f"  💾 Progress: +{new_races} races")
            else:
                # No more races this year
                if new_races > 0 or len(collected_rounds) > 5:
                    break

        except Exception as e:
            if "RATE_LIMIT" in str(e):
                # Save what we have and stop
                if all_data:
                    combined = pd.concat(all_data, ignore_index=True)
                    save_year_data(year, combined)
                raise Exception("RATE_LIMIT")
            else:
                continue

    # Final save
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        save_year_data(year, combined)

        total_races = combined['Round'].nunique()
        print(f"\n✅ {year}: {total_races} races, {len(combined)} records")
        print(f"   New: +{new_races} races")
        return combined
    else:
        print(f"\n⚠️ {year}: No data")
        return None


def create_combined_dataset():
    """Combine all year files into one master dataset."""
    all_data = []

    for year in CONFIG['years']:
        filepath = os.path.join(CONFIG['output_dir'], f'f1_data_{year}.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            all_data.append(df)
            print(f"✓ Loaded {year}: {len(df)} records")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(CONFIG['output_dir'], 'f1_data_combined.csv')
        combined.to_csv(output_path, index=False)

        print(f"\n{'='*70}")
        print(f"COMBINED DATASET")
        print(f"{'='*70}")
        print(f"Total records: {len(combined)}")
        print(f"Years: {sorted(combined['Year'].unique())}")
        print(f"\nBreakdown:")
        for year in sorted(combined['Year'].unique()):
            year_data = combined[combined['Year'] == year]
            races = year_data['Round'].nunique()
            records = len(year_data)
            print(f"  {year}: {races:2d} races, {records:3d} records")
        print(f"\nSaved to: {output_path}")
        print(f"{'='*70}")

        return combined
    else:
        print("❌ No data found")
        return None


# ============================================================================
# MAIN COLLECTION LOGIC
# ============================================================================

def run_collection():
    """Main collection function - handles everything automatically."""
    print("\n" + "="*70)
    print("F1 SMART DATA COLLECTOR")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Years to collect: {CONFIG['years']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print("="*70)

    rate_limit_hit = False

    for i, year in enumerate(CONFIG['years']):
        try:
            collect_year_smart(year)

            # Wait between seasons (except after last one)
            if i < len(CONFIG['years']) - 1:
                wait_time = CONFIG['delay_between_seasons']
                print(f"\n⏸️ Waiting {wait_time}s before next season...")
                time.sleep(wait_time)

        except Exception as e:
            if "RATE_LIMIT" in str(e):
                print(f"\n{'='*70}")
                print(f"⏸️ RATE LIMIT HIT")
                print(f"{'='*70}")
                print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                print(f"Stopped at: {year}")
                print(f"All progress has been saved!")
                print(f"\n💡 Wait 1 hour, then run this script again.")
                print(f"   It will automatically resume from where it stopped.")
                print(f"{'='*70}")
                rate_limit_hit = True
                break

    # Create combined dataset
    if not rate_limit_hit:
        print(f"\n{'='*70}")
        print("Creating combined dataset...")
        print(f"{'='*70}")
        combined = create_combined_dataset()

        if combined is not None:
            print(f"\n✅ COLLECTION COMPLETE!")
    else:
        print(f"\nℹ️ Partial collection saved. Run again after 1 hour.")

    print(f"\n{'='*70}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    run_collection()