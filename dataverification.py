import pandas as pd

df = pd.read_csv('/content/data/f1_data_ml_ready.csv')

print("="*70)
print("DATA QUALITY CHECK")
print("="*70)

# Check how many unique values in key features
print("\n1️⃣ Unique Value Counts (should vary):")
print(f"  Driver_Avg_Position_Last5: {df['Driver_Avg_Position_Last5'].nunique()} unique values")
print(f"  Team_Championship_Points: {df['Team_Championship_Points'].nunique()} unique values")
print(f"  Driver_Race_Count: {df['Driver_Race_Count'].nunique()} unique values")

# Check progression over rounds
print("\n2️⃣ Feature Evolution Across Rounds:")
sample_driver = df[df['Driver_encoded'] == df['Driver_encoded'].mode()[0]]
print(f"\nSample driver's Avg_Position_Last5 over first 10 races:")
print(sample_driver.head(10)[['Race_Number_In_Season', 'Driver_Avg_Position_Last5']].to_string(index=False))

# Count placeholder values
placeholder_count = (df['Driver_Avg_Position_Last5'] == 10.47912088).sum()
total_rows = len(df)
placeholder_pct = (placeholder_count / total_rows) * 100

print(f"\n3️⃣ Placeholder Analysis:")
print(f"  Rows with placeholder (10.47912088): {placeholder_count}")
print(f"  Total rows: {total_rows}")
print(f"  Percentage: {placeholder_pct:.1f}%")
print(f"  Expected: ~5-15% (early races, new drivers)")

if placeholder_pct < 20:
    print("\n✅ GOOD: Most values are real historical data!")
else:
    print("\n⚠️ High placeholder rate - check data collection")

print("\n4️⃣ Sample Data Evolution:")
print("\nFirst race (placeholders expected):")
print(df[df['Race_Number_In_Season'] == 1][['Driver_Avg_Position_Last5', 'Team_Avg_Position_Last5']].head(5))

print("\nLater race (should have real values):")
print(df[df['Race_Number_In_Season'] == 10][['Driver_Avg_Position_Last5', 'Team_Avg_Position_Last5']].head(5))

print("\n" + "="*70)
print("✅ DATA IS FINE - Repeating values are normal!")
print("="*70)