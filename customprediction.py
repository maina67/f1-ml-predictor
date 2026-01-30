# ============================================================================
# MAKE YOUR OWN PREDICTION
# ============================================================================

print("\n" + "="*70)
print("🏎️ MAKE YOUR OWN RACE PREDICTION")
print("="*70)

# CUSTOMIZE THIS: Enter upcoming race qualifying results
# Format: (Driver Code, Qualifying Position, Grid Position)

custom_qualifying = [
    ('VER', 1, 1),
    ('NOR', 2, 2),
    ('PIA', 3, 3),
    ('RUS', 4, 4),
    ('LEC', 5, 5),
    ('ALO', 6, 6),
    ('BOR', 7, 7),
    ('OCO', 8, 8),
    ('HAD', 9, 9),
    ('TSU', 10, 10),
    ('BEA', 11, 11),
    ('SAI', 12, 12),
    ('LAW', 13, 13),
    ('ANT', 14, 14),
    ('STR', 15, 15),
    ('HAM', 16, 16),
    ('ALB', 17, 17),
    ('HUL', 18, 18),
    ('GAS', 19, 19),
    ('COL', 20, 20),
]

custom_circuit = "Abu Dhabi Grand Prix"  # Change this!

print(f"\n🏁 Predicting: {custom_circuit}")
print("="*70)

my_prediction = predict_race(custom_circuit, custom_qualifying, df_full)

print("\n📊 PREDICTED RESULTS:")
print(my_prediction.to_string(index=False))

print("\n🎯 Podium Prediction:")
podium = my_prediction.head(3)
for i, row in podium.iterrows():
    medals = ["🥇", "🥈", "🥉"]
    print(f"{medals[i]} P{i+1}: {row['Driver']} ({row['Team']})")