# ============================================================================
# WHAT-IF SCENARIOS
# ============================================================================

print("\n" + "="*70)
print("🔬 WHAT-IF SCENARIO ANALYSIS")
print("="*70)

# Scenario 1: What if Hamilton qualifies on pole?
print("\n📌 Scenario 1: Hamilton on Pole vs P5")

quali_pole = [('HAM', 1, 1)] + [(d, i+2, i+2) for i, d in enumerate(['VER', 'LEC', 'SAI', 'PER'])]
quali_p5 = [('VER', 1, 1), ('LEC', 2, 2), ('SAI', 3, 3), ('PER', 4, 4), ('HAM', 5, 5)]

pred_pole = predict_race(example_circuit, quali_pole[:5], df_full)
pred_p5 = predict_race(example_circuit, quali_p5, df_full)

ham_pole_finish = pred_pole[pred_pole['Driver'] == 'HAM']['Predicted_Finish'].values[0]
ham_p5_finish = pred_p5[pred_p5['Driver'] == 'HAM']['Predicted_Finish'].values[0]

print(f"  Hamilton from Pole: Predicted P{ham_pole_finish:.1f}")
print(f"  Hamilton from P5: Predicted P{ham_p5_finish:.1f}")
print(f"  Difference: {abs(ham_pole_finish - ham_p5_finish):.1f} positions")

print("\n💡 This shows how important qualifying is!")