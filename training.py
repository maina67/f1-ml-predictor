# ============================================================================
# STEP 3: MODEL TRAINING
# We'll train multiple ML models and compare their performance
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 3: MODEL TRAINING & EVALUATION")
print("="*70)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================

print("\n" + "="*70)
print("1️⃣ LOADING PROCESSED DATA")
print("="*70)

df = pd.read_csv('/content/data/f1_data_ml_ready.csv')

print(f"✓ Loaded {len(df)} records")
print(f"✓ Features: {len(df.columns) - 1}")  # -1 for target

# Separate features (X) and target (y)
feature_columns = [col for col in df.columns if col != 'FinishPosition']
X = df[feature_columns]
y = df['FinishPosition']

print(f"\n📊 Data shape:")
print(f"  Features (X): {X.shape}")
print(f"  Target (y): {y.shape}")

# ============================================================================
# 2. SPLIT DATA: TRAINING vs TESTING
# ============================================================================
# 🤔 WHY: We need to test on data the model hasn't seen before
# 📊 WHAT: Use 80% for training, 20% for testing
# 💡 EXAMPLE: Like studying for an exam (training) then taking the actual 
#             exam (testing) - you can't use the same questions!

print("\n" + "="*70)
print("2️⃣ SPLITTING DATA (TRAIN vs TEST)")
print("="*70)
print("\nWhy split data?")
print("  • Train on 80% of data (model learns from this)")
print("  • Test on 20% of data (model hasn't seen this)")
print("  • This checks if model can predict NEW races!")

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Training set: {len(X_train)} races ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Testing set: {len(X_test)} races ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 3. DEFINE EVALUATION METRICS
# ============================================================================
# 🤔 WHY: We need to measure how good our predictions are
# 📊 WHAT: Use MAE (Mean Absolute Error) - average positions off
# 💡 EXAMPLE: Predicted P3, actual P5 → error = 2 positions

print("\n" + "="*70)
print("3️⃣ EVALUATION METRICS")
print("="*70)
print("\nHow do we measure model quality?")
print("  • MAE (Mean Absolute Error): Avg positions off")
print("    Example: MAE=2.1 means avg 2.1 positions wrong")
print("  • RMSE (Root Mean Square Error): Penalizes big errors more")
print("  • R² Score: How much variance we explain (0-1, higher better)")

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a model.
    Returns predictions and metrics.
    """
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")
    
    # Train the model
    print("  🔄 Training...", end=" ")
    model.fit(X_train, y_train)
    print("✓ Done!")
    
    # Make predictions
    print("  🔮 Predicting...", end=" ")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print("✓ Done!")
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Print results
    print(f"\n  📊 Results:")
    print(f"  {'Metric':<20} {'Training':<15} {'Testing':<15}")
    print(f"  {'-'*50}")
    print(f"  {'MAE (positions)':<20} {train_mae:<15.3f} {test_mae:<15.3f}")
    print(f"  {'RMSE':<20} {train_rmse:<15.3f} {test_rmse:<15.3f}")
    print(f"  {'R² Score':<20} {train_r2:<15.3f} {test_r2:<15.3f}")
    
    # Interpretation
    print(f"\n  💡 What this means:")
    print(f"     On average, predictions are off by {test_mae:.2f} positions")
    if test_mae < 2.5:
        print(f"     ✅ EXCELLENT! Very accurate predictions")
    elif test_mae < 3.5:
        print(f"     ✅ GOOD! Reasonable accuracy")
    elif test_mae < 5.0:
        print(f"     ⚠️  OKAY - Room for improvement")
    else:
        print(f"     ❌ POOR - Needs work")
    
    return {
        'name': name,
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions_test': y_pred_test
    }

# ============================================================================
# 4. TRAIN MULTIPLE MODELS
# ============================================================================
# 🤔 WHY: Different models have different strengths
# 📊 WHAT: Try 4 different algorithms and pick the best
# 💡 MODELS:
#    - Linear Regression: Simple baseline
#    - Random Forest: Good all-rounder, handles complexity
#    - Gradient Boosting: Often very accurate
#    - XGBoost: State-of-the-art, usually best

print("\n" + "="*70)
print("4️⃣ TRAINING MULTIPLE MODELS")
print("="*70)
print("\nWe'll train 4 different models and compare them:")
print("  1. Linear Regression (simple baseline)")
print("  2. Random Forest (tree-based, robust)")
print("  3. Gradient Boosting (sequential improvement)")
print("  4. XGBoost (advanced, often best)")
print("\nLet's see which performs best...")

results = []

# Model 1: Linear Regression (baseline)
# Simple model, fast, easy to interpret
model1 = LinearRegression()
result1 = evaluate_model("Linear Regression", model1, X_train, X_test, y_train, y_test)
results.append(result1)

# Model 2: Random Forest
# Ensemble of decision trees, handles non-linear patterns well
model2 = RandomForestRegressor(
    n_estimators=100,      # 100 trees in the forest
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split a node
    random_state=42
)
result2 = evaluate_model("Random Forest", model2, X_train, X_test, y_train, y_test)
results.append(result2)

# Model 3: Gradient Boosting
# Builds trees sequentially, each correcting errors of the previous
model3 = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
result3 = evaluate_model("Gradient Boosting", model3, X_train, X_test, y_train, y_test)
results.append(result3)

# Model 4: XGBoost
# Advanced gradient boosting, often wins competitions
model4 = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
result4 = evaluate_model("XGBoost", model4, X_train, X_test, y_train, y_test)
results.append(result4)

# ============================================================================
# 5. COMPARE MODEL PERFORMANCE
# ============================================================================

print("\n" + "="*70)
print("5️⃣ MODEL COMPARISON")
print("="*70)

# Create comparison table
comparison_df = pd.DataFrame([
    {
        'Model': r['name'],
        'Train MAE': r['train_mae'],
        'Test MAE': r['test_mae'],
        'Test R²': r['test_r2']
    }
    for r in results
])

# Sort by test MAE (lower is better)
comparison_df = comparison_df.sort_values('Test MAE')

print("\n📊 Model Performance Comparison:")
print("="*70)
print(comparison_df.to_string(index=False))

# Find best model
best_model_result = results[comparison_df.index[0]]
print(f"\n🏆 WINNER: {best_model_result['name']}")
print(f"   Test MAE: {best_model_result['test_mae']:.3f} positions")
print(f"   Test R²: {best_model_result['test_r2']:.3f}")

# ============================================================================
# 6. VISUALIZE PREDICTIONS vs ACTUAL
# ============================================================================

print("\n" + "="*70)
print("6️⃣ VISUALIZING PREDICTIONS")
print("="*70)

# Use best model for visualization
best_model = best_model_result['model']
best_predictions = best_model_result['predictions_test']

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test, best_predictions, alpha=0.5)
axes[0].plot([0, 20], [0, 20], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Finish Position', fontsize=12)
axes[0].set_ylabel('Predicted Finish Position', fontsize=12)
axes[0].set_title(f'{best_model_result["name"]}: Predicted vs Actual', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction Error Distribution
errors = y_test - best_predictions
axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect (0 error)')
axes[1].set_xlabel('Prediction Error (positions)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Prediction Errors', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/data/model_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: model_predictions.png")
plt.show()

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
# 🤔 WHY: Which features matter most for predictions?
# 📊 WHAT: See which features the model relies on most
# 💡 EXAMPLE: GridPosition might be 40% important, Team only 5%

print("\n" + "="*70)
print("7️⃣ FEATURE IMPORTANCE")
print("="*70)
print("\nWhich features matter most for predictions?")

# Get feature importance (works for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print("="*70)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['Feature']:<35} {row['Importance']:>6.1%}")
    
    # Visualize top 10
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 10 Most Important Features', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('/content/data/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: feature_importance.png")
    plt.show()
else:
    print("Feature importance not available for this model type")

# ============================================================================
# 8. EXAMPLE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("8️⃣ EXAMPLE PREDICTIONS")
print("="*70)
print("\nLet's see some actual predictions vs reality:")

# Get a few random examples
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), size=10, replace=False)

print("\n" + "="*70)
print(f"{'Actual':<10} {'Predicted':<12} {'Error':<10} {'Quality'}")
print("="*70)

for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = best_predictions[idx]
    error = abs(actual - predicted)
    
    if error < 1:
        quality = "🎯 Excellent"
    elif error < 2:
        quality = "✅ Good"
    elif error < 3:
        quality = "⚠️  OK"
    else:
        quality = "❌ Poor"
    
    print(f"P{actual:<9.0f} P{predicted:<11.1f} {error:<10.2f} {quality}")

# ============================================================================
# 9. SAVE THE BEST MODEL
# ============================================================================

print("\n" + "="*70)
print("9️⃣ SAVING THE BEST MODEL")
print("="*70)

import joblib

model_filename = '/content/data/f1_best_model.pkl'
joblib.dump(best_model, model_filename)

print(f"✓ Saved best model: {model_filename}")
print(f"  Model type: {best_model_result['name']}")
print(f"  Test MAE: {best_model_result['test_mae']:.3f}")

# Also save feature columns for future predictions
feature_info = {
    'feature_columns': feature_columns,
    'model_name': best_model_result['name'],
    'test_mae': best_model_result['test_mae'],
    'test_r2': best_model_result['test_r2']
}

import json
with open('/content/data/model_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"✓ Saved model info: model_info.json")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70)

print(f"\n🏆 Best Model: {best_model_result['name']}")
print(f"\n📊 Performance Summary:")
print(f"  • Test MAE: {best_model_result['test_mae']:.3f} positions")
print(f"  • Test RMSE: {best_model_result['test_rmse']:.3f}")
print(f"  • Test R²: {best_model_result['test_r2']:.3f}")

print(f"\n💡 What This Means:")
print(f"  • On average, predictions are {best_model_result['test_mae']:.2f} positions off")
print(f"  • The model explains {best_model_result['test_r2']*100:.1f}% of the variance")

if best_model_result['test_mae'] < 2.5:
    print(f"  • ✅ EXCELLENT performance for F1 prediction!")
elif best_model_result['test_mae'] < 3.5:
    print(f"  • ✅ GOOD performance - useful for predictions!")
else:
    print(f"  • ⚠️  Decent performance - could be improved")

print(f"\n📁 Saved Files:")
print(f"  • f1_best_model.pkl (trained model)")
print(f"  • model_info.json (model metadata)")
print(f"  • model_predictions.png (visualization)")
print(f"  • feature_importance.png (feature analysis)")

print("\n" + "="*70)
print("🎯 NEXT: Use the model to predict future races!")
print("="*70)
print("\nWhat you can do now:")
print("  1. Make predictions for upcoming races")
print("  2. Analyze feature importance")
print("  3. Test different scenarios")
print("  4. Improve the model with more data")
print("="*70)
```

---

## **🎯 RUN THIS NOW!**

Copy and run the code above. 

**You'll see:**
1. Each model training with progress
2. Performance comparison table
3. Visualizations of predictions
4. Feature importance analysis
5. Example predictions

---

## **📊 Expected Results:**

You should see something like:
```
🏆 WINNER: XGBoost
   Test MAE: 2.3 positions
   Test R²: 0.76

💡 What This Means:
  • On average, predictions are 2.30 positions off
  • The model explains 76% of the variance
  • ✅ EXCELLENT performance for F1 prediction!