import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from rul_plotting import plot_rul_prediction_comparison

print("Starting Turbofan RUL ensemble pipeline with Meta-Learner...")

WINDOW_SIZE = 30

# 1. Load dataset
# ... (rest of data loading remains same)
print("Loading C-MAPSS FD001 dataset...")
train_df = pd.read_csv('dataset/train_FD001.txt', sep=r'\s+', header=None)
test_df  = pd.read_csv('dataset/test_FD001.txt',  sep=r'\s+', header=None)
rul_test = pd.read_csv('dataset/RUL_FD001.txt', header=None)

# 2. Build training labels with piece-wise capped RUL
def calculate_piecewise_rul(df, window_size=WINDOW_SIZE):
    rul = []
    for unit in df[0].unique():
        unit_cycles = len(df[df[0] == unit])
        unit_rul = np.minimum(range(unit_cycles, 0, -1), 125)
        rul.extend(unit_rul[window_size - 1:])
    return np.array(rul)

y_train = calculate_piecewise_rul(train_df)

# 3. Feature engineering
def create_features(df, window_size=WINDOW_SIZE):
    features = []
    sensor_cols = range(5, 26)
    for unit in df[0].unique():
        unit_data = df[df[0] == unit].iloc[:, sensor_cols]
        if len(unit_data) < window_size:
            continue
        for i in range(window_size - 1, len(unit_data)):
            window = unit_data.iloc[i - window_size + 1 : i + 1]
            stats = np.hstack([
                window.mean().values,
                window.std().values,
                window.min().values,
                window.max().values,
            ])
            features.append(stats)
    return np.array(features)

# 4. Create training features
print("Creating training features...")
X_train = create_features(train_df)
print(f"Training features shape: {X_train.shape}")

assert len(X_train) == len(y_train), f"Feature/label mismatch: {len(X_train)} vs {len(y_train)}"

# 5. Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# 7. Define base regressors and Stacking Ensemble
base_models = [
    ('SVR',       SVR(kernel='rbf', C=10, gamma=0.1)),
    ('XGBoost',   XGBRegressor(n_estimators=200, max_depth=6, random_state=42)),
    ('LinearReg', LinearRegression()),
    ('GBR',       GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
]

# The Meta-Learner is a LinearRegression model that learns weights for each base model
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5,
    passthrough=False  # Only use base model predictions as features for meta-learner
)

print("\nTraining individual base models for comparison...")
individual_rmse = {}
for name, model in base_models:
    print(f"  Evaluating {name}...")
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_val), 0, 125)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    individual_rmse[name] = rmse
    print(f"    {name} RMSE: {rmse:.2f}")

print("\nTraining Stacking Ensemble (Meta-Learner)...")
stacking_model.fit(X_tr, y_tr)

# 8. Evaluate Ensemble on validation split
ensemble_pred = np.clip(stacking_model.predict(X_val), 0, 125)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
print(f"\nValidation Stacking Ensemble RMSE: {ensemble_rmse:.2f} cycles")

# 9. Persist models and scaler
joblib.dump(stacking_model, 'ensemble_models.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Stacked model and scaler saved.")

# 10. Visualization: RMSE comparison
fig, ax = plt.subplots(figsize=(12, 8))
all_names = list(individual_rmse.keys()) + ['Stacking Ensemble']
all_rmse  = list(individual_rmse.values()) + [ensemble_rmse]
colors    = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'purple']

bars = ax.bar(all_names, all_rmse, color=colors, alpha=0.7, edgecolor='black')
for bar, rmse in zip(bars, all_rmse):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{rmse:.1f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('RMSE (Cycles)', fontsize=12, fontweight='bold')
ax.set_title('Base Models vs Stacking Ensemble (Meta-Learner)\nNASA C-MAPSS FD001 Dataset', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300)
print("RMSE comparison saved.")

# 11. Evaluate on official test set
print("\nEvaluating on official test set...")
X_test_full   = create_features(test_df, window_size=WINDOW_SIZE)
X_test_scaled = scaler.transform(X_test_full)

# Meta-learner prediction
raw_preds = np.clip(stacking_model.predict(X_test_scaled), 0, 125)

final_rul_predictions = []
true_test_rul         = rul_test.values.flatten()

prediction_idx = 0
for unit in sorted(test_df[0].unique()):
    unit_rows    = test_df[test_df[0] == unit]
    unit_windows = len(unit_rows) - WINDOW_SIZE + 1
    if unit_windows <= 0:
        final_rul_predictions.append(0.0)
        continue
    prediction_idx += unit_windows
    final_rul_predictions.append(raw_preds[prediction_idx - 1])

final_rul_predictions = np.array(final_rul_predictions)
test_rmse = np.sqrt(mean_squared_error(true_test_rul, final_rul_predictions))
test_mae  = np.mean(np.abs(true_test_rul - final_rul_predictions))
print(f"Final test set RMSE: {test_rmse:.2f} cycles")

# Save results
results_df = pd.DataFrame({
    'Engine_ID': range(1, len(true_test_rul) + 1),
    'True_RUL': true_test_rul,
    'Predicted_RUL': final_rul_predictions.round(1),
})
results_df.to_csv('rul_results.csv', index=False)

# 12. Visualization
plot_rul_prediction_comparison(true_test_rul, final_rul_predictions)

print(f"\nSummary:\n\tValidation RMSE: {ensemble_rmse:.2f}\n\tTest RMSE: {test_rmse:.2f}\n\tTest MAE: {test_mae:.2f}")