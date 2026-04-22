import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from rul_plotting import plot_rul_prediction_comparison

print("Starting Turbofan RUL ensemble pipeline...")

WINDOW_SIZE = 30

# 1. Load dataset
print("Loading C-MAPSS FD001 dataset...")
train_df = pd.read_csv('dataset/train_FD001.txt', sep=r'\s+', header=None)
test_df  = pd.read_csv('dataset/test_FD001.txt',  sep=r'\s+', header=None)
rul_test = pd.read_csv('dataset/RUL_FD001.txt', header=None)

# 2. Build training labels with piece-wise capped RUL
def calculate_piecewise_rul(df, window_size=WINDOW_SIZE):
    """
    Calculate RUL with early-life cap at 125 cycles.
    Labels are sliced to match feature windows (first window_size-1 rows
    per engine are dropped to stay aligned with create_features output).
    """
    rul = []
    for unit in df[0].unique():
        unit_cycles = len(df[df[0] == unit])
        unit_rul = np.minimum(range(unit_cycles, 0, -1), 125)
        # Drop labels without a corresponding sliding window.
        rul.extend(unit_rul[window_size - 1:])
    return np.array(rul)

y_train = calculate_piecewise_rul(train_df)

# 3. Feature engineering
def create_features(df, window_size=WINDOW_SIZE):
    """
    Extract statistical features from sliding sensor windows.
    Engines with fewer than window_size cycles are skipped consistently
    in both this function and calculate_piecewise_rul.
    """
    features = []
    sensor_cols = range(5, 26)

    for unit in df[0].unique():
        unit_data = df[df[0] == unit].iloc[:, sensor_cols]
        if len(unit_data) < window_size:
            # Must mirror label skipping to avoid feature/label mismatch.
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

# Validate alignment between extracted windows and labels.
assert len(X_train) == len(y_train), (
    f"Feature/label mismatch: {len(X_train)} features vs {len(y_train)} labels"
)

# 5. Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# 7. Train base regressors
models = {
    'SVR':       SVR(kernel='rbf', C=10, gamma=0.1),
    'XGBoost':   XGBRegressor(n_estimators=200, max_depth=6, random_state=42),
    'LinearReg': LinearRegression(),
    'GBR':       GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
}

trained_models  = {}
individual_rmse = {}

print("\nTraining 4 regression models...")
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_tr, y_tr)
    trained_models[name] = model

    pred = model.predict(X_val)
    pred = np.clip(pred, 0, 125)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    individual_rmse[name] = rmse
    print(f"    {name} RMSE: {rmse:.2f}")

# 8. Ensemble averaging on validation split
ensemble_pred = np.mean(
    [np.clip(model.predict(X_val), 0, 125) for model in trained_models.values()],
    axis=0
)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
print(f"\nValidation ensemble RMSE: {ensemble_rmse:.2f} cycles")

# 9. Persist models and scaler
joblib.dump(trained_models, 'ensemble_models.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models and scaler saved.")

# 10. Visualization: RMSE comparison
fig, ax = plt.subplots(figsize=(12, 8))

model_names  = list(individual_rmse.keys())
rmse_values  = list(individual_rmse.values())
all_names    = model_names + ['Ensemble']
all_rmse     = rmse_values + [ensemble_rmse]
colors       = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'purple']
alphas       = [0.7, 0.7, 0.7, 0.7, 0.8]

bars = ax.bar(all_names, all_rmse, color=colors, alpha=0.7, edgecolor='black')
bars[-1].set_alpha(0.8)
bars[-1].set_linewidth(2)

for bar, rmse in zip(bars, all_rmse):
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        bar.get_height() + 0.5,
        f'{rmse:.1f}',
        ha='center', va='bottom', fontweight='bold'
    )

ax.set_ylabel('RMSE (Cycles)', fontsize=12, fontweight='bold')
ax.set_title(
    '4-Model Ensemble vs Individual Performance\nNASA C-MAPSS FD001 Dataset',
    fontsize=14, fontweight='bold', pad=20
)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(all_rmse) * 1.15)
plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("RMSE comparison saved as 'rmse_comparison.png'")

# 11. Evaluate on official test set
print("\nEvaluating on official test set...")

X_test_full   = create_features(test_df, window_size=WINDOW_SIZE)
X_test_scaled = scaler.transform(X_test_full)
print(f"Test features shape: {X_test_full.shape}")

# Predict all windows, then map each engine to its final window.
raw_preds = np.mean(
    [np.clip(model.predict(X_test_scaled), 0, 125) for model in trained_models.values()],
    axis=0
)

final_rul_predictions = []
true_test_rul         = rul_test.values.flatten()

prediction_idx = 0
for unit in sorted(test_df[0].unique()):
    unit_rows    = test_df[test_df[0] == unit]
    unit_windows = len(unit_rows) - WINDOW_SIZE + 1

    if unit_windows <= 0:
        # Fallback for engines shorter than the configured window.
        final_rul_predictions.append(0.0)
        continue

    # Last window represents the current engine health state.
    prediction_idx += unit_windows
    engine_rul = raw_preds[prediction_idx - 1]
    final_rul_predictions.append(engine_rul)

final_rul_predictions = np.array(final_rul_predictions)

# Test metrics
test_rmse = np.sqrt(mean_squared_error(true_test_rul, final_rul_predictions))
test_mae  = np.mean(np.abs(true_test_rul - final_rul_predictions))
print(f"Final test set RMSE: {test_rmse:.2f} cycles")
print(f"First 10 RUL predictions: {final_rul_predictions[:10].round(1)}")

# Save results
results_df = pd.DataFrame({
    'Engine_ID':     range(1, len(true_test_rul) + 1),
    'True_RUL':      true_test_rul,
    'Predicted_RUL': final_rul_predictions.round(1),
})
results_df.to_csv('rul_results.csv', index=False)

# 12. Generate RUL Prediction Comparison Visualization
print("\nGenerating RUL prediction comparison plots...")
plot_rul_prediction_comparison(true_test_rul, final_rul_predictions)

print("\nFinal results summary:")
print(f"\tValidation RMSE : {ensemble_rmse:.2f} cycles")
print(f"\tTest RMSE       : {test_rmse:.2f} cycles")
print(f"\tTest MAE        : {test_mae:.2f} cycles")
print("Project complete. Files saved:")
print("\trmse_comparison.png")
print("\trul_results.csv")
print("\tensemble_models.pkl")
print("\tscaler.pkl")