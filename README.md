# Turbofan RUL Prediction

Predictive maintenance model for estimating the **Remaining Useful Life (RUL)** of turbofan engines using machine learning ensemble methods. This project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001 dataset.

## Project Overview

This project implements a **Stacking Ensemble (Stacked Generalization)** approach to predict the remaining useful life of aircraft engines. Unlike simple averaging, this method uses a **Meta-Learner** to intelligently combine predictions from multiple base models:
- **Support Vector Regression (SVR)** — Robust non-linear modeling
- **XGBoost** — High-performance gradient boosting
- **Linear Regression** — Baseline linear modeling
- **Gradient Boosting Regressor** — Alternative tree-based boosting

The meta-learner (Linear Regression) is trained on out-of-fold predictions from these base models, learning to prioritize the most accurate predictors for different engine states.

### Key Features

- **Exploratory Data Analysis (EDA)** — Sensor correlation analysis and PCA engine state mapping
- **Feature Engineering** — Sliding window-based statistical extraction (Window Size: 30)
- **Stacking Ensemble** — Meta-learner optimization using `StackingRegressor`
- **RUL Prediction** — Piece-wise capped RUL (125 cycles) for realistic early-life modeling

## Dataset

The project uses the **C-MAPSS FD001 dataset**, which contains sensor readings from turbofan engines under degradation.

### Dataset Source

- **Download Location**: [NASA's Prognostics Center of Excellence](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

### Dataset Files Required

Place the following files in the `dataset/` directory:
- `train_FD001.txt` — Training data with sensor readings
- `test_FD001.txt` — Test data with sensor readings
- `RUL_FD001.txt` — Ground truth RUL values for test data

## Installation

### Step 1: Clone and Setup
```bash
git clone https://github.com/AvisharSharan/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

### Step 2: Virtual Environment
```bash
python -m venv turbofan_env
source turbofan_env/bin/activate  # Or turbofan_env\Scripts\activate on Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Exploratory Data Analysis
```bash
python eda_analysis.py
```
Outputs detailed visualizations in `eda_outputs/` including PCA projections and correlation heatmaps.

### 2. Train Stacking Ensemble
```bash
python main.py
```
This runs the full pipeline:
1. Feature extraction and scaling.
2. Training base models (SVR, XGBoost, etc.).
3. Training the **Meta-Learner** via 5-fold cross-validation.
4. Final evaluation on the official test set.

## Project Structure

```
turbofan-rul-prediction/
├── main.py                    # Stacking Ensemble pipeline
├── eda_analysis.py            # Data exploration and PCA
├── rul_plotting.py            # Visualization utilities
├── requirements.txt           # Python dependencies
├── dataset/                   # FD001 data files
├── eda_outputs/               # EDA plots and summaries
├── ensemble_models.pkl        # Saved Stacking Regressor
└── rul_results.csv            # Final predictions on test set
```

## Model Performance

The **Meta-Learner** approach provides a significant improvement in validation reliability by learning the strengths of each base model.

| Metric | Stacking Ensemble Value |
|------|---------|
| **Validation RMSE** | ~4.5 Cycles |
| **Test RMSE** | ~19.3 Cycles |
| **Test MAE** | ~14.2 Cycles |

*Note: Validation RMSE is significantly lower as the meta-learner optimizes for the training distribution; Test RMSE reflects generalization to unseen NASA engine profiles.*

## Configuration

Key parameters in `main.py`:
- `WINDOW_SIZE = 30`: Length of the temporal window for sensor stats.
- `RUL_CAP = 125`: The cycle count where RUL is capped during healthy operation.
- `cv = 5`: Number of folds for training the meta-learner.

## References

- **NASA C-MAPSS Dataset**: NASA Prognostics Center of Excellence.
- **Stacked Generalization**: Wolpert, D. H. (1992). Stacked generalization.

## License

Educational use only.
