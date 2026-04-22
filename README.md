# Turbofan RUL Prediction

Predictive maintenance model for estimating the **Remaining Useful Life (RUL)** of turbofan engines using machine learning ensemble methods. This project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001 dataset.

## Project Overview

This project implements an ensemble learning approach to predict remaining useful life for aircraft engines. It combines multiple machine learning models including:
- Support Vector Regression (SVR)
- XGBoost
- Linear Regression
- Gradient Boosting Regressor

### Key Features

- **Exploratory Data Analysis (EDA)** — Sensor correlation analysis and dataset profiling
- **Feature Engineering** — Sliding window-based statistical feature extraction
- **Model Ensemble** — Multiple model comparison and optimization
- **RUL Prediction** — Predictive maintenance using real sensor data

## Dataset

The project uses the **C-MAPSS FD001 dataset**, which contains sensor readings from turbofan engines under degradation.

### Dataset Source

- **Download Location**: [NASA's Prognostics Center of Excellence](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **Alternative**: [GitHub Mirror](https://github.com/cathysiyu/Turbofan-engine-RUL-prediction)

### Dataset Files Required

Place the following files in the `dataset/` directory:
- `train_FD001.txt` — Training data with sensor readings
- `test_FD001.txt` — Test data with sensor readings
- `RUL_FD001.txt` — Ground truth RUL values for test data

### Dataset Structure

Each row contains:
- **Unit ID**: Engine identifier (1-100 for training)
- **Cycle**: Operating cycle number
- **Operating Settings**: 3 columns (OS1, OS2, OS3)
- **Sensor Readings**: 21 sensors (S1-S21)

## Installation

### Prerequisites

- Python 3.8+
- Windows/Linux/macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

### Step 2: Create Virtual Environment

```bash
# On Windows (PowerShell)
python -m venv turbofan_env
.\turbofan_env\Scripts\Activate.ps1

# On Windows (Command Prompt)
python -m venv turbofan_env
turbofan_env\Scripts\activate

# On macOS/Linux
python -m venv turbofan_env
source turbofan_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **numpy** — Numerical computing
- **pandas** — Data manipulation
- **scikit-learn** — Machine learning algorithms
- **xgboost** — Gradient boosting
- **matplotlib** — Visualization
- **seaborn** — Statistical plotting
- **scipy** — Scientific computing
- **joblib** — Model persistence

## Project Setup

### 1. Download and Prepare Dataset

```bash
# Create dataset directory
mkdir dataset

# Download FD001 files from NASA repository and place in dataset/ folder
# - train_FD001.txt
# - test_FD001.txt
# - RUL_FD001.txt
```

### 2. Run Exploratory Data Analysis

```bash
python eda_analysis.py
```

This generates:
- `eda_outputs/dataset_summary.csv` — Dataset statistics
- `eda_outputs/sensor_rul_correlations.csv` — Sensor-RUL correlations
- Correlation heatmaps and distribution plots

### 3. Train and Evaluate Models

```bash
python main.py
```

This performs:
- Data loading and preprocessing
- Feature engineering with sliding windows
- Model training (SVR, XGBoost, Linear Regression, Gradient Boosting)
- Model evaluation and comparison
- Saves predictions to `rul_results.csv`

### 4. Visualize Results

Predictions are automatically plotted showing:
- Actual vs Predicted RUL values
- Model comparison plots
- Error distributions

## Project Structure

```
turbofan-rul-prediction/
├── main.py                    # Main training pipeline
├── eda_analysis.py            # Exploratory data analysis
├── rul_plotting.py            # Visualization utilities
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── dataset/                   # Data directory (download from NASA)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── eda_outputs/               # EDA results
│   ├── dataset_summary.csv
│   └── sensor_rul_correlations.csv
│
├── rul_results.csv            # Model predictions (generated)
│
└── turbofan_env/              # Virtual environment (auto-created)
```

## Usage

### Quick Start

```bash
# Activate environment
.\turbofan_env\Scripts\Activate.ps1  # Windows PowerShell
# or
source turbofan_env/bin/activate    # macOS/Linux

# Run full pipeline
python main.py
```

### Run Individual Components

```bash
# EDA only
python eda_analysis.py

# Add your own analysis
python -i main.py  # Interactive mode for exploration
```

## Configuration

Key parameters in `main.py`:

- `WINDOW_SIZE = 30` — Sliding window size for feature extraction
- `RUL_CAP = 125` — Maximum RUL value (capped early-life degradation)

## Model Performance

The ensemble approach combines:
- **SVR** — Robust non-linear modeling
- **XGBoost** — Gradient boosting with regularization
- **Linear Regression** — Baseline model
- **Gradient Boosting** — Alternative boosting method

Performance metrics include:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Core training pipeline with ensemble models |
| `eda_analysis.py` | Data exploration, correlation analysis, visualizations |
| `rul_plotting.py` | Visualization functions for RUL predictions |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore configuration |

## Troubleshooting

### Dataset Not Found
```
Error: FileNotFoundError: 'dataset/train_FD001.txt'
```
**Solution**: Download FD001 files from NASA repository and place in `dataset/` directory.

### Memory Issues
If running out of memory:
- Reduce `WINDOW_SIZE` parameter
- Process data in batches
- Use subset of training data

### Package Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## References

- **NASA C-MAPSS Dataset**: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Scikit-Learn**: https://scikit-learn.org/

## License

This project is provided for educational purposes.

## Contributing

Contributions welcome! Feel free to:
- Add new models
- Improve feature engineering
- Enhance visualizations
- Optimize performance
