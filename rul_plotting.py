import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_rul_prediction_comparison(true_rul: np.ndarray, pred_rul: np.ndarray) -> None:
    """
    Create visualization comparing true vs predicted RUL with two plots:
    1. Scatter plot: Predicted vs True RUL
    2. Line plot: RUL across test engines
    
    Args:
        true_rul: Array of true RUL values
        pred_rul: Array of predicted RUL values
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_rul, pred_rul))
    mae = mean_absolute_error(true_rul, pred_rul)
    r2 = r2_score(true_rul, pred_rul)

    # Create a 1x2 subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Scatter plot: Predicted vs True
    ax1 = axes[0]
    ax1.scatter(true_rul, pred_rul, alpha=0.6, s=50, color="#2A9D8F", edgecolors="black", linewidth=0.5)
    min_val, max_val = min(true_rul.min(), pred_rul.min()), max(true_rul.max(), pred_rul.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax1.set_xlabel("True RUL (cycles)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted RUL (cycles)", fontsize=11, fontweight="bold")
    ax1.set_title("Predicted vs True RUL", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Line plot: Sequential comparison
    ax2 = axes[1]
    x_range = np.arange(len(true_rul))
    ax2.plot(x_range, true_rul, label="True RUL", color="#264653", linewidth=2, alpha=0.8)
    ax2.plot(x_range, pred_rul, label="Predicted RUL", color="#E76F51", linewidth=2, alpha=0.8, linestyle="--")
    ax2.fill_between(x_range, true_rul, pred_rul, alpha=0.2, color="gray")
    ax2.set_xlabel("Engine ID", fontsize=11, fontweight="bold")
    ax2.set_ylabel("RUL (cycles)", fontsize=11, fontweight="bold")
    ax2.set_title("RUL Across Test Engines", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add metrics text box
    metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}"
    fig.text(0.98, 0.02, metrics_text, fontsize=11, ha="right", va="bottom",
             bbox=dict(boxstyle="round", facecolor="#E8F4F8", alpha=0.8, pad=0.5))

    fig.suptitle("RUL Prediction Performance Analysis (Test Set)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig('rul_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("RUL comparison visualization saved as 'rul_prediction_comparison.png'")


if __name__ == "__main__":
    # Generate sample test data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Create realistic RUL values (0-125 cycles)
    true_rul = np.random.uniform(10, 125, n_samples)
    
    # Add some noise to create predicted values
    pred_rul = true_rul + np.random.normal(0, 8, n_samples)
    pred_rul = np.clip(pred_rul, 0, 125)
    
    # Generate plots
    print("Running RUL Plotting Module (Demo Mode)")
    print(f"Sample size: {n_samples}")
    print(f"True RUL range: [{true_rul.min():.1f}, {true_rul.max():.1f}]")
    print(f"Predicted RUL range: [{pred_rul.min():.1f}, {pred_rul.max():.1f}]")
    plot_rul_prediction_comparison(true_rul, pred_rul)

