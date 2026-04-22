import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_DIR = "dataset"
OUTPUT_DIR = "eda_outputs"
RUL_CAP = 125


def load_fd001_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.txt"), sep=r"\s+", header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_FD001.txt"), sep=r"\s+", header=None)
    rul_df = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"), header=None)

    columns = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    train_df.columns = columns
    test_df.columns = columns
    rul_df.columns = ["rul"]
    return train_df, test_df, rul_df


def add_rul_columns(train_df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    df = train_df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul_uncapped"] = max_cycle - df["cycle"]
    df["rul_capped"] = df["rul_uncapped"].clip(upper=rul_cap)
    df["life_pct"] = df["cycle"] / max_cycle
    return df


def choose_top_sensors(df: pd.DataFrame, n: int = 6) -> List[str]:
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    corr = df[sensor_cols + ["rul_capped"]].corr(numeric_only=True)["rul_capped"].drop("rul_capped")
    top = corr.abs().sort_values(ascending=False).head(n)
    return top.index.tolist()


def save_engine_life_distribution(train_df: pd.DataFrame) -> None:
    life_by_engine = train_df.groupby("unit")["cycle"].max()

    plt.figure(figsize=(10, 6))
    sns.histplot(life_by_engine, bins=20, kde=True, color="#2A9D8F")
    plt.title("FD001 Engine Lifetime Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Cycles Until Failure")
    plt.ylabel("Number of Engines")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_engine_lifetime_distribution.png"), dpi=300)
    plt.close()


def save_rul_distribution(train_with_rul: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_with_rul["rul_capped"], bins=30, kde=True, color="#E76F51")
    plt.title("Capped RUL Label Distribution (Training Rows)", fontsize=14, fontweight="bold")
    plt.xlabel("RUL (cycles, capped at 125)")
    plt.ylabel("Number of Observations")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_capped_rul_distribution.png"), dpi=300)
    plt.close()


def save_degradation_trajectories(train_with_rul: pd.DataFrame, top_sensors: List[str]) -> None:
    sensors_to_plot = top_sensors[:4]
    trend_df = train_with_rul.copy()
    trend_df["life_bin"] = pd.cut(trend_df["life_pct"], bins=np.linspace(0, 1, 21), include_lowest=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for idx, sensor in enumerate(sensors_to_plot):
        grouped = trend_df.groupby("life_bin", observed=False)[sensor].mean().reset_index()
        grouped["life_mid"] = grouped["life_bin"].apply(lambda interval: interval.mid if pd.notna(interval) else np.nan)
        axes[idx].plot(grouped["life_mid"], grouped[sensor], marker="o", linewidth=2, color="#264653")
        axes[idx].set_title(f"{sensor} vs Normalized Life", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Lifecycle Progress (0=start, 1=failure)")
        axes[idx].set_ylabel("Mean Sensor Value")
        axes[idx].grid(alpha=0.3)

    fig.suptitle("Average Sensor Degradation Trajectories", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_degradation_trajectories.png"), dpi=300)
    plt.close()


def save_correlation_heatmap(train_with_rul: pd.DataFrame) -> None:
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    corr_series = (
        train_with_rul[sensor_cols + ["rul_capped"]]
        .corr(numeric_only=True)["rul_capped"]
        .drop("rul_capped")
        .sort_values()
    )

    corr_df = corr_series.to_frame(name="Correlation with RUL")
    plt.figure(figsize=(6, 9))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar=True)
    plt.title("Sensor Correlation with Capped RUL", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_sensor_rul_correlation_heatmap.png"), dpi=300)
    plt.close()

    corr_df.reset_index().rename(columns={"index": "sensor"}).to_csv(
        os.path.join(OUTPUT_DIR, "sensor_rul_correlations.csv"), index=False
    )


def save_sensor_by_rul_bin_boxplots(train_with_rul: pd.DataFrame, top_sensors: List[str]) -> None:
    sensors_to_plot = top_sensors[:4]
    plot_df = train_with_rul.copy()

    plot_df["rul_band"] = pd.cut(
        plot_df["rul_capped"],
        bins=[-1, 30, 60, 90, 125],
        labels=["0-30", "31-60", "61-90", "91-125"],
    )

    melted = plot_df.melt(id_vars=["rul_band"], value_vars=sensors_to_plot, var_name="sensor", value_name="value")

    g = sns.catplot(
        data=melted,
        x="rul_band",
        y="value",
        col="sensor",
        kind="box",
        col_wrap=2,
        height=4,
        aspect=1.4,
        sharey=False,
        palette="Set2",
    )
    g.fig.suptitle("Top Sensors Across RUL Bands", y=1.02, fontsize=15, fontweight="bold")
    g.set_axis_labels("RUL Band (cycles)", "Sensor Value")
    g.savefig(os.path.join(OUTPUT_DIR, "05_sensor_boxplots_by_rul_band.png"), dpi=300)
    plt.close("all")


def save_pca_projection(train_with_rul: pd.DataFrame) -> None:
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    sampled = train_with_rul.sample(min(12000, len(train_with_rul)), random_state=42).copy()

    sampled["rul_band"] = pd.cut(
        sampled["rul_capped"],
        bins=[-1, 30, 60, 90, 125],
        labels=["Near Failure (0-30)", "Mid-Late (31-60)", "Mid (61-90)", "Healthy (91-125)"],
    )

    X = sampled[sensor_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "PC1": comps[:, 0],
            "PC2": comps[:, 1],
            "rul_band": sampled["rul_band"].values,
        }
    )

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="rul_band",
        alpha=0.55,
        s=28,
        palette="viridis",
    )
    explained = pca.explained_variance_ratio_
    plt.title(
        f"PCA View of Engine States (PC1+PC2 Explained: {(explained[0] + explained[1]) * 100:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_pca_engine_state_map.png"), dpi=300)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    train_df, test_df, rul_df = load_fd001_data()
    train_with_rul = add_rul_columns(train_df)
    top_sensors = choose_top_sensors(train_with_rul, n=6)

    print("Top sensors by absolute correlation with capped RUL:")
    print(top_sensors)

    save_engine_life_distribution(train_df)
    save_rul_distribution(train_with_rul)
    save_degradation_trajectories(train_with_rul, top_sensors)
    save_correlation_heatmap(train_with_rul)
    save_sensor_by_rul_bin_boxplots(train_with_rul, top_sensors)
    save_pca_projection(train_with_rul)

    # Keep a compact dataset-level summary for paper writing.
    summary = pd.DataFrame(
        {
            "metric": ["num_train_engines", "num_test_engines", "num_train_rows", "mean_engine_life", "std_engine_life"],
            "value": [
                train_df["unit"].nunique(),
                test_df["unit"].nunique(),
                len(train_df),
                train_df.groupby("unit")["cycle"].max().mean(),
                train_df.groupby("unit")["cycle"].max().std(),
            ],
        }
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"), index=False)

    print("\nEDA complete. Files saved in 'eda_outputs':")
    for name in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {name}")


if __name__ == "__main__":
    main()