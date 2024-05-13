import os
import pathlib
import json
import argparse
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

dataframes_folder_path = "./dataframes/"
results_folder_path = "./results/"
BENCHMARK_RESULTS_PATH = os.environ["BENCHMARK_RESULTS_PATH"]

sns.set_style("white")
sns.color_palette("deep")


def save_plot(filename:str):
    images_folder_path = "./images/"
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path, exist_ok=True)
    plt.savefig(os.path.join(images_folder_path, filename), format="pdf")


def split_df(data_df):
    splits = math.ceil(len(data_df) / 20)

    total_rows = len(data_df)
    rows_per_part = math.ceil(total_rows / splits)

    split_dfs = [
        data_df.iloc[i * rows_per_part : (i + 1) * rows_per_part] for i in range(splits)
    ]

    num_rows = math.ceil(splits / 3)
    num_cols = min(splits, 3)
    return split_dfs, num_rows, num_cols


def calculate_q_error(estimated, real):
        return (estimated / real).combine(real / estimated, max)


def get_mysql_q_error_for_workload(workload, sub_plans=False):
    if (sub_plans):
        results_folder_path = f"{BENCHMARK_RESULTS_PATH}/{workload}/sub_plan_queries/baseline"
    else:
        results_folder_path = f"{BENCHMARK_RESULTS_PATH}/{workload}/baseline"
    if (not os.path.exists(results_folder_path)):
        return None
    dfs = {}
    df_combined = pd.DataFrame()
    for path in pathlib.Path(results_folder_path).glob("*.json"):
        with open(str(path)) as f:
            data = json.load(f)
            name = str(path.stem)
            estimated_cardinality = data["inputs"][0]["estimated_rows"]
            actual_cardinality = data["inputs"][0]["actual_rows"]
            df = pd.DataFrame({"query": name, "estimated": [estimated_cardinality], "real": [actual_cardinality]})
            df["q-error"] = calculate_q_error(df['estimated'], df['real'])
            dfs[name] = df
            
    dfs = {key: dfs[key] for key in sorted(dfs)}
    for key, df in dfs.items():
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    
    return df_combined


def plot_model_training(save=False):
    df = pd.read_pickle(os.path.join(dataframes_folder_path, "model_training.pkl"))
    df["epochs"] = range(1, len(df) + 1)
    df_training = df[["epochs", "training_loss"]]
    df_validation = df[["epochs", "validation_loss"]]
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_training, x="epochs", y="training_loss")
    plt.yticks(np.arange(0, df_training["training_loss"].max() + 1, 10.0))
    plt.xlabel("Epochs")
    plt.ylabel("Mean Q-Error")
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_validation, x="epochs", y="validation_loss")
    plt.title("Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Q-Error")
    plt.tight_layout()
    if save:
        save_plot("model_training.pdf")
    else:
        plt.show()


def plot_q_error_for_workload(workload, plot_type="box", sampling=False, sub_plans=False, save=False):
    dfs = {}
    df_combined = pd.DataFrame()
    if (sampling):
        df = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}-sampling.csv"), names=["estimated", "real"])
    elif (sub_plans):
        df = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}-sub-queries.csv"), names=["estimated", "real"])
    else:
        df = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}.csv"), names=["estimated", "real"])
    df["q-error"] = calculate_q_error(df["estimated"], df["real"])
    df["query"] = df.index + 1
    dfs["MSCN"] = df
    df_baseline = get_mysql_q_error_for_workload(workload, sub_plans)
    dfs["MySQL"] = df_baseline
    dfs = {key: dfs[key] for key in sorted(dfs)}
    for key, df in dfs.items():
        if df is None: continue
        df["model"] = key
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    plt.figure(figsize=(20, 12))
    if (plot_type == "box"):
        sns.boxplot(data=df_combined, x="model", y="q-error", hue="model")
    elif (plot_type == "bar"):
        sns.barplot(data=df_combined, x="query", y="q-error", hue="model", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Q-Error")
    plt.tight_layout()
    if (save):
        if (sub_plans):
            save_plot(f"q-error-{workload}-sub-queries.pdf")
        else:
            save_plot(f"q-error-{workload}.pdf")
    else:
        plt.show()


def plot_q_error_for_workload_compare_with_sampling(workload, plot_type="bar", sub_plans=False, save=False):
    dfs = {}
    df_combined = pd.DataFrame()
    if (sub_plans):
        df = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}-sub-queries.csv"), names=["estimated", "real"])
    else:
        df = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}.csv"), names=["estimated", "real"])
    df["q-error"] = calculate_q_error(df["estimated"], df["real"])
    df["query"] = df.index + 1
    dfs["MSCN"] = df
    if (sub_plans):
        df_sampling = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}-sub-queries-sampling.csv"), names=["estimated", "real"])
    else:
        df_sampling = pd.read_csv(os.path.join(results_folder_path, f"predictions_{workload}-sampling.csv"), names=["estimated", "real"])
    df_sampling["q-error"] = calculate_q_error(df_sampling["estimated"], df_sampling["real"])
    df_sampling["query"] = df_sampling.index + 1
    dfs["MSCN (with sampling)"] = df_sampling
    df_baseline = get_mysql_q_error_for_workload(workload, sub_plans)
    dfs["MySQL"] = df_baseline
    dfs = {key: dfs[key] for key in sorted(dfs)}
    for key, df in dfs.items():
        if df is None: continue
        df["model"] = key
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    plt.figure(figsize=(20, 12))
    if (plot_type == "box"):
        sns.boxplot(data=df_combined, x="model", y="q-error", hue="model")
    elif (plot_type == "bar"):
        sns.barplot(data=df_combined, x="query", y="q-error", hue="model", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Q-Error")
    plt.tight_layout()
    if (save):
        if (sub_plans):
            save_plot(f"q-error-{workload}-compare-sampling-sub-queries.pdf")
        else:
            save_plot(f"q-error-{workload}-compare-sampling.pdf")
    else:
        plt.show()


def plot_q_error_all_workloads(save=False):
    dfs = {}
    for path in pathlib.Path(results_folder_path).glob("*.csv"):
        with open(str(path)) as f:
            df = pd.read_csv(str(path), names=["estimated", "real"])
            name = str(path.stem).removeprefix("predictions_")
            df["q-error"] = calculate_q_error(df['estimated'], df['real'])
            dfs[name] = df
    
    df_combined = pd.DataFrame()
    dfs = {key: dfs[key] for key in sorted(dfs)}
    for key, df in dfs.items():
        df["workload"] = key
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    
    plt.figure(figsize=(20, 12))
    sns.boxplot(data=df_combined, x="workload", y="q-error", hue="workload")
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Q-Error")
    plt.tight_layout()
    if (save):
        save_plot("q-error-all-workloads.pdf")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", help="synthetic, synthetic-sampling, scale, scale-sampling, job-light, job-light-sampling (default: job-light)", type=str, default="job-light")
    parser.add_argument("--plot-type", help="bar, box (default: bar)", type=str, default="box")
    parser.add_argument("--all-workloads", help="plot metrics for all workloads", action="store_true")
    parser.add_argument("--sub-plans", help="use sub-plans", action="store_true")
    parser.add_argument("--sampling", help="use workload with materialized base samples", action="store_true")
    parser.add_argument("--compare-sampling", help="plot metrics for workload with and without sampling", action="store_true")
    parser.add_argument("--save-plot", help="save figure", action="store_true")
    args = parser.parse_args()
    if (args.all_workloads):
        plot_q_error_all_workloads(args.save_plot)
    elif (args.compare_sampling):
        plot_q_error_for_workload_compare_with_sampling(args.workload, args.plot_type, args.sub_plans, args.save_plot)
    else:
        plot_q_error_for_workload(args.workload, args.plot_type, args.sampling, args.sub_plans, args.save_plot)
    

if __name__ == "__main__":
    main()