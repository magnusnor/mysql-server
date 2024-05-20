import os
import pathlib
import json
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

DATAFRAMES_PATH = "./dataframes/"
ML_RESULTS_PATH = "./results/"
BENCHMARK_RESULTS_PATH = os.environ["BENCHMARK_RESULTS_PATH"]

sns.set_style("white")
sns.color_palette("deep")

preferred_hue_order = ["MySQL", "MSCN", "MSCN-sampling"]

def save_plot(filename:str):
    images_folder_path = "./images/"
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path, exist_ok=True)
    plt.savefig(os.path.join(images_folder_path, filename), format="pdf", bbox_inches="tight")


def calculate_q_error(estimated, real):
        return (estimated / real).combine(real / estimated, max)
    
def load_and_process_prediction(workload, suffix):
    filename = f"predictions_{workload}{suffix}.csv"
    file_path = os.path.join(ML_RESULTS_PATH, filename)
    df = pd.read_csv(file_path, names=["estimated", "real"])
    
    df["q-error"] = calculate_q_error(df["estimated"], df["real"])
    df["query"] = (df.index + 1).astype(str)

    return df


def combine_predictions(workload, sampling=False, sub_plans=False):
    suffixes = ['']
    
    if sampling:
        suffixes.append('-sampling')
    
    dfs = {}
    
    for suffix in suffixes:
        if sub_plans:
            dfs[f"MSCN{suffix}"] = load_and_process_prediction(workload, f"{suffix}-sub-queries")
        else:
            dfs[f"MSCN{suffix}"] = load_and_process_prediction(workload, suffix)
    
    df_baseline = get_mysql_q_error_for_workload(workload, sub_plans)
    dfs["MySQL"] = df_baseline

    df_combined = pd.DataFrame()
    
    for key, df in dfs.items():
        if df is not None:
            df["model"] = key
            df_combined = pd.concat([df_combined, df], ignore_index=True)
    
    return df_combined


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
    df = pd.read_pickle(os.path.join(DATAFRAMES_PATH, "model_training.pkl"))
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
    suffixes = []

    df_combined = combine_predictions(workload, sampling, sub_plans)
    distinct_models = df_combined['model'].unique()
    hue_order = [model for model in preferred_hue_order if model in distinct_models]
    df_combined["model"] = pd.Categorical(df_combined['model'], categories=hue_order, ordered=True)

    if (plot_type == "box"):
        sns.boxplot(data=df_combined, x="model", y="q-error", hue="model")
        plt.yscale("log")
        plt.xlabel("")
        plt.ylabel("Q-Error")
    elif (plot_type == "bar"):
        df_combined.sort_values(by="q-error", inplace=True, ascending=False)
        sns.barplot(data=df_combined, x="q-error", y="query", hue="model", dodge=False, alpha=0.7, edgecolor='black', orient="h")
        plt.xscale("log")
        plt.xlabel("Q-Error")
        plt.ylabel("Query")

    plt.tight_layout()

    if sub_plans:
        suffixes.append("-sub-queries")
    if sampling:
        suffixes.append("-compare-sampling")
    if plot_type == "bar":
        suffixes.append("-bar")

    suffix = ''.join(suffixes)
    
    if save:
        filename = f"q-error-{workload}{suffix}.pdf"
        save_plot(filename)
    else:
        plt.show()


def plot_q_error_all_workloads(save=False):
    dfs = {}
    for path in pathlib.Path(ML_RESULTS_PATH).glob("*.csv"):
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
    parser.add_argument("--model-training", help="plot model training", action="store_true")
    parser.add_argument("--all-workloads", help="plot metrics for all workloads", action="store_true")
    parser.add_argument("--sub-plans", help="use sub-plans", action="store_true")
    parser.add_argument("--compare-sampling", help="compare workload with and without materialized base samples", action="store_true")
    parser.add_argument("--save-plot", help="save figure", action="store_true")
    args = parser.parse_args()
    if (args.all_workloads):
        plot_q_error_all_workloads(args.save_plot)
    elif (args.model_training):
        plot_model_training(args.save_plot)
    else:
        plot_q_error_for_workload(args.workload, args.plot_type, args.compare_sampling, args.sub_plans, args.save_plot)
    

if __name__ == "__main__":
    main()