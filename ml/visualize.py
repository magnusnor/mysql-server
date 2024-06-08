import os
import pathlib
import json
import argparse
import math
import re

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

DATAFRAMES_PATH = "dataframes/"
ML_RESULTS_PATH = "results/"
BENCHMARK_RESULTS_PATH = os.environ["BENCHMARK_RESULTS_PATH"]

sns.set_style("white")
sns.color_palette("deep")

def save_plot(filename:str):
    images_folder_path = "./images/"
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path, exist_ok=True)
    plt.savefig(os.path.join(images_folder_path, filename), format="pdf", bbox_inches="tight")

def calculate_q_error(estimated, real):
        return (estimated / real).combine(real / estimated, max)


def custom_sort(key):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', key)]

def split_df(data_df, cols, rows):
    splits = math.ceil(len(data_df) / 20)

    total_rows = len(data_df)
    rows_per_part = math.ceil(total_rows / splits)

    split_dfs = [
        data_df.iloc[i * rows_per_part : (i + 1) * rows_per_part] for i in range(splits)
    ]

    num_rows = math.ceil(splits / rows)
    num_cols = min(splits, cols)
    return split_dfs, num_rows, num_cols

def find_keys(data, search_keys):
    result = []
    if isinstance(data, dict):
        if data.get('access_type') == 'join':
            found = {key: data[key] for key in search_keys if key in data}
            if found:
                result.append(found)
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result.extend(find_keys(value, search_keys))
    elif isinstance(data, list):
        for item in data:
            result.extend(find_keys(item, search_keys))
    return result

def get_mscn_predictions_dfs(workload):
    files = [
        f"predictions_{workload}.csv",
        f"predictions_{workload}-sampling.csv",
    ]
    labels = [
        "MSCN",
        "MSCN-original",
    ]
    
    dfs = []
    for file, label in zip(files, labels):
        file_path = os.path.join(ML_RESULTS_PATH, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, names=["estimated", "real"])
            df["q-error"] = calculate_q_error(df["estimated"], df["real"])
            df["query"] = (df.index + 1).astype(str)
            df["source"] = label
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def get_mysql_df(workload):
    results_folder_path = f"{BENCHMARK_RESULTS_PATH}/{workload}/baseline"
    if (not os.path.exists(results_folder_path)):
        return None
    
    mscn_predictions = get_mscn_predictions_dfs(workload)
    
    data = []
    for path in pathlib.Path(results_folder_path).glob("*.json"):
        with open(str(path)) as f:
            name = str(path.stem)
            file_content = f.read()
            json_start_index = file_content.find('EXPLAIN: {')

            true_cardinality = mscn_predictions[mscn_predictions["query"] == name]["real"].unique()[0]

            if json_start_index != -1:
                json_content = file_content[json_start_index + len('EXPLAIN: '):]
                json_data = json.loads(json_content)
                json_data["query"] = path.stem
                if (json_data["inputs"][0]["access_type"] == "join"):
                    estimated_cardinality = json_data["inputs"][0]["estimated_rows"]
                    df = pd.DataFrame({"query": name, "estimated": [estimated_cardinality], "real": [true_cardinality]})
                    df["q-error"] = calculate_q_error(df['estimated'], df['real'])
                    df.drop(columns=["estimated", "real"], inplace=True)
                    data.append(df)
    dfs = [
        query for query in data
    ]
    return pd.concat(dfs, ignore_index=True)

def get_mysql_df_sub_plans(workload):
    results_folder_path = f"{BENCHMARK_RESULTS_PATH}/{workload}/baseline_plans"
    if (not os.path.exists(results_folder_path)):
        return None
    
    data = []
    for path in pathlib.Path(results_folder_path).glob("*.json"):
        with open(str(path)) as f:
            file_content = f.read()
            json_start_index = file_content.find('EXPLAIN: {')

            if json_start_index != -1:
                json_content = file_content[json_start_index + len('EXPLAIN: '):]
                json_data = json.loads(json_content)
                json_data["query"] = path.stem
                data.append(json_data)
    dfs = [
        pd.json_normalize(query) for query in data
    ]
    return pd.concat(dfs, ignore_index=True)

def get_mysql_plan_q_errors_df(workload):
    df = get_mysql_df_sub_plans(workload)

    search_keys = ["actual_rows", "estimated_rows_ml", "estimated_rows_original"]
    query_plan = df["inputs"].apply(lambda x: find_keys(x, search_keys))

    def extract_q_errors(found_list):
        extracted_values = {
            'estimated_rows_ml': [],
            'estimated_rows_original': [],
            'actual_rows': [],
        }
        q_errors = {
            'q_error_ml': [],
            'q_error_baseline': []
        }
        
        for found_dict in found_list:
            if 'actual_rows' in found_dict:
                extracted_values['actual_rows'].append(found_dict['actual_rows'])
            if 'estimated_rows_ml' in found_dict:
                extracted_values['estimated_rows_ml'].append(found_dict['estimated_rows_ml'])
            if 'estimated_rows_original' in found_dict:
                extracted_values['estimated_rows_original'].append(found_dict['estimated_rows_original'])

        for actual, ml, baseline in zip(extracted_values['actual_rows'], extracted_values['estimated_rows_ml'], extracted_values['estimated_rows_original']):
            q_error_ml = max(actual / ml, ml / actual) if ml else None
            q_error_original = max(actual / baseline, baseline / actual) if baseline else None
            q_errors['q_error_ml'].append(q_error_ml)
            q_errors['q_error_baseline'].append(q_error_original)
            
        return q_errors

    df['q_errors'] = query_plan.apply(extract_q_errors)

    df['MySQL'] = df['q_errors'].apply(lambda x: x['q_error_baseline'])
    df['MSCN'] = df['q_errors'].apply(lambda x: x['q_error_ml'])

    expanded_data = []
    for i, row in df.iterrows():
        query = row['query']
        for level, q_error in enumerate(row['MySQL'], start=1):
            expanded_data.append([query, 'MySQL', level, q_error])
        for level, q_error in enumerate(row['MSCN'], start=1):
            expanded_data.append([query, 'MSCN', level, q_error])

    return pd.DataFrame(expanded_data, columns=['Query', 'Model', 'Level', 'Q-Error']).sort_values(by="Query")


def get_benchmark_timing_dfs(workload):
    results_folder_path = f"{BENCHMARK_RESULTS_PATH}/{workload}/benchmark"
    if not os.path.exists(results_folder_path):
        return None
    data = []
    for path in pathlib.Path(results_folder_path).glob("*.json"):
        with open(str(path)) as f:
            json_data = json.load(f)
            json_data["query"] = path.stem
            data.append(json_data)
    
    dfs = [
        pd.json_normalize(query["results"], sep="_").assign(
            query=query["query"]
        )
        for query in data
    ]
    return pd.concat(dfs, ignore_index=True)

def plot_model_training(save=False):
    df = pd.read_pickle(os.path.join(DATAFRAMES_PATH, "model_training.pkl"))
    df["epochs"] = range(1, len(df) + 1)
    df_melted = df.melt(id_vars=["epochs"], value_vars=["training_loss", "validation_loss"], 
                    var_name="Loss Type", value_name="Loss")
    df_melted["Loss Type"] = df_melted["Loss Type"].map({"training_loss": "Training", "validation_loss": "Validation"})
    sns.lineplot(data=df_melted, x="epochs", y="Loss", hue="Loss Type", style="Loss Type")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Q-Error")
    plt.tight_layout()
    if save:
        save_plot("model_training.pdf")
        plt.close()
    else:
        plt.show()

def plot_total_q_error(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))
    melted_df = pd.melt(new_df, id_vars=["query"], 
                value_vars=["q-error_mysql", "q-error_mscn"],
                var_name="model", value_name="q-error")

    source_replacements = {
        "q-error_mysql": "MySQL",
        "q-error_mscn": "MSCN"
    }
    melted_df['model'] = melted_df['model'].replace(source_replacements)

    sns.boxplot(data=melted_df, x="model", y="q-error", hue="model")
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Q-Error")

    plt.tight_layout()

    if save:
        filename = f"total-q-error-{workload}.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_total_q_error_compare_original(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_original = dfs[dfs["source"] == "MSCN-original"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))
    df_original = df_original.rename(columns=lambda x: f"{x}_mscn_original" if x != "query" else x)
    new_df = pd.merge(new_df, df_original, on="query", suffixes=("", "_mscn_original"))
    melted_df = pd.melt(new_df, id_vars=["query"], 
                value_vars=["q-error_mysql", "q-error_mscn", "q-error_mscn_original"],
                var_name="model", value_name="q-error")

    source_replacements = {
        "q-error_mysql": "MySQL",
        "q-error_mscn": "MSCN",
        "q-error_mscn_original": "MSCN-original"
    }
    melted_df['model'] = melted_df['model'].replace(source_replacements)

    sns.boxplot(data=melted_df, x="model", y="q-error", hue="model")
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Q-Error")

    plt.tight_layout()

    if save:
        filename = f"total-q-error-{workload}-compare-original.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_per_query_no_split_sort_by_query(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))

    sorted_df = new_df.sort_values(by="query", key=lambda x: x.apply(custom_sort))

    plt.figure(figsize=(10, 10))

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=True,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=True,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    bar_mscn.set_xlabel("Q-Error")
    bar_mscn.set_ylabel("Query")
    bar_mscn.set_xscale("log")

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-per-query-no-split-sort-by-query.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_per_query_no_split_sort_by_mysql(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))

    sorted_df = new_df.sort_values(by="q-error_mysql")

    plt.figure(figsize=(10, 10))

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=True,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=True,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    bar_mscn.set_xlabel("Q-Error")
    bar_mscn.set_ylabel("Query")
    bar_mscn.set_xscale("log")

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-per-query-no-split-sort-by-mysql.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_per_query_no_split_sort_by_mscn(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))

    sorted_df = new_df.sort_values(by="q-error_mscn")

    plt.figure(figsize=(10, 10))

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=True,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=True,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    bar_mscn.set_xlabel("Q-Error")
    bar_mscn.set_ylabel("Query")
    bar_mscn.set_xscale("log")

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-per-query-no-split-sort-by-mscn.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_per_query_split(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))

    sorted_df = new_df.sort_values(by="q-error_mysql")

    split_dfs, num_rows, num_cols = split_df(sorted_df, 2, 2)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()
    for part_df, ax in zip(split_dfs, axes):
        bar_mysql = sns.barplot(
            x="q-error_mysql",
            y="query",
            data=part_df,
            legend=False,
            label="MySQL",
            edgecolor="black",
            linestyle="dotted",
            ax=ax,
        )

        bar_mscn = sns.barplot(
            x="q-error_mscn",
            y="query",
            data=part_df,
            legend=False,
            label="MSCN",
            alpha=0.7,
            edgecolor="black",
            ax=ax,
        )

        bar_mscn.set_xscale("log")
        bar_mscn.set_xlabel("Q-Error")
        bar_mscn.set_ylabel("Query")

        handles, labels = axes[0].get_legend_handles_labels()
        unique_labels = list(set(labels))
        combined_handles = [handles[labels.index(label)] for label in unique_labels]
        fig.legend(
            combined_handles,
            unique_labels,
            ncol=2,
            loc="lower center",
            frameon=True,
        )

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-per-query-split.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_per_query_no_split_compare_original(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_original = dfs[dfs["source"] == "MSCN-original"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))
    df_original = df_original.rename(columns=lambda x: f"{x}_mscn_original" if x != "query" else x)
    new_df = pd.merge(new_df, df_original, on="query", suffixes=("", "_mscn_original"))

    sorted_df = new_df.sort_values(by="query", key=lambda x: x.apply(custom_sort))

    plt.figure(figsize=(10, 10))

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=True,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=True,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    bar_mscn_original = sns.barplot(
        x="q-error_mscn_original",
        y="query",
        data=sorted_df,
        legend=True,
        label="MSCN-original",
        alpha=0.7,
        edgecolor="black",
    )

    bar_mscn_original.set_xlabel("Q-Error")
    bar_mscn_original.set_ylabel("Query")
    bar_mscn_original.set_xscale("log")

    sns.despine(left=True, bottom=True)

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-per-query-no-split-compare-original.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_per_query_split_compare_original(workload, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_original = dfs[dfs["source"] == "MSCN-original"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))
    df_original = df_original.rename(columns=lambda x: f"{x}_mscn_original" if x != "query" else x)
    new_df = pd.merge(new_df, df_original, on="query", suffixes=("", "_mscn_original"))

    sorted_df = new_df.sort_values(by="q-error_mysql")

    split_dfs, num_rows, num_cols = split_df(sorted_df, 2, 2)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()
    for part_df, ax in zip(split_dfs, axes):
        bar_mysql = sns.barplot(
            x="q-error_mysql",
            y="query",
            data=part_df,
            legend=False,
            label="MySQL",
            edgecolor="black",
            linestyle="dotted",
            ax=ax
        )

        bar_mscn = sns.barplot(
            x="q-error_mscn",
            y="query",
            data=part_df,
            legend=False,
            label="MSCN",
            alpha=0.7,
            edgecolor="black",
            ax=ax,
        )

        bar_mscn_original = sns.barplot(
            x="q-error_mscn_original",
            y="query",
            data=part_df,
            legend=False,
            label="MSCN-original",
            alpha=0.7,
            edgecolor="black",
            ax=ax,
        )

        bar_mscn_original.set_xscale("log")
        bar_mscn_original.set_xlabel("Q-Error")
        bar_mscn_original.set_ylabel("Query")

        handles, labels = axes[0].get_legend_handles_labels()
        unique_labels = list(set(labels))
        combined_handles = [handles[labels.index(label)] for label in unique_labels]
        fig.legend(
            combined_handles,
            unique_labels,
            ncol=2,
            loc="lower center",
            frameon=True,
            bbox_to_anchor=(0.5, -0.05)
        )

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-per-query-split-compare-original.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_top_n_best_queries(workload, n, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))

    sorted_df = new_df.sort_values(by="q-error_mscn").head(n)
    sorted_df_mysql = new_df.sort_values(by="q-error_mysql").head(n)

    print(f"Sorted by MSCN: {sorted_df.round({'q-error_mysql': 3, 'q-error_mscn': 3})}")
    print(f"Sorted by MySQL: {sorted_df_mysql.round({'q-error_mysql': 3, 'q-error_mscn': 3})}")

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=False,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=False,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    handles, labels = bar_mysql.get_legend_handles_labels()

    plt.legend(handles, labels, loc='upper right', title='Model', bbox_to_anchor=(1.25, 1))

    plt.xlabel("Q-Error")
    plt.ylabel("Query")
    plt.xscale("log")

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-top-{n}-best-queries.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_top_n_best_queries_relative(workload, n, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    df_merged = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))
    df_merged["relative_difference"] = (df_merged["q-error_mscn"] - df_merged["q-error_mysql"]).abs() / df_merged["q-error_mscn"]

    sorted_df = df_merged.sort_values(by="relative_difference", ascending=False).head(n)

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=False,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=False,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    handles, labels = bar_mysql.get_legend_handles_labels()

    plt.legend(handles, labels, loc='upper right', title='Model', bbox_to_anchor=(1.25, 1))

    plt.xlabel("Q-Error")
    plt.ylabel("Query")
    plt.xscale("log")

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-top-{n}-best-queries-relative.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_top_n_worst_queries(workload, n, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    new_df = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))

    sorted_df = new_df.sort_values(by="q-error_mscn", ascending=False).head(n)
    sorted_df_mysql = new_df.sort_values(by="q-error_mysql", ascending=False).head(n)

    print(f"Sorted by MSCN: {sorted_df.round({'q-error_mysql': 3, 'q-error_mscn': 3})}")
    print(f"Sorted by MySQL: {sorted_df_mysql.round({'q-error_mysql': 3, 'q-error_mscn': 3})}")

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=False,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=False,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    handles, labels = bar_mysql.get_legend_handles_labels()

    plt.legend(handles, labels, loc='upper right', title='Model', bbox_to_anchor=(1.25, 1))

    plt.xlabel("Q-Error")
    plt.ylabel("Query")
    plt.xscale("log")

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-top-{n}-worst-queries.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_top_n_worst_queries_relative(workload, n, save=False):
    dfs = get_mscn_predictions_dfs(workload)
    df = dfs[dfs["source"] == "MSCN"]
    df_mysql = get_mysql_df(workload)

    df_merged = pd.merge(df_mysql, df, on="query", suffixes=("_mysql", "_mscn"))
    df_merged["relative_difference"] = (df_merged["q-error_mscn"] - df_merged["q-error_mysql"]).abs() / df_merged["q-error_mysql"]

    sorted_df = df_merged.sort_values(by="relative_difference", ascending=False).head(n)

    bar_mysql = sns.barplot(
        x="q-error_mysql",
        y="query",
        data=sorted_df,
        legend=False,
        label="MySQL",
        edgecolor="black",
        linestyle="dotted",
    )

    bar_mscn = sns.barplot(
        x="q-error_mscn",
        y="query",
        data=sorted_df,
        legend=False,
        label="MSCN",
        alpha=0.7,
        edgecolor="black",
    )

    handles, labels = bar_mysql.get_legend_handles_labels()

    plt.legend(handles, labels, loc='upper right', title='Model', bbox_to_anchor=(1.25, 1))

    plt.xlabel("Q-Error")
    plt.ylabel("Query")
    plt.xscale("log")

    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-top-{n}-worst-queries-relative.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_sub_plans_per_query(workload, save=False):

    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))

    unique_queries = df['Query'].unique()
    
    def plot_queries(queries, df):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        axes = axes.flatten()

        hue_order = ['MySQL', 'MSCN']

        for ax, query in zip(axes, queries):
            query_data = df[df['Query'] == query]
            sns.barplot(data=query_data, x="Level", y="Q-Error", hue="Model", hue_order=hue_order, ax=ax)
            ax.set_yscale("log")
            ax.set_title(f"{query}.sql", fontsize=10)
            ax.set_xlabel('Level')
            ax.set_ylabel('Q-Error')
            ax.legend().remove()
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=True, bbox_to_anchor=(0.5, -0.05))

        for i in range(len(queries), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            filename = f"q-error-{workload}-sub-plans-queries-{queries[0]}-to-{queries[-1]}.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

    chunks_queries = [unique_queries[i:i + 9] for i in range(0, len(unique_queries), 9)]

    for chunk in chunks_queries:
        plot_queries(chunk, df)

def plot_q_error_sub_plans_top_n_best_queries(workload, n, save=False):
    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))
    aggregate_q_error = df.groupby(['Query', 'Model'])['Q-Error'].median().reset_index()
    pivot_df = aggregate_q_error.pivot(index='Query', columns='Model', values='Q-Error').reset_index()
    filtered_queries = pivot_df[pivot_df['MSCN'] < pivot_df['MySQL']]
    sorted_filtered_queries = filtered_queries.sort_values(by='MSCN')
    top_n_queries = sorted_filtered_queries.head(n)['Query']
    top_n_entries = df[df['Query'].isin(top_n_queries)]

    unique_queries = top_n_entries['Query'].unique()
    
    def plot_queries(queries, df):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        axes = axes.flatten()

        hue_order = ['MySQL', 'MSCN']

        for ax, query in zip(axes, queries):
            query_data = df[df['Query'] == query]
            sns.barplot(data=query_data, x="Level", y="Q-Error", hue="Model", hue_order=hue_order, ax=ax)
            ax.set_yscale("log")
            ax.set_title(f"{query}.sql", fontsize=10)
            ax.set_xlabel('Level')
            ax.set_ylabel('Q-Error')
            ax.legend().remove()
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=True, bbox_to_anchor=(0.5, -0.05))

        for i in range(len(queries), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            filename = f"q-error-{workload}-sub-plans-top-{n}-best-queries.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

    chunks_queries = [unique_queries[i:i + 9] for i in range(0, len(unique_queries), 9)]

    for chunk in chunks_queries:
        plot_queries(chunk, df)

def plot_q_error_sub_plans_top_n_worst_queries(workload, n, save=False):
    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))
    aggregate_q_error = df.groupby(['Query', 'Model'])['Q-Error'].median().reset_index()
    pivot_df = aggregate_q_error.pivot(index='Query', columns='Model', values='Q-Error').reset_index()
    filtered_queries = pivot_df[pivot_df['MSCN'] > pivot_df['MySQL']]
    sorted_filtered_queries = filtered_queries.sort_values(by='MSCN')
    top_n_queries = sorted_filtered_queries.head(n)['Query']
    top_n_entries = df[df['Query'].isin(top_n_queries)]

    unique_queries = top_n_entries['Query'].unique()
    
    def plot_queries(queries, df):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        axes = axes.flatten()

        hue_order = ['MySQL', 'MSCN']

        for ax, query in zip(axes, queries):
            query_data = df[df['Query'] == query]
            sns.barplot(data=query_data, x="Level", y="Q-Error", hue="Model", hue_order=hue_order, ax=ax)
            ax.set_yscale("log")
            ax.set_title(f"{query}.sql", fontsize=10)
            ax.set_xlabel('Level')
            ax.set_ylabel('Q-Error')
            ax.legend().remove()
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=True, bbox_to_anchor=(0.5, -0.05))

        for i in range(len(queries), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            filename = f"q-error-{workload}-sub-plans-top-{n}-worst-queries.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

    chunks_queries = [unique_queries[i:i + 9] for i in range(0, len(unique_queries), 9)]

    for chunk in chunks_queries:
        plot_queries(chunk, df)

def plot_total_q_error_sub_plans_levels(workload, save=False):

    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))

    df = df.sort_values(by=["Level"])
    unique_levels = df['Level'].unique()

    def plot_levels(levels, df):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes = axes.flatten()

        hue_order = ['MySQL', 'MSCN']
        order = ['MySQL', 'MSCN']

        for ax, level in zip(axes, levels):
            level_data = df[df['Level'] == level]
            sns.boxplot(data=level_data, x="Model", y="Q-Error", hue="Model", legend=False, hue_order=hue_order, order=order, ax=ax)
            ax.set_yscale('log')
            ax.set_title(f"Level: {level}", fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Q-Error')

        for i in range(len(levels), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            filename = f"total-q-error-{workload}-sub-plans-levels.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

    chunks_levels = [unique_levels[i:i + 4] for i in range(0, len(unique_levels), 4)]

    for chunk in chunks_levels:
        plot_levels(chunk, df)

def plot_q_error_sub_plans_levels_per_query(workload, save=False):
    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))

    df = df.sort_values(by=["Level"])
    unique_levels = df['Level'].unique()

    def plot_levels(levels, df):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes = axes.flatten()

        hue_order = ['MySQL', 'MSCN']

        for ax, level in zip(axes, levels):
            level_data = df[df['Level'] == level]
            sns.barplot(data=level_data, x="Query", y="Q-Error", hue="Model", hue_order=hue_order, ax=ax)
            ax.set_yscale('log')
            ax.set_title(f"Level: {level}", fontsize=10)
            ax.set_xlabel('Query')
            ax.set_ylabel('Q-Error')
            ax.legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="lower center", frameon=True)

        for i in range(len(levels), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save:
            filename = f"q-error-{workload}-sub-plans-levels-per-query.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

    chunks_levels = [unique_levels[i:i + 4] for i in range(0, len(unique_levels), 4)]

    for chunk in chunks_levels:
        plot_levels(chunk, df)

def plot_q_error_sub_plans_levels_for_query(workload, query, save=False):
    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))

    df = df.sort_values(by=["Level"])

    hue_order = ['MySQL', 'MSCN']

    query_data = df[df['Query'] == str(query)]

    sns.barplot(data=query_data, x="Level", y="Q-Error", hue="Model", hue_order=hue_order)
    plt.yscale("log")
    plt.title(f"{query}.sql", fontsize=10)
    plt.xlabel("Level")
    plt.ylabel("Q-Error")
    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-sub-plans-levels-for-query-{query}.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_sub_plans_level_trend(workload, save=False):

    df = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))

    hue_order = ["MySQL", "MSCN"]

    sns.lineplot(data=df, x="Level", y="Q-Error", hue="Model", style="Model", markers=['o', 's'], hue_order=hue_order)
    plt.yscale("log")
    plt.tight_layout()
    if save:
        filename = f"q-error-{workload}-sub-plans-levels-trend.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_q_error_correlation(workload, save=False):

    df_subplans = get_mysql_plan_q_errors_df(workload).sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))
    df_subplans_mscn = df_subplans[df_subplans["Model"] == "MSCN"]
    df_subplans_mscn = df_subplans_mscn.groupby(['Query'], as_index=False)["Q-Error"].median()
    df_mscn = get_mscn_predictions_dfs(workload)
    df_mscn = df_mscn[df_mscn["source"] == "MSCN"]
    df_mscn.rename(columns={"query": "Query", "q-error": "Q-Error"}, inplace=True)

    df_merged = pd.merge(df_subplans_mscn, df_mscn, on='Query', suffixes=("_subplans", "_plan"))
    df_melted = pd.melt(df_merged, id_vars=["Query"], 
                value_vars=["Q-Error_subplans", "Q-Error_plan"],
                var_name="Type", value_name="Q-Error")
    
    df_sorted = df_merged.sort_values(by=["Query"], key=lambda x: x.apply(custom_sort))

    plt.figure(figsize=(10, 10))

    corr = sns.lmplot(x='Q-Error_subplans', y='Q-Error_plan', data=df_sorted)
    
    corr.set_xlabels("Subplans")
    corr.set_ylabels("Plan")

    plt.xscale("log")
    plt.yscale("log")
    
    plt.tight_layout()

    if save:
        filename = f"q-error-{workload}-correlation.pdf"
        save_plot(filename)
        plt.close()
    else:
        plt.show()

def plot_exec_time_no_split(workload, save=False):
    dfs = get_benchmark_timing_dfs(workload)
    if dfs is not None:
        df_without = dfs[dfs["command"] == "without_ml"]
        df_with = dfs[dfs["command"] == "with_ml"]

        df_with = df_with.sort_values(by=["query"])
        df_without = df_without.sort_values(
            by=["query"]
        )
        new_df = pd.merge(df_with, df_without, on="query", suffixes=("_with", "_without"))

        sorted_df = new_df.sort_values(by="median_without")

        plt.figure(figsize=(10, 10))

        bar_without = sns.barplot(
            x="median_without",
            y="query",
            data=sorted_df,
            legend=True,
            label="Without",
            edgecolor="black",
            linestyle="dotted",
        )

        bar_with = sns.barplot(
            x="median_with",
            y="query",
            data=sorted_df,
            legend=True,
            label="With",
            alpha=0.7,
            edgecolor="black",
        )
        bar_with.set_xlabel("Median seconds")
        bar_with.set_ylabel("Query")

        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        if save:
            filename = f"exec-time-{workload}-no-split.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

def plot_exec_time_split(workload, save=False):
    dfs = get_benchmark_timing_dfs(workload)
    if dfs is not None:
        df_without = dfs[dfs["command"] == "without_ml"]
        df_with = dfs[dfs["command"] == "with_ml"]

        df_with = df_with.sort_values(by=["query"])
        df_without = df_without.sort_values(
            by=["query"]
        )
        new_df = pd.merge(df_with, df_without, on="query", suffixes=("_with", "_without"))

        sorted_df = new_df.sort_values(by="median_without")

        split_dfs, num_rows, num_cols = split_df(sorted_df, 2, 2)

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        for part_df, ax in zip(split_dfs, axes):
            bar_without = sns.barplot(
                x="median_without",
                y="query",
                data=part_df,
                legend=False,
                label="Without",
                edgecolor="black",
                linestyle="dotted",
                ax=ax,
            )

            bar_with = sns.barplot(
                x="median_with",
                y="query",
                data=part_df,
                legend=False,
                label="With",
                alpha=0.7,
                edgecolor="black",
                ax=ax, 
            )
            bar_with.set_xlabel("Median seconds")
            bar_with.set_ylabel("Query")

            handles, labels = axes[0].get_legend_handles_labels()
            unique_labels = list(set(labels))
            combined_handles = [handles[labels.index(label)] for label in unique_labels]
            fig.legend(
                combined_handles,
                unique_labels,
                ncol=2,
                loc="lower center",
                frameon=True,
            )

        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        if save:
            filename = f"exec-time-{workload}-split.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

def plot_exec_time_top_n_fastest_queries(workload, n, save=False):
    dfs = get_benchmark_timing_dfs(workload)
    if dfs is not None:
        df_without = dfs[dfs["command"] == "without_ml"]
        df_with = dfs[dfs["command"] == "with_ml"]

        df_with = df_with.sort_values(by=["query"])
        df_without = df_without.sort_values(
            by=["query"]
        )
        new_df = pd.merge(df_with, df_without, on="query", suffixes=("_with", "_without"))

        sorted_df = new_df.nsmallest(n, "median_with")

        bar_without = sns.barplot(
            x="median_without",
            y="query",
            data=sorted_df,
            legend=False,
            label="Without",
            edgecolor="black",
            linestyle="dotted",
        )

        bar_with = sns.barplot(
            x="median_with",
            y="query",
            data=sorted_df,
            legend=False,
            label="With",
            alpha=0.7,
            edgecolor="black",
        )

        bar_with.set_xlabel("Median seconds")
        bar_with.set_ylabel("Query")

        handles, labels = bar_with.get_legend_handles_labels()

        plt.legend(handles, labels, loc='upper right', title='Command', bbox_to_anchor=(1.3, 1))

        plt.tight_layout()

        if save:
            filename = f"exec-time-{workload}-top-{n}-fastest-queries.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

def plot_exec_time_top_n_slowest_queries(workload, n, save=False):
    dfs = get_benchmark_timing_dfs(workload)
    if dfs is not None:
        df_without = dfs[dfs["command"] == "without_ml"]
        df_with = dfs[dfs["command"] == "with_ml"]

        df_with = df_with.sort_values(by=["query"])
        df_without = df_without.sort_values(
            by=["query"]
        )
        new_df = pd.merge(df_with, df_without, on="query", suffixes=("_with", "_without"))

        sorted_df = new_df.nlargest(n, "median_with")

        bar_without = sns.barplot(
            x="median_without",
            y="query",
            data=sorted_df,
            legend=False,
            label="Without",
            edgecolor="black",
            linestyle="dotted",
        )

        bar_with = sns.barplot(
            x="median_with",
            y="query",
            data=sorted_df,
            legend=False,
            label="With",
            alpha=0.7,
            edgecolor="black",
        )

        bar_with.set_xlabel("Median seconds")
        bar_with.set_ylabel("Query")

        handles, labels = bar_with.get_legend_handles_labels()

        plt.legend(handles, labels, loc='upper right', title='Command', bbox_to_anchor=(1.3, 1))

        plt.tight_layout()

        if save:
            filename = f"exec-time-{workload}-top-{n}-slowest-queries.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

def plot_exec_time_top_n_fastest_queries_relative(workload, n, save=False):
    dfs = get_benchmark_timing_dfs(workload)
    if dfs is not None:
        df_without = dfs[dfs["command"] == "without_ml"]
        df_with = dfs[dfs["command"] == "with_ml"]

        df_with = df_with.sort_values(by=["query"])
        df_without = df_without.sort_values(
            by=["query"]
        )
        new_df = pd.merge(df_with, df_without, on="query", suffixes=("_with", "_without"))
        new_df['absolute_difference'] = new_df['median_with'] - new_df['median_without']
        new_df['relative_difference'] = (new_df['absolute_difference'] / new_df['median_without'])

        sorted_df = new_df.nsmallest(n, "relative_difference")

        bar_without = sns.barplot(
            x="median_without",
            y="query",
            data=sorted_df,
            legend=False,
            label="Without",
            edgecolor="black",
            linestyle="dotted",
        )

        bar_with = sns.barplot(
            x="median_with",
            y="query",
            data=sorted_df,
            legend=False,
            label="With",
            alpha=0.7,
            edgecolor="black",
        )

        bar_with.set_xlabel("Median seconds")
        bar_with.set_ylabel("Query")

        handles, labels = bar_with.get_legend_handles_labels()

        plt.legend(handles, labels, loc='upper right', title='Command', bbox_to_anchor=(1.3, 1))

        plt.tight_layout()

        if save:
            filename = f"exec-time-{workload}-top-{n}-fastest-queries-relative.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

def plot_exec_time_top_n_slowest_queries_relative(workload, n, save=False):
    dfs = get_benchmark_timing_dfs(workload)
    if dfs is not None:
        df_without = dfs[dfs["command"] == "without_ml"]
        df_with = dfs[dfs["command"] == "with_ml"]

        df_with = df_with.sort_values(by=["query"])
        df_without = df_without.sort_values(
            by=["query"]
        )
        new_df = pd.merge(df_with, df_without, on="query", suffixes=("_with", "_without"))
        new_df['absolute_difference'] = new_df['median_with'] - new_df['median_without']
        new_df['relative_difference'] = (new_df['absolute_difference'] / new_df['median_without'])

        sorted_df = new_df.nlargest(n, "relative_difference")

        bar_without = sns.barplot(
            x="median_without",
            y="query",
            data=sorted_df,
            legend=False,
            label="Without",
            edgecolor="black",
            linestyle="dotted",
        )

        bar_with = sns.barplot(
            x="median_with",
            y="query",
            data=sorted_df,
            legend=False,
            label="With",
            alpha=0.7,
            edgecolor="black",
        )

        bar_with.set_xlabel("Median seconds")
        bar_with.set_ylabel("Query")

        handles, labels = bar_with.get_legend_handles_labels()

        plt.legend(handles, labels, loc='upper right', title='Command', bbox_to_anchor=(1.3, 1))

        plt.tight_layout()

        if save:
            filename = f"exec-time-{workload}-top-{n}-slowest-queries-relative.pdf"
            save_plot(filename)
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="save plots", action="store_true")
    args = parser.parse_args()
    save = args.save

    job_light = "job-light"
    job_light_sub_queries = "job-light-sub-queries"
    scale = "scale"
    synthetic = "synthetic"

    # Model Training
    plot_model_training(save)

    # JOB-light
    plot_total_q_error(job_light, save)
    plot_total_q_error_compare_original(job_light, save)
    plot_q_error_per_query_no_split_sort_by_query(job_light, save)
    plot_q_error_per_query_no_split_sort_by_mysql(job_light, save)
    plot_q_error_per_query_no_split_sort_by_mscn(job_light, save)
    plot_q_error_per_query_split(job_light, save)
    plot_q_error_per_query_no_split_compare_original(job_light, save)
    plot_q_error_per_query_split_compare_original(job_light, save)
    plot_q_error_top_n_best_queries(job_light, 10, save)
    plot_q_error_top_n_worst_queries(job_light, 10, save)
    plot_q_error_top_n_best_queries_relative(job_light, 10, save)
    plot_q_error_top_n_worst_queries_relative(job_light, 10, save)
    plot_q_error_sub_plans_top_n_best_queries(job_light, 9, save)
    plot_q_error_sub_plans_top_n_worst_queries(job_light, 9, save)
    plot_q_error_sub_plans_per_query(job_light, save)
    plot_total_q_error_sub_plans_levels(job_light, save)
    plot_q_error_sub_plans_levels_per_query(job_light, save)
    plot_q_error_sub_plans_level_trend(job_light, save)
    plot_q_error_correlation(job_light, save)
    plot_exec_time_no_split(job_light, save)
    plot_exec_time_split(job_light, save)
    plot_exec_time_top_n_fastest_queries(job_light, 10, save)
    plot_exec_time_top_n_slowest_queries(job_light, 10, save)
    plot_exec_time_top_n_fastest_queries_relative(job_light, 10, save)
    plot_exec_time_top_n_slowest_queries_relative(job_light, 10, save)

    # JOB-light sub-queries
    plot_total_q_error(job_light_sub_queries, save)
    plot_q_error_top_n_best_queries(job_light_sub_queries, 10, save)
    plot_q_error_top_n_worst_queries(job_light_sub_queries, 10, save)
    plot_q_error_top_n_best_queries_relative(job_light_sub_queries, 10, save)
    plot_q_error_top_n_worst_queries_relative(job_light_sub_queries, 10, save)

    # Scale
    plot_total_q_error(scale, save)
    plot_total_q_error_compare_original(scale, save)
    plot_q_error_top_n_best_queries(scale, 10, save)
    plot_q_error_top_n_worst_queries(scale, 10, save)
    plot_q_error_top_n_best_queries_relative(scale, 10, save)
    plot_q_error_top_n_worst_queries_relative(scale, 10, save)

    # Synthetic
    plot_total_q_error(synthetic, save)
    plot_total_q_error_compare_original(synthetic, save)
    plot_q_error_top_n_best_queries(synthetic, 10, save)
    plot_q_error_top_n_worst_queries(synthetic, 10, save)
    plot_q_error_top_n_best_queries_relative(synthetic, 10, save)
    plot_q_error_top_n_worst_queries_relative(synthetic, 10, save)

if __name__ == "__main__":
    main()