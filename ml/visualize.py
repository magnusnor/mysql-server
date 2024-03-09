import os
import glob
import random
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

dataframes_folder_path = "./dataframes/"

sns.set_theme()
sns.color_palette("colorblind")

def save_plot(filename:str):
    images_folder_path = "./images/"
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path, exist_ok=True)
    plt.savefig(os.path.join(images_folder_path, filename))


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

    plt.tight_layout(pad=4.0)
    if save:
        save_plot("model_training.png")
    plt.show()

def main():
    plot_model_training()

if __name__ == "__main__":
    main()