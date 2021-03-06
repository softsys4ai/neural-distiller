import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import config_reference as cfg
sns.set()

RESULTS_FILE = os.path.join(cfg.processed_csv_path, "logit_entropy_analysis_2020-02-13T02:12:19.743551.csv")
EXP = "ESKD_teacher"
CMAP = "gnuplot2"

INCLUDE_LOGIT_DIFF = False
INCLUDE_SOFT_DIFF = False
INCLUDE_EUCLID_DIST = False
INCLUDE_ENTROPY = True

if INCLUDE_LOGIT_DIFF or INCLUDE_SOFT_DIFF or INCLUDE_EUCLID_DIST:
    # plot logit difference heatmap
    df_plot = pd.read_csv(RESULTS_FILE)
    df = df_plot[df_plot.interval != 0]

    min_epoch = int(df['interval'].min())
    max_epoch = int(df['interval'].max())
    epoch_interval = 5
    epochs = np.arange(min_epoch, max_epoch+epoch_interval-1e-2, epoch_interval)
    print(df.head())
    if INCLUDE_LOGIT_DIFF:
        df_min = df['test_logit_diff'].min()
        df_max = df['test_logit_diff'].max()
        hm_data1 = pd.pivot_table(df_plot, values='test_logit_diff',
                                  index=['temp'],
                                  columns='interval')
        plot = sns.heatmap(hm_data1, cmap=CMAP, vmin=df_min, vmax=df_max)
        plot.invert_yaxis()
        plt.title("Teacher-Student Logit Difference w.r.t Temperature and Epoch Interval")
        fig = plot.get_figure()
        fig.savefig(os.path.join(cfg.figures_path, EXP+"_logit_heatplot.png"))
        plt.show()

    # plot soft difference heatmap
    if INCLUDE_SOFT_DIFF:
        df_min = df['test_soft_diff'].min()
        df_max = df['test_soft_diff'].max()
        hm_data1 = pd.pivot_table(df_plot, values='test_soft_diff',
                                  index=['temp'],
                                  columns='interval')
        plot = sns.heatmap(hm_data1, cmap=CMAP, vmin=df_min, vmax=df_max)
        plot.invert_yaxis()
        plt.title("Teacher-Student Soft Difference w.r.t Temperature and Epoch Interval")
        fig = plot.get_figure()
        fig.savefig(os.path.join(cfg.figures_path, EXP+"_soft_heatplot.png"))
        plt.show()



    # plot min, mean, max of euclidean distance data
    if INCLUDE_EUCLID_DIST:
        if INCLUDE_LOGIT_DIFF:
            # plot min, mean, max of logit differences
            max_logit_diffs = np.zeros(len(epochs))
            min_logit_diffs = np.zeros(len(epochs))
            mean_logit_diffs = np.zeros(len(epochs))
            for i in range(len(epochs)):
                temp_df = df[df.interval == epochs[int(i)]]
                max_logit_diffs[i] = temp_df['test_logit_diff'].max()
                min_logit_diffs[i] = temp_df['test_logit_diff'].min()
                mean_logit_diffs[i] = temp_df['test_logit_diff'].mean()

            plt.plot(epochs, max_logit_diffs, label="Max test distance", color="r")
            plt.plot(epochs, mean_logit_diffs, label="Mean test distance", color="k")
            plt.plot(epochs, min_logit_diffs, label="Min test distance", color="b")
            # plt.hlines(0., min_epoch, max_epoch, colors="r", linestyles="dashed", label="Baseline Max test AR Accuracy")
            # plt.hlines(0., min_epoch, max_epoch, colors='k', linestyles="dashed", label="Baseline Mean test AR Accuracy")
            plt.xlabel("Epoch Interval")
            epochs = np.arange(0, max_epoch+epoch_interval-1e-2, epoch_interval)
            plt.xticks(epochs)
            plt.ylabel("Euclidean Distance")
            plt.legend(loc="lower right")
            # plt.legend(bbox_to_anchor=(1.0, 1.0))
            plt.title("Teacher-Student Logit Difference w.r.t Temperature and Epoch Interval")
            plt.savefig(os.path.join(cfg.figures_path, EXP+"_logit_lineplot.png"))
            plt.show()

        if INCLUDE_SOFT_DIFF:
            max_soft_diffs = np.zeros(len(epochs))
            min_soft_diffs = np.zeros(len(epochs))
            mean_soft_diffs_train = np.zeros(len(epochs))
            for i in range(len(epochs)):
                temp_df = df[df.interval == epochs[int(i)]]
                max_soft_diffs[i] = temp_df['test_soft_diff'].max()
                min_soft_diffs[i] = temp_df['test_soft_diff'].min()
                mean_soft_diffs_train[i] = temp_df['test_soft_diff'].mean()

            # plot min, mean, max of soft differences
            plt.figure(0)
            plt.plot(epochs, max_soft_diffs, label="Max test distance", color="r")
            plt.plot(epochs, mean_soft_diffs_train, label="Mean test distance", color="k")
            plt.plot(epochs, min_soft_diffs, label="Min test distance", color="b")
            # plt.hlines(0., min_epoch, max_epoch, colors="r", linestyles="dashed", label="Baseline Max test AR Accuracy")
            # plt.hlines(0., min_epoch, max_epoch, colors='k', linestyles="dashed", label="Baseline Mean test AR Accuracy")
            plt.xlabel("Epoch Interval")
            epochs = np.arange(0, max_epoch+epoch_interval-1e-2, epoch_interval)
            plt.xticks(epochs)
            plt.ylabel("Euclidean Distance")
            plt.legend(loc="lower right")
            # plt.legend(bbox_to_anchor=(1.0, 1.0))
            plt.title("Teacher-Student Soft Difference w.r.t Temperature and Epoch Interval")
            plt.savefig(os.path.join(cfg.figures_path, EXP+"_soft_lineplot.png"))
            plt.show()

if INCLUDE_ENTROPY:
    ENTROPY_RESULTS_FILE = os.path.join(cfg.processed_csv_path, "logit_entropy_analysis_2020-02-13T02:12:19.743551.csv")
    df_entropy_plot = pd.read_csv(ENTROPY_RESULTS_FILE)
    df_entropy_plot = df_entropy_plot[df_entropy_plot.teacher_entropy_intervals != 0]
    entropy_plot = sns.lineplot(x="teacher_entropy_intervals", y="avg_train_entropy", legend='brief', label="train_entropy", data=df_entropy_plot)
    entropy_plot = sns.lineplot(x="teacher_entropy_intervals", y="avg_test_entropy", legend='brief', label="test_entropy", data=df_entropy_plot)
    plt.xlabel("Epoch Interval")
    plt.ylabel("Entropy")
    plt.legend(loc="upper right")
    # plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.title("Teacher Logit Entropy w.r.t Epoch Interval")
    # fixing dense labels
    # for ind, label in enumerate(entropy_plot.get_xticklabels()):
    #     if ind % 5 == 0:
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)
    plt.savefig(os.path.join(cfg.figures_path, EXP+"_entropy.png"))
    plt.show()
















