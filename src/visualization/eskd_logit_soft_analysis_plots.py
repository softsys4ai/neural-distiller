import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import config_reference as cfg
sns.set()

RESULTS_FILE = os.path.join(cfg.generated_csv_path, "logit_soft_analysis.csv")
EXP = "experiment2"
CMAP = "gnuplot2"

# plot logit difference heatmap
df_plot = pd.read_csv(RESULTS_FILE)
df = df_plot[df_plot.interval != 0]
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

# find max accuracy for each epoch interval
# find min accuracy for each epoch interval
min_epoch = 0
max_epoch = 100
epoch_interval = 5
epochs = np.arange(min_epoch, max_epoch+epoch_interval-1e-2, epoch_interval)

# plot min, mean, max of logit differences
max_logit_diffs = np.zeros(len(epochs))
min_logit_diffs = np.zeros(len(epochs))
mean_logit_diffs = np.zeros(len(epochs))
for i in range(len(epochs)):
    temp_df = df[df.interval == epochs[int(i)]]
    max_logit_diffs[i] = temp_df['test_logit_diff'].max()
    min_logit_diffs[i] = temp_df['test_logit_diff'].min()
    mean_logit_diffs[i] = temp_df['test_logit_diff'].mean()

# plot min, mean, max of adversarial data
plt.figure(0)
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


max_soft_diffs = np.zeros(len(epochs))
min_soft_diffs = np.zeros(len(epochs))
mean_soft_diffs = np.zeros(len(epochs))
for i in range(len(epochs)):
    temp_df = df[df.interval == epochs[int(i)]]
    max_soft_diffs[i] = temp_df['test_soft_diff'].max()
    min_soft_diffs[i] = temp_df['test_soft_diff'].min()
    mean_soft_diffs[i] = temp_df['test_soft_diff'].mean()

# plot min, mean, max of soft differences
plt.figure(0)
plt.plot(epochs, max_soft_diffs, label="Max test distance", color="r")
plt.plot(epochs, mean_soft_diffs, label="Mean test distance", color="k")
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


max_soft_diffs = np.zeros(len(epochs))
min_soft_diffs = np.zeros(len(epochs))
mean_soft_diffs = np.zeros(len(epochs))
for i in range(len(epochs)):
    temp_df = df[df.interval == epochs[int(i)]]
    mean_soft_diffs[i] = temp_df['avg_train_entropy'].mean()

# plot min, mean, max of soft differences
plt.figure(0)
plt.plot(epochs, mean_soft_diffs, label="Entropy Value", color="k")
# plt.hlines(0., min_epoch, max_epoch, colors="r", linestyles="dashed", label="Baseline Max test AR Accuracy")
# plt.hlines(0., min_epoch, max_epoch, colors='k', linestyles="dashed", label="Baseline Mean test AR Accuracy")
plt.xlabel("Epoch Interval")
epochs = np.arange(0, max_epoch+epoch_interval-1e-2, epoch_interval)
plt.xticks(epochs)
plt.ylabel("Entropy")
plt.legend(loc="lower right")
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Teacher Logit Entropy w.r.t Epoch Interval")
plt.savefig(os.path.join(cfg.figures_path, EXP+"_entropy.png"))
plt.show()
















