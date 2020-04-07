
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import config_reference as cfg
sns.set()

# TODO add gaussian noise plots to this file

RESULTS_FILE = os.path.join(cfg.raw_data_path, "resnet_adversarial_results.csv")
EXP = "experiment2"
EPS = "0.050"
SIG = "0.200"
USE_EPS = False
if USE_EPS:
    COL_NAME = 'eps_' + EPS
    ATTACK = "FGM"
else:
    COL_NAME = 'sig_' + SIG
    ATTACK = "GAUSS_NOISE"


FIG_DIR = "figures"
CMAP = "binary"
df = pd.read_csv(RESULTS_FILE)
df_plot = df[df.interval != 0]
df_min = df[COL_NAME].min()
df_max = df[COL_NAME].max()
hm_data1 = pd.pivot_table(df_plot, values=COL_NAME,
                          index=['temp'],
                          columns='interval')
df_min = 0.18
plot = sns.heatmap(hm_data1, cmap=CMAP, vmin=df_min, vmax=df_max)
plot.invert_yaxis()
plt.title("Adversarial Accuracy w.r.t Temperature and Epoch Interval ("+COL_NAME+")")
fig = plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, EXP+"_AR_"+ATTACK+"_"+COL_NAME+"_heatplot.png"))
plt.show()

# find max accuracy for each epoch interval
# find min accuracy for each epoch interval
min_epoch = 5
max_epoch = 200
epoch_interval = 5
epochs = np.arange(min_epoch, max_epoch+epoch_interval-1e-2, epoch_interval)

max_ar_accs = np.zeros(len(epochs))
min_ar_accs = np.zeros(len(epochs))
mean_ar_accs = np.zeros(len(epochs))
for i in range(len(epochs)):
    temp_df = df[df.interval == epochs[int(i)]]
    max_ar_accs[i] = temp_df['eps_'+EPS].max()
    min_ar_accs[i] = temp_df['eps_'+EPS].min()
    mean_ar_accs[i] = temp_df['eps_'+EPS].mean()

# plot min, mean, max of adversarial data
plt.figure(0)
plt.plot(epochs, max_ar_accs, label="Max test AR Accuracy", color="r")
plt.plot(epochs, mean_ar_accs, label="Mean test AR Accuracy", color="k")
plt.plot(epochs, min_ar_accs, label="Min test AR Accuracy", color="b")
plt.hlines(0.1088, min_epoch, max_epoch, colors="r", linestyles="dashed", label="Baseline Max test AR Accuracy")
plt.hlines(0.094179798, min_epoch, max_epoch, colors='k', linestyles="dashed", label="Baseline Mean test AR Accuracy")
plt.hlines(0.0778, min_epoch, max_epoch, colors='b', linestyles="dashed", label="Baseline Mean test AR Accuracy")
plt.xlabel("Epoch Interval")
epochs = np.arange(0, max_epoch+epoch_interval-1e-2, epoch_interval)
plt.xticks(epochs)
plt.xlim((min_epoch, max_epoch))
plt.ylabel("Adversarial Test Accuracy")
plt.legend(loc="lower right")
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Adversarial Robustness Accuracy w.r.t. Temperature and Epoch Interval ("+COL_NAME+")")
plt.savefig(os.path.join(cfg.figures_path, EXP+"_AR_"+ATTACK+"_"+COL_NAME+"_lineplot.png"))
plt.show()






















