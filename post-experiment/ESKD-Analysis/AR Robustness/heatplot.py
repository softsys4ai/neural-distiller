import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

RESULTS_FILE = "experiment3_AR_BIM.csv"
EXP = "experiment2"
EPS = "0.100"
ATTACK = "BIM"
CMAP = "gnuplot2"
df_plot = pd.read_csv(RESULTS_FILE)
df = df_plot[df_plot.interval != 0]
df_min = df['eps_'+EPS].min()
df_max = df['eps_'+EPS].max()
hm_data1 = pd.pivot_table(df_plot, values='eps_' + EPS,
                          index=['temp'],
                          columns='interval')
plot = sns.heatmap(hm_data1, cmap=CMAP, vmin=df_min, vmax=df_max)
plot.invert_yaxis()
plt.title("Adversarial Accuracy w.r.t Temperature and Epoch Interval ("+ATTACK+", Epsilon "+EPS+")")
fig = plot.get_figure()
fig.savefig(EXP+"_AR_"+ATTACK+"_Eps_"+EPS+"_"+CMAP+"_heatmap.png")
plt.show()

# find max accuracy for each epoch interval
# find min accuracy for each epoch interval
min_epoch = 5
max_epoch = 100
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
plt.hlines(0., min_epoch, max_epoch, colors="r", linestyles="dashed", label="Baseline Max test AR Accuracy")
plt.hlines(0., min_epoch, max_epoch, colors='k', linestyles="dashed", label="Baseline Mean test AR Accuracy")
plt.xlabel("Epoch Interval")
epochs = np.arange(0, max_epoch+epoch_interval-1e-2, epoch_interval)
plt.xticks(epochs)
plt.ylabel("Adversarial Test Accuracy")
plt.legend(loc="lower right")
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Adversarial Robustness Accuracy w.r.t. Temperature and Epoch Interval ("+ATTACK+")")
plt.savefig(EXP+"_AR_"+ATTACK+"_Eps_"+EPS+"_lineplot.png")
plt.show()






















