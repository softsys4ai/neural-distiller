import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import config_reference as cfg
sns.set()

LOGIT_DIFFERENCE_PATH = "/Users/blakeedwards/Desktop/Repos/research/neural-distiller/post-experiment/ESKD-Analysis/experiment3_logit_diffs.pkl"
with open(LOGIT_DIFFERENCE_PATH, 'rb') as file:
    logit_differences = pickle.load(file)

train_difference = logit_differences[0][0]
test_difference = logit_differences[0][1]
train_difference = np.transpose(train_difference)
test_difference = np.transpose(test_difference)

cmap = "seismic"
max_heat = 0.1
min_heat = -0.1

# average of columns to get average class difference
train_mean = np.mean(train_difference, axis=1)
test_mean = np.mean(test_difference, axis=1)

train_plot = sns.heatmap([train_mean], vmin=min_heat, vmax=max_heat, cmap=cmap)
train_plot.set(xlabel="Data Sample", ylabel="Class")
plt.title("Teacher-Student Average Logit Difference (Train Data)")
fig = train_plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "average-train-logit-difference.png"))
plt.show()


train_plot = sns.heatmap([test_mean], vmin=min_heat, vmax=max_heat, cmap=cmap)
train_plot.invert_yaxis()
train_plot.set(xlabel="Data Sample", ylabel="Class")
plt.title("Teacher-Student Average Logit Difference (Test Data)")
fig = train_plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "average-test-logit-difference.png"))
plt.show()

train_difference = np.transpose(train_difference)
test_difference = np.transpose(test_difference)

cmap = "seismic"
max_heat = 0.5
min_heat = -0.5

train_start = 0
train_slice = 1000
train_plot = sns.heatmap(train_difference[train_start:train_slice + train_start, :], vmin=min_heat, vmax=max_heat, cmap=cmap)
train_plot.invert_yaxis()
train_plot.set(xlabel="Data Sample", ylabel="Class")
plt.title("Teacher-Student Logit Difference Map (Train Data)")
fig = train_plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "train-logit-difference.png"))
plt.show()

test_start = 0
test_slice = 1000
test_plot = sns.heatmap(test_difference[test_start:test_slice + test_start, :], vmin=min_heat, vmax=max_heat, cmap=cmap)
test_plot.invert_yaxis()
test_plot.set(xlabel="Data Sample", ylabel="Class")
plt.title("Teacher-Student Logit Difference Map (Test Data)")
fig = test_plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "test-logit-difference.png"))
plt.show()

#
# indices = np.arange(1, len(train_difference)+1, 1)
# cols = ["col"+str(i) for i in range(1, len(train_difference[0])+1)]
# df_train = pd.DataFrame(data=train_difference, index=indices, columns=cols)
#
# indices = np.arange(1, len(test_difference)+1, 1)
# cols = ["col"+str(i) for i in range(1, len(test_difference[0])+1)]
# df_test = pd.DataFrame(data=test_difference, index=indices, columns=cols)
#
















