
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import config_reference as cfg

with open(os.path.join(cfg.raw_data_path, "resnet_distillation_results_4.txt")) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

sizes = []
intervals = []
temperatures = []
test_accs = []
train_accs = []
for model in content:
    try:
        size, interval, temp, test_acc, train_acc = re.findall(rf"model_(\d+)_(\d+)\|\d+_(\d+)_(\d+.\d+)_(\d+.\d+).", model)[0]
    except:
        print(f"[ERROR] when parsing: {model}")
    sizes.append(int(size))
    intervals.append(int(interval))
    temperatures.append(float(temp))
    test_accs.append(float(test_acc))
    train_accs.append(float(train_acc))

# create dataframe with the parsed data
df = pd.DataFrame(list(zip(sizes, intervals, temperatures, test_accs, train_accs)),
                     columns=['size', 'interval', 'temp', 'test_acc', 'train_acc'])

# pre-plot manipulations
df_plot = df[df.interval != 0]
df_min = df_plot['test_acc'].min()
df_max = df_plot['test_acc'].max()
df_min = 0.47

hm_data1 = pd.pivot_table(df_plot, values='test_acc',
                          index=['temp'],
                          columns='interval')
plot = sns.heatmap(hm_data1, cmap="binary", vmin=df_min, vmax=df_max)
plot.invert_yaxis()
plt.title("Student Test Accuracy w.r.t Temperature and Epoch Interval")
fig = plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "ESKD_resnet_test_heatplot.png"))
plt.show()


# box plotting
plot = sns.boxplot(x=df_plot["interval"], y=df_plot["test_acc"])
# fixing dense labels
for ind, label in enumerate(plot.get_xticklabels()):
    ind +=1
    if ind % 5 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.hlines(0.4753, 0, 200, colors="r", linestyles="dashed", label="Baseline Max test Accuracy")
plt.hlines(0.44908, 0, 200, colors='k', linestyles="dashed", label="Baseline Mean test Accuracy")
plt.hlines(0.4078, 0, 200, colors='b', linestyles="dashed", label="Baseline Min test Accuracy")
fig = plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "ESKD_resnet_test_boxplot.png"))
plt.show()


# line plotting
plot = sns.lineplot(x=df_plot["interval"], y=df_plot["test_acc"])
# fixing dense labels
plt.hlines(0.4753, 0, 200, colors="r", linestyles="dashed", label="Baseline Max test Accuracy")
plt.hlines(0.44908, 0, 200, colors='k', linestyles="dashed", label="Baseline Mean test Accuracy")
plt.hlines(0.4078, 0, 200, colors='b', linestyles="dashed", label="Baseline Min test Accuracy")
fig = plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "ESKD_resnet_test_lineplot.png"))
plt.show()


