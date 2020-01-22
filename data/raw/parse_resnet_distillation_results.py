
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import config_reference as cfg

with open("resnet_distillation_results.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

sizes = []
intervals = []
temperatures = []
test_accs = []
train_accs = []
for model in content:
    size, interval, temp, test_acc, train_acc = re.findall(rf"model_(\d+)_(\d+)\|\d+_(\d+)_(\d+.\d+)_(\d+.\d+).", model)[0]
    print(test_acc)
    sizes.append(int(size))
    intervals.append(int(interval))
    temperatures.append(float(temp))
    test_accs.append(float(test_acc))
    train_accs.append(float(train_acc))

# create dataframe with the parsed data
df = pd.DataFrame(list(zip(sizes, intervals, temperatures, test_accs, train_accs)),
                     columns=['size', 'interval', 'temp', 'test_acc', 'train_acc'])

# pre-plot manipulations
ATTACK = "fgm"
df_min = df['test_acc'].min()
df_max = df['test_acc'].max()
df_min = 0.44
df_plot = df

hm_data1 = pd.pivot_table(df_plot, values='test_acc',
                          index=['temp'],
                          columns='interval')
plot = sns.heatmap(hm_data1, cmap="gnuplot2", vmin=df_min, vmax=df_max)
plot.invert_yaxis()
plt.title("Student Test Accuracy w.r.t Temperature and Epoch Interval")
fig = plot.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "ESKD_resnet_test_heatplot.png"))
plt.show()

