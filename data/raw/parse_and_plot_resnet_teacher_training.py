
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import config_reference as cfg

with open("resnet_teacher_training.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

sizes = []
intervals = []
test_accs = []
train_accs = []
for model in content:
    size, interval, test_acc, train_acc = re.findall(rf"model_(\d+)_(\d+)\|\d+_(\d+.\d+)_(\d+.\d+).", model)[0]
    print(test_acc)
    sizes.append(int(size))
    intervals.append(int(interval))
    test_accs.append(float(test_acc))
    train_accs.append(float(train_acc))

# create dataframe with the parsed data
df = pd.DataFrame(list(zip(sizes, intervals, test_accs, train_accs)),
                     columns=['size', 'interval', 'test_acc', 'train_acc'])

teacher_training = sns.lineplot(x="interval", y="test_acc", legend='brief', label="test", data=df)
teacher_training = sns.lineplot(x="interval", y="train_acc", legend='brief', label="train", data=df)

plt.title("ResNet Teacher Network Training and Test Accuracy w.r.t. Epoch")
fig = teacher_training.get_figure()
fig.savefig(os.path.join(cfg.figures_path, "ESKD_resnet_teacher_training_lineplot.png"))
plt.show()


