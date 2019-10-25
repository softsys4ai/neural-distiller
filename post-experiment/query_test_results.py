import json

data = []
with open('/Users/blakeedwards/Desktop/Repos/research/neural-distiller-2/run-experiment/Experiment-Logs/experiments_2019-10-23T13:25:50.732460.log') as f:
    for _ in range(1):
        next(f)
    for line in f:
        data.append(json.loads(line))

# average the accuracy and validation accuracy of vanilla size two networks
acc = []
val_acc = []
acc_avg = 0
val_avg = 0
for experiment in data:
    acc.append(experiment['experiment_results'][0]['result'][0]['acc'])
    val_acc.append(experiment['experiment_results'][0]['val_acc'])
    acc_avg += float(experiment['experiment_results'][0]['result'][0]['acc'])
    val_avg += float(experiment['experiment_results'][0]['result'][0]['val_acc'])

print("average acc: %f" % (acc_avg/len(acc)))
print("average val_acc: %f" % (val_avg/len(val_acc)))


