import matplotlib as plt
import numpy as np
from json import loads
from ast import literal_eval

def load_experiments(filepath):
    experiment_data = []
    with open(filepath) as experimentLog:
        count = 0
        for line in experimentLog:
            if line[0] is not "{": # skip first json_result because header of file
                continue
            json_result = loads(line)
            experiment_data.append(json_result)
            count += 1
    return experiment_data

def get_location_of_netsize_result(experiment, net_size):
    experiment_order = experiment['experiment_results']
    count = 0
    for elem in experiment_order:
        if int(elem['net_size']) is net_size:
            return count
        count += 1
    return -1

def return_top_five_validation(experiment_results, net_size):
    top_5 = []
    for experiment in experiment_results:
        # find location of target net size in the current experiment results
        loc = get_location_of_netsize_result(experiment, net_size)
        if loc == -1:  # if net size not in experiment being parsed, skip it!
            continue
        for i, top in enumerate(top_5):
            top_loc = get_location_of_netsize_result(top, net_size)
            float(top['experiment_results'][top_loc]['val_acc'])
            float(experiment['experiment_results'][loc]['val_acc'])
            if float(top['experiment_results'][top_loc]['val_acc']) < float(experiment['experiment_results'][loc]['val_acc']):
                if len(top_5) < 5:
                    top_5.insert(i, experiment)
                    break
                else:
                    top_5[i] = experiment
                    break
        if len(top_5) == 0:
            top_5.append(experiment)
    return top_5


# loading experiment data json objects
filepath = "example-results.log"
experiment_data = load_experiments(filepath)

# net size we are interested in
net_size = 2
# find top 5 highest validation accuracy
top_5 = return_top_five_validation(experiment_data, net_size)

# # print full top 5, not specific network size results
# for elem in top_5:
#     print(elem)


# printing only the network size we are interested in
net_result = []
for elem in top_5:
    loc = get_location_of_netsize_result(elem, net_size)
    net_result.append(elem['experiment_results'][loc])

for result in net_result:
    print(result)










