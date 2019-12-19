from matplotlib import pyplot as plt

import numpy as np

import os
import re

PATH_TO_STUDENT_MODELS = "results/ESKD_Knowledge_Distillation_cifar100_2_16-12-19_23:17:30/models"
PATH_TO_FIGURES = "results/figures"


def group_results_by_interval_to_vacc(min_epoch=0, max_epoch=200, epoch_interval=10) -> list:
    """
    Parse results for validation accuracy and interval and group by interval
    :param min_epoch: Minimum epoch value for results
    :param max_epoch: Maximum epoch value for results
    :param epoch_interval: Iteration Interval for epoch values
    :return: List with sublists for each epoch interval, containing validation accuracy
    """
    # Collecting student model results and building dict for each epoch interval
    student_results = os.listdir(PATH_TO_STUDENT_MODELS)
    results_groups = [[] for i in range(min_epoch, max_epoch + epoch_interval, epoch_interval)]
    for result in student_results:
        interval, vacc = re.findall(rf"model_2_(\d+)\|{max_epoch}_\d+_(\d+.\d+)", result)[0]
        interval = int(interval)
        vacc = float(vacc)
        results_groups[interval//epoch_interval].append(vacc)
    return results_groups


def plot_student_models():
    """
    Plot student models from results of experiment 1.
    Plot 1: Highest validation accuracy for each epoch interval
    Plot 2: Validation Accuracy against epoch interval against temp
    :return:
    """
    # -------- Plot 1 --------: (epoch interval, validation accuracy)
    # Load model directory so that we can parse names
    # Identify and parse highest validation accuracy for each interval
    # Plot epoch interval against max validation accuracy
    plot_epoch_against_vacc()

    # -------- Plot 2 --------: (epoch interval, temperature, validation accuracy)
    # Load model directory so that we can parse names
    # For each saved model, parse name for epoch interval, validation accuracy, and temp
    # Plot for each model: interval against temperature, against validation accuracy
    plot_epoch_against_temp_against_vacc()


def plot_epoch_against_vacc(min_epoch=0, max_epoch=200, epoch_interval=10):
    # Grabbing file names from experiment directory and grouping by iteration interval
    grouped_results = group_results_by_interval_to_vacc(min_epoch=min_epoch, max_epoch=max_epoch, epoch_interval=epoch_interval)
    epoch_intervals = np.arange(min_epoch, max_epoch + epoch_interval, epoch_interval)

    # Calculating max, average, and minimum values for each interval
    vacc_max = np.zeros(len(grouped_results))
    vacc_avg = np.zeros(len(grouped_results))
    vacc_min = np.zeros(len(grouped_results))
    for group, result in enumerate(grouped_results):
        max_vacc = max(result)
        avg_vacc = sum(result) / len(result)
        min_vacc = min(result)
        vacc_max[group] = max_vacc
        vacc_avg[group] = avg_vacc
        vacc_min[group] = min_vacc

    # Plotting min, average, and max on same graph
    plt.figure(0)
    # Plotting max validation accuracy against interval
    plt.plot(epoch_intervals, vacc_max, label="Max Validation Accuracy")
    # Plotting mean validation accuracy against interval
    plt.plot(epoch_intervals, vacc_avg, label="Mean Validation Accuracy")
    # Plotting min validation accuracy against interval
    plt.plot(epoch_intervals, vacc_min, label="Min Validation Accuracy")
    plt.hlines(0.4109, 0, 200, colors="r", linestyles="dashed", label="Baseline Max Validation Accuracy")
    plt.hlines(0.403121739, 0, 200, colors='g', linestyles="dashed", label="Baseline Mean Validation Accuracy")

    plt.xlabel("Epoch Interval")
    plt.xticks(epoch_intervals)
    plt.ylabel("Validation Accuracy")
    plt.ylim((0.39, 0.445))
    plt.xlim((0, 70))
    plt.legend(loc="lower right")
    save_path = os.path.join(PATH_TO_FIGURES, "max_mean_min.png")
    plt.savefig(save_path)
    plt.show()

    # Plotting average and max
    plt.figure(1)
    # Plotting max validation accuracy against interval
    plt.plot(epoch_intervals, vacc_max, label="Max Validation Accuracy")
    # Plotting mean validation accuracy against interval
    plt.plot(epoch_intervals, vacc_avg, label="Mean Validation Accuracy")
    plt.xlabel("Epoch Interval")
    plt.xticks(epoch_intervals)
    plt.ylabel("Validation Accuracy")
    plt.ylim((0.39, 0.445))
    plt.legend(loc="lower right")
    plt.show()

    # Plotting min and average
    plt.figure(2)
    # Plotting min validation accuracy against interval
    plt.plot(epoch_intervals, vacc_min, label="Min Validation Accuracy")
    # Plotting mean validation accuracy against interval
    plt.plot(epoch_intervals, vacc_avg, label="Mean Validation Accuracy")
    plt.xlabel("Epoch Interval")
    plt.xticks(epoch_intervals)
    plt.ylabel("Validation Accuracy")
    plt.ylim((0.39, 0.42))
    plt.legend(loc="lower right")
    plt.show()

    # Plotting min and max
    plt.figure(3)
    # Plotting min validation accuracy against interval
    plt.plot(epoch_intervals, vacc_min, label="Min Validation Accuracy")
    # Plotting max validation accuracy against interval
    plt.plot(epoch_intervals, vacc_max, label="Max Validation Accuracy")
    plt.xlabel("Epoch Interval")
    plt.xticks(epoch_intervals)
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.39, 0.445)
    plt.legend(loc="lower right")
    plt.show()


def group_results_by_interval_to_temp_vacc(min_epoch=0, max_epoch=200, epoch_interval=10,
                                           min_temp = 0, max_temp = 20, temp_interval = 2) -> list:
    """
    Parse results for validation accuracy and interval and group by interval
    :param min_epoch: Minimum epoch value for results
    :param max_epoch: Maximum epoch value for results
    :param epoch_interval: Iteration Interval for epoch values
    :return: List with sublists for each epoch interval, containing validation accuracy
    """
    # Collecting student model results and building dict for each epoch interval
    student_results = os.listdir(PATH_TO_STUDENT_MODELS)
    results_groups = [np.zeros((max_epoch + epoch_interval - min_epoch)//epoch_interval) for i in range(min_temp, max_temp + temp_interval, temp_interval)]
    for result in student_results:
        interval, temp, vacc = re.findall(rf"model_2_(\d+)\|{max_epoch}_(\d+)_(\d+.\d+)", result)[0]
        interval = int(interval)
        temp = int(temp)
        vacc = float(vacc)
        results_groups[temp // temp_interval][interval // epoch_interval] = vacc
    return results_groups


def plot_epoch_against_temp_against_vacc(min_epoch = 0, max_epoch = 200, epoch_interval=10,
                                         min_temp=0, max_temp=20, temp_interval=2):
    grouped_results = group_results_by_interval_to_temp_vacc(min_epoch=min_epoch, max_epoch=max_epoch, epoch_interval=epoch_interval)
    epoch_intervals = np.arange(min_epoch, max_epoch + epoch_interval, epoch_interval)
    temp_intervals = np.arange(min_temp, max_temp + temp_interval, temp_interval)

    fig, ax = plt.subplots()

    ax.set_title("Validation Accuracy w.r.t Temperature and Epoch Interval")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Epoch Interval")

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ax.set_xticklabels(epoch_intervals)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_yticklabels(temp_intervals)

    plt.pcolor(grouped_results, cmap="tab20c", vmin=0.39, vmax=0.445)
    plt.colorbar()
    plt.show()

    save_path = os.path.join(PATH_TO_FIGURES, "epoch_temp_vacc_tab20c.png")
    plt.savefig(save_path)

if __name__ == "__main__":
    plot_epoch_against_vacc()
    plot_epoch_against_temp_against_vacc()