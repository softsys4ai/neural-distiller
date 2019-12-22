from matplotlib import pyplot as plt

import numpy as np

import os
import re

PATH_TO_STUDENT_MODELS = "results/experiment-3/ESKD_Knowledge_Distillation_cifar100_2_18-12-19_18:04:58/models"
PATH_TO_FIGURES = "results/figures"


def group_results_by_interval_to_vacc(min_epoch=0, max_logit_epochs=100, max_epoch=200, epoch_interval=5) -> list:
    """
    Parse results for validation accuracy and interval and group by interval
    :param min_epoch: Minimum epoch value for results
    :param max_epoch: Maximum epoch value for results
    :param epoch_interval: Iteration Interval for epoch values
    :return: List with sublists for each epoch interval, containing validation accuracy
    """
    # Collecting student model results and building dict for each epoch interval
    student_results = os.listdir(PATH_TO_STUDENT_MODELS)
    results_groups = [[] for i in range(min_epoch, max_logit_epochs + epoch_interval, epoch_interval)]
    for result in student_results:
        interval, vacc = re.findall(rf"model_2_(\d+)\|{max_epoch}_\d+_(\d+.\d+)", result)[0]
        interval = int(interval)
        vacc = float(vacc)
        results_groups[interval//epoch_interval].append(vacc)
    return results_groups


def plot_student_models():
    """
    Plot student models from results of experiment 1.
    Plot 1: Highest test accuracy for each epoch interval
    Plot 2: test Accuracy against epoch interval against temp
    :return:
    """
    # -------- Plot 1 --------: (epoch interval, test accuracy)
    # Load model directory so that we can parse names
    # Identify and parse highest test accuracy for each interval
    # Plot epoch interval against max test accuracy
    plot_epoch_against_vacc()

    # -------- Plot 2 --------: (epoch interval, temperature, test accuracy)
    # Load model directory so that we can parse names
    # For each saved model, parse name for epoch interval, test accuracy, and temp
    # Plot for each model: interval against temperature, against test accuracy
    plot_epoch_against_temp_against_vacc()


def plot_epoch_against_vacc(min_epoch=0, max_epoch=200, epoch_interval=10):
    # Grabbing file names from experiment directory and grouping by iteration interval
    grouped_results = group_results_by_interval_to_vacc()
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
    # Plotting max test accuracy against interval
    plt.plot(epoch_intervals, vacc_max, label="Max test Accuracy", color="r")
    # Plotting mean test accuracy against interval
    plt.plot(epoch_intervals, vacc_avg, label="Mean test Accuracy", color="k")
    # Plotting min test accuracy against interval
    plt.plot(epoch_intervals, vacc_min, label="Min test Accuracy", color="b")
    plt.hlines(0.4109, 0, 200, colors="r", linestyles="dashed", label="Baseline Max test Accuracy")
    plt.hlines(0.403121739, 0, 200, colors='k', linestyles="dashed", label="Baseline Mean test Accuracy")

    plt.xlabel("Epoch Interval")
    plt.xticks(epoch_intervals)
    plt.ylabel("Test Accuracy")
    plt.ylim((0.39, 0.46))
    plt.xlim((0, 100))
    plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    save_path = os.path.join(PATH_TO_FIGURES, "max_mean_min.png")
    plt.savefig(save_path)
    plt.show()
    #
    # # Plotting average and max
    # plt.figure(1)
    # # Plotting max test accuracy against interval
    # plt.plot(epoch_intervals, vacc_max, label="Max test Accuracy")
    # # Plotting mean test accuracy against interval
    # plt.plot(epoch_intervals, vacc_avg, label="Mean test Accuracy")
    # plt.xlabel("Epoch Interval")
    # plt.xticks(epoch_intervals)
    # plt.ylabel("Test Accuracy")
    # plt.ylim((0.39, 0.46))
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # # Plotting min and average
    # plt.figure(2)
    # # Plotting min test accuracy against interval
    # plt.plot(epoch_intervals, vacc_min, label="Min test Accuracy")
    # # Plotting mean test accuracy against interval
    # plt.plot(epoch_intervals, vacc_avg, label="Mean test Accuracy")
    # plt.xlabel("Epoch Interval")
    # plt.xticks(epoch_intervals)
    # plt.ylabel("Test Accuracy")
    # plt.ylim((0.39, 0.46))
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # # Plotting min and max
    # plt.figure(3)
    # # Plotting min test accuracy against interval
    # plt.plot(epoch_intervals, vacc_min, label="Min test Accuracy")
    # # Plotting max test accuracy against interval
    # plt.plot(epoch_intervals, vacc_max, label="Max test Accuracy")
    # plt.xlabel("Epoch Interval")
    # plt.xticks(epoch_intervals)
    # plt.ylabel("Test Accuracy")
    # plt.ylim((0.39, 0.46))
    # plt.legend(loc="lower right")
    # plt.show()


def group_results_by_interval_to_temp_vacc(min_epoch=0, max_logit_epochs=100, max_epoch=200, epoch_interval=10,
                                           min_temp = 0, max_temp = 20, temp_interval = 2) -> list:

    """
    Parse results for test accuracy and interval and group by interval
    :param min_epoch: Minimum epoch value for results
    :param max_epoch: Maximum epoch value for results
    :param epoch_interval: Iteration Interval for epoch values
    :return: List with sublists for each epoch interval, containing test accuracy
    """
    # Collecting student model results and building dict for each epoch interval
    student_results = os.listdir(PATH_TO_STUDENT_MODELS)
    # results_groups = [np.zeros((max_epoch + epoch_interval - min_epoch)//epoch_interval) for i in range(min_epoch, max_logit_epochs + epoch_interval, epoch_interval)]
    results_groups = [np.zeros((max_logit_epochs + epoch_interval - min_epoch)//epoch_interval) for i in range(min_temp, max_temp + temp_interval, temp_interval)]
    for result in student_results:
        interval, temp, vacc = re.findall(rf"model_2_(\d+)\|{max_epoch}_(\d+)_(\d+.\d+)", result)[0]
        interval = int(interval)
        temp = int(temp)
        vacc = float(vacc)
        results_groups[temp // temp_interval][interval // epoch_interval] = vacc
    return results_groups


def plot_epoch_against_temp_against_vacc(min_epoch=0, max_logit_epochs=100, max_epoch=200, epoch_interval=10,
                                         min_temp=0, max_temp=20, temp_interval=2):
    grouped_results = group_results_by_interval_to_temp_vacc(min_epoch=min_epoch, max_logit_epochs=max_logit_epochs,
                                                             max_epoch=max_epoch, epoch_interval=epoch_interval,
                                                             min_temp=min_temp, max_temp=max_temp,
                                                             temp_interval=temp_interval)
    epoch_intervals = np.arange(min_epoch, max_logit_epochs + epoch_interval, epoch_interval)
    temp_intervals = np.arange(min_temp, max_temp + temp_interval, temp_interval)

    fig, ax = plt.subplots()

    ax.set_title("Test Accuracy w.r.t Temperature and Epoch Interval")

    ax.set_xlabel("Epoch Interval")
    ax.set_ylabel("Temperature")

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ax.set_xticklabels(epoch_intervals)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_yticklabels(temp_intervals)

    # colormode = "afmhot"
    colormode = "gnuplot2"
    plt.pcolor(grouped_results, cmap=colormode, vmin=0.40420, vmax=0.45020)
    plt.colorbar()

    plotname = "epoch_temp_vacc_"+colormode+".png"
    save_path = os.path.join(PATH_TO_FIGURES, plotname)
    plt.savefig(save_path)

    plt.show()

if __name__ == "__main__":
    plot_epoch_against_vacc(min_epoch=0, max_epoch=100, epoch_interval=5)
    plot_epoch_against_temp_against_vacc(min_epoch=0, max_logit_epochs=100, max_epoch=200, epoch_interval=5,
                                         min_temp=0, max_temp=10, temp_interval=1)