import ast
import json
import datetime
from optparse import OptionParser
from itertools import combinations


def config_option_parser():
    # reading command line input
    usage = """USAGE: %python generate-config.py -s [array of network sizes] -t [min,max,steps] -a [min,max,steps]
            """
    parser = OptionParser(usage=usage)
    parser.add_option('-s', "--sizes-of-nets",
                      action="store",
                      type="string",
                      default="[10,8,6,4,2]",
                      dest="orderLists",
                      help="Lists of student network sizes for multistage experiments")
    parser.add_option('-t', "--temp-config",
                      action="store",
                      type="string",
                      default="[1,50,20]",
                      dest="tempOrderList",
                      help="Temperatures to apply to teacher logits for multistage experiments")
    parser.add_option('-a', "--alpha-config",
                      action="store",
                      type="string",
                      default="[0,1,10]",
                      dest="alphaOrderList",
                      help="Alphas to apply to student KD loss function for multistage experiments")
    parser.add_option('-e', "--num-epochs",
                      action="store",
                      type="int",
                      default=50,
                      dest="epochs",
                      help="Alphas to apply to student KD loss function for multistage experiments")
    (options, args) = parser.parse_args()
    return (options, usage)


def save_json_to_disk(json_obj):
    now = datetime.datetime.now()
    with open("configs/experiment_" + str(now.isoformat()) + ".json", "w+") as f:
        json.dump(json_obj, f)
    print("Saved to file: %s" % ("experiment_" + str(now.isoformat()) + ".json"))


def main():
    # parsing command line input
    (options, usage) = config_option_parser()
    sizes_config = ast.literal_eval(options.orderLists)
    temp_config = ast.literal_eval(options.tempOrderList)
    alpha_config = ast.literal_eval(options.alphaOrderList)

    # creating list of test temperatures
    temp_step_size = (temp_config[1] - temp_config[0])/temp_config[2]
    training_temps = []
    for i in range(temp_config[2]):
        training_temps.append(int(temp_config[0]+i*temp_step_size))
    training_temps.append(temp_config[1])  # in case int conversion rounds
    # creating list of test alphas
    alpha_step_size = (alpha_config[1] - alpha_config[0]) / alpha_config[2]
    training_alphas = []
    for i in range(alpha_config[2] - 1):
        training_alphas.append(round(alpha_config[0] + i * alpha_step_size, 2))
    training_alphas.append(alpha_config[1])  # in case int conversion rounds

    # creating config json object
    config = {}
    config['teacher_name'] = None  # none by default b/c we want to train our own teacher and student nets
    config['epochs'] = options.epochs
    config['temp_config'] = training_temps
    config['alpha_config'] = training_alphas
    config['size_combinations'] = None

    # generating all combinations of provided network sizes
    fixed_start = sizes_config[0]
    del sizes_config[0]
    fixed_end = sizes_config[-1]
    del sizes_config[-1]
    comb_size = len(sizes_config)
    tests_sizes = []
    while (comb_size > -1):
        combs = combinations(sizes_config, comb_size)
        for test in combs:
            test_list = list(test)
            test_list.sort(reverse=True)
            test_list.insert(0, fixed_start)
            test_list.append(fixed_end)
            tests_sizes.append(test_list)
        comb_size -= 1
    config['size_combinations'] = tests_sizes

    # saving the generated configuration object
    save_json_to_disk(config)

    print("- COMPLETE")


if __name__ == "__main__":
    main()