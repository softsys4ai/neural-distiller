# imports
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import adadelta, RMSprop, SGD, Adam
# import wandb
# from wandb.keras import WandbCallback
# wandb.init(project="knowledge-distillation")

# logging
spacer = "--------------------------------"
student_train_spacer = "-----------------"
# custom model attributes
# Set callback functions to early stop training and save the best model so far
# student_callbacks = callbacks=[
#         EarlyStopping(monitor='val_acc', patience=8, min_delta=0.0001),
            #ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, min_lr=0.0001)
#     ]
#
student_callbacks = callbacks=[
        EarlyStopping(monitor='val_acc', patience=8, min_delta=0.00007)
    ]
# WandbCallback(),
#                                                     mode='auto', baseline=None, restore_best_weights=False)
# [EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=10, verbose=0,
#                                                     mode='auto', baseline=None, restore_best_weights=False)]
base_dense_student_size = 16
# optimizers for students
# SGD(lr=1e-1, momentum=0.9, nesterov=True),
# Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
student_optimizer = adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# student_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#student_optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
# student_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#  training
mnist_input_shape = (28, 28, 1)
mnist_number_classes = 10
random_seed = 1
teacher_epochs = 50
test_temperatures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#test_temperatures = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
student_dropout = 0.0
alpha = 0.2
student_epochs = 50
student_batch_size = 256
pruning_epochs = 100
pruning_batch_size = 256
# saved model configurations
temp_serialized_net = "temp_KD_teacher_net.h5"
temp_experiment_configs_dir = "ModelConfigs/TemporaryExperimentConfigs"
teacher_model_dir = "ModelConfigs/TeacherModelConfigs"
student_model_dir = "ModelConfigs/StudentModelConfigs"
lenet_config="99.2_TeacherCNN_2019-08-29_17-25-46"
custom_teacher_config="99.2_TeacherCNN_2019-08-29_17-25-46"
custom_student_config="94.89_StudentDense_2019-08-29_18-10-22"
# logging and system
log_dir="/Experiment-Logs/"
util_dir="/Utils/"
input_dir="/Data/Input/TestData/"
jetson_output_dir="/Data/Output/Jetson/"
tx2_output_dir="/Data/Output/TX2/"
jetson_config_file="/Jetson/Params.py"
tx2_config_file="/TX2/Params.py"
systems = {
    "Jetson": {
        "cpu":
            {
                "cores":
                    {
                        "core0": "cpu0",
                        "core1": "cpu1",
                        "core2": "cpu2",
                        "core3": "cpu3"
                    },
                "frequency":
                    {
                        "available": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies"
                    }
            },
        "gpu": {
            "frequency": {
                "available": "/sys/devices/57000000.gpu/devfreq/57000000.gpu/available_frequencies",
                "current": "/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq"
            },
            "status": "/sys/kernel/debug/clk/gpu/clk_state"
        },
        "emc": {
            "frequency": {
                "available": "/sys/kernel/debug/clk/emc/clk_possible_rates",
                "current": "/sys/kernel/debug/clk/emc/clk_rate"
            },
            "status": "/sys/kernel/debug/clk/emc/clk_state"
        },
        "power_state": {},
        "fan": {},
        "power": {
            "total": "/sys/devices/7000c400.i2c/i2c-1/1-0040/iio:device0/in_power0_input",
            "gpu": "/sys/devices/7000c400.i2c/i2c-1/1-0040/iio:device0/in_power1_input",
            "cpu": "/sys/devices/7000c400.i2c/i2c-1/1-0040/iio:device0/in_power2_input"
        }

    },
    "TX2": {
        "cpu":
            {
                "cores":
                    {
                        "core0": "cpu0",
                        "core1": "cpu3",
                        "core2": "cpu4",
                        "core3": "cpu5"
                    },
                "frequency":
                    {
                        "available": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies"
                    }
            },
        "gpu": {
            "frequency": {
                "available": "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/available_frequencies",
                "current": "/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/cur_freq"
            },
            "status": "/sys/kernel/debug/bpmp/debug/clk/gpu/state"
        },
        "emc": {
            "frequency": {
                "available": "/sys/kernel/debug/bpmp/debug/emc/possible_rates",
                "current": "/sys/kernel/debug/clk/emc/clk_rate"
            },
            "status": "/sys/kernel/debug/bpmp/debug/clk/emc/state"
        },
        "power_state": {},
        "fan": {},
        "power": {
            "total": "/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power0_input",
            "gpu": "/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power1_input",
            "cpu": "/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power2_input"
        }
    }
}