# logging
spacer = "--------------------------------"
student_train_spacer = "-----------------"
#  training
random_seed = 1
temp = 1
test_temperatures = [1, 2, 3, 5, 10, 20, 50, 100, 200]
alpha = 0.1
student_epochs = 50
student_batch_size = 256
pruning_epochs = 100
pruning_batch_size = 256
# logging and system
custom_teacher_config="99.2_TeacherCNN_2019-08-29_17-25-46"
custom_student_config="94.89_StudentDense_2019-08-29_18-10-22"
log_dir="/Logs/"
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