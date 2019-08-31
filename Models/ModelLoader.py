from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from Configuration import Config as cfg
from Models.CustomTeacher import TeacherCNN
from Models.CustomStudent import StudentDense

class ModelLoader(object):
    """This class is used to initialize teacher models
    """
    def __init__(self,
                 logger,
                 model_name):
        self.logger = logger
        self.logger.info("Initializing ModelLoader Class")
        self.model = None
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        """This function is used to load pretrained model
        """
        usage = """USAGE: %python train.py -m [model]
                         Model ResNet50:                  python train.py -o resnet50
                         Model VGG16:                     python train.py -o vgg16
                         Model LeNet-5:                   python train.py -o lenet5
                         Model AlexNet:                   python train.py -o alexnet
                         Model Xception:                  python train.py -o xception
                         Model InceptionV3:               python train.py -o inceptionv3
                """
        try:
            # resnet50
            if self.model_name == "resnet50":
                from tensorflow.python.keras.applications import ResNet50
                from tensorflow.python.keras.applications.resnet50 import preprocess_input
                self.preprocess = preprocess_input
                self.model = ResNet50()
                self.logger.info("Loaded " + self.model_name)
            # vgg16
            elif self.model_name == "vgg16":
                from tensorflow.python.keras.applications import VGG16
                from tensorflow.python.keras.applications.vgg16 import preprocess_input
                self.preprocess = preprocess_input
                self.model = VGG16()
                self.logger.info("Loaded " + self.model_name)
            # vgg19
            elif self.model_name == "lenet5":
                from tensorflow.python.keras.applications import VGG19
                from tensorflow.python.keras.applications.vgg19 import preprocess_input
                self.preprocess = preprocess_input
                self.model = VGG19()
                self.logger.info("Loaded " + self.model_name)
            # xception
            elif self.model_name == "xception":
                from tensorflow.python.keras.applications import Xception
                from tensorflow.python.keras.applications.xception import preprocess_input
                self.preprocess = preprocess_input
                self.model = Xception()
                self.logger.info("Loaded " + self.model_name)
            # inceptionv3
            elif self.model_name == "inceptionv3":
                from tensorflow.python.keras.applications import InceptionV3
                from tensorflow.python.keras.applications.inception_v3 import preprocess_input
                self.preprocess = preprocess_input
                self.model = InceptionV3()
                self.logger.info("Loaded " + self.model_name)
            # alexnet
            elif self.model_name == "alexnet":
                # TODO get a pre-trained alexnet model
                self.logger.info("[ERROR]: Not yet implemented")
            # custom teacher model
            elif self.model_name == "custom_teacher":
                # compiling and training teacher network
                teacher = TeacherCNN()
                teacher.__init__()
                teacher.load(cfg.custom_teacher_config + ".json", cfg.custom_teacher_config + ".h5")
                self.model = teacher.getModel()
                self.logger.info("Loaded " + self.model_name)
                # custom CNN teacher model
            # custom student model
            elif self.model_name == "custom_student":
                # compiling and training teacher network
                student = StudentDense()
                student.__init__()
                student.buildAndCompile()
                # todo change student class to load config but not pre-trained weights
                # student.load(cfg.custom_student_config + ".json")
                self.model = student.getModel()
                self.logger.info("Loaded " + self.model_name)
            else:
                self.logger.error("Invalid model name")
        except Exception as e:
            self.logger.error("Loading of model failed due to {0}".format(str(e)))

    def get_loaded_model(self):
        return self.model