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
        self.logger.info("[STATUS]: Initializing ModelLoader Class")
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
            # vgg16
            elif self.model_name == "vgg16":
                from tensorflow.python.keras.applications import VGG16
                from tensorflow.python.keras.applications.vgg16 import preprocess_input
                self.preprocess = preprocess_input
                self.model = VGG16()
            # vgg19
            elif self.model_name == "lenet5":
                from tensorflow.python.keras.applications import VGG19
                from tensorflow.python.keras.applications.vgg19 import preprocess_input
                self.preprocess = preprocess_input
                self.model = VGG19()
            # xception
            elif self.model_name == "xception":
                from tensorflow.python.keras.applications import Xception
                from tensorflow.python.keras.applications.xception import preprocess_input
                self.preprocess = preprocess_input
                self.model = Xception()
            # inceptionv3
            elif self.model_name == "inceptionv3":
                from tensorflow.python.keras.applications import InceptionV3
                from tensorflow.python.keras.applications.inception_v3 import preprocess_input
                self.preprocess = preprocess_input
                self.model = InceptionV3()
            # alexnet
            elif self.model_name == "alexnet":
                # TODO get a pre-trained alexnet model
                print("[ERROR]: Not yet implemented")
            # custom teacher model
            elif self.model_name == "custom_teacher":
                # compiling and training teacher network
                teacher = TeacherCNN()
                teacher.__init__()
                teacher.load(cfg.custom_teacher_config + ".json", cfg.custom_teacher_config + ".h5")
                self.model = teacher.getModel()
                # custom CNN teacher model
            # custom student model
            elif self.model_name == "custom_student":
                # compiling and training teacher network
                student = StudentDense()
                student.__init__()
                # todo change student class to load config but not pre-trained weights
                student.load(cfg.custom_student_config + ".json")
                self.model = student.getModel()
            else:
                self.logger.error("[ERROR]: invalid model")
        except Exception as e:
            self.logger.error("[ERROR]: teacher model load failed due to {0}".format(str(e)))

    def get_loaded_model(self):
        return self.model