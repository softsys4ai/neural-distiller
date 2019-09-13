from Configuration import Config as cfg
from Data import LoadDataset
from Models.ModelLoader import ModelLoader
from Models import TeacherUtils
from Utils import HelperUtil
from numpy.random import seed
seed(cfg.random_seed)
from tensorflow import set_random_seed
set_random_seed(cfg.random_seed)

def run(logger, options):
    logger.info(cfg.student_train_spacer + "NO STEP STUDENT-TEACHER EXPERIMENT" + cfg.student_train_spacer)
    teacher_model_name = str(options.model)
    # loading training data
    X_train, Y_train, X_test, Y_test = LoadDataset.load_mnist(logger)

    # training student network at a range of temperatures
    logger.info(cfg.student_train_spacer + "NEW STUDENT TRAINING SESSION" + cfg.student_train_spacer)
    cfg.temp = 10
    # setting up custom student network
    ssm = ModelLoader(logger, "custom_student_128")
    student = ssm.get_loaded_model()
    # training and evaluating the student model
    logger.info('Training student network')
    logger.info('Student params: (epochs, batch_size) --> (%s, %s)' % (cfg.student_epochs, cfg.student_batch_size))
    student.fit(X_train, Y_train,
                batch_size=cfg.student_batch_size,
                epochs=cfg.student_epochs,
                verbose=1,
                callbacks=[],
                validation_data=(X_test, Y_test))
    logger.info('Completed student network training')
    # testing student accuracy before reverting the network architecture
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, student, X_train, Y_train,
                                                                  X_test, Y_test)
    logger.info('Student weighted score before revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    # evaluating student performance
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, student, X_train, Y_train,
                                                                  X_test, Y_test)
    logger.info('Student weighted score after revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))