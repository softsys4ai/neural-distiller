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
    logger.info(cfg.student_train_spacer + "THREE STEP STUDENT-TEACHER EXPERIMENT" + cfg.student_train_spacer)
    teacher_model_name = str(options.model)
    # loading training data
    X_train, Y_train, X_test, Y_test = LoadDataset.load_mnist(logger)
    # config callbacks
    callbacks = []
    # setting up teacher model
    stm = ModelLoader(logger, teacher_model_name)
    stm.compile_loaded_model()
    teacher = stm.get_loaded_model()
    # evaluate teacher accuracy and performance
    teacherLoss, teacherAcc = HelperUtil.calculate_weighted_score(logger, teacher, X_train, Y_train, X_test, Y_test)
    logger.info('Teacher weighted score: (acc, loss) --> (%s, %s)' % (teacherAcc, teacherLoss))
    # creating intermediate training data
    Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(teacher, X_train, Y_train, X_test, Y_test)

    cfg.temp = 10
    # perform training of intermediate student network
    logger.info(cfg.student_train_spacer + "custom_student_128 TRAINING SESSION STARTED" + cfg.student_train_spacer)
    ssm = ModelLoader(logger, "custom_student_128")
    intermediateStudent = ssm.get_loaded_model()
    intermediateStudent = HelperUtil.apply_knowledge_distillation_modifications(logger, intermediateStudent)
    logger.info("Intermediate student dense layer size --> " + str(cfg.student_dense_128_size))
    logger.info('Training intermediate student network')
    logger.info('Intermediate student params: (temperature, epochs, batch_size) --> (%s, %s, %s)' % (
        cfg.temp, cfg.student_epochs, cfg.student_batch_size))
    intermediateStudent.fit(X_train, Y_train_new,
                batch_size=cfg.student_batch_size,
                epochs=200,
                verbose=1,
                callbacks=[],
                validation_data=(X_test, Y_test_new))
    logger.info('Completed intermediate student network training')
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, intermediateStudent, X_train, Y_train_new,
                                                                  X_test, Y_test_new)
    logger.info('Intermediate student weighted score before revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    intermediateStudent = HelperUtil.revert_knowledge_distillation_modifications(logger, intermediateStudent)
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, intermediateStudent, X_train, Y_train,
                                                                  X_test, Y_test)
    logger.info('Intermediate student weighted score after revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    # creating the training data for the target student network
    Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(intermediateStudent, X_train, Y_train, X_test,
                                                                     Y_test)

    cfg.temp = 5
    # perform training of intermediate student network
    logger.info(cfg.student_train_spacer + "custom_student_64 TRAINING SESSION STARTED" + cfg.student_train_spacer)
    ssm = ModelLoader(logger, "custom_student_64")
    intermediateStudent = ssm.get_loaded_model()
    intermediateStudent = HelperUtil.apply_knowledge_distillation_modifications(logger, intermediateStudent)
    logger.info("Intermediate student dense layer size --> " + str(cfg.student_dense_64_size))
    logger.info('Training intermediate student network')
    logger.info('Intermediate student params: (temperature, epochs, batch_size) --> (%s, %s, %s)' % (
        cfg.temp, cfg.student_epochs, cfg.student_batch_size))
    intermediateStudent.fit(X_train, Y_train_new,
                batch_size=cfg.student_batch_size,
                epochs=200,
                verbose=1,
                callbacks=[],
                validation_data=(X_test, Y_test_new))
    logger.info('Completed intermediate student network training')
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, intermediateStudent, X_train, Y_train_new,
                                                                  X_test, Y_test_new)
    logger.info('Intermediate student weighted score before revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    intermediateStudent = HelperUtil.revert_knowledge_distillation_modifications(logger, intermediateStudent)
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, intermediateStudent, X_train, Y_train,
                                                                  X_test, Y_test)
    logger.info('Intermediate student weighted score after revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    # creating the training data for the target student network
    Y_train_new, Y_test_new = TeacherUtils.createStudentTrainingData(intermediateStudent, X_train, Y_train, X_test,
                                                                     Y_test)

    cfg.temp = 1
    # perform training of final student network
    logger.info(cfg.student_train_spacer + "custom_student_32 TRAINING SESSION STARTED" + cfg.student_train_spacer)
    ssm.set_model_name("custom_student_32")
    ssm.load_model()  # load model manually
    student = ssm.get_loaded_model()
    student = HelperUtil.apply_knowledge_distillation_modifications(logger, student)
    logger.info("Target student dense layer size --> " + str(cfg.student_dense_32_size))
    logger.info('Training target student network')
    logger.info('Target student params: (temperature, epochs, batch_size) --> (%s, %s, %s)' % (
    cfg.temp, cfg.student_epochs, cfg.student_batch_size))
    student.fit(X_train, Y_train_new,
                batch_size=cfg.student_batch_size,
                epochs=200,
                verbose=1,
                callbacks=[],
                validation_data=(X_test, Y_test_new))
    logger.info('Completed target student network training')
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, student, X_train, Y_train_new,
                                                                  X_test, Y_test_new)
    logger.info('Target student weighted score before revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))
    student = HelperUtil.revert_knowledge_distillation_modifications(logger, student)
    studentLoss, studentAcc = HelperUtil.calculate_weighted_score(logger, student, X_train, Y_train,
                                                                  X_test, Y_test)
    logger.info('Target student weighted score after revert: (acc, loss) --> (%s, %s)' % (studentAcc, studentLoss))