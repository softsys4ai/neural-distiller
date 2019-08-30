# Evolutionary Pruning and Knowledge Distillation (EPK Distillation)
Contributers: Blake Edwards
References: Dr. Pooyan Jamshidi
Description: n/a

Notes:
[!] Find easy way to tune hyperparams of pre-trained models.

- https://keras.io/applications/#available-models
	- if model is of sequential type (keras) it is easy to .pop() the last layer to re-purpose the network
	- not a straightforward way of changing the layers within the sequential model
	- easy to get the intermediate outputs of individual layers of pre-trained models
- https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
	- what parameters can be changed on the pre-trained Keras models?
		- pooling, output classes, input shap, temperature
		- optimizer, learning rate, momentum, batch size, number of epochs, shuffle
		- pop off last layer / add a new output layer + re-train its use of feature output of layer[-2]

Rough-outline:
// train all teacher models that are not pre-trained
// load each teacher model configurations and weights
// create a directory for each run of the program to contain all of the saved metrics and student models
// REFERENCE POINT 1
// perform five different pruning techniques to the teacher network
// save each pruned teacher model as a new student model within a directory named by the iteration number
// modify the student network to accept and produce the correct output
// tune each student model by training it for 100 epochs on hard and soft teacher targets
// compare and rank all of the models with accuracy and validation_accuracy weighed 1:3
// save the rankings of student models and their associated pruning method
// choose the best performing student model and load it
// modify this student model to output in the correct dimensions (original nb_classes)
// decrease the temperature value by 5
// return to REFERENCE POINT 1
// repeat this process 10 times