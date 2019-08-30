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