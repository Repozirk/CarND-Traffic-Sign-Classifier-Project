#**Traffic Sign Classifier** 

##Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (training, evaluation and test data)
* Explore, summarize and visualize the training data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./classes.png "Classes"
[image2]: ./distribution_training_set.png "Distribution"
[image3]: ./my_images.png "My Images"
[image4]: ./prediction_of_my_images.png "Prediction of My Images"
[image5]: ./softmax_probabilities_of_my_images.png "Softmax Probabilities"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here you go! [writup_template](https://github.com/Repozirk/CarND-Traffic-Sign-Classifier-Project/edit/master/writeup_template.md)
My [project code](https://github.com/Repozirk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

By using the numpy library, I got this statistics out of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Number of Validation examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43


####2. Include an exploratory visualization of the dataset.

Using the mathplotlib, I printed out 1 example traffic sign for each class

![alt text][image1]

To get the information, how many traffic signs each class containing

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, the traffic signs were converted to grayscale and then normalized to get a better performance of the neural network
With conversion to grayscale, I was able to run the model on my computer.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The NN Model is the LeNet with 32x32x1 dimension for input and 43x1 dimension for output. I decided to use the elu function as an activation function the improve the performance of the neural network:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   							| 
| Convolution 1x6     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU					|	ELU insteac of RELU for better performance											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 6x16    |  1x1 stride, valid padding, outputs 10x10x16 |
| ELU					|	ELU insteac of RELU for better performance											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| Creates 400x1 dimension form 5x5x16       									|
| Fully connected		| Input = 400. Output = 120,        									|
| ELU					|	ELU insteac of RELU for better performance, dropout									|
| Fully connected		| Input = 120. Output = 84,        									|
| ELU					|	ELU insteac of RELU for better performance, dropout									|
| Fully connected		| Input = 84. Output = 43       									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The best results in accuracy was achieved with the following training parameters:

EPOCHS = 20
BATCH_SIZE = 64
rate = 0.002
optimizer = tf.train.AdamOptimizer()

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Using the normalization method suggested: (pixel-128)/128 did not work for me, because I had serious problems the run a dataset with 32x32x3 on my computer. I tried to run the model on AWS, but finally I decided to greyscale the data and reduce dimension to 32x32x1.
After performing greyscale on the data, I did the noramlization by the known method: X_train_norm = X_train_gray/255 * 0.8 + 0.1

A increas of accurancy was also performed by intrducing dropout to the fully connected layers. The best solution was created with keep_prob: 0.5

With changing the number of Epochs to 20 and thre rate to 0.002, I was able to get a accuracy > 0.93 with the training data and a accuracy of < 0.93 on the validation set.

Finally I replaced the acitvation function RELU with ELU, to speed up the model. The final value of accuracy was:

keep_prob: 0.5
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


