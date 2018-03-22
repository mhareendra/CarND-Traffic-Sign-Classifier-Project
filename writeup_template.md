# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Readme_data/train_exploration.png "Explore Training data"
[image2]: ./Readme_data/valid_exploration.png "Explore Validation data"
[image3]: ./Readme_data/original1.png "Original"
[image4]: ./Readme_data/preprocess1.png "Pre-processed"
[image5]: ./custom/1r.jpg "Traffic Sign 1"
[image6]: ./custom/2r.jpg "Traffic Sign 2"
[image7]: ./custom/3r.jpg "Traffic Sign 3"
[image8]: ./custom/4r.jpg "Traffic Sign 4"
[image9]: ./custom/5r.jpg "Traffic Sign 5"
[image10]: ./custom/preprocess1.png "Preprocessed Traffic sign"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mhareendra/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[Writeup html](https://github.com/mhareendra/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
Bar chart showing number of training samples in each class:

![alt text][image1]

Bar chart showing number of validation samples in each class:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

![alt text][image3]

![alt text][image4]

Shown above is the original vs preprocessed image

The pre-processing operation:
1. Convert image to YUV color space
2. Extract Y channel
3. Perform histogram equalization on the Luminance channel
4. Normalize the result of the above step

YUV corresponds to 1 luminance and 2 chrominance components. The Y component itself can be visualized as a grayscale image which indicates the brightness of colors and the chrominance components represent the colors. Only the Y layer is retained and the other 2 are not used (as described in http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
[Traffic Sign Recognition with Multi-Scale Convolutional Networks, Pierre Sermanet et al])

Histogram equalization is used to improve the contrast of our images (https://docs.opencv.org/3.2.0/d5/daf/tutorial_py_histogram_equalization.html). Intuitively, this is similar to applying uniform illumination throughout the image. This helps to avoid bias caused due to good illumination in less-useful regions in our images and also poor illumination on the useful regions.

Normalizing the resulting image ensures that the pixel values are in the range [-1, +1]. This helps bound the range of values used in the training process.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image   							| 
| Convolution 5x5     	| 32 filters, 1x1 stride, Valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Valid padding, outputs 14x14x32 				|
| Convolution 5x5	    | 64 filters, 1x1 stride, Valid padding, outputs 10x10x64	|
| RELU					|					Activation							|
| Max pooling	      	| 2x2 stride, Valid padding, outputs 5x5x64 				|
| Flatten        | input 5x5x64, output= 1600   |
| Fully connected		|  input=1600, output=800   									|
| RELU |       Activation          |
| Dropout |    keep probability: 0.75  |
| Fully connected		|  input=800, output=200   									|
| RELU |       Activation            |
| Dropout |    keep probability: 0.75  |
|	Fully connected (Logits)					|	outputs = 43											| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

We stick to using a learning rate of 0.001 and consequently 75 epochs. Since our learning rate is not very slow, the number of epochs needed doesn't have  to be high.
The batch-size is set to 128 to ensure optimum memory usage.

The optimizer used is the Adam (Adaptive moment estimation) optimizer which is an extension of the classical stochastic gradient descent optimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999 
* validation set accuracy of 0.961 
* test set accuracy of 0.94

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture chosen was the LeNet architecture used for classifying MNIST data. The reason to choose this was the similarity in dimensions of the images and the similarity in features (shape, geomtery) of the images.

I had also tried out different types of preprocessing, including retaining the original RGB image, converting to grayscale and normalization, converting to YUV and normalization etc. Finally, using the normalized Y layer provided the best results.

* What were some problems with the initial architecture?

The best validation accuracy obtained was 85% by choosing RGB images as input to the model. Also, the training accuracy was not hitting 100%. This clearly indicated that the model was not able to learn enough information from the training samples.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The modification involved increasing the number of filters in the convolutional layers as this directly affected how much information could be extracted from the images. This solved the underfitting problem and the training accuracy reached 100% consistently (solved under-fitting problem).

But this lead to over-fitting since the validation accuracy still did not increase drastically. Adding dropout to the fully connnected layers helped solve this issue to an extent.

* Which parameters were tuned? How were they adjusted and why?

Increasing the number of epochs gave enough time to the model to learn and improve the validation accuracy while maintaining the same learning rate of 0.001. The final number of epochs chosen was 75.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

Pre processed:


![alt text][image10]

It can be observed that the traffic signs occupy majority of the image ensuring that we do not have to apply any detection algorithms to segment the objects of interest.

The images correspond to: 
[Stop, Right-of-way at the next intersection, Road work, No entry, Yield]


It can be observed that the traffic signs occupy majority of the image ensuring that we do not have to apply any detection algorithms to segment the objects of interest.

The images correspond to: 
[Stop, Right-of-way at the next intersection, Road work, No entry, Yield]

We can see that 'Right of way' and 'Road work' signs are quite similar. They share the same outline shape and its color, with a black colored figure inside. This would be a good test for the classifier, but since we supplied a large number of training samples for both these classes (1170 and 1350), we should expect correct classification.

The STOP and Yield signs are also quite similar in terms of shape and the colors of the foreground vs background. If the classifier has captured patterns pertaining to the letters inside the sign, it should be able to differentiate these signs confidently.  


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Yield   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Road work					| Road work											|
| No entry	      		| No entry					 				|
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. In comparison, the accuracy on the test set was 94% (lot more test samples)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Yield  									| 
| 0.0014     				| Stop 										|
| 0.00003					  | No vehicles											|
| ~0	      			| Traffic signals					 				|
| ~0				    | Speed limit (120km/h)						|


As shown in the above cell, the classifier is very confident of its predictions on all images.
The first image was incorrectly classified as 13 (Yield) even though it was 14 (Stop).

At first glance, the No entry sign and the stop sign could be mistaken if not for the 'STOP' word. This may be
because the octagonal  outline of the STOP sign could be confused for a circular outline, especially in low resolution images 
like the ones in our training data set (32x32). Also, the signs are a bit similar in the sense that each is a white foreground
on a red background. 

But interestingly, the Yield sign is not at all related to the shape of the STOP sign. The only possible similarity could be the gradual reduction of the sign's outline towards the bottom of the image. Looking back at the training data, we see that there were 690 samples of the 'Yield' sign whereas we had 1920 samples of the 'STOP' sign. It should be noted that the soft max probabilities reveal that the Stop sign had the second highest probability.


All the other images were correctly classified with maximum probability of 1, which means that the classifier is very sure of
its predictions

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


