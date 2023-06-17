# CNN-AGE-DETECTION-USING-KL-DIVERGENCE

# Project Description
The project aims to perform age estimation using a deep learning model. The model takes images as input and predicts the age of the person in the image. The age prediction is treated as a classification problem, where the model assigns a probability distribution over age classes.

# Code Description
The code begins with importing the necessary libraries and mounting the Google Drive to access the image dataset. It then defines the directory containing the dataset and the desired image size.

Next, it initializes two lists, images and labels, to store the image data and corresponding age labels. The code iterates through each file in the dataset directory and extracts the age from the filename. It creates a label array with a length of 111 (representing age classes from 0 to 110) and assigns a Gaussian probability distribution centered at the extracted age. The label is normalized to sum to 1.0. The image is loaded, resized to the desired size, and appended to the images list, while the label is appended to the labels list.

After processing all the images, the code converts the images and labels lists to numpy arrays and performs normalization by dividing the images by 255.0.

The next step involves splitting the data into training and testing sets using the train_test_split function from sklearn. The data is split with 80% for training and 20% for testing.

The code then proceeds to build the age estimation model using TensorFlow and Keras. The model is defined as a sequential model with several layers: two 2D convolutional layers with ReLU activation, two max-pooling layers, flattening layer, two fully connected (dense) layers with ReLU activation and dropout, and a final dense layer with softmax activation representing the output probabilities for the age classes.

After defining the model architecture, two custom functions are implemented: age_loss and age_accuracy. The age_loss function computes the Kullback-Leibler divergence between the true age labels and the predicted age probabilities. The age_accuracy function calculates the accuracy of age prediction by rounding the predicted mean age and comparing it with the true mean age.

The model is compiled with the Adam optimizer, the age_loss as the loss function, and age_accuracy as the evaluation metric.

Next, the model is trained on the training data using the fit function. The training is performed for 10 epochs with a batch size of 64, and 20% of the training data is used as validation data.

After the training completes, the model is evaluated on the testing data. The predicted age probabilities (y_pred) are used to compute the mean age (n), true mean age (mu) from the labels, and the variance (sigma2) of the true age distribution. The age estimation error is then calculated as the mean of 1 - exp(- (n - mu)^2 / (2 * sigma2)).

Finally, the age estimation error is printed as the output.

# GitHub Description
Title: Age Estimation using Deep Learning

# Description:
This project implements a deep learning model for age estimation. The model takes images as input and predicts the age of the person in the image. Age prediction is treated as a classification problem, where the model assigns a probability distribution over age classes.

The code consists of the following main steps:

Dataset Preparation: The code assumes that the dataset is stored in the Google Drive. The directory containing the dataset is specified, and the images are resized to a desired size.

Data Processing: The images and corresponding age labels are loaded and processed. The age labels are converted into probability distributions using a Gaussian distribution
