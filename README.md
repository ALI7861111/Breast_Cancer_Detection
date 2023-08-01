# Breast Cancer Detection Using Neural Networks

This repository contains the code for a breast cancer detection model. It uses a dataset comprising various features derived from digitalized images of a fine needle aspirate (FNA) of a breast mass. The output of the model is a binary classification indicating whether the cancer is benign or malignant.

# Dataset
The dataset consists of 32 columns, including an ID, a diagnosis (M = malignant, B = benign), and 30 real-valued input features. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe the following characteristics of the cell nuclei present in the image:

radius (mean of distances from center to points on the perimeter)
texture (standard deviation of gray-scale values)
perimeter
area
smoothness (local variation in radius lengths)
compactness (perimeter^2 / area - 1.0)
concavity (severity of concave portions of the contour)
concave points (number of concave portions of the contour)
symmetry
fractal dimension ("coastline approximation" - 1)
The mean, standard error, and "worst" or largest (mean of the three worst/largest values) of these features were computed for each image, resulting in 30 features.

# model
The model is a sequential neural network built using the Keras library. The network has the following structure:

Input Layer: 30 nodes (corresponding to the 30 features in the dataset)
Hidden Layer 1: 32 nodes with ReLU (Rectified Linear Unit) activation function
Hidden Layer 2: 64 nodes with ReLU activation function
Hidden Layer 3: 128 nodes with ReLU activation function
Dropout Layer: With dropout rate of 0.2 to prevent overfitting
Hidden Layer 4: 256 nodes with ReLU activation function
Output Layer: 2 nodes (corresponding to the 2 classes: benign and malignant) with sigmoid activation function
The model is trained with back-propagation and optimized with stochastic gradient descent.

# 
How to Use
The main code for the model can be found in the Breast_Cancer_Diagnosis_Features_Probability.ipynb file. To run the code, simply clone this repository and execute the Breast_Cancer_Diagnosis_Features_Probability.ipynb

# Requirements
This model requires the following Python libraries:

Keras
TensorFlow
Pandas
NumPy
Scikit-learn
Acknowledgements
The dataset used is publicly available and was created by Dr. William H. Wolberg, General Surgery Dept at University of Wisconsin, Clinical Sciences Center in Madison, WI, USA.