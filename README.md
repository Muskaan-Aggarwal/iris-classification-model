# Iris Classification Model
## Overview
*This repository contains a machine learning model for classifying iris flowers based on their features using Python, scikit-learn, and TensorFlow. The model is trained on the classic Iris dataset and achieves an accuracy of 96% on the test set.*

## Dataset
The Iris dataset is a multi-class classification problem, consisting of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample is described by 4 features: sepal length, sepal width, petal length, and petal width.

## Model Architecture
*The model is a neural network with the following architecture:*

1. Input layer: 4 neurons (one for each feature)
2. Hidden layer 1: 64 neurons with ReLU activation
3. Hidden layer 2: 32 neurons with ReLU activation
4. Output layer: 3 neurons with softmax activation (one for each class)
*The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.*

## Usage
*To use the model, follow these steps:*

1. Install the required libraries by running pip install -r requirements.txt
2. Run the model by executing python iris_classification.py
3. The model will be trained and evaluated on the test set
4. The accuracy, classification report, and confusion matrix will be printed to the console

## Author
Muskaan Aggarwal

## Acknowledgments
This model is based on the classic Iris dataset, which is widely used in machine learning tutorials and examples. The model architecture and implementation are inspired by various online resources and tutorials.

## Contributing
If you'd like to contribute to this repository, please fork the repository and submit a pull request. You can also open an issue if you have any questions or suggestions.
