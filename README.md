Project Description
The current research applies a Multilayer Perceptron model to carry out a multi-class classification task on the Iris dataset.

**Objective**
The aim is to develop a model for a neural network that is able to classify iris flower types.

Dataset Used
The dataset is characterized by:
- Number of samples: 150
- Number of features: 4
- Number of classes: 3
- Classes:
  - Setosa
  - Versicolor
  - Virginica

Model Used
The model utilized is MLPClassifier, a part of the scikit-learn library.

Model Specifications
The model is characterized by:
- Number of hidden layers: 1
- Number of neurons in hidden layers: 100
- Activation function: ReLU
- Solver: Adam
- Number of iterations: 500
- Regularization term (alpha): 1e-4
- Train/Test Split: 70/30

Data Preprocessing
The steps taken for data preprocessing include:

- Encoding target variable
- Train/Test Split
- Scaling data using StandardScaler

Rationale for Scalin
Feature scaling is applied to the dataset as it is known that the performance of a neural network is superior with standardized data.


How to Run the Project
1. Install required libraries:
   pip install scikit-learn pandas numpy

2. Run the script:
   python mlp_iris.py

**Results Obtained**

The accuracy is 1.0000, or 100%. According to the classification report, precision, recall, and F1-score for all three classes are perfect.
