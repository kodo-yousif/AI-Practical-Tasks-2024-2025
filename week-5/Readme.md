# 1. KNN sampling method

the app will demonstrate the difference where KNN is used traditional method and the sampling method to reduce the space of the data

the app should give statistical UI with some demonstration on a board where the datas are places and the cluster center points are placed

## Requirements:

- data particle dots are placed on the board
- each data particle has a color according toi the cluster it belongs to
- feel free to use any language/library for UI
- main logic mus be in python
- no library should be used for implementing KNN formula

# 2. Dynamic Bayesian Classifier

The app will accept a Excel file or any other file format which is table, after that the app can take a examples via some form and predict the outcome using Bayesian Classifier.

the app shows the calculated table and also can download it.

## Requirements:

- the data input is a Excel file or any other table based file
- the last column in the file is the labels
- each column is a feature with non-binary value ( means the value can be any thing )
- after calculation show and make the user be able to download the calculated table in Excel or any other table file format
- after the calculation (training), user can enter their own data
- make all the fields that user enters data a dropdown
- don't let user enter the last column because it is the result and app should predict it
- don't use libraries for the bayesian classifier methods and calculations
- feel free to use any language/framework
- main logic mus be in python

# 3. perceptron learning algorithm (in the lecture)

create a app where it shows the perceptron learning algorithm line movement it justifies itself to satisfy all the points.

the app is a white board ( or app or any thing )

use ten data samples of two classes where each class include 5 samples.
the app will draw the separating line between the two classes

then the line starts adjusting toward the better position in a 2fps speed so that the movement be clear

the app will save each iteration table in a file

## Requirements:

- if user can enter parameters in the formula if its possible will be better
- animation speed ( iteration speed ) should be 2fps for clear visualization and can be more than 2fps for better animation
- open another windows to show the live table or save each table in a file
- feel free to use any language/library for UI
- main logic mus be in python

# 4 SVM Model

train SVM model on iris dataset, the app will train the model 3 times showing the difference kernel works in each model ( RBF, Linear, Polynomial) kernels

at the GUI for each kernel show support vectors and separation line and margin lines

show the model accuracy

## requirements

- train svm with 3 different kernels ( RBF, Linear, Polynomial) and know their difference
- use iris dataset in Sklearn
- use 2 dimensions for each data (for simplicity)
- show support vector
- show separation lines
- show margin lines
- use Sklearn in Python for SVM
- show the model accuracy
- main logic mus be in python
- feel free to use any language/library for UI

# 5 Regression model

In the iris dataset use the Sepal length feature to predict the Sepal Width using a regression model.

Then use the Sepal Length and Width to predict the Petal Length and Petal width.

show the model accuracy

## requirements

- use iris dataset in Sklearn
- use Sklearn in Python for regression
- show the model accuracy
- main logic mus be in python
- feel free to use any language/library for UI

# 6 KMeans clustering

Cluster the Iris dataset using KMeans clustering.

Cluster the data 3 times using three different metrics ( Euclidean distance, cosine distance and manhattan distance).

plot each KMeans result to a scatter plot GUI, give each cluster a different color.

show the model accuracy

## requirements

- use iris dataset in Sklearn
- use Sklearn in Python for KMeans
- use three different metrics ( Euclidean distance, cosine distance and manhattan distance)
- show scatter plot for each method with coloring each cluster in a different color
- show the model accuracy
- main logic mus be in python
- feel free to use any language/library for UI
