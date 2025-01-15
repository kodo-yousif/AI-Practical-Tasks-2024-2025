from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import List
from joblib import Parallel, delayed
import math
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = None
scaler = None
trained_models = {}
class TrainRequest(BaseModel):
    validation_method: str

class PredictRequest(BaseModel):
    data_row: List[float]
    model_type: str

# Neural Network for Diabetes prediction
class DiabetesNN(nn.Module):
    def __init__(self, input_size):
        super(DiabetesNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        
        # Increase numbers if your model is underfitting (not learning well).
        # Decrease numbers if your model is overfitting (learning too well on training data but failing on unseen data) or if training is too slow.
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def train_neural_network(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    
    # epochs: Number of complete passes through the training dataset.
    # Higher values allow the model to learn more but can lead to overfitting.

    # batch_size: Number of samples processed at once during training.
    # Smaller values lead to finer weight updates (more noise but better generalization).
    # Larger values make training faster but may miss local minima.
    # Smaller batches require less memory but may increase training time.
    
    # Parameters:
    # Trains the neural network using binary cross-entropy loss and Adam optimizer.
    # model: The neural network instance (DiabetesNN) to be trained.
    # X_train: Training input features as a NumPy array.
    # y_train: Training labels as a NumPy array.
    # X_test: Testing input features (used for evaluation after training).
    # y_test: Testing labels (used for evaluation after training).
    # epochs: Number of training epochs (default: 100).
    # batch_size: Size of the mini-batches (default: 32).
    
    # Dataset and DataLoader:
    # Converts X_train and y_train to PyTorch tensors.

    # Training Loop:
    # For each epoch:
    # Iterate through batches of data from the DataLoader.
    # Forward pass: Compute predictions using the model.
    # Compute loss: Compare predictions to actual labels.
    # Backpropagation: Calculate gradients and update model weights.
    # Accumulate total loss for monitoring.
    
    # Evaluation:
    # Sets the model to evaluation mode (model.eval()).
    # Makes predictions on X_test and thresholds outputs to generate binary classifications (> 0.5 → class 1).
    # Returns predictions and raw probabilities.
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        test_outputs = model(X_test_tensor)
        predictions = (test_outputs > 0.5).float().numpy()
        probabilities = test_outputs.numpy()
    
    return predictions, probabilities

def load_data():
    global data
    try:
        data = pd.read_csv('./diabetes.csv')
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def preprocess_data():
    global scaler

    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    processed_data = data.copy()

    for column in columns_to_process:
        processed_data[column] = processed_data[column].replace(0, np.nan)
        processed_data[column] = processed_data[column].fillna(processed_data[column].mean())

    X = processed_data.iloc[:, :-1].values
    y = processed_data.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def save_model(model, model_name):
    if isinstance(model, DiabetesNN):
        torch.save(model.state_dict(), f"{model_name}.pth")
    else:
        with open(f"{model_name}.pkl", "wb") as f:
            joblib.dump(model, f)

def load_model(model_name):
    try:
        if model_name == "Neural":
            model = DiabetesNN(input_size=8)
            model.load_state_dict(torch.load(f"{model_name}.pth"))
            return model
        else:
            with open(f"{model_name}.pkl", "rb") as f:
                return joblib.load(f)
    except FileNotFoundError:
        return None

# Evaluates the model on the test data using metrics like accuracy, precision, recall, etc.
def evaluate_model(model, X_test, y_test):
    # model: The model to evaluate (either DiabetesNN or a traditional ML model like k-NN).
    # X_test: Test input features.
    # y_test: Test labels.
    
    
    try:
        # Neural Network Evaluation:
        # If the model is an instance of DiabetesNN:
        # Sets the model to evaluation mode (model.eval()).
        # Uses the test features (X_test) to make predictions.
        # Converts outputs to binary predictions (> 0.5) and retrieves probabilities.
        if isinstance(model, DiabetesNN):
            print()
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                outputs = model(X_tensor)
                predictions = (outputs > 0.5).float().numpy().flatten()
                probas = outputs.numpy()
        else:
            # Traditional Model Evaluation:

            # For models like kNN, SVM, or Naive Bayes:
            # Calls model.predict for predictions.
            # If the model supports probability output (predict_proba), retrieves probabilities for positive class.
            predictions = model.predict(X_test)
            probas = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # accuracy: Fraction of correct predictions.
        # precision: Positive predictive value.
        # recall: True positive rate.
        # f1_score: Harmonic mean of precision and recall.
        # roc_auc: Area under the ROC curve (requires probabilities).
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0))
        }
        if probas is not None:
            # For binary classification (single output probability)
            if isinstance(probas, np.ndarray) and len(probas.shape) == 1:
                probas = probas  # Use the single output as-is
            elif isinstance(probas, np.ndarray) and probas.shape[1] == 1:
                probas = probas[:, 0]  # Use the first column if it's reshaped as a 2D array
            elif isinstance(probas, np.ndarray) and probas.shape[1] > 1:
                probas = probas[:, 1]  # Use the second column for positive class probabilities
            else:
                raise ValueError("Unexpected shape for model output probabilities.")

            metrics["roc_auc"] = float(roc_auc_score(y_test, probas))
        else:
            metrics["roc_auc"] = 0.0


        metrics = {k: (v if not math.isnan(v) else 0.0) for k, v in metrics.items()}
        return metrics

    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0
        }

@app.on_event("startup")
async def startup_event():
    if not load_data():
        raise HTTPException(status_code=500, detail="Failed to load initial data")

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    try:
        X, y = preprocess_data()
        validation_method = request.validation_method

        models = {
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "Bayesian": GaussianNB(),
            "SVM": SVC(probability=True),
            "Neural": DiabetesNN(input_size=8)
        }

        if validation_method == "holdout":
            # Holdout Validation:
            # Splits data into 80% training and 20% testing = test_size=0.2 (train_test_split).
            # Trains each model:
            # For neural networks (DiabetesNN), calls train_neural_network.
            # For traditional models, uses model.fit.
            # Evaluates the models on the test set using evaluate_model.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            results = {}

            for name, model in models.items():
                if isinstance(model, DiabetesNN):
                    predictions, _ = train_neural_network(model, X_train, y_train, X_test, y_test)
                else:
                    model.fit(X_train, y_train)
                
                metrics = evaluate_model(model, X_test, y_test)
                print(metrics)
                results[name] = {
                    key: float(value) if not math.isnan(float(value)) else 0.0
                    for key, value in metrics.items()
                }
                save_model(model, name)

            return results

        elif validation_method in ["3-fold", "10-fold"]:
            # Determines the number of splits (3 or 10).
            # Initializes metrics to zero for averaging across folds.
            # For each fold:
            # Splits data into training and test sets.
            # Reinitializes the neural network (DiabetesNN) for each fold to prevent parameter leakage.
            # Trains and evaluates the model on the fold.
            # Aggregates metrics across all valid folds.
            # Averages metrics across folds and returns the results.
            n_splits = 3 if validation_method == "3-fold" else 10
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            results = {}

            for name, model in models.items():
                metrics_sum = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "roc_auc": 0.0
                }
                valid_folds = 0

                for train_index, test_index in kf.split(X):
                    X_train_cv, X_test_cv = X[train_index], X[test_index]
                    y_train_cv, y_test_cv = y[train_index], y[test_index]

                    if isinstance(model, DiabetesNN):
                        model = DiabetesNN(input_size=8)  # Reinitialize for each fold
                        predictions, _ = train_neural_network(model, X_train_cv, y_train_cv, X_test_cv, y_test_cv)
                    else:
                        model.fit(X_train_cv, y_train_cv)
                    
                    fold_metrics = evaluate_model(model, X_test_cv, y_test_cv)

                    for key in metrics_sum:
                        if fold_metrics[key] is not None and not math.isnan(fold_metrics[key]):
                            metrics_sum[key] += fold_metrics[key]
                            valid_folds += 1

                results[name] = {
                    key: float(metrics_sum[key] / valid_folds if valid_folds > 0 else 0.0)
                    for key in metrics_sum
                }
                
                save_model(model, name)

            return results

        else:
            raise HTTPException(status_code=400, detail="Invalid validation method")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    
    try:
        model = load_model(request.model_type)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_type} not found. Please train the model first."
            )

        if scaler is None:
            _, _ = preprocess_data()
            
        data_row_scaled = scaler.transform([request.data_row])
        
        if request.model_type == "Neural":
            # Neural Network (DiabetesNN):
            # Sets the model to evaluation mode (model.eval()).
            # Converts the scaled input to a PyTorch tensor.
            # Computes the output probability (output.item()).
            # Generates a binary prediction (> 0.5 → class 1) and computes class probabilities.
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data_row_scaled)
                output = model(input_tensor)
                prediction = int(output.item() > 0.5)
                probability = [1 - output.item(), output.item()]
        else:
            prediction = model.predict(data_row_scaled)[0]
            probability = model.predict_proba(data_row_scaled)[0].tolist() if hasattr(model, 'predict_proba') else None
            
        response = {
            "prediction": int(prediction),
            "probability": probability
        }
        
        return response

    except Exception as e:
        if "is not fitted yet" in str(e):
            print(e)
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_type} needs to be trained first. Please call /train endpoint."
            )
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)


# sklearn: A library for machine learning that includes algorithms like k-NN, Naive Bayes, and SVM, as well as utility functions for training and evaluation.
# torch: PyTorch is used to implement a neural network for diabetes prediction.

# torch, torch.nn, etc., are used to define and train deep learning models.
# pandas: Used for data manipulation and reading CSV files.

# Global Variables
# data: Stores the diabetes dataset (loaded from a CSV file).
# scaler: Stores the StandardScaler used to scale the features of the dataset.
# trained_models: A dictionary that stores the trained models for later use in predictions.

# /predict: This endpoint is used for making predictions with a trained model. It takes data_row (the features for prediction) and model_type (the model to use for prediction). The model is loaded, the data is scaled, and a prediction is made.
# For the Neural Network, predictions are made by passing the data through the network after converting it to a tensor.
# For other models, predictions are made using predict() and the probability of each class is returned.

# Model Training:
# Three models are trained using sklearn: k-NN, Naive Bayes, and SVM.
# A neural network is trained using PyTorch.


# Holdout Method:
# This is the simplest validation method. The dataset is split into two
# parts: training and testing. By default, 80% of the data is used for training and 20% for testing (train_test_split function)


# What is the role of the neural network in this application?
# The neural network (implemented in the DiabetesNN class) is used to predict the likelihood of a patient having diabetes based on input features such as Glucose, Blood Pressure, BMI, etc. The role of the neural network is:
# Architecture: The network has three fully connected layers (fc1, fc2, fc3)
# with ReLU activation functions between them. The final layer uses the Sigmoid activation function to produce a value between 0 and 1, which is interpreted as the probability of a positive diagnosis (class 1 for diabetes).
# Training: The neural network is trained using backpropagation with a
# loss function (BCELoss), which is suitable for binary classification tasks 
# (since diabetes prediction is a binary classification problem). The optimizer used is Adam, which adjusts the weights to minimize the loss.
# Purpose: It offers an alternative to traditional machine learning algorithms 
# (like k-NN, Naive Bayes, or SVM) by capturing more complex relationships in
# the data. While simpler models can be more interpretable, the neural network may capture non-linearities better and provide improved performance on larger, more complex datasets.


# Why is it necessary to use a scaler like StandardScaler?
# Scaling is crucial for several reasons:
# Improves Performance: Many machine learning algorithms (especially
# those based on distance metrics, like k-NN) perform better when the features
# are on the same scale. If the features have different ranges, the algorithm may 
# be biased towards features with larger values.
# Speeds Up Convergence: For optimization-based models like SVM or
# Neural Networks, scaling can help the model converge faster because 
# the gradients for all features are on a similar scale, preventing the optimizer from taking erratic steps.
# Ensures Model Compatibility: Some algorithms assume the data is 
# standardized (e.g., SVM or logistic regression), so it's essential
# to scale the features before feeding them into such models.


# Evaluation Metrics:

# Accuracy = (True Positives + True Negatives) / Total Instances
# Definition: Accuracy is the proportion of correctly classified instances (both true positives and true negatives) out of all the instances in the dataset.
# it can be misleading when the dataset is imbalanced

# Precision = True Positives / ( True Positives + False Positives )
# Definition: Precision is the proportion of true positive predictions (correctly predicted instances of the positive class) out of all the instances that the model predicted as positive.
# It answers the question: Of all the positive predictions, how many were actually positive?

# Recall (Sensitivity or True Positive Rate)
# Definition: Recall is the proportion of true positive predictions (correctly predicted instances of the positive class) out of all the instances that are actually positive in the dataset.
# Recall = True Positives / ( True Positives + False Negatives )
# It answers the question: Of all the actual positive instances, how many did the model correctly identify?

# F1-Score Definition: The F1-Score is the harmonic mean of Precision and Recall. It is a balanced metric that takes both false positives and false negatives into account, especially useful when the dataset is imbalanced.
# F1-Score = 2 * [ (precision * recall) / ( Precision + recall )]
# The F1-Score ranges from 0 to 1, where 1 is the best possible value (perfect precision and recall) and 0 is the worst.

# ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)
# Definition: ROC-AUC measures the ability of a model to distinguish between positive and negative classes. It calculates the area under the ROC curve, where:
# True Positive Rate (Recall) is plotted on the y-axis.
# False Positive Rate (1 - Specificity) is plotted on the x-axis.


# kNN (KNeighborsClassifier): A k-Nearest Neighbors classifier with n_neighbors=5.
# Bayesian (GaussianNB): Naive Bayes classifier.
# SVM (SVC): Support Vector Machine with probability=True to compute probabilities.
# Neural (DiabetesNN): A custom neural network with 8 input features.