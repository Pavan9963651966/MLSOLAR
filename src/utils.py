import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, param):
    try:
        # Initialize the XGBoost model
        model = XGBClassifier()
        
        # Initialize GridSearchCV with the XGBoost model and parameters
        gs = GridSearchCV(model, param, cv=3, scoring='accuracy')  # Set scoring to accuracy
        gs.fit(X_train, y_train)

        # Set the best parameters and refit the model
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate accuracy scores
        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)

        # Return the accuracy scores and best parameters
        return {
            'train_accuracy': train_model_score,
            'test_accuracy': test_model_score,
            'best_params': gs.best_params_
        }

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

