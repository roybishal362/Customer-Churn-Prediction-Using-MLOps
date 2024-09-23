import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """Save Python object to a file using dill."""
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to a file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate multiple models using RandomizedSearchCV and return performance report."""
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # If the model supports class weighting, add the class_weight='balanced'
            if 'class_weight' in model.get_params().keys():
                model.set_params(class_weight='balanced')

            # RandomizedSearchCV for best hyperparameters
            gs = RandomizedSearchCV(model, para, cv=5, n_jobs=-1, scoring='f1', error_score='raise')
            gs.fit(X_train, y_train)

            # Set the model to the best found parameters and refit
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Performance metrics
        
            train_accuracy=accuracy_score(y_train, y_train_pred)
            
            test_accuracy= accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load a Python object from a file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)