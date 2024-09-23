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

def load_object(file_path):
    """Load a Python object from a file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate multiple models using GridSearchCV and return performance report."""
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            # If the model supports class weighting, add the class_weight='balanced'
            if 'class_weight' in model.get_params().keys():
                model.set_params(class_weight='balanced')

            # GridSearch for best hyperparameters
            rd_cv = RandomizedSearchCV(model, para, cv=5, n_jobs=-1, scoring='f1',error_score='raise')  # Focus on F1 score
            rd_cv.fit(X_train, y_train)

            # Set the model to the best found parameters and refit
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Check if model supports predict_proba for roc_auc_score
            if hasattr(best_model, 'predict_proba'):
                y_test_prob = best_model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC-AUC
            else:
                y_test_prob = None  # Handle cases where predict_proba isn't available

            # Performance metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_f1': f1_score(y_train, y_train_pred, average='binary'),
                'train_precision': precision_score(y_train, y_train_pred, average='binary'),
                'train_recall': recall_score(y_train, y_train_pred, average='binary'),
                
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'test_f1': f1_score(y_test, y_test_pred, average='binary'),
                'test_precision': precision_score(y_test, y_test_pred, average='binary'),
                'test_recall': recall_score(y_test, y_test_pred, average='binary'),
            }

            # ROC-AUC Score (Only calculate if probability estimates are available)
            if y_test_prob is not None:
                metrics['test_rocauc'] = roc_auc_score(y_test, y_test_prob)
            else:
                metrics['test_rocauc'] = None

            # Add the metrics to the report
            report[model_name] = metrics

        return report

    except Exception as e:
        raise CustomException(e, sys)
