import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModeltrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def initiated_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "XGBClassifier": XGBClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            }

            # Define hyperparameters for models
            params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [5, 7, 9, 10, 11, 13, 15],
                },
                "Random Forest": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'max_depth': [5, 7, 9, 10, 11, 13, 15],
                },
                "Gradient Boosting": {
                'loss': ['log_loss', 'exponential'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'criterion': ['squared_error', 'friedman_mse'],
                'max_features': ['sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "XGBClassifier": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost Classifier": {
                'learning_rate': [.1, .01, 0.5, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                }
            }

            
            model_report:dict= evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param = params)
            
            ## To get best model score from dict 
            best_model_score = max(sorted(model_report.values()))
            


            ## to get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.5:
                print("No best model found")
            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # Use the best model for predictions
            predicted = best_model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            roc_auc = roc_auc_score(y_test, predicted)

            return accuracy, f1, precision, recall, roc_auc
        
        except Exception as e:
            raise CustomException(e, sys)