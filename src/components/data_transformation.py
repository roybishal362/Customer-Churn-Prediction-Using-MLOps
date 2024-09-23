import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#Configuration for saving preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    # Function to get the preprocesisng object 
    def get_data_transformer_object(self):
        try:
            # Defining numerical  columns
            numerical_columns = [ 
                'cons_12m', 'cons_gas_12m', 'cons_last_month', 'forecast_cons_12m',
                'forecast_cons_year', 'forecast_discount_energy',
                'forecast_meter_rent_12m', 'forecast_price_energy_off_peak',
                'forecast_price_energy_peak', 'forecast_price_pow_off_peak', 'imp_cons',
                'margin_gross_pow_ele', 'margin_net_pow_ele', 'nb_prod_act',
                'net_margin', 'num_years_antig', 'pow_max', 'date_activ_day',
                'date_activ_month', 'date_activ_year', 'date_end_day', 'date_end_month',
                'date_end_year', 'date_modif_prod_day', 'date_modif_prod_month',
                'date_modif_prod_year', 'date_renewal_day', 'date_renewal_month',
                'date_renewal_year',]
            # Numrical Pipeline 
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            # Pipelines into ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline,numerical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

        # Function to initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get preprocessing object 
            logging.info("Obtaining preprocesing objects")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "churn"
            numerical_columns = ['cons_12m', 'cons_gas_12m', 'cons_last_month', 'forecast_cons_12m',
                'forecast_cons_year', 'forecast_discount_energy',
                'forecast_meter_rent_12m', 'forecast_price_energy_off_peak',
                'forecast_price_energy_peak', 'forecast_price_pow_off_peak', 'imp_cons',
                'margin_gross_pow_ele', 'margin_net_pow_ele', 'nb_prod_act',
                'net_margin', 'num_years_antig', 'pow_max', 'date_activ_day',
                'date_activ_month', 'date_activ_year', 'date_end_day', 'date_end_month',
                'date_end_year', 'date_modif_prod_day', 'date_modif_prod_month',
                'date_modif_prod_year', 'date_renewal_day', 'date_renewal_month',
                'date_renewal_year']
            # Separate input features and target feature for training and test data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object in training and testing dataframes.")

            # Apply transformation to training and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with the target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing objects.")

            # Save the preprocessing object
            save_object(
                file_path = self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)