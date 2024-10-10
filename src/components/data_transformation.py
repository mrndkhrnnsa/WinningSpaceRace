from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self) -> object:
        try:
            logging.info("Data Transformation Initiated")
            # Feature
            categorical_cols = ["cut", "color", "clarity"]
            numerical_cols = ["carat", "table", "carat_per_volume"]
            # Ordinal Ranking
            cut_map = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            clarity_map = ["I1", "SI2", "SI1","VS2", "VS1", "VVS2", "VVS1", "IF"]
            color_map = ["D", "E", "F", "G", "H", "I", "J"]

            logging.info("Data Transformation Pipeline Initiated")
            #--- numerical pipeline ----
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            #--- categorial pipeline ----
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('oridinalencoder', OrdinalEncoder(categories=[cut_map, color_map, clarity_map])),
                    ('scaler', StandardScaler())
                ]
            )
            #--- processor steps pipeline ----
            processor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_cols),
                ("categorical_pipeline", cat_pipeline, categorical_cols)
            ])
            logging.info("Data Transformation Pipeline Completed")
            return processor

        except Exception as e:
            logging.info("Exception occured in Data Transformation")
            raise CustomException(e, sys)
    
    def initiate_data_transform(self,train_data_path:str,test_data_path:str,target = 'price') -> object:
        try:
            logging.info('Starting to Read train and test data')
            drop_cols = ["x","y","z","proportion_score","volume","depth","depth_percentage"]
            non_feature    = [target,'id']

            train_df = pd.read_csv(train_data_path)
            train_df = train_df.drop(drop_cols,axis=1)

            test_df = pd.read_csv(test_data_path)
            test_df = test_df.drop(drop_cols,axis=1)

            logging.info('Read train and test data completed')
            logging.info(f'Train Data first of 5 rows : \n{train_df.head().to_string}')
            logging.info(f'test Data first of 5 rows : \n{test_df.head().to_string}')

            logging.info("Starting preprocessing")
            processing_obj          = self.get_data_transformation()
            
            input_feature_train_df  = train_df.drop(non_feature,axis=1)
            target_feature_train_df = train_df[target]

            input_feature_test_df   = test_df.drop(non_feature,axis=1)
            target_feature_test_df  = test_df[target]

            #--- transformation applied ---
            input_feature_train_arr = processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = processing_obj.transform(input_feature_test_df)

            train_arr               = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr                = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #--- save the pipeline ---
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = processing_obj
            )

            logging.info("Transformation Process Completed")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            raise CustomException(e,sys)


