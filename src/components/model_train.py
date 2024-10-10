import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluation

from dataclasses import dataclass
import os
import sys


@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts','base_model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()
    
    def initiate_model_training(self, train_array:np.array, test_array:np.array):
        try:
            logging.info("Start to split data to train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            y_train = np.log(y_train)
            y_test = np.log(y_test)

            models = {
                    "LinearRegression":LinearRegression(),
                    "RidgeRegression":Ridge(random_state=42),
                    "Lasso":Lasso(random_state=42),
                    "RandomForestRegressor":RandomForestRegressor(random_state=42,max_depth=5),
                    "XGBRegressor":XGBRegressor(random_state=42,max_depth=5)
                }
            
            evaluation_results = evaluation(models, X_train, X_test, y_train, y_test)
            best_model_name    = evaluation_results[evaluation_results["Rank"]==1]["Model Name"][0]
            r2_score           = evaluation_results[evaluation_results["Rank"]==1]["R2 Score Test"][0]
            rmse_values        = evaluation_results[evaluation_results["Rank"]==1]["RMSE Test"][0]
            best_model         = evaluation_results[evaluation_results["Rank"]==1]["Model Function"][0]
            logging.info(f"Model Report : Base Model {best_model_name} best r2 score {r2_score} best rmse score {rmse_values}")

            save_object(
                file_path = self.model_train_config.trained_model_file_path,
                obj       = best_model
            )

        except Exception as e:
            raise CustomException(e,sys)
