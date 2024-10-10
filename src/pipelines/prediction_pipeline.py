import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features:np.array) -> float:
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path        = os.path.join('artifacts','base_model.pkl')

            preprocessor      = load_object(preprocessor_path)
            model             = load_object(model_path)

            data_scaled       = preprocessor.transform(features)
            pred              = model.predict(data_scaled)
            pred              = np.exp(pred).round(0)
            return pred
        
        except Exception as e:
            logging.info("Fail to predict")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                carat:float,
                depth:float,
                table:float,
                x:float,
                y:float,
                z:float,
                cut:str,
                color:str,
                clarity:str):
        
        self.carat   = carat
        self.depth   = depth
        self.table   = table
        self.x       = x
        self.y       = y
        self.z       = z
        self.cut     = cut
        self.color   = color
        self.clarity = clarity

    def get_data_as_dataframe(self) -> pd.DataFrame:
        features = ['carat','cut','color','clarity','table','carat_per_volume']
        try:
            data_input = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x'    : [self.x],
                'y'    : [self.y],
                'z'    : [self.z],
                'cut'  : [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
                }
            df = pd.DataFrame(data_input)
            df["volume"] = df['x'] * df['y'] * df['z']
            df['carat_per_volume'] = df['carat'] / df['volume']
            df = df[features]
            logging.info("Dataframe gathered")
            return df
        except Exception as e:
            logging.info("Dataframe failed gathered")
            raise CustomException(e,sys)
        

