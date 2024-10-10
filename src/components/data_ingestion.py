import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            
            logging.info(f"Data size before cleaning: {df.shape[0]}")
            
            # Filter data where x, y, z are not zero
            df = df[df["x"] != 0]
            df = df[df["y"] != 0]
            df = df[df["z"] != 0].reset_index(drop=True)
            
            # Further filter where z is less than 30
            df = df[df["z"] < 30].reset_index(drop=True)
            
            logging.info(f"Data size after cleaning: {df.shape[0]}")

            # Calculate additional features
            df["volume"] = df['x'] * df['y'] * df['z']
            df['depth_percentage'] = (df['depth'] / df['table']) * 100
            df['carat_per_volume'] = df['carat'] / df['volume']
            df['proportion_score'] = (df['x'] / df['y']) * (df['x'] / df['z']) * (df['y'] / df['z'])
            
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Raw data is created')

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Train and Test data is created')
            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)
