import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/credit_data.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #dropping duplicate records
            df = df.drop_duplicates()
            
            # Treating missing values
            df['Occupation'] = df['Occupation'].fillna('Others')

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            #Feature Engineering and Removing Very low cost vehicle data
            df['TwoWheeler Cost'] = round((df['Loan Amount']/df['LTV Ratio'])*100,0)

            df = df[df['TwoWheeler Cost']>=20000.0]
            
            # Calculate the number of rows to select (10% of the total number of rows)
            sample_size = int(0.10 * len(df))

            # Randomly select 10% of the rows
            random_indices = np.random.choice(df.index, sample_size, replace=False)
            selected_rows = df.loc[random_indices]

            # List of columns to assign "Others" to
            columns_to_assign_others = ['State','City'] 

            # Assign "Others" to the selected rows in the specified columns
            selected_rows[columns_to_assign_others] = 'Other'

            # Update the original DataFrame with the modified selected rows
            df.loc[random_indices] = selected_rows

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
