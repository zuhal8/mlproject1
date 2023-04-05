import os
import sys # because we'll use our CustomException and Logging
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# normally we'd use __init__ to define the class variable, but when we use
# dataclass decorator, we can directly define the class variable. 
# The @dataclass decorator is used to automatically generate several special
#  methods such as '__init__()', '__repr__()', '__eq__()', and others that are 
# commonly used in data classes. Using a data class simplifies the creation
#  and manipulation of instances of this class, as you don't need to write 
# the boilerplate code for these special methods yourself.
@dataclass 
# when we're extracting the data we state where the extracted data goes in
# this class.
class DataIngestionConfig: 
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
 
# If you only define variables in the class, then use dataclass, 
# but if class also have methods then use __init__.
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    #read the data from a database or any other source here.
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #read the data from the path
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # create the artifacts folder, use exist_ok=True->if the folder is already there,
            #  don't delete it or modify it.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # to save the raw data to raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            # save the split data to the path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

# to initiate and execute the all data and model codes here        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))