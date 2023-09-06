import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import  DataIngestion
import os



@dataclass
class DataTransformationconfig:
    train_data_prepro:str=os.path.join('artifacts/preprocessfile','train_pre.csv')
    test_data_prepro:str=os.path.join('artifacts/preprocessfile','test_pre.csv')
class DataTransformation:
    def __init__(self):
        self.preprocess_config=DataTransformationconfig()

    def get_data_transformation(self,train_path,test_path):
        try:
            logging.info('Data Transformation initiated')            
            # Reading train and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info("started preprocessing----->Training Data")
            train_data.dropna(inplace = True)
            train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
            train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
            train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

            # Dep_Time
            train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
            train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
            train_data.drop(["Dep_Time"], axis = 1, inplace = True)

            # Arrival_Time
            train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour
            train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute
            train_data.drop(["Arrival_Time"], axis = 1, inplace = True)


            # Duration
            duration = list(train_data["Duration"])

            for i in range(len(duration)):
                if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
                    if "h" in duration[i]:
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        duration[i] = "0h " + duration[i]           # Adds 0 hour

            duration_hours = []
            duration_mins = []
            for i in range(len(duration)):
                duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

            # Adding Duration column to test set
            train_data["Duration_hours"] = duration_hours
            train_data["Duration_mins"] = duration_mins
            train_data.drop(["Duration"], axis = 1, inplace = True)


            # Categorical data
            Airline = train_data[["Airline"]]
            Airline = pd.get_dummies(Airline, drop_first= True)
            Airline = Airline.applymap(lambda x: 1 if x else 0)
            
            Source = train_data[["Source"]]
            Source = pd.get_dummies(Source, drop_first= True)
            Source = Source.applymap(lambda x: 1 if x else 0)

            Destination = train_data[["Destination"]]
            Destination = pd.get_dummies(Destination, drop_first = True)
            Destination= Destination.applymap(lambda x: 1 if x else 0)

            # Additional_Info contains almost 80% no_info
            # Route and Total_Stops are related to each other
            train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

            # Replacing Total_Stops
            train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

            # Concatenate dataframe --> test_data + Airline + Source + Destination
            train_arr= pd.concat([train_data, Airline, Source, Destination], axis = 1)

            train_arr.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
            train_arr['Fare']=train_arr['Price']
            train_arr.drop(["Price","Airline_Multiple carriers Premium economy","Airline_Trujet","Unnamed: 0"], axis = 1, inplace = True)




            logging.info("Completed Preprocessing------->Training Data")


            ############  test data prepro started
            
            logging.info("started preprocessing----->Test Data")
            test_data.dropna(inplace = True)

            test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
            test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
            test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

            # Dep_Time
            test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
            test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
            test_data.drop(["Dep_Time"], axis = 1, inplace = True)

            # Arrival_Time
            test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
            test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
            test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

            # Duration
            duration = list(test_data["Duration"])

            for i in range(len(duration)):
                if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
                    if "h" in duration[i]:
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        duration[i] = "0h " + duration[i]           # Adds 0 hour

            duration_hours = []
            duration_mins = []
            for i in range(len(duration)):
                duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

            # Adding Duration column to test set
            test_data["Duration_hours"] = duration_hours
            test_data["Duration_mins"] = duration_mins
            test_data.drop(["Duration"], axis = 1, inplace = True)


            # Categorical data
            Airline = test_data[["Airline"]]
            Airline = pd.get_dummies(Airline, drop_first= True)
            Airline = Airline.applymap(lambda x: 1 if x else 0)
       
            Source = test_data[["Source"]]
            Source = pd.get_dummies(Source, drop_first= True)
            Source = Source.applymap(lambda x: 1 if x else 0)
            

            Destination = test_data[["Destination"]]
            Destination = pd.get_dummies(Destination, drop_first = True)
            Destination = Destination.applymap(lambda x: 1 if x else 0)

            # Additional_Info contains almost 80% no_info
            # Route and Total_Stops are related to each other
            test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

            # Replacing Total_Stops
            test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

            # Concatenate dataframe --> test_data + Airline + Source + Destination
            test_arr = pd.concat([test_data, Airline, Source, Destination], axis = 1)

            test_arr.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
            test_arr['Fare']=test_arr['Price']
            
            test_arr.drop(["Price","Unnamed: 0"], axis = 1, inplace = True)

        
            logging.info("Completed Preprocessing------->Training Data")

            train_arr.to_csv(self.preprocess_config.train_data_prepro,index=False,header=True)
            test_arr.to_csv(self.preprocess_config.test_data_prepro,index=False,header=True)

            logging.info('preprocessing  train and test data completed')
            logging.info(f'Train Pre Dataframe Head : \n{train_arr.head().to_string()}')
            logging.info(f'Test Pre Dataframe Head  : \n{test_arr.head().to_string()}')
            

            return (train_arr,test_arr)
            
        except Exception as e:
            logging.info("Exception occured in the Data Preprocessing")

            raise CustomException(e,sys)

        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr= data_transformation.get_data_transformation(train_data_path, test_data_path)
    