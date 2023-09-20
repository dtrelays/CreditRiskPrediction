import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle  
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def get_model_accuracy(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square
    
def evaluate_models(X_train, y_train,X_val,y_val,models,param):
    try:        
        report = []
        
        for i in range(len(list(models))):
    
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            gs = RandomizedSearchCV(model, param_distributions=para, cv=3, n_iter=50,n_jobs=-1)
        
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Evaluate Train and Test dataset
            model_train_mae , model_train_rmse, model_train_r2 = get_model_accuracy(y_train, y_train_pred)

            model_test_mae , model_test_rmse, model_test_r2 = get_model_accuracy(y_val, y_val_pred)

            test_model_score = r2_score(y_val,y_val_pred)
                
            print(list(models.keys())[i])
            
            best_params = gs.best_params_
            best_estimator = gs.best_estimator_
            best_score = gs.best_score_
            
            # Print the best parameters and best score
            print("Best Parameters:", best_params)
            print("Best Score:", best_score)
            
            print('Model performance for Training set')
            print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
            print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
            print("- R2 Score: {:.4f}".format(model_train_r2))

            print('----------------------------------')
            
            print('Model performance for Test set')
            print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
            print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
            print("- R2 Score: {:.4f}".format(model_test_r2))            

            report.append({'Model': list(models.keys())[i], 'BestScore': model_test_r2, 'ModelParams': best_params})
            
            print('='*35)
            print('\n')
              
        data = pd.DataFrame(report)

        # Sort the DataFrame by the 'FloatColumn' in descending order
        data = data.sort_values(by='BestScore', ascending=False)

        # Reset the index to have continuous index values
        data.reset_index(drop=True, inplace=True)
        
        return data

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def get_categorical_variables():
    try:
        
        df_categorical_var = pd.read_csv('notebook/data/credit_data.csv')

        gender = df_categorical_var['Gender'].unique().tolist()

        df_state_city = df_categorical_var[['City', 'State']].drop_duplicates()

        states = df_state_city['State'].unique().tolist()

        df_categorical_var['Occupation'] = df_categorical_var['Occupation'].fillna('Others')

        occupation_list = df_categorical_var['Occupation'].unique().tolist()

        employment_type = df_categorical_var['Employment Profile'].unique().tolist()

        return gender, states, df_state_city, occupation_list, employment_type

    except Exception as e:
        raise CustomException(e, sys)

