import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "XGBRegressor": XGBRegressor(verbosity=0),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "Decision Tree": DecisionTreeRegressor(),
            }
            
            params = {
                    "Linear Regression": {
                        'fit_intercept': [True, False]
                    },
                    "Lasso": {
                        'alpha': [0.1, 0.01, 0.001]
                    },
                    "Ridge": {
                        'alpha': [0.1, 0.01, 0.001]
                    },
                    "Decision Tree": {
                        'criterion': ['absolute_error','squared_error', 'friedman_mse', 'poisson'],
                        'splitter': ['best', 'random'],
                        'max_features': ['sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4, 8],
                    },
                    "Random Forest Regressor": {
                        'n_estimators': [16,64, 128],
                        'max_features': ['sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4],
                    },
                    "XGBRegressor": {
                        'learning_rate': [0.1, 0.01, 0.05],
                        'n_estimators': [8, 16, 32, 64,128],
                        'max_depth': [3, 4, 6,8],
                        'subsample': [0.6, 0.7, 0.8, 0.9],
                    },
                    "CatBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    }
                }

            model_report=evaluate_models(X_train=X_train,y_train=y_train,X_val=X_test,y_val=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = model_report.loc[0,'BestScore']

            ## To get best model name from dict

            best_model_name = model_report.loc[0,'Model']
            
            best_model = models[best_model_name]
            
            best_params = model_report.loc[0,'ModelParams']

            model_and_params = {
                            "model": best_model,
                            "params": best_params
                            }

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_and_params
            )

            model_with_loaded_params = best_model.set_params(**best_params)
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
            
        except Exception as e:
            raise CustomException(e,sys)