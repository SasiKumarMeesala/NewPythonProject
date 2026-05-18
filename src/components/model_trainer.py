import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {'learning_rate': [.1, .01, .05  , .001], 'n_estimators': [8, 16, 32, 64, 128, 256], 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]},
                "Linear Regression": {},
                "XGBRegressor": {'learning_rate': [.1, .01, .05  , .001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
                "CatBoosting Regressor": {'learning_rate': [.1, .01, .05  , .001], 'n_estimators': [8, 16, 32, 64, 128, 256], 'iterations': [30, 50, 100, 200]},
                "AdaBoost Regressor": {'learning_rate': [.1, .01, .05  , .001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 Score for the best model: {r2_square}")
            return r2_square
        except Exception as e:
            logging.info("Error occurred in Model Training")
            raise CustomException(e, sys)