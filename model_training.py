# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:22:45 2023

@author: mladen
"""

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta



def train_model(prepared_data_forec_sales, prepared_data_forec_date):
    
    '''
    The function runs training LGBMRegressor algorithms.Two independent models are trained, 
    one for predicting sales volume and the other for predicting delivery date.
    '''
    
    #create model directory
    os.chdir('D:/MyData/Data Science Mladen/MyGitHubProjects/forecasting_python')

    if not os.path.exists("models"):
        os.makedirs("models")
    
    directory = os.getcwd()
    model_directory  = os.path.join(directory, 'models')
    
    #models file name    
    model_file_path_forec_sales = os.path.join(model_directory, 'lightGBM_forec_sales.joblib')
    model_file_path_forec_date = os.path.join(model_directory, 'lightGBM_forec_date.joblib')
    
    
    # traning for sales volume predictions ------------------------------------------------------------------------------------

    
    X_train = prepared_data_forec_sales.drop(['date',
                                              'deliveries',
                                              'returns',
                                              'sale',
                                              'sale_without_increase',
                                              'days_until_next_date'], axis=1)
    
    
    y_train = prepared_data_forec_sales['sale']
    
    
    
    def read_model(models_file_path):
        return joblib.load(models_file_path)
    
    
    
    last_training_forec_sales = datetime.fromtimestamp(os.path.getmtime(model_file_path_forec_sales))
    date_now = datetime.now()
    
    if (date_now - last_training_forec_sales).days >= 7: 
        #if a week has passed
        cv_split = TimeSeriesSplit(n_splits = 4)
        
        hyp_parameters = {
            "max_depth": [3, 4, 6, 5, 10],
            "num_leaves": [10, 20, 30, 40, 100, 120],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [50, 100, 300, 500, 700, 900, 1000],
            "colsample_bytree": [0.3, 0.5, 0.7, 1],
            "objective": ['tweedie', 'regression']
         } 
        
        
        lgb_regress = LGBMRegressor()
        grid_search = GridSearchCV(estimator = lgb_regress, 
                                   cv = cv_split, 
                                   param_grid = hyp_parameters, 
                                   scoring = "neg_mean_absolute_error",
                                   n_jobs= -1)
        
        grid_search.fit(X_train, y_train)
       
        best_hyp_parameters = grid_search.best_params_
        #best_model = grid_search.best_estimator_
        
        cv_results_forec_sales = pd.DataFrame( grid_search.cv_results_ )
        
        
        lgb_model = LGBMRegressor(**best_hyp_parameters)
        best_model_forec_sales = lgb_model.fit(X_train, y_train)
        
        # save new best model hyperparameters
        joblib.dump(best_hyp_parameters, model_file_path_forec_sales)
        
        print("parameters search after a one week")
        
    else:
        #if it hasn't been a one week - read previous model hyperparameters 
        best_hyp_parameters = read_model(model_file_path_forec_sales)
        lgb_regress = LGBMRegressor()

        cv_split = TimeSeriesSplit(n_splits = 4)

        grid_search = GridSearchCV(estimator = lgb_regress, 
                                   cv = cv_split, 
                                   #param_grid = best_hyp_parameters, 
                                   param_grid = {key: [best_hyp_parameters[key]] for key in best_hyp_parameters},
                                   scoring = "neg_mean_absolute_error",
                                   n_jobs= -1)
        
        grid_search.fit(X_train, y_train)
        
        lgb_model = LGBMRegressor(**best_hyp_parameters)
        best_model_forec_sales = lgb_model.fit(X_train, y_train)
       
        cv_results_forec_sales = pd.DataFrame( grid_search.cv_results_ )
        
        print("model with parameters in the current week")
        
        
    # training for next date of deliveries predictions ---------------------------------------------------------

    last_training_forec_date = datetime.fromtimestamp(os.path.getmtime(model_file_path_forec_date))

    features_for_checks_forec_date = prepared_data_forec_date[['date',
                                                               'days_until_next_date']]
    
    
    X_train_forec_date = prepared_data_forec_date.drop(['date',
                                                        'dayOfWeek',
                                                        'month',
                                                        'quarter',
                                                        'season',
                                                        'week_in_month',
                                                        'holidays',
                                                        'days_until_next_date',
                                                        'returns',
                                                        'sale',
                                                        'sale_without_increase'], axis=1)
    

    
    y_train_forec_date = prepared_data_forec_date['days_until_next_date']
    
    
    
    if (date_now - last_training_forec_date).days >= 7: 
        #if a week has passed
        cv_split = TimeSeriesSplit(n_splits = 4)
        
        hyp_parameters = {
            "max_depth": [3, 4, 6, 5, 10],
            "num_leaves": [10, 20, 30, 40, 100, 120],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [50, 100, 300, 500, 700, 900, 1000],
            "colsample_bytree": [0.3, 0.5, 0.7, 1],
            'objective': ['tweedie', 'regression']
         } 
        
        
        lgb_regress = LGBMRegressor()
        grid_search_forec_date = GridSearchCV(estimator = lgb_regress, 
                                               cv = cv_split, 
                                               param_grid = hyp_parameters, 
                                               scoring = "neg_mean_absolute_error",
                                               n_jobs= -1)
        
        grid_search_forec_date.fit(X_train_forec_date, y_train_forec_date)
       
        best_hyp_parameters_forec_date = grid_search_forec_date.best_params_
        #best_model = grid_search.best_estimator_
        
        cv_results_forec_date = pd.DataFrame( grid_search_forec_date.cv_results_ )
        
        
        lgb_model_forec_date = LGBMRegressor(**best_hyp_parameters_forec_date)
        best_model_forec_date = lgb_model_forec_date.fit(X_train_forec_date, y_train_forec_date)
        
        # save new best model hyperparameters
        joblib.dump(best_hyp_parameters_forec_date, model_file_path_forec_date)
        
        print("parameters search after a one week")
        
    else:
        #if it hasn't been a one week - read previous model hyperparameters
        best_hyp_parameters_forec_date = read_model(model_file_path_forec_date)
        lgb_regress = LGBMRegressor()

        cv_split = TimeSeriesSplit(n_splits = 4)

        grid_search_forec_date = GridSearchCV(estimator = lgb_regress, 
                                               cv = cv_split, 
                                               #param_grid = best_hyp_parameters, 
                                               param_grid = {key: [best_hyp_parameters_forec_date[key]] for key in best_hyp_parameters_forec_date},
                                               scoring = "neg_mean_absolute_error",
                                               n_jobs= -1)
        
        grid_search_forec_date.fit(X_train_forec_date, y_train_forec_date)
        
        lgb_model_forec_date = LGBMRegressor(**best_hyp_parameters_forec_date)
        best_model_forec_date = lgb_model_forec_date.fit(X_train_forec_date, y_train_forec_date)
       
        cv_results_forec_date = pd.DataFrame( grid_search_forec_date.cv_results_ )
        
        print("model with parameters in the current week")



    return {'best_model_forec_sales': best_model_forec_sales, 
            'cv_results_forec_sales': cv_results_forec_sales,
            'best_model_forec_date': best_model_forec_date,
            'cv_results_forec_date': cv_results_forec_date}
    
    
    


    
