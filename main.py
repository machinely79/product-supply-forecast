# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:31:06 2023

@author: mladen

"""

import os
import pandas as pd
import numpy as np

#Project directory
os.chdir('D:/MyData/Data Science Mladen/MyGitHubProjects/forecasting_python')



#myModules
import get_data
import prepare_data
import feature_engineering   
import model_training
import predictions

#set data paths
file_path = r'D:/MyData/Data Science Mladen/MyGitHubProjects/forecasting_python'
deliveries_file = r'data\ot2018.csv'
product_returns_file = r'data\rk2018.csv'
products_category_file = r'data\prod_category.csv'


# Run Functions
data = get_data.load_data(file_path, deliveries_file, product_returns_file, products_category_file)
prepared_data = prepare_data.prepare(data['product_deliveries'], 
                                     data['product_returns'], 
                                     data['products_category'])

prepared_data = feature_engineering.make_engineering(prepared_data)

model_trained = model_training.train_model(prepared_data['prepared_data_forec_sales'], 
                                           prepared_data['prepared_data_forec_date'])

predictions = predictions.make_predictions(model_trained, 
                                           prepared_data['prepared_data_forec_sales'], 
                                           prepared_data['prepared_data_forec_date'], 
                                           prepared_data['season_mapping'], 
                                           prepared_data['product_mapping'], 
                                           prepared_data['delivery_point_mapping'])




