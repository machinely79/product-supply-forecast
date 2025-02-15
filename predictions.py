# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:31:19 2023

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
import holidays
import json


def make_predictions(model_trained, prepared_data_forec_sales, prepared_data_forec_date, season_mapping, product_mapping, delivery_point_mapping):
    
    '''
    The function utilizes a trained model to predict sales quantity and delivery dates for the next day.
    For sales prediction (next delivery), it generates new features such as season, week in the month, holidays, and sales lags, 
    then uses the model to predict the sales quantity for the next day. For delivery dates, it takes the previous delivery date 
    data and predicts when the next delivery will occur, considering holidays. Finally, it generates a JSON structure mapping 
    object and article codes to the predicted delivery date and sales quantity. The output includes a JSON with the predictions
    and a DataFrame with detailed information.
    '''
    
    #for sales predictions -----------------------------
    
    next_day = prepared_data_forec_sales['date'].max() + timedelta(days=1)
    next_day_df = pd.DataFrame({'date': [next_day]})
    
    #get season
    next_day_df['date_offset'] = (next_day_df.date.dt.month*100 + next_day_df.date.dt.day - 320)%1300
    next_day_df['season'] = pd.cut(next_day_df['date_offset'], 
                                     [0, 300, 602, 900, 1300], 
                          labels=['spring', 'summer', 'autumn', 'winter'],
                          include_lowest = True)
    next_day_df.drop(['date_offset'],
                       inplace=True, axis=1)
    
    next_day_df['week_in_month'] = next_day_df['date'].dt.day // 7 + 1

    # add holidays - Serbia
    def add_holidays(df):
        holiday = holidays.Serbia(years=df['date'].dt.year)
        df['holidays'] = np.where(df['date'].dt.date.astype('datetime64').isin(holiday.keys()), 1, 0)
        return df
    
    next_day_df = add_holidays(next_day_df)

    next_day_prediction_df = prepared_data_forec_sales.groupby(['delivery_point_id', 'product_id']).apply(lambda group: pd.DataFrame({
            'delivery_point_id': [group['delivery_point_id'].iloc[0]],
            'product_id': [group['product_id'].iloc[0]],
            'dayOfWeek': [next_day.weekday()],
            'month': [next_day.month],
            'quarter': [next_day.quarter],
            'season': [next_day_df['season'].iloc[0]],
            'week_in_month': [next_day_df['week_in_month'].iloc[0]],
            'holidays': [next_day_df['holidays'].iloc[0]],
            'lag_sales_1': [group['sale'].iloc[-1]],  
            'lag_sales_2': [group['sale'].iloc[-2]], 
            'lag_sales_3': [group['sale'].iloc[-3]], 
            'lag_sales_4': [group['sale'].iloc[-4]], 
            'lag_sales_5': [group['sale'].iloc[-5]], 
            'lag_sales_6': [group['sale'].iloc[-6]], 
            'lag_sales_7': [group['sale'].iloc[-7]], 
            'lag_sales_8': [group['sale'].iloc[-8]], 
            'lag_sales_9': [group['sale'].iloc[-9]], 
            'lag_sales_10': [group['sale'].iloc[-10]], 
            'lag_sales_11': [group['sale'].iloc[-11]],  
            'lag_sales_12': [group['sale'].iloc[-12]], 
            'lag_sales_13': [group['sale'].iloc[-13]], 
            'lag_sales_14': [group['sale'].iloc[-14]], 
            'lag_sales_15': [group['sale'].iloc[-15]], 
            'lag_sales_16': [group['sale'].iloc[-16]], 
            'lag_sales_17': [group['sale'].iloc[-17]], 
            'lag_sales_18': [group['sale'].iloc[-18]], 
            'lag_sales_19': [group['sale'].iloc[-19]], 
            'lag_sales_20': [group['sale'].iloc[-20]], 
            'lag_sales_21': [group['sale'].iloc[-21]],  
            'lag_sales_22': [group['sale'].iloc[-22]], 
            'lag_sales_23': [group['sale'].iloc[-23]], 
            'lag_sales_24': [group['sale'].iloc[-24]], 
            'lag_sales_25': [group['sale'].iloc[-25]], 
            'lag_sales_26': [group['sale'].iloc[-26]], 
            'lag_sales_27': [group['sale'].iloc[-27]], 
            'lag_sales_28': [group['sale'].iloc[-28]], 
            'lag_sales_29': [group['sale'].iloc[-29]], 
            'lag_sales_30': [group['sale'].iloc[-30]],
            'sales_differences_per_day': [group['sale'].diff(1).iloc[-1]],
            'mean_sales_7day': [group['sale'].rolling(7).mean().iloc[-1]], 
            'std_sales_7day': [group['sale'].rolling(7).std().iloc[-1]]   
    })).reset_index(drop=True)
    
    
    # string  to int
    next_day_prediction_df['season'] = next_day_prediction_df['season'].map(season_mapping).astype('int')

    # make predictions
    model_forec_sales = model_trained['best_model_forec_sales'] # get model from dic
    predictions = model_forec_sales.predict(next_day_prediction_df)
    next_day_prediction_df['quantity_predictions'] = predictions

    #inverse mapping
    inv_product_id_map={v: k for k, v in product_mapping.items()}
    inv_delivery_point_id_map={v: k for k, v in delivery_point_mapping.items()}

    inv_mapping = [inv_product_id_map, inv_delivery_point_id_map] 
    columns = ['product_id', 'delivery_point_id']

    for i, j in zip(columns, inv_mapping):
        #print(str(i) + "--> " + str(j))    
        next_day_prediction_df[i] = next_day_prediction_df[i].map(j)


    selected_columns = ['delivery_point_id', 'product_id', 'quantity_predictions']
    df_sales_predictions = next_day_prediction_df[selected_columns]
   
    #df_sales_predictions['date'] = next_day
    df_sales_predictions['quantity_predictions'] = df_sales_predictions['quantity_predictions'].round(0)

    
    #  for date of delivery predictions -----------------------------

    
    last_deliveries_date = prepared_data_forec_date.groupby(['delivery_point_id', 'product_id'])['date'].max().reset_index()
    last_deliveries_date.rename(columns={'date': 'last_deliv_date'}, inplace=True)
    
    for i, j in zip(columns, inv_mapping):
         #print(str(i) + "--> " + str(j))    
         last_deliveries_date[i] = last_deliveries_date[i].map(j)
    

                                                    
    next_deliveries_prediction_df = prepared_data_forec_date.groupby(['delivery_point_id', 'product_id']).apply(lambda group: pd.DataFrame({
                'delivery_point_id': [group['delivery_point_id'].iloc[0]],
                'product_id': [group['product_id'].iloc[0]],
                'lag_num_days_1': [group['days_until_next_date'].iloc[-1]],  
                'lag_num_days_2': [group['days_until_next_date'].iloc[-2]], 
                'lag_num_days_3': [group['days_until_next_date'].iloc[-3]], 
                'lag_num_days_4': [group['days_until_next_date'].iloc[-4]], 
                'lag_num_days_5': [group['days_until_next_date'].iloc[-5]], 
                'lag_num_days_6': [group['days_until_next_date'].iloc[-6]], 
                'lag_num_days_7': [group['days_until_next_date'].iloc[-7]], 
                'lag_num_days_8': [group['days_until_next_date'].iloc[-8]], 
                'lag_num_days_9': [group['days_until_next_date'].iloc[-9]], 
                'lag_num_days_10': [group['days_until_next_date'].iloc[-10]], 
                'lag_num_days_11': [group['days_until_next_date'].iloc[-11]],  
                'lag_num_days_12': [group['days_until_next_date'].iloc[-12]], 
                'lag_num_days_13': [group['days_until_next_date'].iloc[-13]], 
                'lag_num_days_14': [group['days_until_next_date'].iloc[-14]], 
                'lag_num_days_15': [group['days_until_next_date'].iloc[-15]], 
                'lag_num_days_16': [group['days_until_next_date'].iloc[-16]], 
                'lag_num_days_17': [group['days_until_next_date'].iloc[-17]], 
                'lag_num_days_18': [group['days_until_next_date'].iloc[-18]], 
                'lag_num_days_19': [group['days_until_next_date'].iloc[-19]], 
                'lag_num_days_20': [group['days_until_next_date'].iloc[-20]], 
                'lag_num_days_21': [group['days_until_next_date'].iloc[-21]],  
                'lag_num_days_22': [group['days_until_next_date'].iloc[-22]], 
                'lag_num_days_23': [group['days_until_next_date'].iloc[-23]], 
                'lag_num_days_24': [group['days_until_next_date'].iloc[-24]], 
                'lag_num_days_25': [group['days_until_next_date'].iloc[-25]], 
                'lag_num_days_26': [group['days_until_next_date'].iloc[-26]], 
                'lag_num_days_27': [group['days_until_next_date'].iloc[-27]], 
                'lag_num_days_28': [group['days_until_next_date'].iloc[-28]], 
                'lag_num_days_29': [group['days_until_next_date'].iloc[-29]], 
                'lag_num_days_30': [group['days_until_next_date'].iloc[-30]],
                'num_days_differences_per_day': [group['days_until_next_date'].diff(1).iloc[-1]],
                'mean_num_days_20day': [group['days_until_next_date'].rolling(20).mean().iloc[-1]], 
                'std_num_days_20day': [group['days_until_next_date'].rolling(20).std().iloc[-1]]         
    })).reset_index(drop=True)

    
    # make predictions
    model_forec_date = model_trained['best_model_forec_date'] # get model from dic
    predictions_forec_date = model_forec_date.predict(next_deliveries_prediction_df)
    next_deliveries_prediction_df['deliverie_predictions'] = predictions_forec_date

    
    for i, j in zip(columns, inv_mapping):
         #print(str(i) + "--> " + str(j))    
         next_deliveries_prediction_df[i] = next_deliveries_prediction_df[i].map(j)
         
         
    selected_columns = ['delivery_point_id', 'product_id', 'deliverie_predictions']
    df_deliveries_predictions = next_deliveries_prediction_df[selected_columns]
       
    df_deliveries_predictions['deliverie_predictions'] = df_deliveries_predictions['deliverie_predictions'].round(0)
        
    
    df_deliveries_predictions = df_deliveries_predictions.merge(last_deliveries_date, on=['delivery_point_id',
                                                                                          'product_id'],
                                                                                            how='left')
    
    max_date = prepared_data_forec_date['date'].max()
    
    df_deliveries_predictions['max_date'] = max_date

    def add_predicted_date(df):
        if (df['max_date'] - df['last_deliv_date']).days >= df['deliverie_predictions']:
            return df['max_date'] + timedelta(days=1)
        else:
            return df['last_deliv_date'] + timedelta(days=df['deliverie_predictions'])
    
    
    df_deliveries_predictions['predicted_delivery_date'] = df_deliveries_predictions.apply(add_predicted_date, axis=1)
    
    

    def add_predicted_date_holidays(df):
        if (df['max_date'] - df['last_deliv_date']).days >= df['deliverie_predictions']:
            predicted_date = df['max_date'] + timedelta(days=1)
        else:
            predicted_date = df['last_deliv_date'] + timedelta(days=df['deliverie_predictions'])
    
        # Checking if the predicted date falls on a holiday.
        serbia_holidays = holidays.Serbia()
        while predicted_date in serbia_holidays:
            predicted_date += timedelta(days=1)
    
        return predicted_date
    

    df_deliveries_predictions['predicted_date_avoiding_holidays'] = df_deliveries_predictions.apply(add_predicted_date_holidays, axis=1)

    
    
    for_delete = ['max_date', 'deliverie_predictions', 'last_deliv_date']

    df_deliveries_predictions = df_deliveries_predictions.drop(columns=for_delete)

    
    output_predictions = df_sales_predictions.merge(df_deliveries_predictions, on=['delivery_point_id',
                                                                                   'product_id'], how='left')
    
    #trans to JSON file (delivery_point_id -> product_id -> {date of deliveries, delivery quantity})
    nested_tree = {}

    for index, row in output_predictions.iterrows():
        delivery_point_id = row['delivery_point_id']
        product_id = row['product_id']
        sales_prediction = row['quantity_predictions']
        delivery_date = row['predicted_delivery_date']
        delivery_date_avoiding_holidays = row['predicted_date_avoiding_holidays']

        if delivery_point_id not in nested_tree:
            nested_tree[delivery_point_id] = {}

        if product_id not in nested_tree[delivery_point_id]:
            nested_tree[delivery_point_id][product_id] = {'estimated delivery date': delivery_date.strftime('%Y-%m-%d'),
                                                         'estimated delivery date if the store is closed on holidays': delivery_date_avoiding_holidays.strftime('%Y-%m-%d'),
                                                         'estimated quantity': sales_prediction}


    json_predictions = json.dumps(nested_tree, indent=4)
    
    
    return {'json_predictions': json_predictions, 
            'df_output_predictions': output_predictions}
    
    
    
   
    
  
    
