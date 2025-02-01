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


def make_predictions(model_trained, prepared_data_forec_sales, prepared_data_forec_date, sezona_map, artikal_map, objekat_map):
    
    '''
    The function utilizes a trained model to predict sales quantity and delivery dates for the next day.
    For sales prediction (next delivery), it generates new features such as season, week in the month, holidays, and sales lags, 
    then uses the model to predict the sales quantity for the next day. For delivery dates, it takes the previous delivery date 
    data and predicts when the next delivery will occur, considering holidays. Finally, it generates a JSON structure mapping 
    object and article codes to the predicted delivery date and sales quantity. The output includes a JSON with the predictions
    and a DataFrame with detailed information.
    '''
    
    #for sales predictions -----------------------------
    
    next_day = prepared_data_forec_sales['datum'].max() + timedelta(days=1)
    next_day_df = pd.DataFrame({'datum': [next_day]})
    
    #get season
    next_day_df['date_offset'] = (next_day_df.datum.dt.month*100 + next_day_df.datum.dt.day - 320)%1300
    next_day_df['sezona'] = pd.cut(next_day_df['date_offset'], 
                                     [0, 300, 602, 900, 1300], 
                          labels=['spring', 'summer', 'autumn', 'winter'],
                          include_lowest = True)
    next_day_df.drop(['date_offset'],
                       inplace=True, axis=1)
    
    next_day_df['week_in_month'] = next_day_df['datum'].dt.day // 7 + 1

    # add holidays - Serbia
    def add_holidays(df):
        holiday = holidays.Serbia(years=df['datum'].dt.year)
        df['holidays'] = np.where(df['datum'].dt.date.astype('datetime64').isin(holiday.keys()), 1, 0)
        return df
    
    next_day_df = add_holidays(next_day_df)

    next_day_prediction_df = prepared_data_forec_sales.groupby(['sifra_objekta', 'sifra_artikla']).apply(lambda group: pd.DataFrame({
            'sifra_objekta': [group['sifra_objekta'].iloc[0]],
            'sifra_artikla': [group['sifra_artikla'].iloc[0]],
            'dayOfWeek': [next_day.weekday()],
            'month': [next_day.month],
            'quarter': [next_day.quarter],
            'sezona': [next_day_df['sezona'].iloc[0]],
            'week_in_month': [next_day_df['week_in_month'].iloc[0]],
            'holidays': [next_day_df['holidays'].iloc[0]],
            'lag_sales_1': [group['prodaja'].iloc[-1]],  
            'lag_sales_2': [group['prodaja'].iloc[-2]], 
            'lag_sales_3': [group['prodaja'].iloc[-3]], 
            'lag_sales_4': [group['prodaja'].iloc[-4]], 
            'lag_sales_5': [group['prodaja'].iloc[-5]], 
            'lag_sales_6': [group['prodaja'].iloc[-6]], 
            'lag_sales_7': [group['prodaja'].iloc[-7]], 
            'lag_sales_8': [group['prodaja'].iloc[-8]], 
            'lag_sales_9': [group['prodaja'].iloc[-9]], 
            'lag_sales_10': [group['prodaja'].iloc[-10]], 
            'lag_sales_11': [group['prodaja'].iloc[-11]],  
            'lag_sales_12': [group['prodaja'].iloc[-12]], 
            'lag_sales_13': [group['prodaja'].iloc[-13]], 
            'lag_sales_14': [group['prodaja'].iloc[-14]], 
            'lag_sales_15': [group['prodaja'].iloc[-15]], 
            'lag_sales_16': [group['prodaja'].iloc[-16]], 
            'lag_sales_17': [group['prodaja'].iloc[-17]], 
            'lag_sales_18': [group['prodaja'].iloc[-18]], 
            'lag_sales_19': [group['prodaja'].iloc[-19]], 
            'lag_sales_20': [group['prodaja'].iloc[-20]], 
            'lag_sales_21': [group['prodaja'].iloc[-21]],  
            'lag_sales_22': [group['prodaja'].iloc[-22]], 
            'lag_sales_23': [group['prodaja'].iloc[-23]], 
            'lag_sales_24': [group['prodaja'].iloc[-24]], 
            'lag_sales_25': [group['prodaja'].iloc[-25]], 
            'lag_sales_26': [group['prodaja'].iloc[-26]], 
            'lag_sales_27': [group['prodaja'].iloc[-27]], 
            'lag_sales_28': [group['prodaja'].iloc[-28]], 
            'lag_sales_29': [group['prodaja'].iloc[-29]], 
            'lag_sales_30': [group['prodaja'].iloc[-30]],
            'sales_differences_per_day': [group['prodaja'].diff(1).iloc[-1]],
            'mean_sales_7day': [group['prodaja'].rolling(7).mean().iloc[-1]], 
            'std_sales_7day': [group['prodaja'].rolling(7).std().iloc[-1]]   
    })).reset_index(drop=True)
    
    
    # coding string sezona to int
    next_day_prediction_df['sezona'] = next_day_prediction_df['sezona'].map(sezona_map).astype('int')

    # make predictions
    model_forec_sales = model_trained['best_model_forec_sales'] # get model from dic
    predictions = model_forec_sales.predict(next_day_prediction_df)
    next_day_prediction_df['quantity_predictions'] = predictions

    #inverse mapping (artikal and objekat)
    inv_sifra_artikla_map={v: k for k, v in artikal_map.items()}
    inv_sifra_objekta_map={v: k for k, v in objekat_map.items()}

    inv_mapping = [inv_sifra_artikla_map, inv_sifra_objekta_map] 
    columns = ['sifra_artikla', 'sifra_objekta']

    for i, j in zip(columns, inv_mapping):
        #print(str(i) + "--> " + str(j))    
        next_day_prediction_df[i] = next_day_prediction_df[i].map(j)


    selected_columns = ['sifra_objekta', 'sifra_artikla', 'quantity_predictions']
    df_sales_predictions = next_day_prediction_df[selected_columns]
   
    #df_sales_predictions['datum'] = next_day
    df_sales_predictions['quantity_predictions'] = df_sales_predictions['quantity_predictions'].round(0)

    
    #  for date of deliveries predictions -----------------------------

    
    last_deliveries_date = prepared_data_forec_date.groupby(['sifra_objekta', 'sifra_artikla'])['datum'].max().reset_index()
    last_deliveries_date.rename(columns={'datum': 'last_deliv_date'}, inplace=True)
    
    for i, j in zip(columns, inv_mapping):
         #print(str(i) + "--> " + str(j))    
         last_deliveries_date[i] = last_deliveries_date[i].map(j)
    

                                                    
    next_deliveries_prediction_df = prepared_data_forec_date.groupby(['sifra_objekta', 'sifra_artikla']).apply(lambda group: pd.DataFrame({
                'sifra_objekta': [group['sifra_objekta'].iloc[0]],
                'sifra_artikla': [group['sifra_artikla'].iloc[0]],
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
         
         
    selected_columns = ['sifra_objekta', 'sifra_artikla', 'deliverie_predictions']
    df_deliveries_predictions = next_deliveries_prediction_df[selected_columns]
       
    df_deliveries_predictions['deliverie_predictions'] = df_deliveries_predictions['deliverie_predictions'].round(0)
        
    
    df_deliveries_predictions = df_deliveries_predictions.merge(last_deliveries_date, on=['sifra_objekta',
                                                                                          'sifra_artikla'],
                                                                                            how='left')
    #the largest date in the data set
    max_date = prepared_data_forec_date['datum'].max()
    
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
    
        # checking if predicted_date falls on a holiday 
        serbia_holidays = holidays.Serbia()
        while predicted_date in serbia_holidays:
            predicted_date += timedelta(days=1)
    
        return predicted_date
    

    df_deliveries_predictions['predicted_date_avoiding_holidays'] = df_deliveries_predictions.apply(add_predicted_date_holidays, axis=1)

    
    
    for_delete = ['max_date', 'deliverie_predictions', 'last_deliv_date']

    df_deliveries_predictions = df_deliveries_predictions.drop(columns=for_delete)

    
    output_predictions = df_sales_predictions.merge(df_deliveries_predictions, on=['sifra_objekta',
                                                                                   'sifra_artikla'], how='left')
    
    #trans to JSON file (šifra objekta -> šifta artikla -> {datum isporuke, količina isporuke})
    nested_tree = {}

    for index, row in output_predictions.iterrows():
        sifra_objekta = row['sifra_objekta']
        sifra_artikla = row['sifra_artikla']
        sales_prediction = row['quantity_predictions']
        delivery_date = row['predicted_delivery_date']
        delivery_date_avoiding_holidays = row['predicted_date_avoiding_holidays']

        if sifra_objekta not in nested_tree:
            nested_tree[sifra_objekta] = {}

        if sifra_artikla not in nested_tree[sifra_objekta]:
            nested_tree[sifra_objekta][sifra_artikla] = {'estimated delivery date': delivery_date.strftime('%Y-%m-%d'),
                                                         'estimated delivery date if the store is closed on holidays': delivery_date_avoiding_holidays.strftime('%Y-%m-%d'),
                                                         'estimated quantity': sales_prediction}


    json_predictions = json.dumps(nested_tree, indent=4)
    
    
    return {'json_predictions': json_predictions, 
            'df_output_predictions': output_predictions}
    
    
    
   
    
  
    