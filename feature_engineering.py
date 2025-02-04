# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:44:32 2023

@author: mladen
"""

import pandas as pd
import numpy as np
import holidays

def make_engineering(prepared_data):
    
    '''
    The function performs features engineering from available features in data. The data will 
    be split into two sets of data needed for sales forecasting (next delivery) and delivery date forecasting, 
    Function creates new derived features: sliding window, season, holidays, etc.
    '''
    
    def calculate_days_between_dates(data):
        # Convert the 'date' column to datetime format
        data['date'] = pd.to_datetime(data['date'])

        # Sort the data by 'delivery_point_id', 'product_id', and 'date'
        data.sort_values(['delivery_point_id', 'product_id', 'date'], ascending = True)

        # Calculate the difference in days between the current and next date in each group
        data['days_until_next_date'] = data.groupby(['delivery_point_id', 'product_id'])['date'].diff().dt.days

        return data


    prepared_data = calculate_days_between_dates(prepared_data)
    
   
    prepared_data['dayOfWeek'] = prepared_data['date'].dt.dayofweek # 0 = mon ... 6 = sun
    prepared_data['month'] = prepared_data['date'].dt.month         # 1 = jan ... 12 = dec
    prepared_data["quarter"] = prepared_data["date"].dt.quarter     # 1 = 1.q .... 4 = 4.q
    
    #season
    prepared_data['date_offset'] = (prepared_data.date.dt.month*100 + prepared_data.date.dt.day - 320)%1300

    prepared_data['season'] = pd.cut(prepared_data['date_offset'], 
                                     [0, 300, 602, 900, 1300], 
                          labels=['spring', 'summer', 'autumn', 'winter'],
                          include_lowest = True)
    
    prepared_data.drop(['date_offset'],
                       inplace=True, axis=1)

    # week of the month - in order
    prepared_data['week_in_month'] = prepared_data['date'].dt.day // 7 + 1
    
    # add holidays
    def add_holidays(df):
        holiday = holidays.Serbia(years=df['date'].dt.year)
        df['holidays'] = np.where(df['date'].dt.date.astype('datetime64').isin(holiday.keys()), 1, 0)
        return df
    
    prepared_data = add_holidays(prepared_data)
    
    # make dictionary with unique values (index and value)
    season_mapping = {label: idx for idx, label in 
                          enumerate(np.unique(prepared_data['season']))}


    product_mapping = {label: idx for idx, label in 
                          enumerate(np.unique(prepared_data['product_id']))}

    delivery_point_mapping = {label: idx for idx, label in 
                          enumerate(np.unique(prepared_data['delivery_point_id']))}
  
    # mapping
    mapping=[season_mapping, product_mapping, delivery_point_mapping ] 
    columns=['season', 'product_id', 'delivery_point_id']
    
    # encode the values
    for i, j in zip(columns, mapping):
        #print(str(i) + "--> " + str(j))    
        prepared_data[i]=prepared_data[i].map(j).astype('int')
        
      
        
    # copy DataFrame for days_until_next_date
    prepared_data_forec_date = prepared_data.copy() 
    
    
    # sliding windows for sales - lags "sale column"         
    for i in range(1,31):   
        field_name = 'lag_sales_' + str(i)
        prepared_data[field_name] = prepared_data.groupby(['delivery_point_id', 'product_id'])['sale'].shift(i)

 
    # diff, mean, std, min, max - "sale column"
    prepared_data["sales_differences_per_day"] = prepared_data.groupby(['delivery_point_id', 
                                                                'product_id'])['sale'].diff(1)
    #Seasonality with moving average
    prepared_data["mean_sales_7day"] = prepared_data.groupby(['delivery_point_id', 
                                                                'product_id'])['sale'].rolling(7).mean().droplevel(level=[0,1]) 

    prepared_data["std_sales_7day"] =  prepared_data.groupby(['delivery_point_id', 
                                                              'product_id'])['sale'].rolling(7).std().droplevel(level=[0,1]) 

    prepared_data = prepared_data.dropna() 
    
    
    prepared_data_forec_sales = prepared_data.sort_values(by = ['delivery_point_id', 
                                                                'product_id', 
                                                                'date'], ascending = True)
    
    
    
    prepared_data_forec_date.drop(['deliveries'],
                                   inplace=True, axis=1)


        
    # sliding window for next date - lags days_between_dates         
    for i in range(1,31):   
        field_name = 'lag_num_days_' + str(i)
        prepared_data_forec_date[field_name] = prepared_data_forec_date.groupby(['delivery_point_id', 'product_id'])['days_until_next_date'].shift(i)


    #sale - diff, mean, std, min, max
    prepared_data_forec_date["num_days_differences_per_day"] = prepared_data_forec_date.groupby(['delivery_point_id', 
                                                                                                 'product_id'])['days_until_next_date'].diff(1)
    #Seasonality with moving average
    prepared_data_forec_date["mean_num_days_20day"] = prepared_data_forec_date.groupby(['delivery_point_id', 
                                                                                        'product_id'])['days_until_next_date'].rolling(20).mean().droplevel(level=[0,1]) 

    prepared_data_forec_date["std_num_days_20day"] =  prepared_data_forec_date.groupby(['delivery_point_id', 
                                                                                    'product_id'])['days_until_next_date'].rolling(20).std().droplevel(level=[0,1]) 

    prepared_data_forec_date = prepared_data_forec_date.dropna() 
    
    
    
    #  DataFrame for days_until_next_date
    #prepared_data_forec_date = prepared_data_forec_date[prepared_data_forec_date['days_until_next_date'] > 1]


    prepared_data_forec_date = prepared_data_forec_date.sort_values(by = ['delivery_point_id', 
                                                                          'product_id', 
                                                                          'date'], ascending = True)
    
    

    return {'prepared_data_forec_sales': prepared_data_forec_sales, 
            'prepared_data_forec_date': prepared_data_forec_date,
            'season_mapping': season_mapping, 
            'product_mapping': product_mapping,
            'delivery_point_mapping': delivery_point_mapping}





    
    