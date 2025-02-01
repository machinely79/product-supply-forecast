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
        # Convert the 'datum' column to datetime format
        data['datum'] = pd.to_datetime(data['datum'])

        # Sort the data by 'sifra_objekta', 'sifra_artikla', and 'datum'
        data.sort_values(['sifra_objekta', 'sifra_artikla', 'datum'], ascending = True)

        # Calculate the difference in days between the current and next date in each group
        data['days_until_next_date'] = data.groupby(['sifra_objekta', 'sifra_artikla'])['datum'].diff().dt.days

        return data


    prepared_data = calculate_days_between_dates(prepared_data)
    
   
    prepared_data['dayOfWeek'] = prepared_data['datum'].dt.dayofweek # 0 = mon ... 6 = sun
    prepared_data['month'] = prepared_data['datum'].dt.month         # 1 = jan ... 12 = dec
    prepared_data["quarter"] = prepared_data["datum"].dt.quarter     # 1 = 1.q .... 4 = 4.q
    
    #season
    prepared_data['date_offset'] = (prepared_data.datum.dt.month*100 + prepared_data.datum.dt.day - 320)%1300

    prepared_data['sezona'] = pd.cut(prepared_data['date_offset'], 
                                     [0, 300, 602, 900, 1300], 
                          labels=['spring', 'summer', 'autumn', 'winter'],
                          include_lowest = True)
    
    prepared_data.drop(['date_offset'],
                       inplace=True, axis=1)

    # week of the month - in order
    prepared_data['week_in_month'] = prepared_data['datum'].dt.day // 7 + 1
    
    # add holidays
    def add_holidays(df):
        holiday = holidays.Serbia(years=df['datum'].dt.year)
        df['holidays'] = np.where(df['datum'].dt.date.astype('datetime64').isin(holiday.keys()), 1, 0)
        return df
    
    prepared_data = add_holidays(prepared_data)
    
    # make dictionary with unique values (index and value)
    Sezona_map = {label: idx for idx, label in 
                          enumerate(np.unique(prepared_data['sezona']))}


    Artikal_map = {label: idx for idx, label in 
                          enumerate(np.unique(prepared_data['sifra_artikla']))}

    Objekat_map = {label: idx for idx, label in 
                          enumerate(np.unique(prepared_data['sifra_objekta']))}
  
    # mapping
    mapping=[Sezona_map, Artikal_map, Objekat_map ] 
    columns=['sezona', 'sifra_artikla', 'sifra_objekta']
    
    # encode the values
    for i, j in zip(columns, mapping):
        #print(str(i) + "--> " + str(j))    
        prepared_data[i]=prepared_data[i].map(j).astype('int')
        
      
        
    # copy DataFrame for days_until_next_date
    prepared_data_forec_date = prepared_data.copy() 
    
    
    # sliding windows for sales - lags "prodaja column"         
    for i in range(1,31):   
        field_name = 'lag_sales_' + str(i)
        prepared_data[field_name] = prepared_data.groupby(['sifra_objekta', 'sifra_artikla'])['prodaja'].shift(i)

 
    # diff, mean, std, min, max - "prodaja column"
    prepared_data["sales_differences_per_day"] = prepared_data.groupby(['sifra_objekta', 
                                                                'sifra_artikla'])['prodaja'].diff(1)
    #Seasonality with moving average
    prepared_data["mean_sales_7day"] = prepared_data.groupby(['sifra_objekta', 
                                                                'sifra_artikla'])['prodaja'].rolling(7).mean().droplevel(level=[0,1]) 

    prepared_data["std_sales_7day"] =  prepared_data.groupby(['sifra_objekta', 
                                                              'sifra_artikla'])['prodaja'].rolling(7).std().droplevel(level=[0,1]) 

    prepared_data = prepared_data.dropna() 
    
    
    prepared_data_forec_sales = prepared_data.sort_values(by = ['sifra_objekta', 
                                                                'sifra_artikla', 
                                                                'datum'], ascending = True)
    
    
    
    prepared_data_forec_date.drop(['isporuke'],
                                   inplace=True, axis=1)


        
    # sliding window for next date - lags days_between_dates         
    for i in range(1,31):   
        field_name = 'lag_num_days_' + str(i)
        prepared_data_forec_date[field_name] = prepared_data_forec_date.groupby(['sifra_objekta', 'sifra_artikla'])['days_until_next_date'].shift(i)


    #prodaja - diff, mean, std, min, max
    prepared_data_forec_date["num_days_differences_per_day"] = prepared_data_forec_date.groupby(['sifra_objekta', 
                                                                                                 'sifra_artikla'])['days_until_next_date'].diff(1)
    #Seasonality with moving average
    prepared_data_forec_date["mean_num_days_20day"] = prepared_data_forec_date.groupby(['sifra_objekta', 
                                                                                        'sifra_artikla'])['days_until_next_date'].rolling(20).mean().droplevel(level=[0,1]) 

    prepared_data_forec_date["std_num_days_20day"] =  prepared_data_forec_date.groupby(['sifra_objekta', 
                                                                                    'sifra_artikla'])['days_until_next_date'].rolling(20).std().droplevel(level=[0,1]) 

    prepared_data_forec_date = prepared_data_forec_date.dropna() 
    
    
    
    #  DataFrame for days_until_next_date
    #prepared_data_forec_date = prepared_data_forec_date[prepared_data_forec_date['days_until_next_date'] > 1]


    prepared_data_forec_date = prepared_data_forec_date.sort_values(by = ['sifra_objekta', 
                                                                          'sifra_artikla', 
                                                                          'datum'], ascending = True)
    
    

    return {'prepared_data_forec_sales': prepared_data_forec_sales, 
            'prepared_data_forec_date': prepared_data_forec_date,
            'Sezona_map': Sezona_map, 
            'Artikal_map': Artikal_map,
            'Objekat_map': Objekat_map}





    
    