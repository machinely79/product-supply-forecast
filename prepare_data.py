# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:46:44 2023

@author: mladen
"""


import pandas as pd
import numpy as np


def prepare(product_deliveries, product_returns, products_category):
    
    '''
    The prepare function processes and cleans "product deliveries" and "product returns" 
    data by grouping, filtering, and calculating sales. It merges deliveries with returns, 
    selects frequent delivery locations, handles missing values, and adjusts sales figures 
    based on product categories. Finally, it applies a sales increase factor (4% or 6%) and 
    returns a structured dataset.
    '''
    
    # product deliveries -------------------------------------------------------
    product_deliveries['date'] = pd.to_datetime(product_deliveries['date'])

    #product_deliveries = product_deliveries.dropna()

    group_product_deliveries = product_deliveries.groupby(['delivery_point_id',
                                                           'product_id', 
                                                           'date'], sort=False)['quantity'].sum().reset_index()
    

    
    # product returns ----------------------------------------------------------
    product_returns['date'] = pd.to_datetime(product_returns['date'])

    group_product_returns = product_returns.groupby(['delivery_point_id', 
                                                     'product_id', 
                                                     'date'], sort=False)['quantity'].sum().reset_index()

    group_product_returns = group_product_returns.rename({'quantity': 'returns'}, axis=1)

    # merge product deliveries and product returns
    data_prepared = group_product_deliveries.merge(group_product_returns, on=['delivery_point_id',
                                                                               'product_id',
                                                                               'date'],
                                                                                how='left')
    
    # number of deliveries by groups -----------------------------------------------
    group_count = data_prepared.groupby(['delivery_point_id',
                                         'product_id'])['date'].count().sort_values().reset_index()
    
    #selecting only frequent delivery locations (>60)
    group_count['select'] = np.where((group_count['date']) > 60, 1, 0)
    group_count.drop(['date'], inplace=True, axis=1)
    
    data_prepared = data_prepared.merge(group_count, on=['delivery_point_id',
                                                         'product_id'], how='left')
    
    data_prepared = data_prepared[(data_prepared["select"] == 1)]
    data_prepared.drop(['select'], inplace=True, axis=1)
    

    data_prepared['returns'] = data_prepared['returns'].fillna(0)
    data_prepared['returns'] = np.where((data_prepared['returns']) < 1, 0, data_prepared['returns'])

    data_prepared = data_prepared.rename({'quantity': 'deliveries'}, axis=1)

    data_prepared['sale'] = data_prepared['deliveries'] - data_prepared['returns']

    #if sales are less than 0, then transfer deliveries
    data_prepared['sale'] = np.where((data_prepared['sale']) < 0, 
                                        data_prepared['deliveries'], 
                                        data_prepared['sale'])

    #sales correction: sales = if (deliveries == returns) & (sales == 0) then sales equals deliveries, otherwise keep sales."
    data_prepared['sale'] = np.where((data_prepared['deliveries'] == data_prepared['returns']) & (data_prepared['sale'] == 0), 
                                         data_prepared['deliveries'], 
                                         data_prepared['sale'])

    data_prepared = data_prepared.merge(products_category[['product_id',
                                                           'category_description']], 
                                        on='product_id', 
                                        how='left')

    data_prepared['sale_without_increase'] =  data_prepared['sale'] 
    
    def target_sales(df):
        if df['category_description'] == 'Dnevni hleb':
            return df['sale'] * 1.03 # 4% increase
        else: 
            return df['sale'] * 1.04  # 6% increase
        
    data_prepared['sale'] = data_prepared.apply(target_sales, axis=1)

    data_prepared.drop(['category_description'], inplace=True, axis=1)
    
    data_prepared = data_prepared.sort_values(by = ['delivery_point_id', 
                                                   'product_id', 
                                                   'date'], ascending = True)
    
    return(data_prepared)

    
   

   
    
    
