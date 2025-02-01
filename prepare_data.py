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
    product_deliveries['datum'] = pd.to_datetime(product_deliveries['datum'])

    #product_deliveries = product_deliveries.dropna()

    group_product_deliveries = product_deliveries.groupby(['sifra_objekta',
                                                           'sifra_artikla', 
                                                           'datum'], sort=False)['kolicina'].sum().reset_index()
    

    
    # product returns ----------------------------------------------------------
    product_returns['datum'] = pd.to_datetime(product_returns['datum'])

    group_product_returns = product_returns.groupby(['sifra_objekta', 
                                                     'sifra_artikla', 
                                                     'datum'], sort=False)['kolicina'].sum().reset_index()

    group_product_returns = group_product_returns.rename({'kolicina': 'povrati'}, axis=1)

    # merge product deliveries and product returns
    data_prepared = group_product_deliveries.merge(group_product_returns, on=['sifra_objekta',
                                                                               'sifra_artikla',
                                                                               'datum'],
                                                                                how='left')
    
    # number of deliveries by groups -----------------------------------------------
    group_count = data_prepared.groupby(['sifra_objekta',
                                         'sifra_artikla'])['datum'].count().sort_values().reset_index()
    
    #selecting only frequent delivery locations (>60)
    group_count['select'] = np.where((group_count['datum']) > 60, 1, 0)
    group_count.drop(['datum'], inplace=True, axis=1)
    
    data_prepared = data_prepared.merge(group_count, on=['sifra_objekta',
                                                         'sifra_artikla'], how='left')
    
    data_prepared = data_prepared[(data_prepared["select"] == 1)]
    data_prepared.drop(['select'], inplace=True, axis=1)
    

    data_prepared['povrati'] = data_prepared['povrati'].fillna(0)
    data_prepared['povrati'] = np.where((data_prepared['povrati']) < 1, 0, data_prepared['povrati'])

    data_prepared = data_prepared.rename({'kolicina': 'isporuke'}, axis=1)

    data_prepared['prodaja'] = data_prepared['isporuke'] - data_prepared['povrati']

    #if sales are less than 0, then transfer deliveries
    data_prepared['prodaja'] = np.where((data_prepared['prodaja']) < 0, 
                                        data_prepared['isporuke'], 
                                        data_prepared['prodaja'])

    #sales correction: sales = if (deliveries == returns) & (sales == 0) then sales equals deliveries, otherwise keep sales."
    data_prepared['prodaja'] = np.where((data_prepared['isporuke'] == data_prepared['povrati']) & (data_prepared['prodaja'] == 0), 
                                         data_prepared['isporuke'], 
                                         data_prepared['prodaja'])

    data_prepared = data_prepared.merge(products_category[['sifra_artikla',
                                                           'category_description']], 
                                        on='sifra_artikla', 
                                        how='left')

    data_prepared['prodaja_bez_uvećanja'] =  data_prepared['prodaja'] 
    
    def target_sales(df):
        if df['category_description'] == 'Dnevni hleb':
            return df['prodaja'] * 1.04 # 4% increase
        else: 
            return df['prodaja'] * 1.06  # 6% increase
        
    data_prepared['prodaja'] = data_prepared.apply(target_sales, axis=1)

    data_prepared.drop(['category_description'], inplace=True, axis=1)
    
    data_prepared = data_prepared.sort_values(by = ['sifra_objekta', 
                                                   'sifra_artikla', 
                                                   'datum'], ascending = True)
    
    return(data_prepared)

    
   

   
    
    
