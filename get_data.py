# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:28:07 2023

@author: mladen

"""
import os
import pandas as pd
import numpy as np


def load_data(file_path, deliveries_file, product_returns_file, products_category_file):
    
    """
    The load_data function reads and processes CSV files containing product deliveries, returns, and product categories. 
    It constructs file paths, loads data with predefined column names and types, merges deliveries and 
    returns with product categories, and filters only "Rezani hlebovi" and "Dnevni hleb" products. 
    The function returns a dictionary with three cleaned and structured DataFrames.
    """


    full_path_deliveries = os.path.join(file_path, deliveries_file)
    full_path_product_returns = os.path.join(file_path, product_returns_file)
    full_path_products_category = os.path.join(file_path, products_category_file)
    
    
    header = ["sifra_kupca", "sifra_objekta", "datum", "sifra_artikla", "kolicina"]

    type_dict = {'sifra_kupca': 'object', 'sifra_objekta': 'object',
                  'sifra_artikla':'object', 'kolicina': 'float'}
    

    # deliveries ----------------------------------------------------------
    product_deliveries = pd.read_csv(full_path_deliveries, sep='|', 
                                       names=header, 
                                       dtype=type_dict, 
                                       index_col=False)

    product_deliveries = product_deliveries.dropna()

    products_category = pd.read_csv(full_path_products_category, 
                                    sep=',', 
                                    index_col=False)

    products_category = products_category.rename({'procut_code': 'sifra_artikla'}, axis=1)

    product_deliveries = product_deliveries.merge(products_category[['sifra_artikla','product_name','category_description']],
                                                 left_on='sifra_artikla', right_on='sifra_artikla', how='left')

    product_deliveries = product_deliveries.dropna()

    product_deliveries = product_deliveries[(product_deliveries["category_description"] == 'Rezani hlebovi') | (product_deliveries["category_description"] == 'Dnevni hleb')]


    # returns ----------------------------------------------------------
    product_returns = pd.read_csv(full_path_product_returns,
                                       sep='|', 
                                       names=header, 
                                       dtype=type_dict, 
                                       index_col=False)

    product_returns = product_returns.merge(products_category[['sifra_artikla',
                                           'product_name',
                                           'category_description']],
                        left_on='sifra_artikla',
                        right_on='sifra_artikla',
                        how='left')

    product_returns = product_returns.dropna()

    product_returns = product_returns[(product_returns["category_description"] == 'Rezani hlebovi') | (product_returns["category_description"] == 'Dnevni hleb')]

    
    return {'product_deliveries': product_deliveries, 
            'product_returns': product_returns, 
            'products_category': products_category}


