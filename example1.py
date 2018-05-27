#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 11:17:50 2018

@author: eduardo
"""
# http://www.racketracer.com/2016/07/06/pandas-in-parallel/

import pandas as pd
import numpy as np
import time
from multiprocessing import cpu_count, Pool
import seaborn as sns
import sys

num_cores = cpu_count() #Number of CPU cores on your system
num_partitions = num_cores #Define as many partitions as you want

iris = pd.DataFrame(sns.load_dataset('iris'))

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    
    df = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    
    return df
    
def multiply_columns(data):
    data['length_of_word'] = data['species'].apply(lambda x: len(x))
    return data
    
start = time.time()
iris = parallelize_dataframe(iris, multiply_columns)
print time.time() - start
print iris
