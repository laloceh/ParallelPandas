#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:22:48 2018

@author: eduardo
"""
#https://jacquespeeters.github.io/2017/08/21/parallel-pandas-groupby/

from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
#import timeit
import time

def applyParallel(dfGrouped, func):
    processes = cpu_count() 
    p = Pool(processes)
    
    ret_list = p.map(func, [group for name, group in dfGrouped])
    
    p.close()
    p.join()
    return pd.concat(ret_list)
    
    #with Pool(cpu_count()) as p:
    #    ret_list = p.map(func, [group for name, group in dfGrouped])
    #return pd.concat(ret_list)
    
    
# Create a Dataframe for a minimum example
df = pd.DataFrame()
# 5000 users with approx 100 values
df["user_id"] = np.random.randint(5000, size=500000)
# Generate 500000 random integer values
df["value"] = np.random.randint(30, size=500000)
# Create data_chunk based on modulo of user_id
df["data_chunk"] = df["user_id"].mod(cpu_count() * 3)

# Any not optimised and intensive function i want to apply to each group
def group_function(group):
    # Inverse cumulative sum
    group["inv_sum"] = group.iloc[::-1]['value'].cumsum()[::-1].shift(-1).fillna(0)
    return group
    
    
def func_group_apply(df):
    return df.groupby("user_id").apply(group_function)

    
    
start = time.time()
normal = df.groupby("user_id").apply(group_function)
end = time.time()
print("Execution time :" + str(end - start))



'''
    Parallel
'''
start = time.time()
parallel = applyParallel(df.groupby("user_id"), group_function)
end = time.time()
print("Execution time :" + str(end - start))

'''
    Parallel chunks
'''
start = time.time()
parallel_chunk = applyParallel(df.groupby("data_chunk") , func_group_apply)
end = time.time()
print("Execution time :" + str(end - start))



'''
    Check the results
'''
normal = normal.sort_index()
parallel = parallel.sort_index()
parallel_chunk = parallel_chunk.sort_index()

# Check we have same results
print(normal.equals(parallel))
print(normal.equals(parallel_chunk))





