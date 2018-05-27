#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:27:05 2018

@author: eduardo
"""
#https://krstn.eu/paralell_Pandas/

import pandas as pd
import numpy as np
from multiprocessing import Pool
import sys


ts_df = pd.DataFrame(np.random.random(size=(365, 3000)))
ts_df.shape

def feature_calculation(df):
    # create DataFrame and populate with stdDev
    result = pd.DataFrame(df.std(axis=0))
    result.columns = ["stdDev"]
    
    # mean
    result["mean"] = df.mean(axis=0)

    # percentiles
    for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
        result[str(int(i*100)) + "perc"] = df.quantile(q=i)

    # percentile differences / amplitudes
    result["diff_90perc10perc"] = (result["10perc"] - result["90perc"])
    result["diff_75perc25perc"] = (result["75perc"] - result["25perc"])

    # percentiles of lagged time-series
    for lag in [10, 20, 30, 40, 50]:
        for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result["lag" + str(lag) + "_" + str(int(i*100)) + "perc"] = (df - df.shift(lag)).quantile(q=i)

    # fft
    df_fft = np.fft.fft(df, axis=0)  # fourier transform only along time axis
    result["fft_angle_mean"] = np.mean(np.angle(df_fft, deg=True), axis=0)
    result["fft_angle_min"] = np.min(np.angle(df_fft, deg=True), axis=0)
    result["fft_angle_max"] = np.max(np.angle(df_fft, deg=True), axis=0)
    
    return result



def parallel_feature_calculation(df, partitions=10, processes=4):
    # calculate features in parallel by splitting the dataframe into partitions and using parallel processes
    
    pool = Pool(processes)
    
    df_split = np.array_split(df, partitions, axis=1)  # split dataframe into partitions column wise

    df = pd.concat(pool.map(feature_calculation, df_split))
    pool.close()
    pool.join()
    
    return df
    
# Traditional
#ts_features = feature_calculation(ts_df)

ts_features_parallel = parallel_feature_calculation(ts_df, partitions=14)
