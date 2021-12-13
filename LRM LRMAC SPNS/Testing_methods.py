"""
This script will test the accuracy of the LRM, COG-SDM and COG + SDM

1. First we will be creating subsets of our observations that we will  be testing on
2. We will calculculate the error for each method
    2.1 After 15min, 30min, 45min 60min, 1.5h, 2h, ... , 10h up intil we have an observation for our vessels
    2.2 Error measure Harvisine distance from predicted and observed coordinate
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

import scipy as sp
from scipy.interpolate import interp1d
from scipy.interpolate import interp1d
from scipy import interpolate

import joblib as joblib
from joblib import Memory
import warnings
warnings.filterwarnings("ignore")


def plot_vessel(data_in):
    plt.plot(data_in["lat"], data_in["long"], "o-", color = "orange")
    plt.show()

import random
import copy
import logging

import gc
from LRM_SDM_Combined import LRM_APRIORI


import os
cwd = os.getcwd()
print(cwd)
mmsis_to_test = (genfromtxt('./Vessel_set.csv', delimiter=',')).astype(int)
print(mmsis_to_test)
os.chdir("..")


for mmsi_csv_extract in mmsis_to_test:
    print("TESTING MMSI: "+ str(mmsi_csv_extract))
    pd_dataset = pd.read_csv(r"../Vessels Testing/" + str(mmsi_csv_extract) +"_interpolated.csv", index_col = False)
    testing_error_dataset = copy.deepcopy(pd_dataset)
    testing_time_stamps = testing_error_dataset["datetimestamp"].values
    no_obs = len(pd_dataset)
    random.seed(2020)
    rand_vals = random.sample(range(no_obs),int(no_obs*0.005))
    rand_vals = np.sort(rand_vals)

    if(np.sum(rand_vals == 0) == 0): # if the first observation is not there add it
        rand_vals = np.append(0,rand_vals) # Making sure the first observation is always contained

    if(np.sum(rand_vals == testing_time_stamps[-1])): # Adding the last observation
        rand_vals = np.append(rand_vals,testing_time_stamps[-1])


    pd_dataset = pd_dataset.iloc[rand_vals[0:],]
    timestamps = pd_dataset["datetimestamp"].values
    last_time = timestamps[-1]
    total_hours_in_dataset = int(last_time/(60*60)) # chop off the comma

    if(total_hours_in_dataset > 11):
        total_hours_in_dataset = 12


    """
    The second for-loop below will break our trajectory up into strides, so that we can
    evaluate delta t amount of time that has passed on different sections of a trajectory
    Exampkle:  -----x-----x-----x-----x-----x, where each length between x denotes
    an hour passed and then testing the formula on that section.

    """

    stride_size_hour = 1
    window_sizes = np.array([3])
    nearest_Neighbor = np.array([1])
    hours_to_max = np.arange(2,total_hours_in_dataset)
    hour_set = np.append([5,15,30,1*60],hours_to_max*60)

    for NN_size  in nearest_Neighbor:
        for windows_size  in window_sizes:
            for hours in hour_set:
                stride_max = np.arange(1,np.floor(int(total_hours_in_dataset-hours/60)), step = stride_size_hour)
                stride_max = stride_max[stride_max > 0]
                if(len(stride_max) != 0 ):

                # ^^ For every prediction size ^^
                    for stride_length in stride_max:
                        print(stride_max)
                        print(stride_length)
                        print(hours)
                        # Dercrease the size of the dataset by the stride size to have a new set to test on
                        # FOR MORE DETAILS --> Look at notes 17 NOV 2020

                        hour_subset = timestamps >= stride_length*60*(stride_length-1)*60


                        pd_dataset_temp = pd_dataset.loc[hour_subset]
                        testing_hour_subset = testing_time_stamps >= 60*(stride_length-1)*60
                    #    print(testing_hour_subset)
                    #    print(sum(testing_hour_subset))

                        testing_error_dataset_temp = testing_error_dataset.loc[testing_hour_subset]

                        # Reset the date timestamp asif it is a new trajectory for the algorithm
                        pd_datetumestamp_numpy = pd_dataset_temp["datetimestamp"].values

                        #df_copy = copy.deepcopy(pd_dataset_temp)
                        #df_copy.loc[:,"datetimestamp"] = copy.deepcopy(pd_datetumestamp_numpy - pd_datetumestamp_numpy[0])
                        #

                        if(hours*60 <= testing_error_dataset["datetimestamp"].values[-1]):
                            print("\n")
                            print("MMSI:" + str(mmsi_csv_extract) +" Min: " +str(hours)+ " Window Size: " + str(windows_size) + " NN Size: "+str(NN_size) )
                            print("\n")

                            # try:
                            print("sdssdsd:", hours)
                            #print(pd_dataset_temp.head())
                            LRM_OBJ = LRM_APRIORI(testing_error_dataset_temp,
                                                    windows_size,
                                                    True,
                                                    False,
                                                    False, # Using SOG
                                                    hours*60,
                                                    nearest_n = NN_size,
                                                    universal_plotting = False)
                            stride_test = (stride_length-1)*60*60
                        #    print("STRIDE",stride_test)
                        #    print(testing_error_dataset)

                            LRM_OBJ.AIS_linear(testing_error_dataset,stride_test)
                            # except:
                            #     print("------------------------------------------------------------")
                            #     print("------------------------------------------------------------")
                            #     print("------------------------------------------------------------")
                            #     print("------------------------------------------------------------")
                            #     print("ERROR OCCORDED: MMSI", mmsi_csv_extract, " NN_size:",NN_size, " windows_size ",windows_size, "hours: ",hours)
                            #     print("-----------------------^^^^^^^^^-----------------------------------")
                            #     print("------------------------------------------------------------")
                            #     print("------------------------------------------------------------")
                            #     print("------------------------------------------------------------")
                            gc.collect()
                            del pd_dataset_temp

                            del LRM_OBJ
