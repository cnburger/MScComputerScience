import numpy as np
import math
import csv
import pandas as pd
import joblib as joblib
from joblib import Memory
#Improting another class
from postgresql_connection import  PostgreSQL_connection

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import os
import copy

#print(os.chdir())

x = np.empty((2,2))
x[:,:] = None
print(np.isnan(x[0,0]))

SDM_matrix = joblib.load("/home/neil/git/AIS_LRM/SavedMatrices/Real_Cargo_with_COG_2501.sav")
COG_Matrix = joblib.load("/home/neil/git/AIS_LRM/SavedMatrices/Real_Cargo_with_COG_2501.sav")
mean_matrix = COG_Matrix / SDM_matrix # Element wise devision
variance_matrix = copy.deepcopy(mean_matrix)

def get_cog_average(cog_1,cog_2):
    min_val = min(cog_1, cog_2)
    max_val = max(cog_1, cog_2)
    diff = max_val - min_val

    if(np.isnan(cog_1)): # Assuming the first entry is from the grid
        return cog_2

    if(diff > 180):
        new_val = (360 - max_val + min_val)/2 #get average
        new_val += max_val
        if(new_val > 360):
            new_val -= 360
    else:
        new_val = (max_val + min_val)/2
    return(new_val)

def gridData_to_2DMap(  file_name = "/home/neil/git/AIS_LRM/Cargo_COG_SOG_15km_6h.csv",
                        N = 2501,
                        v = 10000,
                        file_save = "/home/neil/git/AIS_LRM/SavedMatrices/Real_Cargo_with_SDM_2501.sav"):

      x = np.linspace(0,10,N,endpoint=True)
      x_value = x + (x[1]-x[2])/2.0
      x_value = x_value[:-1]

      y = np.linspace(0,10,N,endpoint=True)
      y_value = y + (y[1]-y[2])/2.0
      y_value = y_value[:-1]

      matriks = np.zeros((N-1,N-1),dtype=int)
      matriks_COG = np.empty((N-1,N-1))
      matriks_COG[:,:] = None
      print(matriks_COG.shape)
      counter = 0
      with open(file_name) as infile:
           for line in infile:
               if(counter > 30):
                   line_split = line.split(",")
                   # named incorrectly but correct values passed

                   if(line_split[2] != '\n'):
                       COG = np.rad2deg(float(line_split[2].strip()))
                       lat = float(line_split[1]) +10.00 #- 45.00
                       lon = float(line_split[0]) -45.00 #+ 10.00


                       index_x = np.searchsorted(x,lat)
                       index_y = np.searchsorted(y,lon)

                       if (index_x == N):
                          index_x = index_x - 2
                       elif (index_x > 0):
                          index_x = index_x - 1

                       if (index_y == N):
                          index_y = index_y - 2
                       elif (index_y > 0):
                          index_y = index_y - 1

                       matriks[index_y,index_x] += 1
                      # matriks_COG[index_y,index_x] = get_cog_average(matriks_COG[index_y,index_x],COG)
                       if(np.isnan(matriks_COG[index_y,index_x])):
                           matriks_COG[index_y,index_x] = 0


                       matriks_COG[index_y,index_x] += COG #/matriks[index_y,index_x]

                       # Calculate the Variance
                       variance_matrix[index_y,index_x] += (COG - mean_matrix[index_y,index_x])**2


               counter += 1
               if (counter % v == 0):
                  print(COG)
                  # print(lon)
                  # print(lat)
                  # print("COG",COG)
                  print("counter = ",counter)
                  print(COG)
                  # print(matriks[index_y,index_x] )
                  # print(matriks_COG[index_y,index_x])
                  # print(matriks_COG[index_y,index_x] )


      joblib.dump(matriks, file_save)
      joblib.dump(matriks_COG, "/home/neil/git/AIS_LRM/SavedMatrices/Real_Cargo_COG_mat_2501.sav")
      joblib.dump(variance_matrix, "/home/neil/git/AIS_LRM/SavedMatrices/Real_Cargo_variance_15km_6h_mat_2501.sav")


import scipy as sp
import scipy.ndimage as ndimage
#gridData_to_2DMap()

my_mat = joblib.load("/home/neil/git/AIS_LRM/SavedMatrices/Real_Cargo_with_SDM_2501.sav")

my_mat = np.log(my_mat)
my_mat[my_mat<0] = 0

my_mat = ndimage.gaussian_filter(my_mat, sigma = 1, order=0)
plt.imshow(my_mat)
plt.show()

#
#
# class SDM:
#
#     def __init__(self, file_name, N, v, file_save):
#         self.file_name = file_name
#         self.N = N
#         self.v = v
#         self.file_save = file_save
#
#
#     def gridData_to_2DMap(self):
#         print(self.file_name)
#         x = np.linspace(0,10,self.N,endpoint=True)
#         x_value = x + (x[1]-x[2])/2.0
#         x_value = x_value[:-1]
#
#         y = np.linspace(0,10,self.N,endpoint=True)
#         y_value = y + (y[1]-y[2])/2.0
#         y_value = y_value[:-1]
#
#         matriks = np.zeros((self.N-1,self.N-1),dtype=int)
#         counter = 0
#
#
#         with open(self.file_name) as infile:
#            for line in infile:
#
#                line_split = line.split(",")
#                # named incorrectly but correct values passed
#                lon = float(line_split[0]) -45.00
#                lat = float(line_split[1]) + 10
#
#                index_x = np.searchsorted(x,lat)
#                index_y = np.searchsorted(y,lon)
#
#                if (index_x == self.N):
#                   index_x = index_x - 2
#                elif (index_x > 0):
#                   index_x = index_x - 1
#
#                if (index_y == self.N):
#                   index_y = index_y - 2
#                elif (index_y > 0):
#                   index_y = index_y - 1
#
#                matriks[index_y,index_x]+=1
#
#                counter += 1
#                if (counter % self.v == 0):
#                   print("counter = ",counter)
#
#
#         joblib.dump(matriks, self.file_save)
#
# #
# # print(open("Tankers_TEST.csv"))
# # # pd.read_csv("Tankers_TEST.csv")
# # SDM_obj = SDM(file_name = "/home/neil/git/SDM/Tankers_TEST.csv",
# #               N = 10001,
# #               v = 100000,
# #              file_save = "INTER_TANKERS_TEST_10000.sav")
# # SDM_obj.gridData_to_2DMap()
# # # Extract data from text straight to memory "Meory dump"
# # mymat = joblib.load("Cargo_nari.sav")
# # mymat = np.log(mymat) #Logirithmic of the data
# # mymat[mymat < 0] = 0#Set all negative values to 0
# #
# #
# # # Imports for CartoPy
# # import cartopy as cart
# # import cartopy.crs as ccrs
# # import cartopy.feature as cfeature
# # from cartopy.feature import NaturalEarthFeature
# #
# # #
# # # Create figure to add plots to
# # fig = plt.figure()
# # # Adding a subploot and its projection type
# # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# #
# # # Setting where to startthe new projections data (setting the centre of the map)
# #
# # ax.set_extent([-10,0,45.00, 55.00], crs=ccrs.PlateCarree())
# #
# #
# #
# # # Setting the projection
# # ax = plt.axes(projection=ccrs.PlateCarree())
# #
# # # Adding a Coastline
# # ax.add_feature(cfeature.COASTLINE,linewidth=0.5)
#
# #Shift plot with coordiantes
# lon = np.linspace(-10,0, 1000)
# lat = np.linspace(45, 55, 1000)
# lon2d, lat2d = np.meshgrid(lon, lat)
#
# # Plot colourmap
# ax.pcolormesh(lon2d, lat2d, mymat, transform = ccrs.PlateCarree() )
#
#
# plt.show()
