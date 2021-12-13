import pandas as pd
import numpy as np
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import joblib as joblib
from joblib import Memory


# Creating a class
class InterpolateMMSIs:

    def __init__ (self, dataFrame,data_name,pixel_size_deg):
        self.dataset = dataFrame
        self.dataset_Name = data_name #to name any related output files
        self.pixel_size_deg = pixel_size_deg
        self.pixel_distance_m = self.get_pixel_distance(pixel_size_deg)
        print(self.pixel_distance_m)


    def interpolate_vessels(self):
        # Extracting all vessels
        all_vessel_mmsi = np.unique(self.dataset["mmsi"].values)
        all_mmsi_count = len(all_vessel_mmsi)
        counter = 0


        obs1 = self.dataset.iloc[1,:]
        obs2 = self.dataset.iloc[2,:]

        #print("Time diff",self.time_below_threshold(obs1,obs2))

        #going through each observation
        lat = np.array([])
        lon = np.array([])
        cog = np.array([])

        # For evert vessel mmsi do the following
        for mmsi in all_vessel_mmsi:
            mmsi_data = self.get_data_from_mmsi(mmsi) #extracting the data belonging to the current mmsi
            mmsi_length = len(mmsi_data) #in ordder to count each observation
            counter += 1
            print("Progress: (", counter,"/",all_mmsi_count,") \t", (counter/all_mmsi_count)*100, "%")

            # Get the first observation
            obs1 =  mmsi_data.iloc[0]

            for obs in range(1,mmsi_length):
                obs2 = mmsi_data.iloc[obs]

                if(self.distance_below_threshold(obs1,obs2, km = 15)
                            and self.time_below_threshold(obs1,obs2, thres_min = 60*6) # 6 hours between observations
                            and (sum(obs1 == obs2) < 2)): #not equal

                    # Get the interpolated values
                    inter_x, inter_y = self.interpolate_points(obs1,obs2)
                    # Add the first observation to a np array
                    lat = np.append(lat, obs1.loc["lat"])
                    lon = np.append(lon, obs1.loc["lon"])
                    # Add the interpolated values
                    lat = np.append(lat, inter_x)
                    lon = np.append(lon, inter_y)
                else:
                    # If conditions not met add observations and go on
                    lat = np.append(lat, obs1.loc["lat"])
                    lon = np.append(lon, obs1.loc["lon"])



                # Making sure that we copy the value and not the refrencing address
                obs1 = copy.deepcopy(obs2)
            #cog = np.append(cog,self.get_cog(lat,lon))

            #Saving values in a joblib file incase it increases to large
            joblib.dump(lat, "lat.sav")
            joblib.dump(lon, "lon.sav")
            joblib.dump(cog, "cog.sav")
        #Save values as a csv output
        self.save_as_csv()


    def get_cog(self,lat_values, lon_values):
        """This function calculates the COG making use of pandas,
        due to shifting of the calculation we will duplicate the first
         COG with the second observatrion"""
        print(lat_values)
        #lat_values = pd.DataFrame(lat_values, index = None)
        lat_values = np.array(lat_values)
        lon_values = np.array(lon_values)
        #lon_values = pd.DataFrame(lon_values, index = None)



        y_diff = np.diff(lon_values)
        x_diff = np.diff(lat_values)
        cog_calculated = np.arctan2(y_diff,x_diff) % (2*np.pi) # to numpy and convert to 360
        print(cog_calculated)
        cog_calculated = np.append(cog_calculated[0],cog_calculated) # diplicatin the first value

        return cog_calculated

    def save_as_csv(self,lon_file = "lon.sav", lat_file = "lat.sav", cog_file = "cog.sav"):
        # loading the data
        lon = joblib.load(lon_file)
        lat = joblib.load(lat_file)
        # cog = joblib.load(cog_file)
        cog = self.get_cog(lat,lon)

        print(len(lat))
        print(len(cog))

        # Create a dataframe
        latlon_DF = pd.DataFrame({'lon': lon,
                                'lat': lat,
                                'cog': cog})
        # Create the string name
        file_name = str(self.dataset_Name +".csv")

        # Save to the file
        latlon_DF.to_csv(file_name, index = False, header = False)


    def interpolate_points(self, obs1, obs2, inDegrees = True):
        cell_size_m = self.pixel_distance_m
        lat1, lon1 = self.get_lat_lon(obs1)
        lat2, lon2 = self.get_lat_lon(obs2)

        # Gradient m
        m = (lon2-lon1)/(lat2-lat1)

        # Constant C
        c = lon1 - m*lat1

        # Getting the coordinate difference
        lat_diff = lat2-lat1
        distance_in_m = np.ceil(self.get_distance(obs1,obs2))
        # Ceiling due to some varaitions
        steps_to_take = np.ceil(distance_in_m/cell_size_m)

        # Step size:
        step_size = lat_diff/steps_to_take

        # Arange to fill all the vlaues in
        x = np.arange(lat1,lat2, step = step_size)
        x = np.delete(x,0) # delete the inclusive initial value

        # Calculating the y coordinates LONG
        y = m*x + c

        #Returning answer in degrees
        if(inDegrees):
            x = self.rad_to_deg(x)
            y = self.rad_to_deg(y)

        return x, y

    def get_data_from_mmsi(self, mmsi):
        # Get the index values of the current mmsi in the list
        mmsi_index = self.dataset["mmsi"].values == mmsi

        # Extract the data that belongs to the mmsi, rows according to index + all columns
        mmsi_data = self.dataset.iloc[mmsi_index,:]

        return mmsi_data # Returning the data that we want

    def get_lat_lon(self,obs, deg_to_rad = True):
        # Return the coordinates and convert to radians
        if(deg_to_rad == True):
            return self.deg_to_rad(obs.loc['lat']), self.deg_to_rad(obs.loc['lon'])
        else:
            return obs.loc['lat'], obs.loc['lon']


    # Function to determine the distance between two coodinates and if it is below thethreshold
    def distance_below_threshold(self, obs1,obs2, km = 10.0):

        # Average radius of the earth
        R = 6371.0

        # Get coordinates ov the observations
        lat1, lon1 = self.get_lat_lon(obs1)
        lat2, lon2 = self.get_lat_lon(obs2)

        # Difference between long and lat cooridnates
        dlon = lon2-lon1
        dlat = lat2-lat1

        # Haverstine formula
        a = (np.sin(dlat/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2))**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        #Final distance
        distance = R*c

        if(distance < km):
            return True
        else:
            return False

    def get_distance(self, obs1,obs2):
        # Average radius of the earth
        R = 6371.0

        # Get coordinates ov the observations
        lat1, lon1 = self.get_lat_lon(obs1)
        lat2, lon2 = self.get_lat_lon(obs2)

        # Difference between long and lat cooridnates
        dlon = lon2-lon1
        dlat = lat2-lat1

        # Haverstine formula
        a = (np.sin(dlat/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2))**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        #Final distance in m
        distance = R*c*1000

        return distance

    def get_pixel_distance(self, pixel_size):
        # Average radius of the earth
        R = 6371.0

        # Get coordinates ov the observations
        lat1 = 0
        lon1 = 0
        lat2 = 0
        lon2 = self.deg_to_rad(pixel_size) #Radians

        # Difference between long and lat cooridnates
        dlon = lon2-lon1
        dlat = lat2-lat1

        # Haverstine formula
        a = (np.sin(dlat/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2))**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        #Final distance in m
        distance = R*c*1000

        return np.round(distance,2) #round to the second

    # Function that get the time difference threshold
    def time_below_threshold(self,mmsi_obs1, mmsi_obs2, thres_min = 300.0):
        """
        The function determines of the threshold is allowable, we check if
        we can interpolate based on the time threshold

        mmsi_obs1/mmsi_obs2 indicates the different observations in time
            Where mmsi_obs1 = the first observation
        """

        thres_sec = thres_min*60
        # convert from strings to dates
        date1 = datetime.strptime(mmsi_obs1.loc["datetimestamp"],"%Y-%m-%d %H:%M:%S")
        date2 = datetime.strptime(mmsi_obs2.loc["datetimestamp"],"%Y-%m-%d %H:%M:%S")

        # Get the seconds difference
        time_difference = (date2-date1).seconds

        if(time_difference < thres_sec):
            return True # below threshold
        else:
            return False #above threshold


    def deg_to_rad(self, deg):
        return (deg/180)*np.pi

    def rad_to_deg(self,rad):
        return (rad*180)/np.pi

def get_calculate_course(lon_observed, lat_observed):
    """
    This function takes in the longitude, and latutide of the current and previous
    observations. It makes use of the linear regression y = mx +c forula's
    gradient calculation to calculate the angle between the two sets of coordiantes.
    With getting this angle we can make use of arctan, to calculate the angle in degrees (RADIANS),
    gettingthe course over ground of our observations
    """

    print("THIS METHOD IS VERY SLOW")
    print("THIS METHOD IS VERY SLOW")
    print("THIS METHOD IS VERY SLOW")
    print("THIS METHOD IS VERY SLOW")
    cog = np.array([])

    for i in range(1,len(lat_observed)):
        lat_prev = lat_observed[i-1]
        lat_curr = lat_observed[i]

        lon_prev = lon_observed[i-1]
        lon_curr = lon_observed[i]

        gradient =(lat_curr - lat_prev)/(lon_curr - lon_prev) # gradient formula (y2-y1)/(x2-x1)
        course =  np.arctan2(gradient) # get the estimated course in Radians

        cog = np.append(cog, course)

    return cog






MyData = pd.read_csv("/home/neil/git/SDM/Cargo_with_MMSI.csv")
# MyData = MyData.iloc[MyData["mmsi"].values==205439000,:]
# print(MyData.tail())

#print(get_calculate_course(MyData["lat"],MyData["lon"]))


data_obj = InterpolateMMSIs(MyData,"Cargo_COG_SOG_15km_6h",pixel_size_deg = 0.002) #0.002
data_obj.interpolate_vessels()
data_obj.save_as_csv()
# #data_obj.interpolate_vessels()


MyData = pd.read_csv("Real_Cargo_COG_SOG_15km_6h.csv")
print(MyData.head())
