import numpy
import os

import sqlalchemy
from sqlalchemy import create_engine
import psycopg2

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import copy

# 236386000 - close but not yet
# 636014352 - bad
# 566030000

class VesselCompMethod(object):
    """docstring forVesselCompMethod."""
    def __init__(self,vessel_mmsi,search_radius, cog_deviation_deg, prediction_lenght_sec, plot = False):

        # DB Connection
        self.db_connection = self.get_db_connection()

        #Model parameteres
        self.search_radius = search_radius # in metres
        self.delta_l = self.search_radius*2
        self.max_course_deviation = np.deg2rad(cog_deviation_deg)

        # Extracting vessel data
        vessel_data = self.get_vessel_trajectory(vessel_mmsi)
        vessel_og_data = copy.deepcopy(vessel_data)

        time_prediction = prediction_lenght_sec

        while (not isinstance(vessel_og_data,bool)) :
            print("Enought to predict: ? ",prediction_lenght_sec < vessel_og_data.shape[0] )
            print("Enought to predict: ? ",prediction_lenght_sec , vessel_og_data.shape[0] )
            if (prediction_lenght_sec > vessel_og_data.shape[0]):
                # End while
                return

            vessel_data = vessel_og_data.iloc[:time_prediction+1,:]
            print(vessel_data)
            #print(vessel_data)
            vessel_last_time = int(vessel_data["datetimestamp"].values[-1])/1e9

            # Extracting START POSITION & SPEED & COG
            self.current_pos =  np.array([float(vessel_data["lat"][0]),float(vessel_data["lon"][0]),]) #lat long
            self.current_vessel_COG = np.deg2rad(float(vessel_data["cog"].values[0]))
            self.vessel_speed_SOG = float(vessel_data["sog"].values[0])*0.514444444

            # Getting vessels in radius
            self.data_radius = self.get_vessels(self.current_pos[0],self.current_pos[1])

            # Prediction start time
            current_time = 0

            # Save predictions
            time_pred = np.array([])
            lat_pred = np.array([])
            lon_pred = np.array([])

            # Prediction, whiel we are not get at the desired time do:

            while current_time <= vessel_last_time :
                # Find CN's
                self.data_radius = self.get_vessels(self.current_pos[0],self.current_pos[1])

                # Calculate COG and SOG APROIRI
                self.apriori_cog = self.get_apriori_cog()
                self.apriori_sog = self.get_apriori_sog()
                self.current_vessel_COG = copy.deepcopy(self.apriori_cog)

                # Create the new position with prediciton - PREDICTION
                pred_lat_lon = self.predict_position_p()

                # Calculating new time factor
                current_time = copy.deepcopy(self.get_time_update(current_time))

                # Saving predictions and time updates
                time_pred = np.append(time_pred,current_time)
                lat_pred = np.append(lat_pred,pred_lat_lon[0])
                lon_pred = np.append(lon_pred,pred_lat_lon[1])

                # Update the current position to the next one
                self.current_pos = copy.deepcopy(pred_lat_lon)
                if(plot):
                    plt.plot(self.current_pos[0],self.current_pos[1],"ro")
                    plt.plot(self.data_radius["lat"],self.data_radius["lon"], "bo")





            # Fixing the final update so that we have an exact time
            lat_new, lon_new = self.get_prediction_at_timestep(lat_pred,lon_pred,time_pred,vessel_last_time)

            # Convert to float last values, to get distance error metric
            last_lat = float(vessel_data["lat"].values[-1])
            last_lon = float(vessel_data["lon"].values[-1])

            error_distance = self.haversin_distance(lat_new, lon_new,last_lat ,last_lon)
            self.write_to_csv(vessel_mmsi, vessel_last_time,self.apriori_cog,self.apriori_sog,error_distance,prediction_lenght_sec)
            #
            # print("ERROR: ", error_distance)

            vessel_og_data = copy.deepcopy(self.vessel_stride_cut(vessel_og_data,3600))


        print("STRIDES FINISHED")

        if(plot):
            # Plots for testing
            plt.plot(np.array([lat_new, last_lat]),np.array([lon_new, last_lon]),"r-",linewidth=5)
            plt.plot(vessel_data['lat'],vessel_data["lon"])
            plt.plot(lat_new,lon_new,"mo")
            plt.plot(lat_pred,lon_pred,"yo")
            plt.show()

    def write_to_csv(self, mmsi, time,cog,sog, error,prediction_lenght_sec):
        temp_dictionary = [{"MMSI": mmsi,
                            "Time": time,
                            "COG":cog,
                            "SOG":sog,
                            "Error:": error,
                            "Pred Length:": prediction_lenght_sec/60}]

        temp_df = pd.DataFrame.from_dict(temp_dictionary)

        temp_df.to_csv("AIS_based_VTP_ERRORS.csv", mode = "a", header = False)



    def get_prediction_at_timestep(self,lat_pred_arr, lon_pred_arr, time_array, no_seconds_pred):
        """
        This function will correct the prediction up until the right timestep that we need
        as the implemented function inferst the time travelled time += distance/speed
        """

        time_diff = time_array[-1]-time_array[-2] # caclulate the time difference betweren the last and second last observation
        time_calculation =  (no_seconds_pred-time_array[-2]) # calculate the time that should pass between the second last prediction and the actual time

        time_perc = time_calculation/time_diff # caclulating the % time that we need to still predict from the predicted time step, to get the fraction to increase the long an lat by

        lat_diff_way = lat_pred_arr[-2] + np.sign(np.sin(self.apriori_cog))*( np.abs(-lat_pred_arr[-2]+lat_pred_arr[-1]) )*time_perc # adjusting the lat coordinateas a fraction of the distance travelled
        lon_diff_way = lon_pred_arr[-2] + np.sign(np.cos(self.apriori_cog))*( np.abs(-lon_pred_arr[-2]+lon_pred_arr[-1]))*time_perc # adjusting the lon coordinate as a fraction of the distance travelled

        return lat_diff_way, lon_diff_way

    def haversin_distance(self,lat_1, lon_1, lat_2, lon_2, metres = 1):
        """This function will calculate the difference between two coordinates in
        km, for metre, change metres to 1000"""

        lat_1 = lat_1*np.pi/180
        lat_2 = lat_2*np.pi/180
        lon_1 = lon_1*np.pi/180
        lon_2 = lon_2*np.pi/180
        R = 6371

        a = (np.sin(((lat_2-lat_1))/2))**2 + np.cos(lat_1)*np.cos(lat_2)*((np.sin((lon_2-lon_1)/2))**2)
        c = 2*np.arctan2(a**(0.5), (1-a)**2)
        d = R*c

        return d*metres

    def get_time_update(self,current_time):
        return current_time + self.delta_l/self.vessel_speed_SOG # distance/ speed = time



    def get_vessel_trajectory(self,mmsi):
        import time
        QUERY1 = "SELECT ST_X(geom) as lat, ST_Y(geom) as lon, sog,cog, datetimestamp FROM interpolated_geom WHERE mmsi = " + str(mmsi)+" ORDER BY datetimestamp;"

        my_dat = pd.read_sql(QUERY1,self.db_connection)

        my_dat = my_dat.reset_index()





        my_dat = my_dat.reset_index()
        my_dat["datetimestamp"] = pd.to_datetime(my_dat["datetimestamp"], unit='s')
        my_dat["datetimestamp"] = my_dat["datetimestamp"] - my_dat["datetimestamp"][0]


        total_trajectory = (my_dat["datetimestamp"].values.astype(int)/1000000000)[-1]
        print(total_trajectory)
        return my_dat

    def vessel_stride_cut(self, data_in, stride_sec = 3600):
        """This function will do the striding of the dataset"""

        print(float(data_in["datetimestamp"].values[-1].astype(int)/1000000000))

        if(float(stride_sec) > data_in.shape[0]):
            print("FALSE RETURN")
            return False

        data_in = pd.DataFrame(data_in.iloc[stride_sec:,:])
        data_in = data_in.reset_index(drop=True)
        data_in["datetimestamp"] = data_in["datetimestamp"] - data_in["datetimestamp"].values[0]

        return data_in


    def get_vessels(self, lat, lon):
        """
        This function will return the number of vessels in a pre-specified radius
            lon (float): Longitude
            lat (float) : Latitude
            radius (float): Vessel search radius in meatre
        """
        QUERY_part1 =" SELECT mmsi, st_x(geom) as lat, st_y(geom) as lon, sog, cog, datetimestamp FROM interpolated_geom WHERE ST_DWithin(ST_SetSRID(geom, 4326)::geography, ST_MakePoint("
        QUERY_part2 = str(lat) + "," +str(lon) +")::geography, "+str(self.search_radius)+");"
        QUERY = QUERY_part1 + QUERY_part2
        # Extracting the data from the DB
        vessel_data = pd.read_sql(QUERY,self.db_connection)
        course_lower_bound = (self.current_vessel_COG - self.max_course_deviation)
        if(course_lower_bound < 0):
            course_lower_bound = 2*np.pi - course_lower_bound
        course_upper_bound = self.current_vessel_COG + self.max_course_deviation

        if(course_upper_bound > 2*np.pi):
            course_upper_bound = 2*np.pi -course_upper_bound

        #Exclude all other data according to deviation tollerance
        vessels_to_include = np.deg2rad(vessel_data["cog"].values) >  (self.current_vessel_COG - self.max_course_deviation )
        vessel_data = vessel_data.iloc[vessels_to_include,:]
        vessels_to_include = np.deg2rad(vessel_data["cog"].values) <  (self.current_vessel_COG + self.max_course_deviation )
        vessel_data = vessel_data.iloc[vessels_to_include,:]

        if vessel_data.shape[0] > 0:
            return vessel_data
        else:
            return copy.deepcopy(self.data_radius)

        return vessel_data

    def get_apriori_cog(self):
        # Extracing the course over ground
        cog_radians = np.deg2rad(self.data_radius["cog"].values)

        s_bar = np.median(np.sin(cog_radians))
        c_bar = np.median(np.cos(cog_radians))

        if s_bar > 0 and c_bar > 0:
            return np.arctan(s_bar/c_bar)
        elif c_bar < 0:
            return np.arctan(s_bar/c_bar) + np.pi
        elif s_bar < 0 and c_bar > 0:
            return np.arctan(s_bar/c_bar) + 2*np.pi

    def get_apriori_sog(self):
        # GTet the apriori median value
        return np.median(self.data_radius["sog"].values)*0.5144444

    def predict_position_p(self):
        """
        This function will predict the new position of the vessel
        p(k+1) =  p(k) + delta_l[sin(cog_k)*f(lat_k),cos(cog_k)*f(lon_k)]
        """
        p_k = self.current_pos # Get current pos

        # Calculate the updated longitude and latitude
        lat_pred = self.delta_l*np.sin(self.apriori_cog)*0.0000089979
        lon_pred = self.delta_l*np.cos(self.apriori_cog)*0.0000089979

        p_k_1 = p_k + np.array([lat_pred,lon_pred])

        return p_k_1


    def get_db_connection(self):
        db_uname = "postgres"
        db_password = "a1s_data!"
        db_ip = "127.0.0.1:5432"
        db_name = "postgres"
        try:
            db_connection = sqlalchemy.create_engine(
            'postgres+psycopg2://{0}:{1}@{2}/{3}'.format(db_uname,
                                                    db_password,
                                                    db_ip,
                                                    db_name),
            pool_recycle = 1, pool_timeout = 57600).connect()
            return db_connection
        except Exception as err:
            print("Error in connecting to the database", err)
            return False # Return false if it is not working

#---------------------------------------------------------------------------------------------------

for pred_len in [5,15,30,1,2*60,3*60, 5*60]:
    VesselCompMethod(236386000, search_radius = 50,cog_deviation_deg = 25, prediction_lenght_sec = pred_len*60 )
