import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import copy
import math
import datetime

import utm
from sklearn.linear_model import LinearRegression



import joblib as joblib
from joblib import Memory

import scipy.ndimage as ndimage
from scipy.interpolate import interp1d

################################ END OF IMPORTS #########################################

class LRM_APRIORI(object):
    # CONSTANTS
    NEAREST_Neigh_SIZE = 5
    COORD_index_search_range = np.array([])
    COG_Matrix = None
    COG_VARIANCE_matrix = None
    SDM_matrix = None
    flipped_SDM = None

    def __init__(self, df_vessel, window_size, using_apriori_COG,
                                            using_apriori_COGSDM,
                                            use_APRIORI_SOG,
                                            predict_period_sec,
                                            nearest_n = 5,
                                            universal_plotting = False,
                                            initial_speed = 5,
                                            lat_offset = -10, lon_offset = 45):
        super(LRM_APRIORI, self).__init__()

        self.counter_test = 0

        self.df_vessel = df_vessel # Pandas Dataframe of the vessel info
        #print(self.df_vessel.head())

        self.NEAREST_Neigh_SIZE = nearest_n

        self.universal_plotting = universal_plotting

        # Longitude & Latitude offset (due to centring of coordinates)
        self.lat_offset = lat_offset
        self.lon_offset = lon_offset

        self.USE_APRIORI_COG = using_apriori_COG # only use COG
        self.USE_APRIORI_COGSDM = using_apriori_COGSDM # USE COG + CELL COUNTS
        self.USE_APRIORI_SOG = use_APRIORI_SOG

        # CONSTANTS
        self.pixel_size = 0.008 # 0.008 = 1250 0.004 = 2500
        self.grid_size = 1251



        # Variables that will be local t save the adjustment values at each iteration
        self.lat_adjust = 0
        self.lon_adjust = 0

        # Vector lists
        self.decimal_accuracy = 10000 # This variable is for the accuray when converting from the index to the coordinate domains


        self.window_size = window_size


        # prediciton future
        self.pred_period_sec = predict_period_sec

        # Setting matrices
        self.COG_Matrix = copy.deepcopy(self.get_COG_SDM())
        self.COG_VARIANCE_matrix = copy.deepcopy(self.get_variance_SDM())
        self.SDM_matrix  = self.get_SDM()
        # Setting the global variable of the flipped SDM
        self.flipped_SDM = self.get_flipped_SDM()

        self.SOG_SDM = self.get_SOG_SDM()

        # SET INITIAL PARAMETERS:
        self.initialize_parameters()

        self.initial_speed = np.mean(self.df_vessel['sog'].values[:3])


        """The following constant will be the value used to set the confidence in the Apriori update"""
        self.apriori_confidence_val = 0

        # Generate a constant grid to search through, to not generate it each iteration
        self.COORD_index_search_range = copy.deepcopy(np.round(np.arange(0,10 + self.pixel_size, self.pixel_size),6))

################## LRM SDM CLASSS ###########################################################################
################## LRM SDM CLASSS #######################################################################False####
################## LRM SDM CLASSS ###########################################################################
################## LRM SDM CLASSS ###########################################################################
################## LRM SDM CLASSS ###########################################################################


    def is_tanker(self):
        data_type_dr = r"../nari_static.csv"
        my_data = pd.read_csv(data_type_dr)
        my_data_new = my_data[["sourcemmsi","shiptype"]]

        mmsi_no = self.df_vessel["mmsi"].values[0]
        extracted_vessel_info = my_data_new.iloc[my_data_new["sourcemmsi"].values == mmsi_no,1].values[0]

        if(extracted_vessel_info < 80):
            return False
        else:
            return True

    def get_SOG_SDM(self,sigma = 0.1):

        if(self.is_tanker()):
            SOG_SDM = joblib.load("../Saved_Matrices/Tanker_SOG_MEAN_1251.sav")
        else:
            SOG_SDM  = joblib.load("../Saved_Matrices/Cargo_SOG_MEAN_1251.sav")

        SOG_SDM  = copy.deepcopy(np.flip(SOG_SDM,0))
        SOG_SDM  = ndimage.gaussian_filter(SOG_SDM ,sigma, order = 0)
        return SOG_SDM

    def get_variance_SDM(self, sigma = 0.5):
        cargo_var = joblib.load("../Saved_Matrices/Real_Cargo_variance_15km_6h.sav")
        cargo_cog = joblib.load("../Saved_Matrices/Real_Cargo_with_COG.sav")
        COG_VARIANCE_matrix  =  copy.deepcopy(np.flip(cargo_var,0))
        COG_VARIANCE_matrix /= copy.deepcopy(np.flip(cargo_cog,0))
        del cargo_var
        del cargo_cog

        COG_VARIANCE_matrix = np.sqrt(COG_VARIANCE_matrix)
        COG_VARIANCE_matrix  = ndimage.gaussian_filter(COG_VARIANCE_matrix ,sigma, order = 0)
        return COG_VARIANCE_matrix


    def get_COG_SDM(self, sigma = 0.1):
        # Getting the RAW SDM to calculate the AVERAGE COG PER CELL
        SDM_matrix  = joblib.load("../Saved_Matrices/Real_Cargo_with_COG.sav")
        COG_Matrix_temp = joblib.load("../Saved_Matrices/Real_Cargo_COG_1250_mat.sav")
        # The course over grounds are te counts of the COG values per cell, divide by the count of observations
        COG_Matrix_temp /= SDM_matrix # Element wise devision

        COG_Matrix = copy.deepcopy(np.flip(COG_Matrix_temp,0))
        del SDM_matrix
        del COG_Matrix_temp

        COG_Matrix = ndimage.gaussian_filter(COG_Matrix, sigma , order=0)
        COG_Matrix  = np.deg2rad(COG_Matrix)
        return COG_Matrix



    def get_flipped_SDM(self, sigma = 0.05):
        joblib_mat_load = joblib.load("../Saved_Matrices/Real_Cargo_with_COG.sav")
        flipped_SDM = copy.deepcopy(np.flip(joblib_mat_load,0))

        del joblib_mat_load

        # Gaussian Filter - and SET THE SDM
        flipped_SDM =  ndimage.gaussian_filter(flipped_SDM, sigma, order=0)
        return flipped_SDM


    def get_SDM(self, smooth = True, sigma_smooth = 0.5):
        SDM_load_joblib = joblib.load("../Saved_Matrices/Real_Cargo_with_COG.sav")
        np.seterr(divide='ignore')
        if(smooth): #kearnal smoothing
            SDM_load = copy.deepcopy(ndimage.gaussian_filter(SDM_load_joblib, sigma = sigma_smooth, order=0))
            del SDM_load_joblib

        return(SDM_load)

    def get_delta_time(self):
        return self.df_vessel["datetimestamp"].values

    def get_calculate_course(self, lon_observed, lat_observed, current_sec):
        """
        This function takes in the longitude, and latutide of the current and previous
        observations. It makes use of the linear regression y = mx +c forula's
        gradient calculation to calculate the angle between the two sets of coordiantes.
        With getting this angle we can make use of arctan, to calculate the angle in degrees (RADIANS),
        gettingthe course over ground of our observations
        """
        lat_prev = lat_observed[current_sec-1]
        lat_curr = lat_observed[current_sec]

        lon_prev = lon_observed[current_sec-1]
        lon_curr = lon_observed[current_sec]

        lon_arr = np.diff(np.array([lon_prev,lon_curr]))
        lat_arr = np.diff(np.array([lat_prev,lat_curr]))

        #gradient =(lat_curr - lat_prev)/(lon_curr - lon_prev) # gradient formula (y2-y1)/(x2-x1)
        #course =  np.arctan(gradient) # get the estimated course in Radians
        course_2 = np.arctan2(lat_arr,lon_arr) % (2*np.pi)

        return course_2


    def initialize_parameters(self):
        """
        This function will initialise the parameters for the LRM and Apriori predictions.
        We do this so that we have our "Training" set model that is updated dynamically
        The inital parameters are set to the size of the LRM windowsize.
        """
        initial_values = self.window_size # due to exclucivity

        # Set the course to the linear radians version
        #self.course_linear =  self.get_linear_course()[:]
        self.initial_speed_vector = self.get_SOG()[:initial_values]

        # Need the fist two observations to estimate the course
        self.lat_values_init = self.get_lat_values()[:initial_values]
        self.lon_values_init = self.get_lon_values()[:initial_values]

        self.course_est = self.get_calculate_course(self.lat_values_init,self.lon_values_init,initial_values-1)
        ##print("TIME")
        ##print(self.get_delta_time())
        self.initial_time_vector = self.get_delta_time()[:initial_values]

        ##print("Initial time excluded:",self.initial_time_vector )


    def AIS_linear(self,testing_dataset,stride, return_stuff = False):
        #print("AIS LIN")
        #print(testing_dataset.head())
        APRIORI_SEC = 0

        # Variables to save & set throughout
        time_history, speed_history , pred_speed = np.array([]), np.array([]), np.array([])
        X_loc_pred, Y_loc_pred, sec_observed = [], [], []
        # observed_speed_X, observed_speed_Y = [], [] # to save originals

        # Setting initial values  Extracting the values from the dataframe that we observed
        lat_observed = self.get_lat_values()
        lon_observed = self.get_lon_values()

        # Reduce the timing to exclude initial values
        time_observed = self.get_time_observed()[self.window_size:]
        ##print("TIME OBSERVED FROM RPED:",time_observed)
        time_range_regular_intervals = self.generate_regular_time(time_observed)

        # Getting inital values
        #speed = self.get_SOG()
        #course = self.course_linear
        #plt.plot(lat_observed,lon_observed,"yo", markersize = 15)
        lat_new = lat_observed[self.window_size]
        lon_new = lon_observed[self.window_size]

        # Errors to correct any prediction mistake in the prediction/measurement update equations
        #lat_error, lon_error = 0, 0

        # INITIAL Linear Model for the SOG, LON and LAT
        linear_Model_SOG = self.generate_linear_model(self.initial_time_vector,self.initial_speed_vector)
        # Counting the entries to appropriatly measure difference
        counter_sec = 0
        est_course = self.course_est


        for sec in time_range_regular_intervals:

            if(time_observed[counter_sec] == sec):
                """ MEASUREMENT UPDATE - if a value is observed """
                ##print("Measurement Update")

                if(return_stuff):
                    sec_observed.append(sec) # We observed index at this second

                # updating time Speed history
                time_history, speed_history = self.update_speed_time_history(time_history,
                                                    speed_history, speed,sec, counter_sec)

                # Fit a new LRM based on our latest remember history
                linear_Model_SOG = self.generate_linear_model(time_history,speed_history)

                # Get predicted speed, based on the LRM model
                predict_speed_value = self.get_predicted_sog(linear_Model_SOG,sec)
                del linear_Model_SOG


                # Saving SOG in the x and y direction "Longitudional and Latitudinal"
                pred_lat_speed = self.speed_in_latitudional(predict_speed_value[0][0],est_course)
                pred_lon_speed = self.speed_in_longitudional(predict_speed_value[0][0],est_course)

                pred_lat_speed = self.speed_in_latitudional(predict_speed_value[0][0],est_course)
                pred_lon_speed = self.speed_in_longitudional(predict_speed_value[0][0],est_course)

                # Calculate the accurate course
                est_course_update = self.get_calculate_course(lat_observed,lon_observed,counter_sec)

                # Predicted speeds and observed speeds in the x_y directions
                pred_speed = np.append(pred_speed, predict_speed_value)
                # Saving the observed speed values in the respective direction
                # observed_speed_X.append(self.speed_in_latitudional(speed[counter_sec],est_course))
                # observed_speed_Y.append(self.speed_in_longitudional(speed[counter_sec],est_course))
                """
                The next set of functions will be calculating the new predicted coordinate
                making use of the linear method way by adding the predicted value to the
                current value
                """
                # Speed in the x-axis prediction-----------------------------------------------------------------------------
                lat_new = lat_new + pred_lat_speed #+ lat_error
                # Calculating the error of the predicted value and the true observed value
            #    lat_error = lat_observed[counter_sec] - lat_new  # Adjust error for improved prediction

                if(return_stuff):
                    X_loc_pred.append(lat_new)

                # Speed in the y-axis prediction ---------------------------------------------------------
                lon_new = lon_new + pred_lon_speed #+ lon_error
                # Calculating the error of the predicted value and the true observed value
            #    lon_error = lon_observed[counter_sec] - lon_new  # to update next position to the correct spot du to errors
                if(return_stuff):
                    Y_loc_pred.append(lon_new)

                # Counter to walk through each element in the observed list
                counter_sec += 1
            else:
                """
                FUTURE PREDICTION - based on the known model and prediction, when we have no observed value
                """

                future_lat_apriori, future_lon_apriori = np.array([]), np.array([]) # Future lat/lon prediction correction per observation

                # Reset prediction for next prediction period
                predictor_lat_new = lat_new
                predictor_lon_new = lon_new

                pred_lat_speed_old = self.convert_sog(self.real_speed_in_latitudional(self.initial_speed,est_course)*0.0000089)
                pred_lon_speed_old = self.convert_sog(self.real_speed_in_longitudional(self.initial_speed,est_course)*0.0000089)

                # Amount of time we are predicting into the future
                ##print("self.pred_period_sec",self.pred_period_sec)
                ##print("This is sec:",sec)
                for time in range(sec,sec + self.pred_period_sec):
                    """PREDICTOR EQUATIONS - when we have not observed an observation at a timestep"""


                    """---- ----  Below we turn on the APRIORI infrmation for SPEED ---- ----"""
                    if(self.USE_APRIORI_COG):

                        if(self.USE_APRIORI_SOG):

                        #    if(np.isnan(predictor_lat_new) or np.isnan(predictor_lon_new))


                            pred_lat_speed, pred_lon_speed = self.get_Apriori_SOG_prediction(predictor_lat_new,predictor_lon_new,est_course) # ADD SPEED APRIORI CODE HERE
                            ###print("PRED SPEEDS: ",pred_lat_speed,pred_lon_speed)

                            if(np.isnan(pred_lat_speed) or np.isnan(pred_lon_speed)):
                                #pred_lat_speed, pred_lon_speed = self.get_LRM_prediction(linear_Model_SOG,time,pred_speed,est_course)

                                pred_lat_speed = copy.deepcopy(pred_lat_speed_old)
                                pred_lon_speed = copy.deepcopy(pred_lon_speed_old)
                                ###print("PRED SPEEDS: ",pred_lat_speed,pred_lon_speed)

                            else:
                                pred_lat_speed_old = copy.deepcopy(pred_lat_speed)
                                pred_lon_speed_old = copy.deepcopy(pred_lon_speed)
                        else:
                            pred_lat_speed, pred_lon_speed = self.get_LRM_prediction(linear_Model_SOG,time,pred_speed,est_course)


                        #("PRED SPEEDS: (AGAIN) ",pred_lat_speed,pred_lon_speed)

                        # ##print("XX:",pred_lat_speedxx)
                        # pred_lat_speed, pred_lon_speed = self.get_LRM_prediction(linear_Model_SOG,time,pred_speed,est_course)

                    # Predicting the X and Y coordinate, and adjust for the previously caclulated error
                    predictor_lat_new = copy.deepcopy(predictor_lat_new + pred_lat_speed )#+ lat_error)
                    predictor_lon_new = copy.deepcopy(predictor_lon_new + pred_lon_speed )#+ lon_error)

                    # Reset error after the corrections was appliedm - Errors are from the observed observations
                    # lat_error = 0
                    # lon_error = 0



                    """---- ----  Below we turn on the APRIORI infrmation ---- ----"""
                    if(self.USE_APRIORI_COG):
                        lat_corr, lon_corr = self.SDM_Update_V2(predictor_lat_new,predictor_lon_new)
                        predictor_lon_new  = copy.deepcopy(lon_corr)
                        predictor_lat_new  = copy.deepcopy(lat_corr)


                        if(self.universal_plotting):
                            plt.plot(predictor_lat_new, predictor_lon_new,'o', color = "orange")

                        # Calculate APRIORI COURSE
                        est_course =  self.cog_apriori_update(est_course)
                    else:
                        if(self.universal_plotting):
                            plt.plot(predictor_lat_new, predictor_lon_new,'o', color = "orange")

                    """ADDING THE first next timestep PREDICTION TO BE SAVED so that the LRM
                    can still build of this and not reset to the previously known measurement update"""
                    if(time == sec):
                        lat_new = copy.deepcopy(predictor_lat_new)
                        lon_new = copy.deepcopy(predictor_lon_new)

                    # Add the predictions to the lsit

                    #if(return_stuff):
                    Y_loc_pred.append(predictor_lon_new[0][0])
                    X_loc_pred.append(predictor_lat_new[0][0])

                    if(self.universal_plotting):
                        future_lat_apriori = np.append(future_lat_apriori,predictor_lat_new)
                        future_lon_apriori = np.append(future_lon_apriori,predictor_lon_new)



                ##print("Get Test")

                self.alg_testing_V2(X_loc_pred,Y_loc_pred,time_observed,testing_dataset,stride )
                #self.linear_interp_testing(testing_dataset,predictor_lat_new,predictor_lon_new,time,pred_speed )

                ##print("after test")
                if(self.universal_plotting):
                    self.plot_SDM()
                    plt.plot(self.df_vessel['lat'],self.df_vessel['long'],"m")
                    plt.plot(future_lat_apriori,future_lon_apriori, "green")
                    plt.show()


                # Early stopping to test the algorithm piece by piece
                if(return_stuff):
                    return np.array(X_loc_pred), np.array(Y_loc_pred), np.array(pred_speed), np.array(sec_observed)
                else:
                    return


                #   # SDM CORRECTIONS
                # if(sec % APRIORI_SEC == 0): # TO SHOW THE LAST BIT OF THE TRAJECTORY
                    #plt.show()
                    #self.linear_interp_testing(predictor_lat_new,predictor_lon_new,self.pred_period_sec,pred_speed )



        if(return_stuff):
            return np.array(X_loc_pred), np.array(Y_loc_pred), np.array(pred_speed), np.array(sec_observed)

    def get_LRM_prediction(self, linear_Model_SOG, time,pred_speed, est_course):
        predict_speed_value = self.get_predicted_sog(linear_Model_SOG,time)
        pred_speed = np.append(pred_speed, predict_speed_value)

        # Predicted speed in the x- and y-axis
        pred_lat_speed = self.speed_in_latitudional(predict_speed_value,est_course)
        pred_lon_speed = self.speed_in_longitudional(predict_speed_value,est_course)

        return pred_lat_speed, pred_lon_speed


    def alg_testing_V2(self, pred_lat, pred_lon, pred_sec, testing_set,stride):
        ##print("Errror Test 2")
        ##print(self.df_vessel.head())
        ##print(pred_lat, pred_lon, pred_sec)
        #testing_set = testing_set.iloc[testing_set["datetimestamp"].values >= self.df_vessel["datetimestamp"].values[2],:]
        #testing_set = testing_set.iloc[stride:,:]
        #print("Test HEAD",testing_set)
        #testing_set = testing_set.reset_index()
        ##print(testing_set.iloc[pred_sec[0],:].values)
        ##print("LENGHT:", len(pred_lat))
        ##print(pred_lat)

    #    plt.plot(testing_set["lat"],testing_set["long"], "bo")
        #plt.plot(pred_lat,pred_lon, "ro-", markersize = 12)
        ##print(self.df_vessel["datetimestamp"].values[2])
        time_start = pred_sec[0]
        #print(self.pred_period_sec,self.pred_period_sec)
        test_set_lat = testing_set["lat"].values[time_start:(time_start+self.pred_period_sec)]
        test_set_lon = testing_set["long"].values[time_start:(time_start+self.pred_period_sec)]
        #plt.plot(test_set_lat,test_set_lon, "go")
        #
        # plt.plot(pred_lat[-1],pred_lon[-1],"bo")
        # plt.plot(test_set_lat[-1],test_set_lon[-1],"ro")
        # plt.show()

        my_error = self.haversin_distance_error(pred_lat[-1],
                                            pred_lon[-1],
                                            test_set_lat[-1],
                                            test_set_lon[-1])
        print("New Error",my_error)
        mmsi = int(self.df_vessel["mmsi"].values[0])
        temp_dictionary = [{"MMSI": mmsi,
                            "Hour": int(self.pred_period_sec/60),
                            "Error:": float(my_error) ,
                            "Window Size":self.window_size,
                            "Nearest NN":self.NEAREST_Neigh_SIZE}]

        temp_df = pd.DataFrame.from_dict(temp_dictionary)

        #CSV NAME:
        if(self.USE_APRIORI_COGSDM and not self.USE_APRIORI_SOG):
            csv_name = "../LRM Apriori Results/COG SDM/" +str(mmsi)+"_errors_COG_SDM.csv"
        elif(self.USE_APRIORI_COG and not self.USE_APRIORI_SOG):
            csv_name = "../LRM Apriori Results/COG/" +str(mmsi)+"_errors_COG.csv"
        elif(self.USE_APRIORI_SOG):
            csv_name = "../LRM Apriori Results/SOG/" +str(mmsi)+"_errors_SOG.csv"
        else:
            csv_name = "../LRM Apriori Results/LRM/"+str(mmsi)+"_errors_LRM.csv"

        temp_df.to_csv(csv_name, mode = "a", header = False)
        #plt.show()

    def linear_interp_testing(self,testing_dataset,lat_in, lon_in, obs_time,pred_speed, path =  "Vessels Testing/"):
        # This function will compare the predicted to the linear interpolated values
        x = self.df_vessel["datetimestamp"].values
        lon = self.df_vessel['long'].values
        lat = self.df_vessel['lat'].values
        sog = self.df_vessel['sog'].values
        mmsi = int(self.df_vessel["mmsi"].values[0])

        testing_dataset = testing_dataset.reset_index()


        obs_time = int(obs_time + self.initial_time_vector[-1])

        ##print("lat_in",lat_in)

        lat_test = testing_dataset['lat'].values
        lon_test = testing_dataset['long'].values
        if(self.universal_plotting):
            plt.plot(lat,lon,"co-")
            plt.plot(lat_test[obs_time],lon_test[obs_time],"ro-", markersize = 2)
            plt.plot(lat_in,lon_in,"bo", markersize = 10) # predicted obs

        # Plotting the error line
        #plt.plot(lat_test[303], lon_test[303], "go",markersize = 2)
        my_error = self.haversin_distance_error(lat_test[obs_time],
                                            lon_test[obs_time],
                                            lat_in,
                                            lon_in)

        lat_err = np.array([lat_test[obs_time],lat_in])
        lon_err = np.array([lon_test[obs_time],lon_in])
        if(self.universal_plotting):
            plt.plot(lat_err, lon_err,'--', color = 'orange')

        temp_dictionary = [{"MMSI": mmsi,
                            "Hour": int(self.pred_period_sec/60/60),
                            "Error:": float(my_error[0]) ,
                            "Window Size":self.window_size,
                            "Nearest NN":self.NEAREST_Neigh_SIZE}]

        temp_df = pd.DataFrame.from_dict(temp_dictionary)

        #CSV NAME:
        if(self.USE_APRIORI_COGSDM and not self.USE_APRIORI_SOG):
            csv_name = "../LRM Apriori Results/COG SDM/" +str(mmsi)+"_errors_COG_SDM.csv"
        elif(self.USE_APRIORI_COG and not self.USE_APRIORI_SOG):
            csv_name = "../LRM Apriori Results/COG/" +str(mmsi)+"_errors_COG.csv"
        elif(self.USE_APRIORI_SOG):
            csv_name = "../LRM Apriori Results/SOG/" +str(mmsi)+"_errors_SOG.csv"
        else:
            csv_name = "../LRM Apriori Results/LRM/"+str(mmsi)+"_errors_LRM.csv"

        temp_df.to_csv(csv_name, mode = "a", header = False)

        if(self.universal_plotting):
            plt.legend(['Original Trajectory','Real Answer', "Estimated",str("Error: " +str(my_error[[0]]))])
            plt.show()

        del lat_test
        del temp_df
        del lon_err
        del temp_dictionary



    def haversin_distance_error(self,lat_1, lon_1, lat_2, lon_2, metres = 1):
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


    def cog_apriori_update(self, x):

        # Getting the Counts weight
        weight_mat = copy.deepcopy(self.nearest_n_grid)
        max_val = np.max(weight_mat)

        # Getting the variance error

        # Getting the centre value for confidence
        centre = int((weight_mat.shape[0]-1)/2)
        centre_val = weight_mat[centre,centre]

        SDM_VARIANCE_val = self.COG_VARIANCE_nearest_n[centre,centre]

        # if the cente of the matrix is 0, then sstop
        if(centre_val == 0 or SDM_VARIANCE_val  > 10 or np.isnan(SDM_VARIANCE_val)):
            return x

        # Get the centre value of the SMD
        SDM_COG = self.COG_nearest_n[centre,centre]
        if(np.isnan(SDM_COG)):
            return x

        historic_COG = copy.deepcopy(SDM_COG)
        update_val = centre_val/max_val


        # Weight of the apriori COG update caluclated vs estiamted
        COG_update = (x*(1-update_val) + (update_val)*historic_COG)

        return COG_update

    def plot_SDM(self):
        #Smooth the SDM
        SDM_matrix = self.SDM_matrix
        # SDM_matrix[SDM_matrix>= 1 ] = 100
        # SDM_matrix[SDM_matrix<  0 ] = 0
        SDM_matrix = np.log(SDM_matrix)
        SDM_matrix[SDM_matrix<0] = 0

        SDM_matrix = np.flip(SDM_matrix,0)
        #plt.imshow(SDM_matrix, extent=[-10,0,45,55])

    def plot_COG_SDM(self):
        #Smooth the SDM
        SDM_matrix = self.COG_Matrix
        SDM_matrix = np.flip(SDM_matrix,0)
        SDM_matrix = np.rad2deg(SDM_matrix)
        #plt.imshow(SDM_matrix, extent=[-10,0,45,55])

    def adjust_speed_time_history(self,speed_history,time_history):
        """
        We remove an entry (the oldest) to keep the speed history and the
        time history constant to the remember history. For the first n observations
        smalleer than the windows size this will not be true there will only append's
        to build up the lists
        """
        # Remove "old" elements form the list to keep them with the remember history lenght
        return np.delete(speed_history,0), np.delete(time_history,0)

    def update_speed_time_history(self, time_history, speed_history, speed , sec, counter_sec):

        if len(time_history) > self.window_size:
            speed_history, time_history = self.adjust_speed_time_history(speed_history, time_history)

        # Saving the appropriate time and speed values in our remember history
        time_history = np.append(time_history, sec)
        speed_history = np.append(speed_history, speed[counter_sec])

        return time_history, speed_history


    def get_linear_course(self):
        # Extracting the COG
        course = self.df_vessel['cog'].values
        # Converting from degrees to radians for functions
        course =  np.deg2rad(course)
        # Round to the nth desimal
        course = np.round(course,7)

        return course

    def generate_regular_time(self,time_observations):
        return np.arange(0,int(max(time_observations))+1,1)

    def get_lon_values(self):
        return self.df_vessel['long'].values

    def get_lat_values(self):
        return self.df_vessel['lat'].values

    def get_time_observed(self):
        return self.df_vessel['datetimestamp'].values.astype(int)

    def get_SOG(self):
        return self.convert_sog(self.df_vessel['sog'].values*0.0000089) #https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters


    def generate_linear_model(self,x_values,y_values):
        LR_model = LinearRegression()
        LR_model.fit(x_values.reshape(-1,1),
                y_values.reshape(-1,1))

        return LR_model

    def get_predicted_sog(self,reg_model,time):
        return reg_model.predict(np.array([time]).reshape(-1,1))

    def speed_in_latitudional(self, predicted_val, estimated_course):
        predicted_longitudional = predicted_val*np.cos(estimated_course)
        return predicted_longitudional

    def speed_in_longitudional(self, predicted_val, estimated_course):
        predicted_latitudional = predicted_val*np.sin(estimated_course)
        return predicted_latitudional

    def convert_sog(self, data_in):
        return data_in*0.5144444444 #meter per second


    def get_Apriori_SOG_prediction(self,lat_in, lon_in, course_in):

        lat_coord = lat_in - self.lat_offset
        lon_coord = lon_in - self.lon_offset

        ###print(lat_coord, lon_coord)

        lat_index =  self.convert_to_SDM_index_values(lat_coord).astype(int) # Due to centreing)
        lon_index =  1251 - self.convert_to_SDM_index_values(lon_coord).astype(int) # Due to centreing)


        ###print(lon_index,lat_index)
        my_temp_SOG_SDM = self.SOG_SDM[lon_index,lat_index]

        neighborhood_size_sog = 2
        upper_bound_lon = lon_index+neighborhood_size_sog+1
        lower_bound_lon = lon_index-neighborhood_size_sog

        lower_bound_lat = lat_index-neighborhood_size_sog
        upper_bound_lat = lat_index+neighborhood_size_sog +1


        my_temp_SOG_SDM = self.SOG_SDM[lower_bound_lon:upper_bound_lon,lower_bound_lat:upper_bound_lat]
        my_temp_SOG_SDM = my_temp_SOG_SDM[np.isnan(my_temp_SOG_SDM) == False]
        my_temp_SOG_SDM = np.median(my_temp_SOG_SDM)

        lat_SOG = self.convert_sog(self.real_speed_in_latitudional(my_temp_SOG_SDM,course_in)*0.0000089)#*0.0000089*0.5144444444
        lon_SOG = self.convert_sog(self.real_speed_in_longitudional(my_temp_SOG_SDM,course_in)*0.0000089)#*0.0000089*0.5144444444

        return lat_SOG, lon_SOG


    def real_speed_in_latitudional(self, predicted_val, estimated_course):
        predicted_longitudional = predicted_val*np.cos(estimated_course)
        return predicted_longitudional

    def real_speed_in_longitudional(self, predicted_val, estimated_course):
        predicted_longitudional = predicted_val*np.sin(estimated_course)
        return predicted_longitudional

############## APRIORI INFORMATION ##############################

    def get_weight_mat(self,lat_coord, lon_coord):
        # Convert Coordinate values to index values
        lat_coord = lat_coord - self.lat_offset
        lon_coord = lon_coord - self.lon_offset

    #    ##print("LAT COORD", lat_coord)

        lat_index =  self.convert_to_SDM_index_values(lat_coord).astype(int) # Due to centreing)
        lon_index = self.grid_size - self.convert_to_SDM_index_values(lon_coord).astype(int) # Due to centreing)

        # Calculater the bounds to extract the data
        upper_bound_lon = lon_index+self.NEAREST_Neigh_SIZE+1
        lower_bound_lon = lon_index-self.NEAREST_Neigh_SIZE

        lower_bound_lat = lat_index-self.NEAREST_Neigh_SIZE
        upper_bound_lat = lat_index+self.NEAREST_Neigh_SIZE +1


        weights = self.flipped_SDM[lower_bound_lon:upper_bound_lon,lower_bound_lat:upper_bound_lat]
        # get transposed due to flipping
        self.nearest_n_grid = copy.deepcopy(weights) # Making a global copy

        self.COG_nearest_n = self.COG_Matrix[lower_bound_lon:upper_bound_lon,lower_bound_lat:upper_bound_lat]

        self.COG_VARIANCE_nearest_n = self.COG_VARIANCE_matrix[lower_bound_lon:upper_bound_lon,lower_bound_lat:upper_bound_lat]

        # Confidence - adjusting how much we should trust the LRM over the A Priori information
        """
        Adding the confidence in our prediction, if we predict in a cell that we know is a good prediction, the
        apriori information should be less, that is why we say 1 - (centre weight grid value)/(max value in weight grid)
        """
        # ##print(lon_index,lat_index)
        centre_weight =  self.flipped_SDM[lon_index,lat_index]
        max_weight = np.max(weights)

        # Update confidence
        aprioiri_conf = 1 - (centre_weight/max_weight)

        return weights, aprioiri_conf


    def SDM_Update_V2(self,pred_lat,pred_lon):
        """
        This function will do SDM updates on coordiantes
        """

        plot_char = False # Plot how newtons method is pulling data

        weights, apriori_conf = self.get_weight_mat(pred_lat,pred_lon) # Checked 3 NOV

        lat_adjusted, lon_adjusted = self.getting_distance(weights,pred_lat,pred_lon,plot_char)
        ###print("lat adj", lat_adjusted)

        # We minus from the lat (x-coordiante) because "positive and negative have different meanings"
        lat_sign = -1*(pred_lat < 0) + 1*(pred_lat >=  0)
        lon_sign = -1*(pred_lon < 0) + 1*(pred_lon >=  0)

        # Update confedence and if Weights should be included
        update_conf = apriori_conf*np.sum(self.USE_APRIORI_COGSDM)
        if(np.isnan(update_conf)):
            update_conf = 0
        # Adding the adjusted prediction + our confidence in our prediction, scaled to the pixel size
        lat_adjusted =  pred_lat + lat_sign*(lat_adjusted*self.pixel_size)*update_conf
        lon_adjusted =  pred_lon + lon_sign*(lon_adjusted*self.pixel_size)*update_conf
    #    ##print("lat adj", lat_adjusted)

        return lat_adjusted, lon_adjusted

    def convert_to_SDM_index_values(self,coordinate_in):
        index_val = np.sum(self.COORD_index_search_range < coordinate_in)
        my_ans = np.array(index_val)
        return my_ans


    """
    Important information to note:
    The SDM matrix is upside down with the observations starting from 0 --> 5000 on the y-axis from top to bottom
    and the x-axis starting from 0 --> 5000 left to right

    """

    def haversin_distance(self,lat_1, lon_1, lat_2, lon_2):

        lat_1 = lat_1*np.pi/180
        lat_2 = lat_2*np.pi/180
        lon_1 = lon_1*np.pi/180
        lon_2 = lon_2*np.pi/180
        R = 6371

        a = (np.sin(((lat_2-lat_1))/2))**2 + np.cos(lat_1)*np.cos(lat_2)*((np.sin((lon_2-lon_1)/2))**2)
        c = 2*np.arctan2(a**(0.5), (1-a)**2)
        d = R*c
        return(d)


    def euclidean_distance(self,lat_1, lon_1, lat_2, lon_2):
        d = ((lat_1-lat_2)**2+(lon_1-lon_2)**2)**0.5
        return(d)

    def get_gravity(self, weights, distances,lat_dist,lon_dist):
        """
        This function makes use of newtons formula to calculate a gravity force of pulling
        F_{1,2} = (weight1*weight2)/(distance**2)
        """

        midpoint = int((weights.shape[0]+1)/2 -1)
        mid_weight = weights[midpoint,midpoint]

        # If there are small distances make them small if very close
        mean_dist = np.mean(distances)
        if(mean_dist == 0.0):
            distances[distances == 0] = 1
        else:
            distances[distances == 0] = mean_dist
            #distances = distances/np.sum(distances) # Normalisz


        #*mid_weight # G1*G2 of newtons formula
        Force = weights/(distances**2)

        Force_normalization = np.sum(Force)
        if(Force_normalization != 0):
            Force = Force/np.sum(Force)

        n_length = len(weights)**2

        weight_sum = np.sum(weights)
        if(weight_sum == 0):
            return 0 ,0 # Do no adjustment

        # NORMALIZE WEIGHTS
        weights = weights/weight_sum

        weights_reshaped = weights.reshape(n_length)
        Force_reshaped = Force.reshape(n_length)
        Force_reshaped = weights_reshaped

        lat_pull =  np.sum((Force_reshaped[::-1]*lat_dist.reshape(n_length)))
        lon_pull =  np.sum((Force_reshaped*lon_dist.reshape(n_length)))

        return lat_pull ,lon_pull


    def getting_distance(self,weight_matrix,lat, lon, graphs = False):
        """Note the pixel size is in LAT/LON coordinates
            This function will measure the distance from the cartesian logic, where we measure the distance
            from left to right in the positive plane and right to left in the negative plane
        """

        pixel_size = self.pixel_size
        NN_size = self.NEAREST_Neigh_SIZE

        # Making new memory copies
        lat_update = copy.deepcopy(lat)
        lon_update = copy.deepcopy(lon)

        # Getting the location of where the coordinate is in the square to get an accurate distance vector
        if(graphs):
            plt.figure(figsize=(10,10))

        # Getting matrix info
        matrix_width_size = 1+2*NN_size

        # Creating empty matrices for saving information
        distance_mat = np.zeros((matrix_width_size,matrix_width_size))
        coordinate_mat_lat = np.zeros(distance_mat.shape)
        coordinate_mat_lon = np.zeros(distance_mat.shape)

        # Getting the middle coordinate
        midpoint = int((distance_mat.shape[0] +1 )/2 ) -1

        # To keep the sign and the right side of the pixel we add a boolean argument
        # and to shift the coordiantes back to the middle
        # maing use of modulo to keep the centre of the matrix/ Neighbour grid
        lat_rest =  lat % pixel_size/2
        lon_rest =  lon % pixel_size/2

        # If the coordinates are false, measure differently because modulo is directional in this distance measure case
        if(lat < 0 and lat_rest != 0):
            lat_rest = lat_rest - pixel_size/2

        if(lon < 0 and lon_rest != 0):
            lon_rest = lon_rest  - pixel_size/2


        # Calculating the distance to the adjacent block & remaining distances
        square_factor = pixel_size*NN_size
        distance_to_midpoints = np.arange(-square_factor,square_factor+0.5*pixel_size,pixel_size)

        # Counts for populating the matrices
        j_count = 0
        i_count = 0

        # Loops to populate matrices
        for i in distance_to_midpoints:  # LAT
            j_count = 0
            for j in distance_to_midpoints[::-1]:   # LON COUNT IN REVERSE, due to the centering neccesary --->> NOTES: 3 NOV 2020

                # Populating the coordinate distances
                coordinate_mat_lat[j_count,i_count] = i*1.0
                coordinate_mat_lon[j_count, i_count] = j*1.0

                # Calculating the Euclidean distances & Adding to matrix
            #    my_dist = self.euclidean_distance(lat_rest,lon_rest,i,j)
                #haversin_distance(self,lat_1, lon_1, lat_2, lon_2):
                my_dist = self.haversin_distance(lat_rest, lon_rest, i, j)
                weight = weight_matrix[j_count,i_count]

                distance_mat[j_count,i_count] = my_dist
                # If want to plot
                if(graphs):
                    plt.plot(np.array([lat_rest,i]),np.array([lon_rest,j]), "--", color ="orange")
                    #plt.text((lat_rest+i*2)/2,(lon_rest+j*2)/2,weight)
                    plt.text((lat_rest+i*2)/2,(lon_rest+j*2)/2,my_dist[0])
                    #plt.plot(i,j,"bo")
                j_count += 1
            i_count += 1

        adj_lat, adj_lon = self.get_gravity(weight_matrix, # Weights
                                       distance_mat, # Distances
                                       coordinate_mat_lat, # Longitudional Distances
                                       coordinate_mat_lon ) # Latitudional Distances

        lat_adjust = lat_rest + adj_lat
        lon_adjust = lon_rest + adj_lon


        if(graphs):
            extent_val = self.pixel_size*NEAREST_Neigh_SIZE
    #        plt.imshow(weight_matrix,extent=[-extent_val,extent_val,-extent_val,extent_val])
            #plt.imshow(distance_mat,extent=[-extent_val,extent_val,-extent_val,extent_val])
            #plt.plot(lat_adjust,lon_adjust,"ko", markersize = 15)
            #plt.plot(lat_rest,lon_rest,"ro")
            plt.tight_layout()
            plt.grid()
            plt.show()



        return lat_adjust, lon_adjust
