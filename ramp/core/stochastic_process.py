# -*- coding: utf-8 -*-
#%% Import required libraries
import numpy as np
import numpy.ma as ma
import random 
import math
from ramp.core.initialise import Initialise_model, Initialise_inputs, calibration_parameters
#from core import Appliance as App, User as Us

def calc_peak_time_range(user_list, peak_enlarg):
    """
    Calculation of the peak time range, which is used to discriminate between off-peak and on-peak coincident switch-on probability
    Calculates first the overall Peak Window (taking into account all User classes).
    The peak window is just a time window in which coincident switch-on of multiple appliances assumes a higher probability than off-peak
    Within the peak window, a random peak time is calculated and then enlarged into a peak_time_range following again a random procedure
    """
    Tot_curve = np.zeros(1440)  # creates an empty daily profile
    for Us in user_list:
        Tot_curve = Tot_curve + Us.windows_curve  # adds the User's theoretical max profile to the total theoretical max comprising all classes
    peak_window = np.transpose(np.argwhere(Tot_curve == np.amax(Tot_curve)))  # Find the peak window within the theoretical max profile
    peak_time = round(random.normalvariate(round(np.average(peak_window)), 1 / 3 * (peak_window[0, -1] - peak_window[0, 0])))  # Within the peak_window, randomly calculate the peak_time using a gaussian distribution
    rand_peak_enlarge = round(math.fabs(peak_time - random.gauss(peak_time, peak_enlarg * peak_time)))
    return np.arange(peak_time - rand_peak_enlarge , peak_time + rand_peak_enlarge)  # the peak_time is randomly enlarged based on the calibration parameter peak_enlarg

def randomise_variables(var):
    return random.uniform((1 - var), (1 + var))

def calc_random_cycle(time_1, power_1, time_2, power_2, r_c):
    return np.concatenate((np.ones(int(time_1 * r_c))* power_1, np.ones(int(time_2 * r_c))* power_2))

#%% Core model stochastic script
def Stochastic_Process(j):
    Profile, num_profiles = Initialise_model()
    peak_enlarg, mu_peak, s_peak, op_factor, Year_behaviour, User_list = Initialise_inputs(j)
    peak_time_range = calc_peak_time_range(user_list = User_list, peak_enlarg = peak_enlarg)

    '''
    The core stochastic process starts here. For each profile requested by the software user, 
    each Appliance instance within each User instance is separately and stochastically generated
    '''
    for prof_i in range(num_profiles): #the whole code is repeated for each profile that needs to be generated
        Tot_Classes = np.zeros(1440) #initialise an empty daily profile that will be filled with the sum of the hourly profiles of each User instance
        for Us in User_list: #iterates for each User instance (i.e. for each user class)
            Us.load = np.zeros(1440) #initialise empty load for User instance
            for i in range(Us.num_users): #iterates for every single user within a User class. Each single user has its own separate randomisation
                rand_daily_pref = 0 if Us.user_preference == 0 else random.randint(1, Us.user_preference)
                for App in Us.App_list: #iterates for all the App types in the given User class
                    #initialises variables for the cycle
                    tot_time = 0
                    App.daily_use = np.zeros(1440)
                    if ((random.uniform(0, 1) > App.occasional_use)  # evaluates if occasional use happens or not
                            or (App.Pref_index != 0 and rand_daily_pref != App.Pref_index)  # evaluates if daily preference coincides with the randomised daily preference number
                            or (App.wd_we != Year_behaviour[prof_i] and App.wd_we != 2)):  # checks if the app is allowed in the given yearly behaviour pattern
                        continue

                    #recalculate windows start and ending times randomly, based on the inputs
                    rand_window_1 = App.calc_rand_window(window_num=1)
                    rand_window_2 = App.calc_rand_window(window_num=2)
                    rand_window_3 = App.calc_rand_window(window_num=3)

                    #redefines functioning windows based on the previous randomisation of the boundaries
                    random_windows = [rand_window_1, rand_window_2, rand_window_3]
                    if App.flat == 'yes': #if the app is "flat" the code stops right after filling the newly created windows without applying any further stochasticity
                        total_power_value = App.POWER[prof_i]*App.number
                        for rand_window in random_windows:
                            App.daily_use[rand_window[0]:rand_window[1]] = np.full(np.diff(rand_window),total_power_value)
                        Us.load = Us.load + App.daily_use
                        continue
                    else: #otherwise, for "non-flat" apps it puts a mask on the newly defined windows and continues    
                        for rand_window in random_windows:
                            App.daily_use[rand_window[0]:rand_window[1]] = np.full(np.diff(rand_window),0.001)

                    App.daily_use_masked = np.zeros_like(ma.masked_not_equal(App.daily_use,0.001))
                    App.power = App.POWER[prof_i]

                    #random variability is applied to the total functioning time and to the duration of the duty cycles, if they have been specified
                    random_var_t = randomise_variables(App.r_t)
                    if App.activate == 1:
                        App.p_11 = App.P_11 * randomise_variables(App.Thermal_P_var) #randomly variates the power of thermal apps, otherwise variability is 0
                        App.p_12 = App.P_12 * randomise_variables(App.Thermal_P_var) #randomly variates the power of thermal apps, otherwise variability is 0
                        rand_r_c1 = randomise_variables(App.r_c1)
                        random_cycle1 = calc_random_cycle(App.t_11,App.p_11,App.t_12,App.p_12,rand_r_c1)
                        random_cycle3 = random_cycle2 = random_cycle1
                    elif App.activate == 2:
                        App.p_11 = App.P_11 * randomise_variables(App.Thermal_P_var)  # randomly variates the power of thermal apps, otherwise variability is 0
                        App.p_12 = App.P_12 * randomise_variables(App.Thermal_P_var)
                        App.p_21 = App.P_21 * randomise_variables(App.Thermal_P_var)
                        App.p_22 = App.P_22 * randomise_variables(App.Thermal_P_var)
                        rand_r_c1 = randomise_variables(App.r_c1)
                        rand_r_c2 = randomise_variables(App.r_c2)
                        random_cycle1 = calc_random_cycle(App.t_11, App.p_11, App.t_12, App.p_12, rand_r_c1) #randomise also the fixed cycle
                        random_cycle2 = calc_random_cycle(App.t_21, App.p_21, App.t_22, App.p_22, rand_r_c2)
                        random_cycle3 = random_cycle1
                    elif App.activate == 3:
                        App.p_11 = App.P_11 * randomise_variables(App.Thermal_P_var)  # randomly variates the power of thermal apps, otherwise variability is 0
                        App.p_12 = App.P_12 * randomise_variables(App.Thermal_P_var)
                        App.p_21 = App.P_21 * randomise_variables(App.Thermal_P_var)
                        App.p_22 = App.P_22 * randomise_variables(App.Thermal_P_var)
                        App.p_31 = App.P_31 * randomise_variables(App.Thermal_P_var)
                        App.p_32 = App.P_32 * randomise_variables(App.Thermal_P_var)
                        rand_r_c1 = randomise_variables(App.r_c1)
                        rand_r_c2 = randomise_variables(App.r_c2)
                        rand_r_c3 = randomise_variables(App.r_c3)
                        random_cycle1 = random.choice([calc_random_cycle(App.t_11, App.p_11, App.t_12, App.p_12, rand_r_c1),calc_random_cycle(App.t_12, App.p_12, App.t_11, App.p_11, rand_r_c1)]) # randomise also the fixed cycle
                        random_cycle2 = random.choice([calc_random_cycle(App.t_21, App.p_21, App.t_22, App.p_22, rand_r_c2),calc_random_cycle(App.t_22, App.p_22, App.t_21, App.p_21, rand_r_c2)]) # this is to avoid that all cycles are syncronous
                        random_cycle3 = random.choice([calc_random_cycle(App.t_31, App.p_31, App.t_32, App.p_32, rand_r_c3),calc_random_cycle(App.t_32, App.p_32, App.t_31, App.p_31, rand_r_c3)])
                    else:
                        random_cycle1 = random_cycle2 = random_cycle3 = None

                    rand_time = round(random.uniform(App.func_time,int(App.func_time*random_var_t)))
                    #control to check that the total randomised time of use does not exceed the total space available in the windows
                    if rand_time > 0.99*(np.diff(rand_window_1)+np.diff(rand_window_2)+np.diff(rand_window_3)):
                        rand_time = int(0.99*(np.diff(rand_window_1)+np.diff(rand_window_2)+np.diff(rand_window_3)))
                    max_free_spot = rand_time #free spots are used to detect if there's still space for switch_ons. Before calculating actual free spots, the max free spot is set equal to the entire randomised func_time

                    while tot_time <= rand_time:  # this is the key cycle, which runs for each App until the switch_ons and their duration equals the randomised total time of use of the App
                        switch_on = App.switch_on
                        if App.daily_use[switch_on] != 0.001: #control to check if the app is not already on at the randomly selected switch-on time
                            continue  # if the random switch_on falls somewhere where the App has been already turned on, tries the following conditions
                        if switch_on in range(rand_window_1[0], rand_window_1[1]):
                            indexes = App.calc_indexes_for_rand_switch_on(switch_on, rand_time, max_free_spot, rand_window_1)
                        elif switch_on in range(rand_window_2[0], rand_window_2[1]):
                            indexes = App.calc_indexes_for_rand_switch_on(switch_on, rand_time, max_free_spot,rand_window_2)
                        else:
                            indexes = App.calc_indexes_for_rand_switch_on(switch_on, rand_time, max_free_spot,rand_window_3)
                        if indexes is None:
                            continue
                        tot_time = tot_time + indexes.size  # the count of total time is updated with the size of the indexes array

                        if tot_time > rand_time:  # control to check when the total functioning time is reached. It will be typically overcome, so a correction is applied to avoid this
                            indexes_adj = indexes[:-(tot_time - rand_time)]  # corrects indexes size to avoid overcoming total time
                            coincidence = App.calc_coincident_switch_on(s_peak, mu_peak, op_factor, peak_time_range,index=indexes_adj)
                            App.daily_use_masked = App.calc_app_daily_use_masked(random_cycle1, random_cycle2,random_cycle3, coincidence, App.power,index=indexes_adj)
                            #tot_time = (tot_time - indexes.size) + indexes_adj.size  # updates the total time correcting the previous value
                            break  # exit cycle and go to next App
                        else:  # if the tot_time has not yet exceeded the App total functioning time, the cycle does the same without applying corrections to indexes size
                            coincidence = App.calc_coincident_switch_on(s_peak, mu_peak, op_factor, peak_time_range,index=indexes)
                            App.daily_use_masked = App.calc_app_daily_use_masked(random_cycle1, random_cycle2,random_cycle3, coincidence, App.power,index=indexes)
                            tot_time = tot_time  # no correction applied to previously calculated value

                        free_spots = []  # calculate how many free spots remain for further switch_ons
                        try:
                            free_spots.extend(j.stop - j.start for j in ma.notmasked_contiguous(App.daily_use_masked))

                        except TypeError:
                            free_spots = [0]
                        max_free_spot = max(free_spots)
                    Us.load = Us.load + App.daily_use  # adds the App profile to the User load
            Tot_Classes = Tot_Classes + Us.load #adds the User load to the total load of all User classes
        Profile.append(Tot_Classes) #appends the total load to the list that will contain all the generated profiles
        print('Profile',prof_i+1,'/',num_profiles,'completed') #screen update about progress of computation
    return(Profile)
