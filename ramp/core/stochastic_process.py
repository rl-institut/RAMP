# -*- coding: utf-8 -*-
#%% Import required libraries
import numpy as np
import numpy.ma as ma
import random 
import math
from ramp.core.initialise import Initialise_model, Initialise_inputs, user_defined_inputs

def calc_peak_time_range(user_list, peak_enlarge):
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
    rand_peak_enlarge = round(math.fabs(peak_time - random.gauss(peak_time, peak_enlarge * peak_time)))
    return np.arange(peak_time - rand_peak_enlarge , peak_time + rand_peak_enlarge)  # the peak_time is randomly enlarged based on the calibration parameter peak_enlarge

def randomise_variable(var, norm=1):
    return norm * random.uniform((1 - var), (1 + var))

def calc_random_cycle(time_1, power_1, time_2, power_2, r_c):
    return np.concatenate((np.ones(int(time_1 * r_c))* power_1, np.ones(int(time_2 * r_c))* power_2))

def randomise_cycle(App):
    """
    calculates the new randomised cycles taking the random variability in the duty cycle duration
    """
    App.p_11 = randomise_variable(var=App.Thermal_P_var, norm=App.P_11) # randomly variates the power of thermal apps, otherwise variability is 0
    App.p_12 = randomise_variable(var=App.Thermal_P_var, norm=App.P_12)
    rand_r_c1 = randomise_variable(var=App.r_c1)
    random_cycle1 = calc_random_cycle(App.t_11, App.p_11, App.t_12, App.p_12, rand_r_c1) # randomise also the fixed cycle
    random_cycle2 = random_cycle1
    random_cycle3 = random_cycle1
    if App.activate >= 2:
        App.p_21 = randomise_variable(var=App.Thermal_P_var, norm=App.P_21)  # randomly variates the power of thermal apps, otherwise variability is 0
        App.p_22 = randomise_variable(var=App.Thermal_P_var, norm=App.P_22)
        rand_r_c2 = randomise_variable(var=App.r_c2)
        random_cycle2 = calc_random_cycle(App.t_21, App.p_21, App.t_22, App.p_22, rand_r_c2) #randomise also the fixed cycle
        if App.activate == 3:
            App.p_31 = randomise_variable(var=App.Thermal_P_var, norm=App.P_31)  # randomly variates the power of thermal apps, otherwise variability is 0
            App.p_32 = randomise_variable(var=App.Thermal_P_var, norm=App.P_32)
            rand_r_c3 = randomise_variable(var=App.r_c3)
            random_cycle1 = random.choice([calc_random_cycle(App.t_11, App.p_11, App.t_12, App.p_12, rand_r_c1),calc_random_cycle(App.t_12, App.p_12, App.t_11, App.p_11,rand_r_c1)])  # randomise also the fixed cycle
            random_cycle2 = random.choice([calc_random_cycle(App.t_21, App.p_21, App.t_22, App.p_22, rand_r_c2),calc_random_cycle(App.t_22, App.p_22, App.t_21, App.p_21,rand_r_c2)])  # this is to avoid that all cycles are syncronous
            random_cycle3 = random.choice([calc_random_cycle(App.t_31, App.p_31, App.t_32, App.p_32, rand_r_c3),calc_random_cycle(App.t_32, App.p_32, App.t_31, App.p_31, rand_r_c3)])
    return random_cycle1, random_cycle2, random_cycle3


def generate_profile (prof_i ,User_list, peak_time_range, Year_behaviour):
    """
    generates a single load profile taking all the user types into consideration
    """

    Tot_Classes = np.zeros(1440) #initialise an empty daily profile that will be filled with the sum of the hourly profiles of each User instance
    for Us in User_list: #iterates for each User instance (i.e. for each user class)
        Us.load=Us.generate_user_load(prof_i, peak_time_range, Year_behaviour)
        Tot_Classes = Tot_Classes + Us.load #adds the User load to the total load of all User classes
    return Tot_Classes

def Stochastic_Process(j):
    """
    Generates a stochastic load profile fo each profile requested by the software user taking each appliance instance assosiated with every user
    """
    Profile, num_profiles = Initialise_model()
    User_list = user_defined_inputs(j)
    peak_enlarge, mu_peak, s_peak, op_factor, Year_behaviour = Initialise_inputs()
    peak_time_range = calc_peak_time_range(User_list, peak_enlarge)


    for prof_i in range(num_profiles):  # the whole code is repeated for each profile that needs to be generated
        Tot_Classes = generate_profile(prof_i ,User_list, peak_time_range, Year_behaviour)
        Profile.append(Tot_Classes)  # appends the total load to the list that will contain all the generated profiles
        print('Profile', prof_i + 1, '/', num_profiles, 'completed')  # screen update about progress of computation
    return Profile

