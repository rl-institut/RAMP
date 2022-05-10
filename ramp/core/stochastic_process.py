# -*- coding: utf-8 -*-
#%% Import required libraries
import numpy as np
import random 
import math
from ramp.core.initialise import Initialise_model, Initialise_inputs, user_defined_inputs

def calc_peak_time_range(user_list, peak_enlarge):
    """
    Calculate the peak time range, which is used to discriminate between off-peak and on-peak coincident switch-on probability
    Calculate first the overall Peak Window (taking into account all User classes).
    The peak window is just a time window in which coincident switch-on of multiple appliances assumes a higher probability than off-peak
    Within the peak window, a random peak time is calculated and then enlarged into a peak_time_range following again a random procedure

    Parameters
    ----------
    user_list: list
        list containing all the user types
    peak_enlarge: float
        percentage random enlargement or reduction of peak time range length

    Returns
    -------
    peak time range: numpy array
    """

    Tot_curve = np.zeros(1440)  # creates an empty daily profile
    for Us in user_list:
        Tot_curve = Tot_curve + Us.windows_curve  # adds the User's theoretical max profile to the total theoretical max comprising all classes
    peak_window = np.transpose(np.argwhere(Tot_curve == np.amax(Tot_curve)))  # Find the peak window within the theoretical max profile
    peak_time = round(random.normalvariate(round(np.average(peak_window)), 1 / 3 * (peak_window[0, -1] - peak_window[0, 0])))  # Within the peak_window, randomly calculate the peak_time using a gaussian distribution
    rand_peak_enlarge = round(math.fabs(peak_time - random.gauss(peak_time, peak_enlarge * peak_time)))
    return np.arange(peak_time - rand_peak_enlarge , peak_time + rand_peak_enlarge)  # the peak_time is randomly enlarged based on the calibration parameter peak_enlarge

def randomise_variable(var, norm=1):
    """
    create a random variable with a uniform distribution using given variable

    Parameters
    ----------
    var: float
        variable that is to be randomised
    norm: float
        multiplication factor for randomisation, default = 1

    Returns
    -------
    randomised value of the input variable
    """
    return norm * random.uniform((1 - var), (1 + var))

def calc_random_cycle(time_1, power_1, time_2, power_2, r_c):
    """
    randomise the fixed cycle related to an appliance

    Parameters
    ----------
    time_1: int
        duration of first part of the duty cycle
    power_1: float
        power absorbed during first part of the duty cycle
    time_2: int
        duration of second part of the duty cycle
    power_2: float
        power absorbed during second part of the duty cycle
    r_c: float
        random variability of duty cycle segments duration

    Returns
    -------
    random cycle: numpy array
        specific cycles that are randomised with the random variability fed as input
    """
    return np.concatenate((np.ones(int(time_1 * r_c)) * power_1, np.ones(int(time_2 * r_c)) * power_2))


def generate_profile (prof_i ,User_list, peak_time_range, Year_behaviour):
    """
    generates a single load profile taking all the user types into consideration

    Parameters
    ----------
    prof_i: int
        ith profile requested by the user
    User_list: list
       list containing all the user types
    peak_time_range: numpy array
        randomised peak time range calculated using calc_peak_time_range function
    Year_behaviour: numpy array
        array consisting of a yearly pattern of weekends and weekdays

    Returns
    -------
    Tot_Classes: numpy array
        a single array consisting of the sum of profiles of each User instance
    """

    Tot_Classes = np.zeros(1440) #initialise an empty daily profile that will be filled with the sum of the hourly profiles of each User instance
    for Us in User_list: #iterates for each User instance (i.e. for each user class)
        Us.load=Us.generate_user_load(prof_i, peak_time_range, Year_behaviour)
        Tot_Classes = Tot_Classes + Us.load #adds the User load to the total load of all User classes
    return Tot_Classes

def Stochastic_Process(j):
    """
    Generate a stochastic load profile for each profile requested by the software user taking each appliance instance assosiated with every user

    Parameters
    ----------
    j: int
        input file number

    Returns
    -------
    Profile: numpy array
        total number of stochastically randomised profiles requested by the user
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

