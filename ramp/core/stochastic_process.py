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

def randomise_variable(var):
    return random.uniform((1 - var), (1 + var))

def calc_random_cycle(time_1, power_1, time_2, power_2, r_c):
    return np.concatenate((np.ones(int(time_1 * r_c))* power_1, np.ones(int(time_2 * r_c))* power_2))

def randomise_cycle(App):
    App.p_11 = App.P_11 * randomise_variable(App.Thermal_P_var)  # randomly variates the power of thermal apps, otherwise variability is 0
    App.p_12 = App.P_12 * randomise_variable(App.Thermal_P_var)  # randomly variates the power of thermal apps, otherwise variability is 0
    rand_r_c1 = randomise_variable(App.r_c1)
    random_cycle1 = calc_random_cycle(App.t_11, App.p_11, App.t_12, App.p_12, rand_r_c1) # randomise also the fixed cycle
    random_cycle2 = random_cycle1
    random_cycle3 = random_cycle1
    if App.activate >= 2:
        App.p_21 = App.P_21 * randomise_variable(App.Thermal_P_var) #randomly variates the power of thermal apps, otherwise variability is 0
        App.p_22 = App.P_22 * randomise_variable(App.Thermal_P_var)
        rand_r_c2 = randomise_variable(App.r_c2)
        random_cycle2 = calc_random_cycle(App.t_21, App.p_21, App.t_22, App.p_22, rand_r_c2) #randomise also the fixed cycle
        if App.activate == 3:
            App.p_31 = App.P_31 * randomise_variable(App.Thermal_P_var)
            App.p_32 = App.P_32 * randomise_variable(App.Thermal_P_var)
            rand_r_c3 = randomise_variable(App.r_c3)
            random_cycle1 = random.choice([calc_random_cycle(App.t_11, App.p_11, App.t_12, App.p_12, rand_r_c1),calc_random_cycle(App.t_12, App.p_12, App.t_11, App.p_11,rand_r_c1)])  # randomise also the fixed cycle
            random_cycle2 = random.choice([calc_random_cycle(App.t_21, App.p_21, App.t_22, App.p_22, rand_r_c2),calc_random_cycle(App.t_22, App.p_22, App.t_21, App.p_21,rand_r_c2)])  # this is to avoid that all cycles are syncronous
            random_cycle3 = random.choice([calc_random_cycle(App.t_31, App.p_31, App.t_32, App.p_32, rand_r_c3),calc_random_cycle(App.t_32, App.p_32, App.t_31, App.p_31, rand_r_c3)])
    return random_cycle1, random_cycle2, random_cycle3

def get_single_user_profile(prof_i, Us, peak_time_range, Year_behaviour, rand_daily_pref):
    for App in Us.App_list:  # iterates for all the App types in the given User class
        App.daily_use = np.zeros(1440)
        if (random.uniform(0, 1) > App.occasional_use  # evaluates if occasional use happens or not
                or (App.Pref_index != 0 and rand_daily_pref != App.Pref_index)  # evaluates if daily preference coincides with the randomised daily preference number
                or App.wd_we not in [Year_behaviour[prof_i],2]):  # checks if the app is allowed in the given yearly behaviour pattern
            continue
        # recalculate windows start and ending times randomly, based on the inputs
        rand_window_1 = App.calc_rand_window(window_num=1)
        rand_window_2 = App.calc_rand_window(window_num=2)
        rand_window_3 = App.calc_rand_window(window_num=3)

        # redefines functioning windows based on the previous randomisation of the boundaries
        random_windows = [rand_window_1, rand_window_2, rand_window_3]
        if App.flat == 'yes':  # if the app is "flat" the code stops right after filling the newly created windows without applying any further stochasticity
            total_power_value = App.POWER[prof_i] * App.number
            for rand_window in random_windows:
                App.daily_use[rand_window[0]:rand_window[1]] = np.full(np.diff(rand_window), total_power_value)
            Us.load = Us.load + App.daily_use
            continue
        else:  # otherwise, for "non-flat" apps it puts a mask on the newly defined windows and continues
            for rand_window in random_windows:
                App.daily_use[rand_window[0]:rand_window[1]] = np.full(np.diff(rand_window), 0.001)

        App.daily_use_masked = np.zeros_like(ma.masked_not_equal(App.daily_use, 0.001))
        App.power = App.POWER[prof_i]

        # random variability is applied to the total functioning time and to the duration of the duty cycles, if they have been specified
        random_var_t = randomise_variable(App.r_t)
        rand_time = round(random.uniform(App.func_time, int(App.func_time * random_var_t)))
        random_cycle1 = []
        random_cycle2 = []
        random_cycle3 = []
        if 1 <= App.activate <= 3:
            random_cycle1, random_cycle2, random_cycle3 = randomise_cycle(App)

        # control to check that the total randomised time of use does not exceed the total space available in the windows
        if rand_time > 0.99 * (np.diff(rand_window_1) + np.diff(rand_window_2) + np.diff(rand_window_3)):
            rand_time = int(0.99 * (np.diff(rand_window_1) + np.diff(rand_window_2) + np.diff(rand_window_3)))
        max_free_spot = rand_time  # free spots are used to detect if there's still space for switch_ons. Before calculating actual free spots, the max free spot is set equal to the entire randomised func_time
        App.daily_use = App.get_app_profile(rand_time, rand_window_1, rand_window_2, rand_window_3, max_free_spot,
                                            peak_time_range, random_cycle1, random_cycle2, random_cycle3,
                                            power=App.power)
        Us.load = Us.load + App.daily_use  # adds the App profile to the User load
    return Us.load

def generate_user_load(prof_i, Us, peak_time_range, Year_behaviour):
    Us.load = np.zeros(1440)  # initialise empty load for User instance
    for _ in range(Us.num_users):  # iterates for every single user within a User class. Each single user has its own separate randomisation
        rand_daily_pref = 0 if Us.user_preference == 0 else random.randint(1, Us.user_preference)
        Us.load = get_single_user_profile(prof_i, Us, peak_time_range, Year_behaviour, rand_daily_pref)
    return Us.load

#generate_daily_profile
def generate_profile (prof_i ,User_list, peak_time_range, Year_behaviour):
    Tot_Classes = np.zeros(1440) #initialise an empty daily profile that will be filled with the sum of the hourly profiles of each User instance
    for Us in User_list: #iterates for each User instance (i.e. for each user class)
        Us.load=generate_user_load(prof_i, Us, peak_time_range, Year_behaviour)
        Tot_Classes = Tot_Classes + Us.load #adds the User load to the total load of all User classes
    return Tot_Classes

def Stochastic_Process(j):
    Profile, num_profiles = Initialise_model()
    User_list = user_defined_inputs(j)
    peak_enlarge, mu_peak, s_peak, op_factor, Year_behaviour = Initialise_inputs()
    peak_time_range = calc_peak_time_range(User_list, peak_enlarge)

    '''
    The core stochastic process starts here. For each profile requested by the software user, 
    each Appliance instance within each User instance is separately and stochastically generated
    '''
    for prof_i in range(num_profiles):  # the whole code is repeated for each profile that needs to be generated
        Tot_Classes = generate_profile(prof_i ,User_list, peak_time_range, Year_behaviour)
        Profile.append(Tot_Classes)  # appends the total load to the list that will contain all the generated profiles
        print('Profile', prof_i + 1, '/', num_profiles, 'completed')  # screen update about progress of computation
    return Profile

