# -*- coding: utf-8 -*-

#%% Import required libraries
import numpy as np
import numpy.ma as ma
import pandas as pd
import random
import math

#%% Definition of Python classes that constitute the model architecture
'''
The code is based on two concatenated python classes, namely 'User' and
'Appliance', which are used to define at the outer level the User classes and 
at the inner level all the available appliances within each user class, with 
their own characteristics. Within the Appliance class, some other functions are
created to define windows of use and, if needed, specific duty cycles
'''

#Define the outer python class that represents 'User classes'
class User():
    
    def __init__(self, name = "", n_users = 1, us_pref = 0):
        self.user_name = name
        self.num_users = n_users #specifies the number of users within the class
        self.user_preference = us_pref #allows to check if random number coincides with user preference, to distinguish between various appliance_use options (e.g. different cooking options)
        self.App_list = [] #each instance of User (i.e. each user class) has its own list of Appliances

    def add_appliance(self, *args, **kwargs):
        return Appliance(*args, **kwargs)

    @property
    def windows_curve(self):
        windows_curve = np.zeros(1440)
        for App in self.App_list:
            # Calculate windows curve, i.e. the theoretical maximum curve that can be obtained, for each app, by switching-on always all the 'n' apps altogether in any time-step of the functioning windows
            single_wcurve = App.single_wcurve  # this computes the curve for the specific App
            windows_curve = np.vstack([windows_curve, single_wcurve])  # this stacks the specific App curve in an overall curve comprising all the Apps within a User class
        return np.transpose(np.sum(windows_curve, axis=0)) * self.num_users

"""
The Appliance is added the the App_list whenever its window method is called
Applicance does not need to be part of User class ..., this simplifies a bit the 

This could also be achieved by add a method add_appliance to User class
"""

#Define the inner class for modelling user's appliances within the correspoding user class
class Appliance():

    def __init__(self, user, number = 1, power = 0, num_windows = 1, func_time = 0, time_fraction_random_variability = 0, func_cycle = 1, fixed = 'no', fixed_cycle = 0, occasional_use = 1, flat = 'no', thermal_P_var = 0, pref_index = 0, wd_we_type = 2, P_series = False):
        self.user = user #user to which the appliance is bounded
        self.number = number #number of appliances of the specified kind
        self.num_windows = num_windows #number of functioning windows to be considered
        self.func_time = func_time #total time the appliance is on during the day
        self.r_t = time_fraction_random_variability #percentage of total time of use that is subject to random variability
        self.func_cycle = func_cycle #minimum time the appliance is kept on after switch-on event
        self.fixed = fixed #if 'yes', all the 'n' appliances of this kind are always switched-on together
        self.activate = fixed_cycle #if equal to 1,2 or 3, respectively 1,2 or 3 duty cycles can be modelled, for different periods of the day
        self.occasional_use = occasional_use #probability that the appliance is always (i.e. everyday) included in the mix of appliances that the user actually switches-on during the day
        self.flat = flat #allows to model appliances that are not subject to any kind of random variability, such as public lighting
        self.Thermal_P_var = thermal_P_var #allows to randomly variate the App power within a range
        self.Pref_index = pref_index #defines preference index for association with random User daily preference behaviour
        self.wd_we = wd_we_type #defines if the App is associated with weekdays or weekends | 0 is wd 1 is we 2 is all week
        if P_series == False and not isinstance(power, pd.DataFrame): #check if the user defined P as timeseries
            self.POWER = power*np.ones(365) #treat the power as single value for the entire year
        else:
            self.POWER = power.values[:,0] #if a timeseries is given the power is treated as so

    def windows(self, w1 = np.array([0,0]), w2 = np.array([0,0]),r_w = 0, w3 = np.array([0,0])):
        self.random_var_w = r_w  # percentage of variability in the start and ending times of the windows
        self.daily_use = np.zeros(1440)  # create an empty daily use profile
        for count, window_x in enumerate((w1,w2,w3),start=1):
            self.__setattr__(f'window_{count}',window_x) #array of start and ending time for window of use #1,2 & 3
            self.daily_use[window_x[0]:window_x[1]] = np.full(np.diff(window_x), 0.001) #fills the daily use profile with infinitesimal values that are just used to identify the functioning windows
            self.__setattr__(f'random_var_{count}', int(r_w*np.diff(window_x))) #calculate the random variability of the windows, i.e. the maximum range of time they can be enlarged or shortened
        self.daily_use_masked = np.zeros_like(ma.masked_not_equal(self.daily_use,0.001)) #apply a python mask to the daily_use array to make only functioning windows 'visibile'
        self.user.App_list.append(self) #automatically appends the appliance to the user's appliance list

        if self.activate == 1:
            self.cw11 = self.window_1
            self.cw12 = self.window_2

    def calc_rand_window(self, window_num=1, max_window_range=np.array([0, 1440])):
        _window = self.__getattribute__(f'window_{window_num}')
        _random_var = self.__getattribute__(f'random_var_{window_num}')
        rand_window = np.array(
            [
                int(random.uniform(_window[0] - _random_var, _window[0] + _random_var)),
                int(random.uniform(_window[1] - _random_var, _window[1] + _random_var))
            ]
        )
        if rand_window[0] < max_window_range[0]:
            rand_window[0] = max_window_range[0]
        if rand_window[1] > max_window_range[1]:
            rand_window[1] = max_window_range[1]

        return rand_window

    def switch_on(self):
        rand_window_1 = self.calc_rand_window(window_num=1)
        rand_window_2 = self.calc_rand_window(window_num=2)
        rand_window_3 = self.calc_rand_window(window_num=3)

        # check how many windows to consider
        if self.num_windows == 1:
            return int(random.choice([random.uniform(rand_window_1[0], (rand_window_1[1]))]))

        elif self.num_windows == 2:
            return int(random.choice(np.concatenate((np.arange(rand_window_1[0], rand_window_1[1]), np.arange(rand_window_2[0], rand_window_2[1])), axis=0)))

        else:
            return int(random.choice(np.concatenate((np.arange(rand_window_1[0], rand_window_1[1]), np.arange(rand_window_2[0], rand_window_2[1]), np.arange(rand_window_3[0], rand_window_3[1]),), axis=0)))

    def calc_indexes_for_rand_switch_on(self, switch_on, rand_time, max_free_spot, rand_window):
        if np.any(self.daily_use[switch_on:rand_window[1]] != 0.001):  # control to check if there are any other switch on times after the current one
            next_switch = [switch_on + k[0] for k in np.where(self.daily_use[switch_on:] != 0.001)]  # identifies the position of next switch on time and sets it as a limit for the duration of the current switch on
            if (next_switch[0] - switch_on) >= self.func_cycle and max_free_spot >= self.func_cycle:
                upper_limit = min((next_switch[0] - switch_on), min(rand_time, rand_window[1] - switch_on))
            elif (next_switch[0] - switch_on) < self.func_cycle and max_free_spot >= self.func_cycle:  # if next switch_on event does not allow for a minimum functioning cycle without overlapping, but there are other larger free spots, the cycle tries again from the beginning
                return None
            else:
                upper_limit = next_switch[0] - switch_on  # if there are no other options to reach the total time of use, empty spaces are filled without minimum cycle restrictions until reaching the limit
        else:
            upper_limit = min(rand_time, rand_window[1] - switch_on)  # if there are no other switch-on events after the current one, the upper duration limit is set this way
        return np.arange(switch_on, switch_on + (int(random.uniform(self.func_cycle, upper_limit)))) if upper_limit >= self.func_cycle \
            else np.arange(switch_on, switch_on + upper_limit)

    def calc_coincident_switch_on(self, s_peak, mu_peak, op_factor, peak_time_range, index):
        if np.in1d(peak_time_range,index).any() and self.fixed == 'no':  # check if indexes are in peak window and if the coincident behaviour is locked by the "fixed" attribute
            return min(self.number, max(1, math.ceil(random.gauss((self.number * mu_peak + 0.5), (s_peak * self.number * mu_peak)))))# calculates coincident behaviour within the peak time range
        elif np.in1d(peak_time_range,index).any() == False and self.fixed == 'no':  # check if indexes are off-peak and if coincident behaviour is locked or not
            Prob = random.uniform(0, (self.number - op_factor) / self.number)  # calculates probability of coincident switch_ons off-peak
            array = np.arange(0, self.number) / self.number
            try:
                on_number = np.max(np.where(Prob >= array)) + 1
            except ValueError:
                on_number = 1
            return on_number #randomly selects how many apps are on at the same time for each app type based on the above probabilistic algorithm
        else:
            return self.number #this is the case when App.fixed is activated. All 'n' apps of an App instance are switched_on altogether

    def calc_app_daily_use_masked(self, random_cycle1, random_cycle2, random_cycle3, coincidence, power, index):
        if self.activate > 0:  # evaluates if the app has some duty cycles to be considered
            evaluate = np.round(np.mean(index)) if index.size > 0 else 0

            if evaluate in range(self.cw11[0], self.cw11[1]) or evaluate in range(self.cw12[0], self.cw12[1]):
                np.put(self.daily_use, index, (random_cycle1 * coincidence))
                np.put(self.daily_use_masked, index, (random_cycle1 * coincidence), mode='clip')
            elif evaluate in range(self.cw21[0], self.cw21[1]) or evaluate in range(self.cw22[0], self.cw22[1]):
                np.put(self.daily_use, index, (random_cycle2 * coincidence))
                np.put(self.daily_use_masked, index, (random_cycle2 * coincidence), mode='clip')
            else:
                np.put(self.daily_use, index, (random_cycle3 * coincidence))
                np.put(self.daily_use_masked, index, (random_cycle3 * coincidence), mode='clip')
        else:  # if no duty cycles are specified, a regular switch_on event is modelled
            np.put(self.daily_use, index, (power * (random.uniform((1 - self.Thermal_P_var), (1 + self.Thermal_P_var))) * coincidence))  # randomises also the App Power if Thermal_P_var is on
            np.put(self.daily_use_masked, index,(power * (random.uniform((1 - self.Thermal_P_var), (1 + self.Thermal_P_var))) * coincidence),
                   mode='clip')
        return np.zeros_like(ma.masked_greater_equal(self.daily_use_masked,0.001))  # updates the mask excluding the current switch_on event to identify the free_spots for the next iteration

    @property
    def single_wcurve(self):
        return self.daily_use * np.mean(self.POWER) * self.number

        #if needed, specific duty cycles can be defined for each Appliance, for a maximum of three different ones
    def specific_cycle_1(self, p_1 = 0, t_1 = 0, p_2 = 0, t_2 = 0, r_c = 0):
        self.P_11 = p_1 #power absorbed during first part of the duty cycle
        self.t_11 = t_1 #duration of first part of the duty cycle
        self.P_12 = p_2 #power absorbed during second part of the duty cycle
        self.t_12 = t_2 #duration of second part of the duty cycle
        self.r_c1 = r_c #random variability of duty cycle segments duration
        self.fixed_cycle1 = np.concatenate(((np.ones(t_1)*p_1),(np.ones(t_2)*p_2))) #create numpy array representing the duty cycle

    def specific_cycle_2(self, P_21 = 0, t_21 = 0, P_22 = 0, t_22 = 0, r_c2 = 0):
        self.P_21 = P_21 #same as for cycle1
        self.t_21 = t_21
        self.P_22 = P_22
        self.t_22 = t_22
        self.r_c2 = r_c2
        self.fixed_cycle2 = np.concatenate(((np.ones(t_21)*P_21),(np.ones(t_22)*P_22)))

    def specific_cycle_3(self, P_31 = 0, t_31 = 0, P_32 = 0, t_32 = 0, r_c3 = 0):
        self.P_31 = P_31 #same as for cycle1
        self.t_31 = t_31
        self.P_32 = P_32
        self.t_32 = t_32
        self.r_c3 = r_c3
        self.fixed_cycle3 = np.concatenate(((np.ones(t_31)*P_31),(np.ones(t_32)*P_32)))

    #different time windows can be associated with different specific duty cycles
    def cycle_behaviour(self, cw11 = np.array([0,0]), cw12 = np.array([0,0]), cw21 = np.array([0,0]), cw22 = np.array([0,0]), cw31 = np.array([0,0]), cw32 = np.array([0,0])):
        self.cw11 = cw11 #first window associated with cycle1
        self.cw12 = cw12 #second window associated with cycle1
        self.cw21 = cw21 #same for cycle2
        self.cw22 = cw22
        self.cw31 = cw31 #same for cycle 3
        self.cw32 = cw32

