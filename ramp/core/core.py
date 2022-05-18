# -*- coding: utf-8 -*-

#%% Import required libraries
import numpy as np
import numpy.ma as ma
import pandas as pd
import random
import math
from .initialise import calibration_parameters
from .stochastic_process import randomise_variable, calc_random_cycle

#%% Definition of Python classes that constitute the model architecture
'''
The code is based on two concatenated python classes, namely 'User' and
'Appliance', which are used to define the User classes and all the available 
appliances within each user class, with their own characteristics. 
Within the Appliance class, some other functions are
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

    def get_single_user_profile(self, prof_i, peak_time_range, Year_behaviour, rand_daily_pref):
        """
        generates a single load profile for a single user taking all the appliances associated with the particular user
        into consideration
        """
        for App in self.App_list:  # iterates for all the App types in the given User class
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
                self.load = self.load + App.daily_use
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
                random_cycle1, random_cycle2, random_cycle3 = App.randomise_cycle(App)

            # control to check that the total randomised time of use does not exceed the total space available in the windows
            if rand_time > 0.99 * (np.diff(rand_window_1) + np.diff(rand_window_2) + np.diff(rand_window_3)):
                rand_time = int(0.99 * (np.diff(rand_window_1) + np.diff(rand_window_2) + np.diff(rand_window_3)))
            max_free_spot = rand_time  # free spots are used to detect if there's still space for switch_ons. Before calculating actual free spots, the max free spot is set equal to the entire randomised func_time
            App.daily_use = App.get_app_profile(rand_time, rand_window_1, rand_window_2, rand_window_3, max_free_spot,
                                                peak_time_range, random_cycle1, random_cycle2, random_cycle3,power=App.power)
            self.load = self.load + App.daily_use  # adds the App profile to the User load
        return self.load

    def generate_user_load(self, prof_i, peak_time_range, Year_behaviour):
        """
        generates an aggregate load profile for every single user within a user class

        Parameters
        ----------
        prof_i: int
            ith profile requested by the user
        peak_time_range: numpy array
            randomised peak time range calculated using calc_peak_time_range function
        Year_behaviour: numpy array
            array consisting of a yearly pattern of weekends and weekdays peak_time_range

        Returns
        -------
        User.load : numpy array
            the aggregate load profile of all the users within a user class
        """

        self.load = np.zeros(1440)  # initialise empty load for User instance
        for _ in range(self.num_users):  # iterates for every single user within a User class. Each single user has its own separate randomisation
            rand_daily_pref = 0 if self.user_preference == 0 else random.randint(1, self.user_preference)
            self.load = self.get_single_user_profile(prof_i, peak_time_range, Year_behaviour, rand_daily_pref)
        return self.load

"""
The Appliance is added the the App_list whenever its window method is called
Applicance does not need to be part of User class ..., this simplifies a bit the 

This could also be achieved by add a method add_appliance to User class
"""

#Define a class for modelling user's appliances
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

    def randomise_cycle(self, App):
        """
        calculates the new randomised cycles taking the random variability in the duty cycle duration
        """
        self.p_11 = randomise_variable(var=self.Thermal_P_var,norm=App.P_11)  # randomly variates the power of thermal apps, otherwise variability is 0
        self.p_12 = randomise_variable(var=self.Thermal_P_var, norm=App.P_12)
        rand_r_c1 = randomise_variable(var=App.r_c1)
        random_cycle1 = calc_random_cycle(self.t_11, self.p_11, self.t_12, self.p_12,rand_r_c1)  # randomise also the fixed cycle
        random_cycle2 = random_cycle1
        random_cycle3 = random_cycle1
        if self.activate >= 2:
            self.p_21 = randomise_variable(var=self.Thermal_P_var, norm=App.P_21)  # randomly variates the power of thermal apps, otherwise variability is 0
            self.p_22 = randomise_variable(var=self.Thermal_P_var, norm=App.P_22)
            rand_r_c2 = randomise_variable(var=App.r_c2)
            random_cycle2 = calc_random_cycle(self.t_21, self.p_21, self.t_22, self.p_22,rand_r_c2)  # randomise also the fixed cycle
            if self.activate == 3:
                self.p_31 = randomise_variable(var=self.Thermal_P_var, norm=App.P_31)  # randomly variates the power of thermal apps, otherwise variability is 0
                self.p_32 = randomise_variable(var=self.Thermal_P_var, norm=App.P_32)
                rand_r_c3 = randomise_variable(var=App.r_c3)
                random_cycle1 = random.choice([calc_random_cycle(self.t_11, self.p_11, self.t_12, self.p_12, rand_r_c1),
                                               calc_random_cycle(self.t_12, self.p_12, self.t_11, self.p_11,rand_r_c1)])  # randomise also the fixed cycle
                random_cycle2 = random.choice([calc_random_cycle(self.t_21, self.p_21, self.t_22, self.p_22, rand_r_c2),
                                               calc_random_cycle(self.t_22, self.p_22, self.t_21, self.p_21,rand_r_c2)])  # this is to avoid that all cycles are syncronous
                random_cycle3 = random.choice([calc_random_cycle(self.t_31, self.p_31, self.t_32, self.p_32, rand_r_c3),
                                               calc_random_cycle(self.t_32, self.p_32, self.t_31, self.p_31, rand_r_c3)])
        return random_cycle1, random_cycle2, random_cycle3

    @property
    def single_wcurve(self):
        return self.daily_use * np.mean(self.POWER) * self.number

    def calc_rand_window(self, window_num=1, max_window_range=np.array([0, 1440])):
        _window = self.__getattribute__(f'window_{window_num}')
        _random_var = self.__getattribute__(f'random_var_{window_num}')
        rand_window = np.array([int(random.uniform(_window[0] - _random_var, _window[0] + _random_var)),
                                int(random.uniform(_window[1] - _random_var, _window[1] + _random_var))])
        if rand_window[0] < max_window_range[0]:
            rand_window[0] = max_window_range[0]
        if rand_window[1] > max_window_range[1]:
            rand_window[1] = max_window_range[1]

        return rand_window

    # if needed, specific duty cycles can be defined for each Appliance, for a maximum of three different ones
    def specific_cycle_1(self, p_1=0, t_1=0, p_2=0, t_2=0, r_c=0):
        self.P_11 = p_1  # power absorbed during first part of the duty cycle
        self.t_11 = t_1  # duration of first part of the duty cycle
        self.P_12 = p_2  # power absorbed during second part of the duty cycle
        self.t_12 = t_2  # duration of second part of the duty cycle
        self.r_c1 = r_c  # random variability of duty cycle segments duration
        self.fixed_cycle1 = np.concatenate(((np.ones(t_1) * p_1), (np.ones(t_2) * p_2)))  # create numpy array representing the duty cycle

    def specific_cycle_2(self, P_21=0, t_21=0, P_22=0, t_22=0, r_c2=0):
        self.P_21 = P_21  # same as for cycle1
        self.t_21 = t_21
        self.P_22 = P_22
        self.t_22 = t_22
        self.r_c2 = r_c2
        self.fixed_cycle2 = np.concatenate(((np.ones(t_21) * P_21), (np.ones(t_22) * P_22)))

    def specific_cycle_3(self, P_31=0, t_31=0, P_32=0, t_32=0, r_c3=0):
        self.P_31 = P_31  # same as for cycle1
        self.t_31 = t_31
        self.P_32 = P_32
        self.t_32 = t_32
        self.r_c3 = r_c3
        self.fixed_cycle3 = np.concatenate(((np.ones(t_31) * P_31), (np.ones(t_32) * P_32)))

    # different time windows can be associated with different specific duty cycles
    def cycle_behaviour(self, cw11=np.array([0, 0]), cw12=np.array([0, 0]), cw21=np.array([0, 0]),
                        cw22=np.array([0, 0]), cw31=np.array([0, 0]), cw32=np.array([0, 0])):
        self.cw11 = cw11  # first window associated with cycle1
        self.cw12 = cw12  # second window associated with cycle1
        self.cw21 = cw21  # same for cycle2
        self.cw22 = cw22
        self.cw31 = cw31  # same for cycle 3
        self.cw32 = cw32

    @property
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

    def update_daily_use(self, random_cycle1, random_cycle2, random_cycle3, coincidence, power, index):
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
            np.put(self.daily_use_masked, index,(power * (random.uniform((1 - self.Thermal_P_var), (1 + self.Thermal_P_var))) * coincidence),mode='clip')
        self.daily_use_masked = np.zeros_like(ma.masked_greater_equal(self.daily_use_masked,0.001))  # updates the mask excluding the current switch_on event to identify the free_spots for the next iteration
        return self.daily_use, self.daily_use_masked

    def get_app_profile(self, rand_time, rand_window_1, rand_window_2, rand_window_3, max_free_spot,peak_time_range, random_cycle1, random_cycle2, random_cycle3, power):

        _, mu_peak, s_peak, op_factor = calibration_parameters()
        tot_time = 0
        while tot_time <= rand_time:  # this is the key cycle, which runs for each App until the switch_ons and their duration equals the randomised total time of use of the App
            switch_on = self.switch_on
            if self.daily_use[switch_on] != 0.001:
                continue  # if the random switch_on falls somewhere where the App has been already turned on, tries again from beginning of the while cycle
            if switch_on in range(rand_window_1[0], rand_window_1[1]):
                indexes = self.calc_indexes_for_rand_switch_on(switch_on=switch_on, rand_time=rand_time,max_free_spot=max_free_spot, rand_window=rand_window_1)
            elif switch_on in range(rand_window_2[0], rand_window_2[1]):
                indexes = self.calc_indexes_for_rand_switch_on(switch_on, rand_time, max_free_spot, rand_window_2)
            else:
                indexes = self.calc_indexes_for_rand_switch_on(switch_on, rand_time, max_free_spot, rand_window_3)
            if indexes is None:
                continue
            tot_time += indexes.size  # the count of total time is updated with the size of the indexes array
            if tot_time > rand_time:  # control to check when the total functioning time is reached. It will be typically overcome, so a correction is applied to avoid this
                indexes_adj = indexes[:-(tot_time - rand_time)]  # corrects indexes size to avoid overcoming total time
                coincidence = self.calc_coincident_switch_on(s_peak, mu_peak, op_factor, peak_time_range,index=indexes_adj)
                self.daily_use, self.daily_use_masked = self.update_daily_use(random_cycle1, random_cycle2, random_cycle3,coincidence, power, index=indexes_adj)
                break  # exit cycle and go to next App
            else:  # if the tot_time has not yet exceeded the App total functioning time, the cycle does the same without applying corrections to indexes size
                coincidence = self.calc_coincident_switch_on(s_peak, mu_peak, op_factor, peak_time_range, index=indexes)
                self.daily_use, self.daily_use_masked = self.update_daily_use(random_cycle1, random_cycle2, random_cycle3,coincidence, power, index=indexes)
                tot_time = tot_time  # no correction applied to previously calculated value

            free_spots = []  # calculate how many free spots remain for further switch_ons
            try:
                free_spots.extend(j.stop - j.start for j in ma.notmasked_contiguous(self.daily_use_masked))

            except TypeError:
                free_spots = [0]
            max_free_spot = max(free_spots)

        return self.daily_use



