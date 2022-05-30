# -*- coding: utf-8 -*-

#%% Definition of the inputs
'''
Input data definition 
'''


from ramp.core.core import User, np

#number_of_app = int(np.random.normal(6,1))
#power_of_app = int(np.random.normal(7,2))
User_list = []

'''
This example input file represents an whole village-scale community,
adapted from the data used for the Journal publication. It should provide a 
complete guidance to most of the possibilities ensured by RAMP for inputs definition,
including specific modular duty cycles and cooking cycles. 
For examples related to "thermal loads", see the "input_file_2".
'''

#Create new user classes
HI = User("high income",50)
User_list.append(HI)

#High-Income
HI_indoor_bulb = HI.Appliance(HI,n=6, P=7, w=2, t = 120,r_t = 0.2, c = 10)
HI_indoor_bulb.windows([1170,1440],[0,30],0.35)

HI_Freezer2 = HI.Appliance(HI,1,200,1,1440,0,30,'yes',3)
HI_Freezer2.windows([0,1440],[0,0])
HI_Freezer2.specific_cycle_1(200,20,5,10)
HI_Freezer2.specific_cycle_2(200,15,5,15)
HI_Freezer2.specific_cycle_3(200,10,5,20)
HI_Freezer2.cycle_behaviour([480,1200],[0,0],[300,479],[0,0],[0,299],[1201,1440])

HI_Mixer = HI.Appliance(HI,1,50,3,30,0.1,1,occasional_use = 0.33)
HI_Mixer.windows([420,480],[660,750],0.35,[1140,1200])


