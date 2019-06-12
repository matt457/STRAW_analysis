import numpy as np
import matplotlib.pylab as plt
import re
import csv
from tables import open_file
from statistics import median, mode
from scipy.optimize import curve_fit, minimize
from math import acos, degrees, log
from matplotlib.colors import LogNorm
from scipy import interpolate

class run(object):
#run_time requires 
#1)absolute_timestamp after initial 150 indices are removed
#2)percentage of high jumps
#3)mean of timestamp difference obtained in 'clean' module 
    def run_time(self, abs_timestamp, jumps, dt_mean):
    #removing negative timestamps
        negative_index_values = [] #stores indices of negative timestamps
        for r in range(0, abs_timestamp.size):
            if abs_timestamp[r] < 0 and abs_timestamp[r] > -1e7:
                negative_index_values.append(r)#appending negative timestamp indices 
         
    #Condition if there are no huge jumps in the data       
        if len(negative_index_values) != 0:
            r_negative_stamp = abs_timestamp[max(negative_index_values)+1:] #eliminating negative timestamps 
			
        else:
            r_negative_stamp = abs_timestamp[:]
			
        store_index = [] #stores indices of r_negative_timestamp where the jumps have occured
        for index, value in enumerate(r_negative_stamp):
            if r_negative_stamp[index] <= -1e7 or r_negative_stamp[index] >= 1e11:
                store_index.append(index)
                #print(index, value)
				
        #high jumps
        store_sum = ([])
        if jumps == 0.0:
            sum_1 = r_negative_stamp[r_negative_stamp.size - 1] - r_negative_stamp[0]
            Ncuts_h_size = 0
            
        else:
            Ncuts_h = [] #stores the indices of store_index
            time = ([]) #stores time difference 
        #Calculating the time difference of timestamp after the one set of high jumps and timestamp before the next set of jump
            for values in range(0, len(store_index)):
                if values == 0 and store_index[values] != 0:
                    time_diff = r_negative_stamp[store_index[values]-1] - r_negative_stamp[0]
                    time = np.append(time, time_diff)
                    print(store_index[values]-1)
                    print()
                    
                if values == len(store_index) - 1:
                    time_diff = r_negative_stamp[r_negative_stamp.size - 1] - r_negative_stamp[store_index[values]+1]
                    time = np.append(time, time_diff)
                    print(store_index[values]+1)
                else:
                    if store_index[values+1] - store_index[values]!= 1:
                        Ncuts_h.append(values+1)
                        time_diff = r_negative_stamp[store_index[values+1]-1] - r_negative_stamp[store_index[values]+1]
                        time = np.append(time, time_diff)
                        print(store_index[values+1]-1, store_index[values]+1)
            print(time[0:1000])
            print(r_negative_stamp[store_index[0]-1] - r_negative_stamp[0])
            print(r_negative_stamp[r_negative_stamp.size-1] - r_negative_stamp[store_index[len(store_index) - 1]+1])
            print(store_index[0]-1)
            Ncuts_h_size = len(Ncuts_h) #the number of cuts ie how many sets of huge jumps were removed
            
            sum_1 = time.sum() #summing all the time difference gives the run time without considering the high jumps
        r_high_jumps = r_negative_stamp[(r_negative_stamp < 1e11) & (r_negative_stamp > -1e7)] #removing the huge jumps
        r_high_diff = r_high_jumps[1:] - r_high_jumps[:-1] #timestamp time difference 
        
        s_jump_index = [] #stores indices where the small jumps have occurred 
        list_1 = ([]) #stores all the values of absolute_timestamp where the small jumps occurred
        
    #The loop appends all the indices of the array r_high_diff whose values are less than zero
        for r in range(0, r_high_diff.size):
            if r_high_diff[r] < 0:  
                s_jump_index.append(r)
                #print(s_jump_index)
                
    #The loop appends the list of indices where the jumps occurred        
        for t in range(0, len(s_jump_index)):
            #print("index",s_jump_index[t])
            #print("index value", abs_elim_diff_2[s_jump_index[t]])
            select =r_high_diff[s_jump_index[t] - 10:s_jump_index[t]] #selecting 10 values of r_high_diff
            #print("select", select)
            x = s_jump_index[t] - (10 - (np.abs(select+r_high_diff[s_jump_index[t]])).argmin())#the indices where the jump occurs
            #print("argmin", np.abs(select+abs_elim_diff_2[s_jump_index[t]]).argmin())
            #print(x)
            
            jump_length = s_jump_index[t] - x #jump_length tells in how many values the jump was observed 
            
            #print("jump length", jump_length)
            
            # print("abs_elim_2.size",abs_elim_2.size)
        #appending list_1 for different jump_lengths
            if jump_length == 1:
                list_1 = np.append(list_1,[x+1])
            if jump_length == 2:
                list_1 = np.append(list_1,[x+1, x+2])
            if jump_length == 3:
                list_1 = np.append(list_1,[x+1, x+2, x+3])
            if jump_length == 4:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4])
            if jump_length == 5:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4, x+5])
            if jump_length == 6:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4, x+5, x+6])
            if jump_length == 7:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4, x+5, x+6, x+7])
            if jump_length == 8:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8])
            if jump_length == 9:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9])
            if jump_length == 10:
                list_1 = np.append(list_1,[x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9, x+10])
         
    #Condition if there are no small jumps in the data        
        if len(list_1) == 0:
            run_time = sum_1
            Ncuts = Ncuts_h_size
            error_rtime = (2 * Ncuts * dt_mean) * 1e-9
        
        else:
            list_int = list_1.astype(int) #converting list_1 into an integer
        
            Ncuts_s = [] #stores indices where the jumps occurred
        
            for values_s in range(0, len(list_1)):
                if values_s == len(list_1) - 1:
                    continue
                else:
                    if list_1[values_s+1] - list_1[values_s]!= 1:
                        Ncuts_s.append(values_s)
            print(len(Ncuts_s))
    
            Ncuts = Ncuts_h_size + len(Ncuts_s)
            print(Ncuts)
            error_rtime = (2 * Ncuts * dt_mean) * 1e-9
        
        #Calculating the time difference between the timestamp before the first set of small jumps and timestamp after the same set of small jumps
            # small jumps
            store_sum_1 = [] #stores indices of timestamp before the set of small jumps
            store_sum_2 = ([]) #stores indices of timestamp after the same set of small jumps being considered
            for k in range(0, len(list_int)):
                if k == 0 and list_int[k] != 0:
                    po = r_high_jumps[list_int[k]-1]
                    store_sum_1.append(po)
                    print(k,list_int[k], po, 'p')
                    if list_int[k] != list_int[k+1] - 1:
                        po = r_high_jumps[list_int[k+1]-1] 
                        qo = r_high_jumps[list_int[k] + 1]
                        print(k, list_int[k], po, 'p')
                        print(k, list_int[k], qo, 'q')
                        store_sum_1.append(po)
                        store_sum_2 = np.append(store_sum_2, qo)
                elif k == len(list_int) - 1:
                    qo = r_high_jumps[list_int[k]+1]
                    store_sum_2 = np.append(store_sum_2, qo)
                    print(k,list_int[k],qo, 'q')
                elif list_int[k+1] == list_int[k]+1:
                    continue
                elif list_int[k] != list_int[k+1] - 1:
                    po = r_high_jumps[list_int[k+1]-1]
                    qo = r_high_jumps[list_int[k] + 1]
                    print(k, list_int[k], po, 'p')
                    print(k, list_int[k], qo, 'q')
                    store_sum_1.append(po)
                    store_sum_2 = np.append(store_sum_2, qo)
                #print(k, p)

                #store_sum_1.append(po)
                #store_sum_2 = np.append(store_sum_2, qo)
        
            sum_2 = sum(store_sum_2 - store_sum_1) #summing difference between the timestamps gives the dead time of the readout
        #run time without considering both huge jumps and small jumps
            run_time = sum_1 - sum_2 #subtracting the dead time caused by small jumps from the run time calculated without considering the high jumps
            print('time removed(small jumps) - ', sum_2)
     
        print('timestamp of the last event - ', abs_timestamp[abs_timestamp.size - 1])
        print('eliminating high jumps and summing - ', sum_1)
        print('run time - ', run_time)
    
        self.run_time = run_time * 1e-9 #converting run time from nanoseconds to seconds 
        return self.run_time, error_rtime
    #returs
    #1)run time in seconds 
    #2)error on runtime 
    
#angl_dist requires
#1)POCAM used that is returned in module 'clean'
#2)sDOM used that is returned in module 'clean'
#3)PMT used that is returned in module 'clean' (up or down)
    def angl_dist(self, POCAM_num, SDOM_num, PMT):
    #sizes of POCAMs and sDOMs
        SDOM_size = 0.3 
        POCAM_size = 0.2
        
    #location of POCAM and sDOM on the string
        POCAM2_loc = 107.66 
        POCAM1_loc = 109.79
    
    #Considering only the upper PMT of the sDOM and correcting for the location of sDOMs on the strings
        SDOM1up_loc = 69.79 - SDOM_size
        SDOM2up_loc = 49.40 - SDOM_size
        SDOM3up_loc = 29.98 - SDOM_size
        SDOM4up_loc = 29.96 - SDOM_size
        SDOM5up_loc = 69.10 - SDOM_size
    
        string_distance = 37 #distance between the two strings
    
    #pyth calculates the distance between sDOM and POCAM which are on two different strings 
    #pyth requires 
    #1)location of POCAM on the String
    #2)location of sDOM 
    #3)distance between the strings
        def pyth(a, b, c):
            d = a - b
            e = np.sqrt(d**2 + c**2)
            return e
    
        if POCAM_num == ['P2'] and SDOM_num == ['SDOM1'] and PMT == 'up' :
            self.distance = pyth(POCAM2_loc, SDOM1up_loc, string_distance) #Calculating the distance between sDOM and POCAM 
            angle = degrees(acos((POCAM2_loc - SDOM1up_loc)/self.distance)) #Calculating the angle between sDOM and POCAM
        
        if POCAM_num == ['P2'] and SDOM_num == ['SDOM2'] and PMT == 'up' :
            self.distance = pyth(POCAM2_loc, SDOM2up_loc, string_distance)
            angle = degrees(acos((POCAM2_loc - SDOM2up_loc)/self.distance))
        
        if POCAM_num == ['P2'] and SDOM_num == ['SDOM3'] and PMT == 'up' :
            self.distance = pyth(POCAM2_loc, SDOM3up_loc, string_distance)
            angle = degrees(acos((POCAM2_loc - SDOM3up_loc)/self.distance))
    
        if POCAM_num == ['P2'] and SDOM_num == ['SDOM5'] and PMT == 'up' :
            self.distance = POCAM2_loc - SDOM5up_loc
            angle = 0
        
        if POCAM_num == ['P1'] and SDOM_num == ['SDOM1'] and PMT == 'up' :
            self.distance = POCAM1_loc - SDOM1up_loc
            angle = 0
        
        if POCAM_num == ['P1'] and SDOM_num == ['SDOM4'] and PMT == 'up' :
            self.distance = pyth(POCAM1_loc, SDOM4up_loc, string_distance)
            angle = degrees(acos((POCAM1_loc - SDOM4up_loc)/self.distance))
        
        if POCAM_num == ['P1'] and SDOM_num == ['SDOM5'] and PMT == 'up' :
            self.distance = pyth(POCAM1_loc, SDOM5up_loc, string_distance)
            angle = degrees(acos((POCAM1_loc - SDOM5up_loc)/self.distance))
        
    #Intensity of light produced by the POCAM depends on the angle
    #The following calculates the intensity of the POCAM for a given angle
        data=np.array([-65, 0.9461212989738365,
        -60, 0.9559174264331389,
        -55, 0.9627747156546507,
        -50, 0.9715912303680229,
        -45, 0.9764892940976742,
        -40.000000000000014, 0.9823669705732556,
        -34.999999999999986, 0.9892242597947672,
        -30.000000000000014, 0.992163098032558,
        -25, 0.9931427107784883,
        -20.000000000000014, 0.9941223235244185,
        -15, 0.9951019362703488,
        -10, 0.9951019362703488,
        -5, 0.9970611617622093,
        5, 0.996081549016279,
        10, 0.997061161762209,
        14.999999999999972, 0.9970611617622093,
        20, 0.9970611617622093,
        25, 0.9951019362703488,
        30, 0.9951019362703488,
        35.00000000000003, 0.9911834852866278,
        40, 0.9872650343029068,
        45, 0.9813873578273253,
        49.99999999999997, 0.9725708431139531,
        55, 0.9657135538924414,
        60, 0.9588562646709297,
        64.99999999999997, 0.9500397499575575])
        
        fcn = interpolate.interp1d(data[::2], data[1::2])
        
        self.angle_cr = fcn(angle)
        
        return self.distance, self.angle_cr
    #returns
    #1)distance between sDOM and POCAM 
    #2)Intensity correction for a given angle

#correction requires
#1)number POCAM events obtained in module 'residual'
#2)number of noise hits obtained in module 'residual'
    def correction(self, num_events, noise_events):
        error_nevents = (num_events/self.angle_cr) #poisson error in the number of POCAM hits after correcting for the angle
        eve_p_sec = (num_events/self.angle_cr)/self.run_time #correcting the run time and intensity of different sDOMs (POCAM hits)
        n_eve_p_sec = (noise_events)/self.run_time #correcting  for the run time of different sDOMs (noise hits)
        return eve_p_sec, error_nevents, n_eve_p_sec
    #returns 
    #1)events per second
    #2)error in events per second 
    #3)noise events per second 