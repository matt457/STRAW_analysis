import numpy as np
import matplotlib.pylab as plt
import re
import csv
from tables import open_file
from statistics import median, mode
from scipy.optimize import curve_fit, minimize
from math import acos, degrees, log
from matplotlib.colors import LogNorm

class clean(object):
    def __init__(self, file_name): #__init__ requires file name
        self.file_name = file_name
        self.file_path = 'Data/POSEIDON1/' #change self.file_path to a suitable one
        l = open_file(self.file_path + self.file_name) #reads the contents in the file
        list(l)
    
    #selecting the arrays from index 150 as initial readings are negative or the clock goes to zero abruptly
        self.atstamp = l.root.absolute_timestamp[150:].flatten()    #absolute_timestamp
        self.r_0 = l.root.rising_0[150:].flatten()                  #rising_0
        self.r_1 = l.root.rising_1[150:].flatten()                  #rising_1
        self.r_2 = l.root.rising_2[150:].flatten()                  #rising_2
        self.r_3 = l.root.rising_3[150:].flatten()                  #rising_3
        self.f_0 = l.root.falling_0[150:].flatten()                 #falling_0
        self.f_1 = l.root.falling_1[150:].flatten()                 #falling_1
        self.f_2 = l.root.falling_2[150:].flatten()                 #falling_2
        self.f_3 = l.root.falling_3[150:].flatten()                 #falling_3
        self.sub_time = l.root.subevent_timestamp[150:].flatten()   #subevent_timestamp
        self.sub_id = l.root.subevent_id[150:].flatten()            #subevent_id
    
        
    #self.p_jumps is the percentage of high jumps that occur in the file. 
        self.p_jumps = ((self.atstamp.size - self.atstamp[(self.atstamp < 1e11) & (self.atstamp > -1e7)].size)/
                     self.atstamp.size) * 100 
        print('percentage of high jumps in the file -', self.p_jumps)
    
    def P_S_used(self):
    #finding POCAM, sDOM, PMT, LED and voltage used                                   
        POCAM_re = re.compile('P[1-2]')
        SDOM_re = re.compile('SDOM[1-5]')
        frequency_re = re.compile('[0-9][0-9][0-9][0-9]Hz')
        voltage_re = re.compile('[0-9][0-9]V')
        flash_time_re = re.compile('[0-9][0-9]s')
        LED_re = re.compile('P[0-9]_[a-z]')
        PMT_re = re.compile('hld_[a-z]')
        
    #The code reads the file_name to identifies the required information
        self.SDOM_num = SDOM_re.findall(self.file_name)                      #sDOM num used
        self.POCAM_num = POCAM_re.findall(self.file_name)                    #POCAM_num used
        self.frequency = frequency_re.findall(self.file_name)                #frequency used
        voltage = voltage_re.findall(self.file_name)                         #voltage applied
        flash_time = flash_time_re.findall(self.file_name)                   #flash time
        LED = LED_re.findall(self.file_name)                                 #LED used
        PMT = PMT_re.findall(self.file_name)                                 #PMT used
    
        if PMT == ['hld_d']:
            self.PMT = 'down'
        else:
            self.PMT = 'up'
    
    #Finding the color of LED used in POCAM1
        if LED == ['P1_b']:
            LED = 'blue'
        if LED == ['P1_v']:
            LED = 'violet'
        if LED == ['P1_o']:
        	LED = 'orange'
        if LED == ['P1_u']:
        	LED == 'uv'
        
    #Finding the color of LED used in POCAM1
        if LED == ['P2_b']:
            LED = 'blue'
        if LED == ['P2_v']:
            LED = 'violet'	
        if LED == ['P2_o']:
        	LED = 'orange'
        if LED == ['P2_u']:
        	LED == 'uv'
        
    #graph_title stores the important information from the file name as a list
        graph_title = [self.POCAM_num, self.SDOM_num, self.PMT, LED, voltage, self.frequency]
    #values joins the graph_title and is used as plot titles
        values = ','.join(str(v) for v in graph_title)
         
    #timestamp graph
        plt.figure(figsize=(10,9))
        plt.title(values, fontsize = 22)
        plt.ylabel('absolute_timestamps(ns)', fontsize = 19)
        plt.xlabel('index', fontsize = 19)
        plt.plot(self.atstamp, '.')
        plt.savefig(self.file_path + '/graphs/' + values + 'high_jumps.jpeg', dpi = 200)
    
    #cleaning large jumps
        elim_h_jumps = (self.atstamp < 1e11) & (self.atstamp > -1e7) #boolean for eliminating both negative and positive jumps
        abs_elim = self.atstamp[elim_h_jumps]       #absolute_timestamp 
        rising_0_elim = self.r_0[elim_h_jumps]      #rising_0 
        rising_1_elim = self.r_1[elim_h_jumps]      #rising_1
        rising_2_elim = self.r_2[elim_h_jumps]      #rising_2
        rising_3_elim = self.r_3[elim_h_jumps]      #rising_3
        falling_0_elim = self.f_0[elim_h_jumps]     #falling_0
        falling_1_elim = self.f_1[elim_h_jumps]     #falling_1
        falling_2_elim = self.f_2[elim_h_jumps]     #falling_2
        falling_3_elim = self.f_3[elim_h_jumps]     #falling_3
        sub_time_elim = self.sub_time[elim_h_jumps] #subevent_timestamp
        sub_id_elim = self.sub_id[elim_h_jumps]     #subevent_id
    
    #timestamp graph after huge jumps are removed
        plt.figure(figsize=(10,9))
        plt.title(values, fontsize = 22)
        plt.ylabel('absolute_timestamp(ns)', fontsize = 19)
        plt.xlabel('index', fontsize = 19)
        plt.plot(abs_elim, '.')
        plt.savefig(self.file_path + '/graphs/' + values + 'high_jumps_cleaned.jpeg', dpi = 200)
        plt.show()
   
   #Checking for negative time stamps
        plt.figure(figsize=(10,9))
        plt.title(values + ' Negative Timestamps', fontsize = 18)
        plt.ylabel('absolute_timestamp(ns)', fontsize = 19)
        plt.plot(abs_elim, '.')
        plt.ylim(-1e7, 0)
        plt.savefig(self.file_path + '/graphs/' + values + 'negative_values.jpeg', dpi = 200) #for all plt.savefig in module clean, residual and run change the path where the images are stored
        plt.show()

    #time difference graph
        abs_elim_diff = abs_elim[1:] - abs_elim[:-1] #subtracting consecutive timestamps to obtain time difference 
    
        abs_elim_bool = abs_elim[:-1][abs_elim_diff < 0] # selecting abs_elim_diff events less than zero, these events indicate small jumps 
        
    #plotting timestamp differences ie abs_elim_diff
        plt.figure(figsize=(10,9))
        plt.title(values, fontsize = 22)
        plt.ylabel('absolute_timestamp difference(ns)', fontsize = 19)
        plt.xlabel('index', fontsize = 19)
        plt.plot(abs_elim_diff, '.')
        plt.savefig(self.file_path + '/graphs/' + values + 'timestamp_differences.jpeg', dpi = 200)
        plt.show()
        
    #When the file contains small jumps - plot timestamp difference graph and plot zoomed in version of timestamp difference graph to show the small jumps
        if abs_elim_bool.size != 0:
            plt.figure(figsize=(10,9))
            plt.ylabel('absolute_timestamp(ns)', fontsize = 19)
            plt.xlabel('index', fontsize = 19)
            plt.title(values + ' Jumps in Timestamps', fontsize = 16)
            plt.plot(abs_elim[abs_elim_diff.argmin()-10:abs_elim_diff.argmin()+10], '.')  #Zoomed in version of the time difference graph
            plt.savefig(self.file_path + '/graphs/' + values + 'small_jumps.jpeg', dpi = 200)
            plt.show()
    
    #cleaning small jumps
        abs_elim_diff_2 = abs_elim[1:] - abs_elim[:-1]
        
        s_jump_index = [] #indices of negative abs_elim_diff_2 are stored 
        s_jump_1 = [] # negative abs_elim_diff_2 values are stored
        list_1 = np.array([]) # stores all the values of absolute_timestamp where the jump occurred

    #The loop appends all the indices of the array abs_elim_diff_2 whose values are less than zero
        for r in range(0, abs_elim_diff_2.size):
            if abs_elim_diff_2[r] < 0:
                s_jump_index.append(r)
                s_jump_1.append(abs_elim_diff_2[r])

	#The loop appends the list of indices where the jumps occurred
        for t in range(0, len(s_jump_index)):
            select = abs_elim_diff_2[s_jump_index[t] - 10:s_jump_index[t]] #selecting 10 values of abs_elim_diff_2
            x = s_jump_index[t] - (10 - (np.abs(select+abs_elim_diff_2[s_jump_index[t]])).argmin()) #the indices where the jump occurs
            jump_length = s_jump_index[t] - x #jump_length tells in how many values the jump was observed 
    
    #appending list_1 for different jump_lengths
            if jump_length == 1:
                list_1 = np.append(list_1, [x+1])
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
   
        print(list_1)
        
      #deleting the indices where the small jumps occurred
        abs_elim_3e = np.delete(abs_elim, list_1)                  #absolute_timestamp
        self.abs_elim_diff_3 = abs_elim_3e[1:] - abs_elim_3e[:-1]  #timestamp difference
            	        
        rising_0_elim_3e = np.delete(rising_0_elim, list_1)        #rising_0 
        rising_1_elim_3e = np.delete(rising_1_elim, list_1)        #rising_1
        rising_2_elim_3e = np.delete(rising_2_elim, list_1)        #rising_2
        rising_3_elim_3e = np.delete(rising_3_elim, list_1)        #rising_3
        falling_0_elim_3e = np.delete(falling_0_elim, list_1)      #falling_0
        falling_1_elim_3e = np.delete(falling_1_elim, list_1)      #falling_1
        falling_2_elim_3e = np.delete(falling_2_elim, list_1)      #falling_2
        falling_3_elim_3e = np.delete(falling_3_elim, list_1)      #falling_3
        sub_time_elim_3e = np.delete(sub_time_elim, list_1)        #subevent_timestamp
        sub_id_elim_3e = np.delete(sub_id_elim, list_1)            #subevent_id
        
    #plotting the self.abs_elim_diff_3 ie timestamp differences to check if all the small jumps are removed
        plt.figure(figsize=(10,9))
        plt.title(graph_title, fontsize = 22)
        plt.ylabel('absolute_timestamp difference(ns)', fontsize = 19)
        plt.xlabel('absolute_timestamp', fontsize = 19)
        #plt.ylim(-10000, 0)
        plt.plot(abs_elim_3e[1:], self.abs_elim_diff_3, '.')
        plt.savefig(self.file_path + '/graphs/' + values + 'small_jumps_cleaned.jpeg', dpi = 200)
        plt.show()
        
        dt_mean = (abs_elim_3e[1:] - abs_elim_3e[:-1]).mean() #mean of timestamp difference used in 'run' module
        
    #boolean considers events which have nonzero falling_0 values and subtracting falling edge from rising edge is never negative.
        boolean = ((falling_0_elim_3e != 0) & ((falling_1_elim_3e - rising_1_elim_3e) >= 0) & 
                   ((falling_2_elim_3e - rising_2_elim_3e) >= 0) & ((falling_3_elim_3e - rising_3_elim_3e) >= 0))
        abs_elim_3 = abs_elim_3e[boolean]                      #absolute_timestamp
        rising_0_elim_3 = rising_0_elim_3e[boolean]            #rising_0
        rising_1_elim_3 = rising_1_elim_3e[boolean]            #rising_1
        rising_2_elim_3 = rising_2_elim_3e[boolean]            #rising_2 
        rising_3_elim_3 = rising_3_elim_3e[boolean]            #rising_3
        falling_0_elim_3 = falling_0_elim_3e[boolean]          #falling_0
        falling_1_elim_3 = falling_1_elim_3e[boolean]          #falling_1
        falling_2_elim_3 = falling_2_elim_3e[boolean]          #falling_2
        falling_3_elim_3 = falling_3_elim_3e[boolean]          #falling_3
        sub_time_elim_3 = sub_time_elim_3e[boolean]            #subevent_timestamp
        sub_id_elim_3 = sub_id_elim_3e[boolean]                #subevent_id
        
        delete = [] #list stores events when falling edge is non zero when corresponding rising edge is zero
    #appending indices when falling edge is non zero when corresponding rising edge is zer
        # for i in range(0, abs_elim_3.size):
        #     if ((falling_1_elim_3[i] != 0 and rising_1_elim_3[i] == 0)
        #         or (falling_2_elim_3[i] != 0 and rising_2_elim_3[i] == 0)
        #         or (falling_3_elim_3[i] != 0 and rising_3_elim_3[i] == 0)
        #         ):  
        #         delete.append(i)
                
    #deleting indices of faulty falling and rising values 
        self.abs_elim_3 = np.delete(abs_elim_3, delete)                 #absolute_timestamp
        self.rising_0_elim_3 = np.delete(rising_0_elim_3, delete)       #rising_0
        self.rising_1_elim_3 = np.delete(rising_1_elim_3, delete)       #rising_1
        self.rising_2_elim_3 = np.delete(rising_2_elim_3, delete)       #rising_2
        self.rising_3_elim_3 = np.delete(rising_3_elim_3, delete)       #rising_3
        self.falling_0_elim_3 = np.delete(falling_0_elim_3, delete)     #falling_0
        self.falling_1_elim_3 = np.delete(falling_1_elim_3, delete)     #falling_1
        self.falling_2_elim_3 = np.delete(falling_2_elim_3, delete)     #falling_2
        self.falling_3_elim_3 = np.delete(falling_3_elim_3, delete)     #falling_3  
        self.sub_time_elim_3 = np.delete(sub_time_elim_3, delete)       #subevent_timestamp
        self.sub_id_elim_3 = np.delete(sub_id_elim_3, delete)           #subevent_id
    
    #f_r_percent_error is the percentage of number of faulty rising and falling time events divided by number of events that have no faults
        f_r_percent_error = (abs_elim_3e.size - self.abs_elim_3.size)/(self.abs_elim_3.size) * 100
        print(f_r_percent_error)
        return (self.abs_elim_3,self.rising_0_elim_3, self.rising_1_elim_3, self.rising_2_elim_3, self.rising_3_elim_3,
                self.falling_0_elim_3, self.falling_1_elim_3, self.falling_2_elim_3, self.falling_3_elim_3, 
                self.POCAM_num, values, self.atstamp, self.p_jumps, dt_mean, f_r_percent_error,
                self.file_path, self.SDOM_num, self.PMT, self.sub_time_elim_3, self.sub_id_elim_3)
    #returns 
    #1)absolute_timestamp after the data is cleaned
    #2)rising_0 after the data is cleaned
    #3)rising_1 after the data is cleaned
    #4)rising_2 after the data is cleaned
    #5)rising_3 after the data is cleaned
    #6)falling_0 after the data is cleaned 
    #7)falling_1 after the data is cleaned 
    #8)falling_2 after the data is cleaned
    #9)falling_3 after the data is cleaned
    #10)POCAM used
    #11)values - used for graph title
    #12)absolute_timestamp after initial 150 indices are removed
    #13)percentage of high jumps
    #14)mean of timestamp difference 
    #15)percentage of faulty rising and falling values
    #16)file path
    #17)sDOM used
    #18)upper or lower PMT used
    #19)subevent_timestamp after the data is cleaned
    #20)subevent_id after the data is cleaned

    