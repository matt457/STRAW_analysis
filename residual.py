import numpy as np
import matplotlib.pylab as plt
import re
import csv
from tables import open_file
from statistics import median, mode
from scipy.optimize import curve_fit, minimize
from math import acos, degrees, log
from matplotlib.colors import LogNorm

class residual(object):
#__init__ requires 
#1)absolute_stamp after the data is cleaned
#2)rising_0, rising_1 or rising_2, or rising_3 after the data is cleaned 
#3, 4, 5) x-axis range and the number of steps for numpy linspace to plot the histogram of time difference (np.linspace(lspace_lower, lspace_upper, lspace))
#6,7)lower and upper bound respectively while selecting the events in peak(refer to events_in_peak)(Depends on the frequency of the POCAM)
#8, 9, 10, 11) xlim range and ylim range respectively for time_residuals vs timestamps plot
#12)file path
#13)values - used for graph title
#14)POCAM_num
    def __init__(self, abs_elim_3, rising_a_elim_3, lspace_lower, lspace_upper, lspace, lower_bound, upper_bound,
                xlim_min, xlim_max, ylim_min, ylim_max, file_path, values, POCAM_num, rising_0_elim_3, falling_0_elim_3):
    #self.t_threshold adds timestamp values for which the given rising edge value > 0 and rising edge values > 0
        self.t_threshold = abs_elim_3[rising_a_elim_3 > 0] + rising_a_elim_3[rising_a_elim_3 > 0]
        t_diff = self.t_threshold[1:] - self.t_threshold[:-1] #difference between consecutive threshold values
        self.events_in_peak = self.t_threshold[:-1][(t_diff[:] > lower_bound) & (t_diff[:] < upper_bound)] #selecting events in peak
    
    #plotting time difference histogram
        plt.figure(figsize=(10,9))
        _ = plt.hist(t_diff, np.linspace(lspace_lower,lspace_upper, lspace), log = True)
        plt.title(values + '- Time Difference', fontsize = 19)
        plt.xlabel('time difference(ns)', fontsize = 16)
        
    #The time between consecutive POCAM flashes(ie POCAM interval) varies as the POCAM used changes.
    #Time residual graph is very sensitive to the POCAM interval
        if POCAM_num ==['P1']:
            self.estimate_peak = 400100.71     #an approximation of POCAM interval for POCAM_1 # 200100.71 for 5000Hz, 400100.71 for 2500Hz
        if POCAM_num == ['P2']:
            self.estimate_peak = 400101.33     #an approximation of POCAM interval for POCAM_2 # 200101.33 for 5000Hz, 400101.33 for 2500Hz
            
        self.estimate_residual = self.events_in_peak%self.estimate_peak #an approximation of time residuals obtained using approximate POCAM intervals
    
    #plotting time residuals vs timestamps
        plt.figure(figsize=(10,9))
        plt.plot(self.events_in_peak, self.estimate_residual, '.')
        plt.title(values + '-time residuals of threshold 1', fontsize = 19)
        plt.xlabel('timestamps', fontsize = 16)
        plt.ylabel('time residuals', fontsize = 16)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min,ylim_max)

        # Time over threshold 0 and associated number of photons in events
        time_over_threshold, weights = self.TimeOverThreshold(rising_0_elim_3, falling_0_elim_3) 

        self.abs_elim_3 = abs_elim_3
        self.rising_a_elim_3 = rising_a_elim_3
        self.values = values
        self.file_path = file_path
        self.weights = weights
        self.ToT = time_over_threshold
        self.save_path = 'Data/MINOS1/uv/Measured_arrival_times/'

# TimeOverThreshold accounts for multi-photon events by assigning weights to each event
# requires corresponding rising and falling edges after cleaning
    def TimeOverThreshold(self, rising_a_elim_3, falling_a_elim_3):
        time_threshold = 10 
        time_over_threshold = falling_a_elim_3-rising_a_elim_3
        weights = np.array([2 if i>time_threshold else 1 for i in time_over_threshold])
        return time_over_threshold, weights
    #returns time over threshold and a weight (number of photons) corresponding to each event

# GetGausPeak produces the POCAM interval, gaus_peak
# An alternative to the function minimizer; GetGausPeak no longer relies on events_in_peak (which can be empty)
# Input absolute timestamp range as [min,max]
# Output POCAM interval that maximizes the counts around the peak of the arrival time distribution for events in timestamp_range
    def GetGausPeak(self, timestamp_range):
        window = (self.abs_elim_3>timestamp_range[0]) & (self.abs_elim_3<timestamp_range[1])
        event_timestamps = (self.abs_elim_3+self.rising_a_elim_3)[window]
        
        def func_2_minimize(gaus_peak):
            residuals = event_timestamps%gaus_peak
            hist,bins = np.histogram(residuals,bins=80000) # bins is dependent on POCAM frequency
            max_index = np.argmax(hist)
            my_sum = np.sum(hist[max_index-2:max_index+3])
            return 1./my_sum
        
        result = minimize(func_2_minimize, [400100.72], method='Powell')
        gaus_peak = result.x
        print(gaus_peak)
        return gaus_peak
    
#minimizer requires the timestamp range for the longest stable run
    def minimizer(self, select_min, select_max):
    
        #Minimizer
        tmin = select_min
        tmax = select_max
        def dfunc2(delta_t):
            x = self.events_in_peak
            tres = x%delta_t
            selection = (self.events_in_peak>tmin)*(self.events_in_peak<tmax) #selecting events in peak given tmin and tmax
            return np.sum((tres[selection]-tres[selection].mean())**2)
        from scipy.optimize import minimize
        m = minimize(dfunc2, [400100.], method='Powell')
        #print(m)
        print(m.x)
        gaus_peak = m.x
        return gaus_peak
    #returns an accurate POCAM_interval stored as gaus_peak

#res requires 
#1)gaus_peak obtained from the minimizer
#2)when difference between consecutive self.estimate_residual values est_res_diff is calculated, est_res_diff tells when the jumps
   #in time residuals occurred, the argument given here is the jump length and decides the individual runs from the time residual graph 
#3)med_bound decides width of the gaussian formed when time residual histogram is plotted.  
    def res(self, gaus_peak, greater_than, med_bound, bin_size=1):
        print(self.events_in_peak.size)
        est_res_diff = abs(self.estimate_residual[1:] - self.estimate_residual[:-1]) 
        jump_index = ([]) #stores indices of est_res_diff which are greater than the given jump size
        jump_index = np.append(jump_index, 0) #appending zero jump_index so that events from timestamp zero to timestamp where the first jump occurred can be included
        for r in range(0, est_res_diff.size):
            if abs(est_res_diff[r]) > greater_than:
                jump_index = np.append(jump_index, r)
            
        JumpIndex = jump_index.astype(int) #converting jump_index to intergers
            
        print('jump_index', jump_index)
        print('jump_index size', jump_index.size)
    
        self.peak_1 = [] #stores time_res_all values
        t_res_all = np.array([]) #stores time residuals of events_in_peak 
        t_res_all_all = np.array([]) #stores time residuals of all the events recorded by the sDOM
        weights_all = np.array([])
        keep = self.rising_a_elim_3 >= 0
        abs_elim = self.abs_elim_3[keep] + self.rising_a_elim_3[keep]
        abs_time = self.abs_elim_3[keep]
        weights_keep = self.weights[keep] 
        
        for p in range(0, len(JumpIndex)):    

            if p == 0: #events from timestamp zero to timestamp where the first jump occurred
                print("index_p", p)
            #v selects individual runs from time residual vs timestamp graph (POCAM EVENTS ONLY)
                v = self.events_in_peak[:][(self.events_in_peak[:] >= self.events_in_peak[0]) & 
                                      (self.events_in_peak[:] < self.events_in_peak[JumpIndex[p+1]+1])]
                print('Jump Indices',0,JumpIndex[p+1]+1)
                if v.size == 0: #events from timestamp where the last jump occurred to the last timestamp
                    continue
                print(v.size)
                b = self.events_in_peak[JumpIndex[1]+1]
                a = min(v)
            #run_time selects individual runs from absolute_elim, ie considers all the events that crossed the given threshold
                select = (abs_elim >= 0) & (abs_elim < b)
                run_time = abs_elim[select]
                run_abs_time = abs_time[select]
                weights = weights_keep[select]
            
            elif p == JumpIndex.size - 1: #
                print("index_p", p)
                print('Jump Index size =', JumpIndex.size, self.events_in_peak.size - 1)
                print([JumpIndex[p]+1])
                v = self.events_in_peak[:][(self.events_in_peak[:] >= self.events_in_peak[JumpIndex[p]+1]) & 
                                      (self.events_in_peak[:] <= self.events_in_peak[self.events_in_peak.size - 1])]
            
                print('Jump Indices', JumpIndex[p]+1, self.events_in_peak.size - 1)
                a = min(v)
                select = (abs_elim >= a) & (abs_elim <= abs_elim[abs_elim.size-1])
                run_time = abs_elim[select]
                run_abs_time = abs_time[select]
                weights = weights_keep[select]
        
            else: 
                print("index_p", p)
                v = self.events_in_peak[:][(self.events_in_peak[:] >= self.events_in_peak[JumpIndex[p]+1]) & 
                                      (self.events_in_peak[:] < self.events_in_peak[JumpIndex[p+1]+1])]
                print('Jump Indices', JumpIndex[p]+1, JumpIndex[p+1]+1)
                print(v.size)
                b = self.events_in_peak[JumpIndex[p+1]+1]
                a = min(v)
                select = (abs_elim >= a) & (abs_elim < b)
                run_time = abs_elim[select]
                run_abs_time = abs_time[select]
                weights = weights_keep[select]
        
            # Calculate gaus_peak for individual run
            if v.size == 1:
                gaus_peak_exact = gaus_peak
            else:
                gaus_peak_exact = self.minimizer(run_abs_time[0], run_abs_time[-1])

            time_res = v%gaus_peak_exact #calculates the residual of events in peak for individual runs (POCAM EVENTS ONLY)
            time_res_all = run_time%gaus_peak_exact #calculates the residual of all the events for individual runs
            #print('time_res length', time_res.size)
            
        #plotting time residual graph for individual runs (POCAM EVENTS ONLY)
            plt.figure(figsize=(5,4))           
            plt.plot(time_res, '.')
            plt.ylabel('time_residual')
            plt.show()
    
        #Gaussian fit
            med = median(time_res) #POCAM EVENTS ONLY                    
            med_all = median(time_res_all)
            peak = time_res[(time_res >= med - med_bound) & (time_res <= med + med_bound)] #selecting width of the time residual gaussian(POCAM EVENTS ONLY)
            peak_all = time_res_all[(time_res_all >= med_all - med_bound) & (time_res_all <= med_all + med_bound)] #selecting the width of the time residual gaussian 
    
            
            def gaussian(x, mean, amplitude, standard_deviation):
                return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)

            bins = np.linspace(med-med_bound, med + med_bound, 11) #POCAM EVENTS ONLY   
            bins_all = np.linspace(med-med_bound, med + med_bound, 11)
            data_entries_1, bins_1, _ = plt.hist(peak, bins, alpha = 0.5) #POCAM EVENTS ONLY   
            #plt.show()
            data_entries_1_all, bins_1_all, _ = plt.hist(peak_all, bins_all, alpha = 0.5)
            plt.close()
    
            data = peak #POCAM EVENTS ONLY   
            data_all = peak_all
            bincenters = ((bins[:-1]+bins[1:])/2) #POCAM EVENTS ONLY   
            bincenters_all = ((bins_all[:-1]+bins_all[1:])/2)
    
            from scipy.optimize import curve_fit
            #curve fit (POCAM EVENTS ONLY)   
            data_entries = data_entries_1
            try:
                popt, pcov = curve_fit(gaussian, xdata = bincenters, 
                                    ydata = data_entries,  
                                    absolute_sigma = True, 
                                    p0 = (med, 10, 5),
                                    sigma = np.sqrt(data_entries))
            except RuntimeError:
                continue

            # popt, pcov = curve_fit(gaussian, xdata = bincenters, 
            #                         ydata = data_entries,  
            #                         absolute_sigma = True, 
            #                         p0 = (med, 10, 5),
            #                         sigma = np.sqrt(data_entries))
            
            #curve fit                        
            data_entries_all = data_entries_1_all
            popt_all, pcov_all = curve_fit(gaussian, xdata = bincenters_all, 
                                            ydata = data_entries_all,  
                                            absolute_sigma = True, 
                                            p0 = (med, 10, 5),
                                            sigma = np.sqrt(data_entries_all))
    
    #shifting the time residual peak to zero
            time_res_sub = time_res - popt[0] #subtracting time_residual with the mean obtained by using curve fit(POCAM EVENTS ONLY)   
            time_res_sub_all = time_res_all - popt_all[0] #subtracting time_residual with the mean obtained by using curve fit all events crossing the given rising edge considered
    
            self.peak_1.append(time_res_all) #peak_1 is for function CheckPeak

            #print(popt)
            #print(popt_all)
            
        #appending time residual value after shifted to zero
            t_res_all =np.append(t_res_all, time_res_sub) #POCAM EVENTS ONLY
            t_res_all_all =np.append(t_res_all_all, time_res_sub_all)
            weights_all = np.append(weights_all, weights)
            
    #plotting time residual histogram(POCAM EVENTS ONLY)
        # plt.figure(figsize=(10,9))
        # n, bins, patches = plt.hist(t_res_all, 480,
        #                             #np.linspace(-10, 40, 100), 
        #                             log = True)
        # plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        # plt.xlabel('time_ns', fontsize = 16)
        # plt.ylabel('bincount', fontsize = 16)
        # plt.axvline(color = 'r')
    
    #plotting time residual histogram x limit (-100, 200)
        plt.figure(figsize=(10,9))
        n, bins, patches = plt.hist(t_res_all_all, 
                    #500,
                    np.arange(-100, 400,bin_size), 
                    log = True,
                    weights = weights_all)   
        plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        plt.xlabel('time_ns', fontsize = 16)
        plt.ylabel('bincount', fontsize = 16)
        plt.axvline(x = 0, color = 'r')
        plt.axvline(x = -10, color = 'k')
        plt.axvline(x = 10, color = 'k')
        plt.savefig(self.file_path + '/graphs/' + self.values + 'time resi graph', dpi = 200)

        # save n and bins (centers) as arrival time data
        filename = self.values + '.csv'
        path = self.save_path #'Data/POSEIDON1/Measured_arrival_times/'
        with open(path+filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(bins[:-1])
            writer.writerow(n)
        csvfile.close()
    
    #plotting time residual histogram x limit (-1000, 20000)
        # plt.figure(figsize=(10,9))
        # n, bins, patches = plt.hist(t_res_all_all, 
        #                             #500,
        #                             np.linspace(-1000, 20000, 480), 
        #                             log = True)
        # plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        # plt.xlabel('time_ns', fontsize = 16)
        # plt.ylabel('bincount', fontsize = 16)
                                    
    #plotting time residual histogram all events
        plt.figure(figsize=(10,9))
        n, bins, patches = plt.hist(t_res_all_all, 
                                    500,
                                    #np.linspace(-1000, 20000, 480), 
                                    log = True,
                                    weights = weights_all)
        plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        plt.xlabel('time_ns', fontsize = 16)
        plt.ylabel('bincount', fontsize = 16)
    
    #plotting time residuals after each individual runs are shifted to zero manually
        plt.figure(figsize=(10,9))
        plt.axhline(t_res_all[(t_res_all<25000)].mean(), 0, 1, color='k')
        plt.plot(t_res_all, '.')
    
        #what_peak = abs_elim[(t_res_all_all > 24) & (t_res_all_all < 40)]
        num_events = t_res_all_all[(t_res_all_all > -10) & (t_res_all_all < 10)] #selecting POCAM events with 20ns width
        noise_events = t_res_all_all[(t_res_all_all < -10000) ^ (t_res_all_all > 10000)] #selecting noise events
        return t_res_all_all, num_events.size, noise_events.size
    #returns 
    #1)time_residuals for all the events
    #2)number of POCAM events for given width
    #3)number of noise hits
    
#HIST2D requires
#1)Number of bins used for the 2D histogram
#2)gaus_peak obtained from the minimizer
#3)sDOM_num
    def HIST2D(self, BinsHist, gaus_peak, SDOM_num):
    #plotting 2D histogram of time residuals vs timestamps
        plt.figure(figsize=(12,12))
        keep = self.rising_a_elim_3 >= 0
        self.abs_elim = self.abs_elim_3[keep] + self.rising_a_elim_3[keep]
        self.weights_keep = self.weights[keep]
        x = self.abs_elim
        y = self.abs_elim % gaus_peak
    #Log-scale
        h, self.xedges, self.yedges, img = plt.hist2d(x, y,
                                            BinsHist,
                                            #[np.linspace(0.0e10,0.2e10, 150), np.linspace(97620, 102000, 150)],
                                            #[np.linspace(2.68e10, 3e10, 150), np.linspace(13410, 13500, 150)],
                                            #cmin = 4 , 
                                            norm = LogNorm() 
                                            )
        cb = plt.colorbar()
        plt.xlabel('timestamps')
        plt.ylabel('time residuals')
        
    #No log-scale 
        plt.figure(figsize=(12,12))
        h, self.xedges, self.yedges, img = plt.hist2d(x, y,
                                            BinsHist,
                                            #[np.linspace(0.0e10,0.2e10, 150), np.linspace(97620, 102000, 150)],
                                            #[np.linspace(2.68e10, 3e10, 150), np.linspace(13410, 13500, 150)],
                                            #cmin = 4 , 
                                            #norm = LogNorm() 
                                            )
        cb = plt.colorbar()
        plt.xlabel('timestamps')
        plt.ylabel('residuals')
    
    #the loop scans the graph to find the brightest bins - the brightest bins (ie bins with more events) generally correspond to the POCAM events
        self.POCAM_bins = ([]) #stores the bins that contain POCAM events
        for j in range (0, BinsHist):
            #print(j, j+1)
            bins = h[j:j+1, 0:].flatten()
            max_ind = np.argmax(bins) #selecting the brightest bin
            #moving along x-axis the POCAM_bins correspond to the bin number along y_axis
            self.POCAM_bins = np.append(self.POCAM_bins, max_ind)
        
        POCAM_diff = abs(self.POCAM_bins[1:] - self.POCAM_bins[:-1]) #Subtracting the brightest bin number to observe where the jumps in time residuals have occurred
    
    #Conditions for selecting runs according to each sDOM
        if SDOM_num == ['SDOM5']:
            #Mode = mode(POCAM_diff[POCAM_diff > 1]) #in case this doesn't work use Mode = 1
            Mode = 1
            jump_index = np.where((POCAM_diff > Mode - 25) * (POCAM_diff < Mode + 25))
        if SDOM_num == ['SDOM1']:
            Mode = 1
            jump_index = np.where((POCAM_diff > Mode - 25) * (POCAM_diff < Mode + 25))
        if SDOM_num == ['SDOM2']:
            Mode = 1
            jump_index = np.where((POCAM_diff > Mode - 25) * (POCAM_diff < Mode + 25))
        if SDOM_num == ['SDOM3']:
            Mode = 1
            jump_index = np.where((POCAM_diff > Mode - 25) * (POCAM_diff < Mode + 25))
        
        self.JumpIndex = (np.array(jump_index).flatten()) + 1
        #print(self.JumpIndex)    
    
        plt.plot(self.POCAM_bins, '.') #plotting POCAM_bins as a scatter plot to check if the right bins are bein picked
        
        self.gaus_peak =  gaus_peak #storing gaus_peak value as self.gaus_peak to use for calc_res function
        return self.abs_elim, BinsHist, self.JumpIndex, self.xedges, self.yedges, self.POCAM_bins, POCAM_diff
    #returns
    #1)self.abs_elim
    #2)Number of bins used to plot the 2D histogram
    #3)Indices of POCAM_diff where the jump occurred
    #4)xedges of the 2D histogram
    #5)yedges of the 2D histogram 
    #6)POCAM_bins
    #7)POCAM_diff
    
#calc_res requires
#1)Number of bins used to plot the 2D histogram
#2)med_bound decides width of the gaussian formed when time residual histogram is plotted.
#3)after the mean of  y_axis is calculated the lboud is subtracted from it 
#4)after the mean of  y_axis is calculated the uboud is added to it 
    def calc_res(self, BinsHist, med_bound, yaxis_lbound, yaxis_ubound, bin_size=1):
        #gaus_peak = 200100.33417353552
        a = 0
        Min = 0
        Max = self.JumpIndex[0]
        t_res_all = np.array([]) #stores the time residual values
        weights_all = np.array([])
        self.peak_1 = [] #stores time_res_sub values
        for d in range(0, self.JumpIndex.size + 1):
            #print('run#', d)
            if d == self.JumpIndex.size:
                b = BinsHist
                Max = BinsHist
                #print('b', b, 'Max', Max)
                select_criteria = (self.abs_elim >= a) & (self.abs_elim <= self.xedges[b])
                select = self.abs_elim[select_criteria] % self.gaus_peak #selecting timestamp values 
                weights = self.weights_keep[select_criteria]
                y_axis = int(self.POCAM_bins[Min:Max].mean())
                #print('yaxis - ', y_axis)
                lower_bound = y_axis -  yaxis_lbound #defining the width of the selection of events along y-axis(time residuals)(how many number of bins below the mean)
                upper_bound = y_axis + yaxis_ubound #defining the width of the selection of events along y-axis(time residuals)
                time_res = select[(select >= self.yedges[lower_bound]) & (select <= self.yedges[upper_bound])] #selecting timestamp events that have the defined time residual value(how many number if above the mean
            else:
                b = self.JumpIndex[d] 
                Max = self.JumpIndex[d]
                #print('b', b, 'Max', Max)
                select_criteria = (self.abs_elim >= a) & (self.abs_elim < self.xedges[b])
                select = self.abs_elim[select_criteria] % self.gaus_peak
                weights = self.weights_keep[select_criteria]
                #print('select size', select.size)
                y_axis = int(self.POCAM_bins[Min:Max].mean())
                lower_bound = y_axis -  yaxis_lbound
                upper_bound = y_axis + yaxis_ubound
                time_res = select[(select >= self.yedges[lower_bound]) & (select <= self.yedges[upper_bound])]
                
            if time_res.size == 0:
                continue
            else:
            #Gaussian Fit 
                med = median(time_res)
                #print('median - ', med)
                peak = time_res[(time_res >= med - med_bound) & (time_res <= med + med_bound)]#selecting the width of the time residual gaussian 
                
                def gaussian(x, mean, amplitude, standard_deviation):
                    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)
                    
                bins = np.linspace(med - med_bound, med + med_bound, 11)
                data_entries_1, bins_1, _ = plt.hist(peak, bins, alpha = 0.5)
                
                data = peak
                bincenters = ((bins[:-1]+bins[1:])/2)
                
                from scipy.optimize import curve_fit
                #curve_fit
                data_entries = data_entries_1
                try:
                    popt, pcov = curve_fit(gaussian, xdata = bincenters, 
                                        ydata = data_entries,  
                                        absolute_sigma = True, 
                                        p0 = (med, 10, 5),
                                        sigma = np.sqrt(data_entries))
                    recenter = popt[0]
                except RuntimeError:
                    recenter = bincenters[np.argmax(data_entries)]
                    continue
                # popt, pcov = curve_fit(gaussian, xdata = bincenters, 
                #                         ydata = data_entries,
                #                         absolute_sigma = True,
                #                         p0 = (med, 10, 5),
                #                         sigma = np.sqrt(data_entries))  

                time_res_sub = select - recenter #subtracting time_residual with the mean obtained by using curve fit all events crossing the given rising edge considered
                
                self.peak_1.append(time_res_sub) #peak_1 is for function CheckPeak
                
            #appending time residual value after shifted to zero
                t_res_all =np.append(t_res_all, time_res_sub)
                weights_all = np.append(weights_all, weights)
                
            a = self.xedges[b]
            Min = Max
            #print('a', b, Min, 'min')
        
    #plotting time residual histogram x limit (-100, 200)
        plt.figure(figsize=(10,9))
        n, bins, patches = plt.hist(t_res_all, 
                               #500,
                               np.arange(-100, 400, bin_size), 
                               log = True,
                               weights=weights_all)
        plt.axvline(x = 0, color = 'r')
        plt.axvline(x = -10, color = 'k')
        plt.axvline(x = 10, color = 'k')
        plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        plt.xlabel('time_ns', fontsize = 16)
        plt.ylabel('bincount', fontsize = 16)
        plt.savefig(self.file_path + '/graphs/' + self.values + 'time resi graph', dpi = 200)

        # save to csv
        filename = self.values+'.csv'
        path = self.save_path #'Data/POSEIDON1/Measured_arrival_times/'
        with open(path+filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(bins[:-1])
            writer.writerow(n)
        csvfile.close()
    
        
    #plotting time residual histogram x limit (-1000, 20000)
        # plt.figure(figsize=(10,9))
        # _ = plt.hist(t_res_all,
        #           #500,
        #           np.linspace(-1000, 20000, 480),
        #           log = True)
        # plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        # plt.xlabel('time_ns', fontsize = 16)
        # plt.ylabel('bincount', fontsize = 16)
        # plt.show()
    
    #plotting time residual histogram all events
        plt.figure(figsize=(10,9))
        _ = plt.hist(t_res_all, 
                    500,
                    #np.linspace(-100500, -99500, 480), 
                    log = True,
                    weights=weights_all)
    
        num_events = t_res_all[(t_res_all > -10) & (t_res_all < 10)] #selecting POCAM events with 20ns width
        noise_events = t_res_all[(t_res_all < -10000) ^ (t_res_all > 10000)] #selecting noise events
    
    #Check to see if all events are considered or not 
        if t_res_all.size != self.abs_elim.size:
            print('T_RES_ALL.SIZE != ABS_ELIM.SIZE')
            print('T_RES_ALL.SIZE / ABS_ELIM.SIZE = ', t_res_all.size/self.abs_elim.size)

        return t_res_all,num_events.size, noise_events.size
    #returns
    #1)time residuals 
    #2)number of POCAM events for given width
    #3)number of noise hits
       
#requires no argument 
#plotting time residual for separate runs 
    def CheckPeak(self):
        plt.figure(figsize=(10,9))
        for b in range(0, len(self.peak_1)):
            plt.figure(figsize=(10,9)) #commenting this line out will produce the results in a single graph
            n, bins, patches = plt.hist(self.peak_1[b], 500, alpha = 0.4, log = True)
        plt.title(self.values + '-time residuals of threshold 1', fontsize = 19)
        plt.xlabel('time_ns', fontsize = 16)
        plt.ylabel('bincount', fontsize = 16)
            