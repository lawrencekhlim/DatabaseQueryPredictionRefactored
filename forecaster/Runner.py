import csv
import time
from time import mktime
from datetime import datetime
import numpy as np

class Runner:
    
    def __init__ (self, path="../data/electricity.csv"):
        self.path = path
        f = open (path, "r")
        reader = csv.reader (f)
    
        self.title_row = next (reader)
        f.close ()
        self.np_arr = None
    
    def get_data (self, training=(0,0.5)):
        if self.np_arr == None:
            f = open (self.path, "r")
            reader = csv.reader (f)
            
            ctr = 0
            self.title_row = None
            self.times_arr = []
            arr = []
            for row in reader:
                if ctr == 0:
                    self.title_row = row
                else:
                    self.times_arr.append (datetime.fromtimestamp(mktime(time.strptime(row[0], '%Y-%m-%d %H:%M:%S'))))
                    arr.append ([float (i.replace (",", ".")) for i in row[1:]])
                ctr+=1
            f.close ()
        
            np_arr = np.array (arr)
            self.np_arr = np_arr
            
            # Below is not commented out and is necessary if there are queries with no standard deviation.
            # In such a case with zero std dev, then the linear algebra does not check out (and may cause the program
            # to crash.)
            #"""
            transpose = np.transpose (np_arr)
            transpose2 = np.transpose (np_arr)
            self.means = []
            self.stds = []
            for i in range (len(transpose)-1, -1, -1):
                mean = transpose[i].mean (axis=0)
                self.means.append (mean)
                transpose[i] -= mean
                std = transpose[i].std (axis=0)
                self.stds.append (std)
                transpose[i] /= std
                
                training_std = transpose[i][int (training[0]* len (transpose[i])): int(training[1]* len (transpose[i]))].std (axis=0)
                
                #print (str (i) + " " + str (training_std))
                
                if training_std < 0.01:
                    transpose2 = np.delete (transpose2, i, axis=0)
                    print (str (i) + " " + str (training_std))
            self.np_arr = np.transpose (transpose2)
            """
            #"""
            
        return self.np_arr

    def get_times (self):
        return self.times_arr

    def get_num_queries (self):
        return len (self.title_row) -1

    def get_title_row (self):
        return self.title_row
