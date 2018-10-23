import csv
import matplotlib.pyplot as plt
import numpy as np
from time import mktime
from datetime import datetime
import time

class Visuals:



    def load_data (self, prediction_path, actual_path, has_title=False):
        self.title_row = None
        self.times_arr = []
        self.predictions = []
        self.actual = []
        with open (prediction_path, "r") as f:
            reader = csv.reader (f)
            ctr = 0
            for row in reader:
                if ctr == 0 and has_title:
                    self.title_row = row
                else:
                    self.times_arr.append (datetime.fromtimestamp(mktime(time.strptime(row[0], '%Y-%m-%d %H:%M:%S'))))
                    self.predictions.append ([float (i) for i in row[1:]])
                ctr+=1
        with open (actual_path, "r") as f:
            reader = csv.reader (f)
            ctr = 0
            for row in reader:
                if ctr != 0 or not has_title:
                    self.actual.append ([float (i) for i in row[1:]])
                ctr+=1
    
    def graph (self, query_num):
        
        arr1 = [row[query_num] for row in self.predictions]
        arr2 = [row[query_num] for row in self.actual]
        plt.plot (np.array(self.times_arr), arr2, 'b', label='Actual')
        plt.plot (np.array(self.times_arr), arr1, 'r', label='Prediction')
        plt.xlabel ("Time")
        plt.ylabel ("Popularity")
        plt.title ("Autoregression Predictions")
        if not self.title_row == None:
            plt.title ("Autoregression Predictions for " +self.title_row[query_num+1])
        plt.legend()
        plt.show()


