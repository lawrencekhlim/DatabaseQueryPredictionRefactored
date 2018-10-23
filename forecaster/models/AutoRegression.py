
import scipy
import numpy as np
import time
from time import mktime
import datetime
import sys
import matplotlib.pyplot as plt
import csv
from .QLearn import QLearn
#Resources
#https://www.kaggle.com/carrie1/ecommerce-data/data
#https://archive.ics.uci.edu/ml/machine-learning-databases/00396/
#https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
#https://www.youtube.com/watch?v=eHqhJylvIs4&app=desktop


class AutoRegression:


    def __init__ (self, bias=False, window_size=7):
        self.trained = False
        self.X = []
        self.Y = []
        
        self.pinv_time = 0
        self.window_size = window_size
    
    #----------------------- Training -------------------------


    def train (self, data):

        self.output_size = len (data[0])
        
        self.regressions = []
        for i in range (self.output_size):
            model = QLearn (window_size=self.window_size)
            model.train(data=np.array([np.transpose(data)[i]]).transpose())
            self.regressions.append(model)

    #-------------------------- Prediction and testing ------------------
    
    def predict (self, input, horizon=1):
        output = []
        for i in range (len (input[0])):
            input_val = [input[col][i] for col in range (0, self.window_size)]
            output.append (self.predict_one_query(input_val, i, horizon=horizon))
        return output
    
    def predict_one_query (self, input, query_num, horizon=1):
        input_val = np.array ([input])
        return self.regressions[query_num].predict (input_val, horizon=horizon)[0]

    def test_model (self, data, horizon=1, verbose=False, times=None, title_row=None):
        
        
        total_l2_error = 0
        total_l1_error = 0
        predictions = []
        output = []
        
        
        for i in range (len (data)-self.window_size-horizon):
            prediction = self.predict (data[i:i+self.window_size], horizon=horizon)
            predictions.append (prediction)
            actual = data[i+self.window_size+horizon]
            output.append (actual)
            err = self.l2_error (actual, prediction)
            total_l2_error += err
            err = self.l1_error (actual, prediction)
            total_l1_error += err
            
            if verbose:
                print (str (i+1)+ ") ")
                print ("\tActual\t\tPredicted")
                for prod in range (0, len (output[i])):
                    print ("\t"+str(output [i][prod])+"\t\t"+str(prediction[prod]))
                print ("L2 Error:  " + str (err))
                print ("Std Dev:   " + str ((err/len (output[i])) ** (0.5)))
                print ("")

        total_l2_error = total_l2_error / len (output)
        print ("Average Error L2: " + str (total_l2_error))
        
        rmsd = (total_l2_error / len (output[0]))**0.5
        print ("RMSD: " + str(rmsd))
        
        total_l1_error = total_l1_error / len (output)
        print ("Average Error L1: " + str (total_l2_error))
        
        average_deviation = (total_l1_error / len (output[0]))
        print ("Average Deviation per Query: " + str(average_deviation))

        
        
        #if not times is None:
        #    print (times)
        #times = [time.struct_time(i).strftime("%Y-%m-%d %H:%M:%S") for i in times]

        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        prediction_path = "results/"+timestr+"prediction.csv"
        with open (prediction_path, "w") as f:
            csvwriter = csv.writer (f)
            if not title_row == None:
                csvwriter.writerow (title_row)
            i = 0
            for row in predictions:
                if times is None:
                    csvwriter.writerow (row)
                else:
                    #print (times[i])
                    csvwriter.writerow ( [times[i]] + row)
                i +=1
        actual_path = "results/" + timestr +"actual.csv"
        with open (actual_path, "w") as f:
            csvwriter = csv.writer (f)
            if not title_row == None:
                csvwriter.writerow (title_row)
            i = 0
            for row in output:
                if times is None:
                    csvwriter.writerow (row)
                else:
                    arr = [times[i]]
                    arr.extend(row)
                    csvwriter.writerow ( arr)
                i +=1

        return (prediction_path, actual_path)



    #---------------------- Error Metrics -----------------------
    def l1_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += abs(real[i] - prediction[i])
        return error
    def l2_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += (real[i] - prediction[i]) ** 2
        return error
    
    def mean (self, list):
        return sum (list) / len (list)
    
    def variance (self, list):
        mean = self.mean (list)
        variance = 0
        for value in list:
            variance += (value - mean) ** 2
        return variance / len(list)



