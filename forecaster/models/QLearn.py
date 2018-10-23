
import scipy
import numpy as np
import time
from time import mktime
import datetime
import sys
import matplotlib.pyplot as plt
import csv
#Resources
#https://www.kaggle.com/carrie1/ecommerce-data/data
#https://archive.ics.uci.edu/ml/machine-learning-databases/00396/
#https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
#https://www.youtube.com/watch?v=eHqhJylvIs4&app=desktop


class QLearn:


    def __init__ (self,  bias=False, svd=False, window_size=24):
        self.trained = False
        self.X = [] # 2d array, first dimension is the data point, second dimension is the values of the data point
        self.Y = []
        

        self.regularization_rate = 3
        self.bias = bias
        self.pinv_time = 0
        self.svd = svd
        self.svd_time = 0
        self.window_size = window_size
    
    
    #----------------------- Training -------------------------

    def to_matrix (self, data):
        data = data.tolist()
        week_data = []
        input = []
        output = []
        for days in range (0, self.window_size):
            today = data[days]
            week_data = week_data+ today
        
        
        for days in range (self.window_size, len (data)):
            today = data[days]
            input.append (np.array(week_data))
            output.append (np.array(today))
            week_data = week_data+ today
            week_data = week_data [len (data[0]):]

        return (input, output)
            
    def set_encode (self, data):
        np_arr = np.array (data)
        transpose = np.transpose (np_arr)
        self.means = []
        self.stds = []
        for i in range (len (transpose)):
            mean = transpose[i].mean (axis=0)
            self.means.append (mean)
            transpose[i] -= mean
            std = transpose[i].std (axis=0)
            self.stds.append (std)
            transpose[i] /= std
        return np.transpose (transpose)
    
    def encode (self, data):
        transpose = np.transpose (data)
        for i in range (len (transpose)):
            transpose [i] -= self.means[i]
            transpose [i] /= self.stds [i]
        return np.transpose (transpose)

    def decode (self, np_arr):
        transpose = np.transpose (np_arr)
        for i in range (len (transpose)):
            transpose[i] *= self.stds[i]
            transpose[i] += self.means[i]
            return np.transpose (transpose)
    
    def set_training_data (self, data):
        data = self.set_encode (data)
        (x, y) = self.to_matrix (data)
        self.X = x
        self.Y = y


    def train (self, data):
        
        self.set_training_data (data=data)
        X = self.X
        Y = self.Y
        start_pinv = time.time()

        if not self.bias:
            X1 = np.array(X).transpose()

            Y1 = np.array(Y).transpose()
            #"""
            #    First implementation
            # Reliable and computationally faster than SVD
            inv = np.linalg.pinv(X1)

        else:
            X1 = np.matrix ([[1.0] + x for x in X]).transpose().tolist()
            Y1 = np.matrix (Y).transpose().tolist()
        
            np_X1 = np.matrix (X1)
            np_X1_transpose = np.matrix (X1).transpose()
            x_times_x_transpose = np.matmul (np_X1, np_X1_transpose)
            x_times_x_transpose = x_times_x_transpose.tolist()
        
            #print ("Completed x times x transpose")
        
            # Uncomment the following
            for row in range (0, len(x_times_x_transpose)):
                x_times_x_transpose [row][row] += self.regularization_rate
        
            inv = np.matmul (np.matrix(X1).transpose(), np.linalg.inv(np.matrix(x_times_x_transpose)))
        
        self.Z = np.matmul(Y1, inv)
        
        end_pinv = time.time()
        self.trained = True
        #print ("Z Calculation time: " + str (end_pinv - start_pinv))
        
        if self.svd:
            start_svd = time.time()
            self.decomp = np.linalg.svd (self.Z, full_matrices=False)
            end_svd = time.time ()
        
            U = self.decomp[0]
            singularValues = self.decomp [1]
            VT = self.decomp [2]
        
        
            # Do dimension reduction
            truncated = self.dimension_reduction (U, singularValues, VT, truncate=-1)
            self.truncated = (truncated[0], self.createSingularValuesMatrix (truncated[0], truncated[1], truncated[2]), truncated[2])
            print ("self.truncated")
            print ("U = (Length x width), Sigma = (Length x width), VT = (Length x width)")
            print ("U = ("+str(len(self.truncated[0].tolist()))+" x "+str(len(self.truncated[0].tolist()[0]))+")")
            print ("Sigma = ("+str(len(self.truncated[1].tolist()))+" x "+str(len(self.truncated[1].tolist()[0]))+")")
            print ("VT = ("+str(len(self.truncated[2].tolist()))+" x "+str(len(self.truncated[2].tolist()[0]))+")")
            print ("SVD Calculation time: " + str (end_svd - start_svd))
     
     
    #-------------------------- Prediction and testing ------------------
    def try_ktruncations (self, input, output):
        kstart = 1
        kend = 50
        
        total_l2_error = []
        total_l1_error = []
        total_time = []
        
        truncations = []
        for i in range (kstart, kend):
            trunc = self.dimension_reduction (self.decomp[0], self.decomp[1], self.decomp[2], i)
            truncations.append ((trunc[0],self.createSingularValuesMatrix (trunc[0], trunc[1], trunc[2]), trunc[2]))
            total_l2_error.append (0)
            total_l1_error.append (0)
            total_time.append (0)
        
        
        for i in range (len (output)):
            for b in range (kstart, kend):
                index = b-kstart
            
                time_start = time.time()
                inp = np.matrix ([input[i]]).transpose()
                mul1 = truncations[index][2] * inp
                mul2 = truncations[index][1]* (mul1)
                mul3 = truncations[index][0]* (mul2)
                prediction = np.matrix(mul3).transpose().getA()[0].tolist()
            
                time_end = time.time()
                
                total_time[index] += time_end-time_start

                total_l2_error[index] += self.l2_error (output[i], prediction)
                total_l1_error[index] += self.l1_error (output[i], prediction)
        
        print ("trunc\t\tL2 Err\t\tRMSD\t\tL1 Err\t\tAve Dev\t\tTime")
        
        for b in range (kstart, kend):
            index = b-kstart
        
            average_l2_error = total_l2_error[index] / len (output)
        
            rmsd = (average_l2_error / len (output[0]))**0.5
        
            average_l1_error = total_l1_error[index] / len (output)
        
            average_deviation = (average_l1_error / len (output[0]))
            
            average_time = (total_time[index]/len (input))
            
            print (str (b)+"\t\t" + str ("{:6.5f}".format(average_l2_error)) + "\t" + str ("{:6.5f}".format(rmsd)) + "\t\t" + str ("{:6.5f}".format(average_l1_error)) + "\t" + str ("{:6.5f}".format(average_deviation)) + "\t\t" + str ("{:6.5f}".format(average_time)))
        
        #coeffs = self.coeff_of_determination (output, predictions)
        #print ("Coefficients of determination: " + str (coeffs))
        
    
    
    def predict (self, input, horizon=1):
        input = self.encode (input.transpose()).transpose().tolist()
        if (self.trained):
            for i in range (horizon):
                if self.bias:
                    input = [1.0] + input
                
                if (self.svd == False):
                    pinv_start = time.time()
                    vec = np.matrix(self.Z * np.matrix (input).transpose()).transpose().getA()[0].tolist()
                
                    pinv_end = time.time()
                    self.pinv_time += pinv_end-pinv_start
                else:
                    svd_start = time.time()
                    vec = np.matrix(self.truncated[0]* (self.truncated[1]* (self.truncated[2] * np.matrix ([input]).transpose()))).transpose().getA()[0].tolist()
                
                    svd_end = time.time()
                    self.svd_time += svd_end-svd_start
            
            
                #print (input)
                if self.bias:
                    input[0].pop (0)
                for i in range (len (vec)):
                    input[0].pop(0)
                    input[0].append (vec [i])
                #print (input)
            vec = self.decode (vec)
            return vec
    
        else:
            return 0
    


    def test_model (self, input, output, verbose=True, horizon=1, times=None):
        total_l2_error = 0
        total_l1_error = 0
        predictions = []
        
        for i in range (len (output)):
            prediction = self.predict (input[i], horizon=horizon)
            predictions.append (prediction)
            err = self.l2_error (output[i], prediction)
            total_l2_error += err
            err = self.l1_error (output[i], prediction)
            total_l1_error += err
            if verbose:
                print (str (i+1)+ ") ")
                print ("\tActual\t\tPredicted")
                for prod in range (0, len (output[i])):
                    print ("\t"+str(output [i][prod])+"\t\t"+str(prediction[prod]))
                #print ("Actual:    " + str(output[i]))
                #print ("Predicted: "+ str(prediction))
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

        #coeffs = self.coeff_of_determination (output, predictions)
        #print ("Coefficients of determination: " + str (coeffs))

        print ("[TP, FP, FN, TN]")
        cntng_table = self.contingency_table (output, predictions)
        print (cntng_table[0])

        print ("Average pseudoinverse time: " + str (self.pinv_time/len (input)))

        if self.svd:
            print ("Average SVD time: " + str (self.svd_time/ len (input)))

        with open ("dev.txt", "a")as f:
            f.write (str(average_deviation)+"\n")
        
        
        #if not times is None:
        #    print (times)
        #times = [time.struct_time(i).strftime("%Y-%m-%d %H:%M:%S") for i in times]

        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        with open (timestr+"prediction.csv", "w") as f:
            csvwriter = csv.writer (f)
            i = 0
            for row in predictions:
                if times is None:
                    csvwriter.writerow (row)
                else:
                    #print (times[i])
                    #print (row)
                    csvwriter.writerow ( [times[i]] + row)
                i +=1
        with open (timestr +"actual.csv", "w") as f:
            csvwriter = csv.writer (f)
            i = 0
            for row in output:
                if times is None:
                    csvwriter.writerow (row)
                else:
                    arr = [times[i]]
                    arr.extend(row)
                    csvwriter.writerow ( arr)
                i +=1

        timesteps = range (0, len (predictions))
        if not times is None:
            timesteps = times
            #print (timesteps)
        
        #"""
        
        arr1 = [row[0] for row in predictions]
        #print (arr1)
        arr2 = [row[0] for row in output]
        plt.plot (np.array(timesteps), arr2, 'b', label='Actual')
        plt.plot (np.array(timesteps), arr1, 'r', label='Prediction')
        plt.xlabel ("Time")
        plt.ylabel ("Normalized Data")
        plt.title ("Electricity Dataset Matrix Factorization Predictions")
        plt.legend()
        plt.show()
        """
        #"""
     
     
    #------------------------ SVD --------------------------
    
    def createSingularValuesMatrix (self, U, sigma, VT):
        numCols = len(VT)   # number of columns is equal to the number of rows of VT
        numRows = len(U[0])       # number of rows is equal to the number of columns of U

        arr = [[0 for col in range (numCols)] for row in range (numRows)]
    
        smaller = numCols
        if numRows < smaller:
            smaller = numRows
        for i in range (smaller):
            arr[i][i] = sigma[i]
        return np.array(arr)

    def find_inverse (self, U, sigma, VT):
        # Finding the pseudoinverse using the singular value decomposition
        # The pseudoinverse using an SVD is equal to
        # V * sigma^-1 * UT
        V = np.matrix(VT).transpose().getA()
        UT = np.matrix(U).transpose().getA()
        numCols = len (UT)      # number of columns is equal to the number of rows of UT
        numRows = len (V[0])       # number of rows is equal to the number of columns of V
    
        arr = [[0 for col in range (numCols)] for row in range (numRows)]
    
        smaller = numCols
        if numRows < smaller:
            smaller = numRows
        for i in range (smaller):
            if (sigma[i] != 0): # Avoid Divide by zero
                arr[i][i] = 1/sigma[i]
            #arr [i][i] = 1/sigma[i]
        # returns V, sigma^-1, UT
        return (V, arr, UT)

    # dimensionality reduction
    # Preconditions: U, sigma, VT to be numpy arrays
    # Postconditions: numpy arrays that are truncated size
    def dimension_reduction (self, U, sigma, VT, truncate=-1):
        U = U.tolist()
        sigma = sigma.tolist()
        VT = VT.tolist()
        
        if truncate == -1:
            ktruncate = 0
            while (ktruncate < len (sigma) and sigma[ktruncate] > 0.01):
                ktruncate+= 1
        else:
            ktruncate = truncate
        
        # U will become an (m by r) size matrix (m rows and r columns)
        # sigma will become an (r by r) size matrix (r rows and r columns)
        # VT will become a (r by n) size matrix (r rows and n columns)

        Utruncated = [[U[row][col] for col in range (ktruncate)] for row in range (len(U))]

        VTtruncated = [[VT[row][col] for col in range (len (VT[row]))] for row in range (ktruncate)]

        """
        arr = [[0 for col in range (ktruncate)] for row in range (ktruncate)]
        for i in range (len(ktruncate)):
            arr[i][i] = sigma[i]
        """
        return (np.array(Utruncated), np.array(sigma), np.array(VTtruncated))
    
    
    def print_concepts (self, number_of_concepts=1):
        print (self.truncated[1])
        """
        rank = len (self.truncated[2]) # The rank is equal to the number of rows in V transpose
        print ("Rank = " + str(rank))
        
        num = number_of_concepts
        if (rank < num):
            num = rank
        
        for i in range (num):
            Ucol = np.matrix([row[i] for row in self.truncated[0]]).transpose()
            sigma = self.truncated [1][i]
            VTrow = self.truncated [2][i]
            concept = np.matrix(Ucol) * np.matrix (VTrow)
            concept = np.array(concept) * sigma
            
            print ("Concept " + str (i) + ") with sigma = " + str (sigma))
            print (concept.tolist())
            print ("")
            print ("")
        """
    


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


    """
        Returns an R^2 term that shows how much the model explains the variance
    """
    def coeff_of_determination (self, real, prediction):

        #sums_of_columns_real = [ sum(x) for x in zip(*real) ]
        #sums_of_columns_prediction = [ sum(x) for x in zip(*prediction) ]
        #print (prediction)
        
        r_squared = []
        for i in range (0, len (real[0])):
            col_real = [row[i] for row in real]
            col_prediction = [row[i] for row in prediction]

            mean = self.mean(col_real)
            #ss_total = self.variance (col_real) * len (col_real)
            
            ss_total = 0
            ss_reg = 0
            for testnum in range (0, len(col_prediction)):
                ss_total += (col_real[testnum] - mean) ** 2
                ss_reg += (col_prediction[testnum] - mean) ** 2

            if (ss_total == 0):
                r_squared.append (-1)
            else:
                r_squared.append (ss_reg/ss_total)
        return r_squared





