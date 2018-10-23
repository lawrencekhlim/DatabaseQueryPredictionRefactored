import csv
from models.QLearn import QLearn
from models.AutoRegression import AutoRegression

from Runner import Runner
from Visuals import Visuals
import numpy as np
import sys


class RunnerPrediction:
    def __init__ (self, runner, window_size=24):
        self.data = []
        self.training = (0, 0.5)
        self.validation = (0.5, 0.8)
        self.testing = (0.8, 0.95)
        self.data_size = window_size
        self.num_queries = num_queries
        self.ds = runner
        self.data = self.ds.get_data()
        self.dates = self.ds.get_times()
        self.num_queries = self.ds.get_num_queries()


    
    def new_data (self):
        # partitioning the data into multiple parts
        data = self.data
        self.training_data = data[int (self.training[0]* len (data)): int(self.training[1]* len (data))]
        self.validation_data = data[int (self.validation[0]* len (data)): int(self.validation[1]* len (data))]
        self.testing_data = data[int (self.training[0] * len (data)): int(self.training[1] * len (data))]
        
        self.val_dates = self.dates [int (self.validation[0]* len (data)): int(self.validation[1]* len (data))]
        
        self.predictor1 = AutoRegression(bias=True)

    
    
    def validate_predictor (self, horizon=1):

        validation_data = self.validation_data.tolist()
        
        print ()
        print ("Linear Algebra Model")
        (p_path, a_path) = self.predictor1.test_model (data=self.validation_data, horizon=horizon, verbose=False, times=self.val_dates, title_row=self.ds.get_title_row())
        self.p_path = p_path
        self.a_path = a_path

    
    def train_data (self):

        training_data = self.training_data.tolist()
        
        print ("")
        print ("Window size: " + str (self.data_size))
        print ("")
        print ("Training Model...")

        self.predictor1.train(training_data)
        
        print ("... Done Training")

    def graph_interface (self):
        v = Visuals ()
        v.load_data (self.p_path, self.a_path, has_title=True)
        i = int (input ("Enter query number (0-"+str (self.num_queries-1)+", else to quit): "))
        while (i >= 0 and i < self.num_queries):
            v.graph (i)
            i = int (input ("Enter query number (0-"+str (self.num_queries-1)+", else to quit): "))


if __name__== "__main__":
    if (len (sys.argv) < 3):
        print ("Usage: python3 forecaster/RunnerPrediction.py <path-to-data-file> <window-size>")
    
    else:
        DATA_DIR = sys.argv [1]
        WINDOW_SIZE = sys.argv [2]
        runner = Runner (DATA_DIR)
        TOTAL_QUERIES = runner.get_num_queries()
        num_queries = runner.get_num_queries()
        
        test = RunnerPrediction (runner, WINDOW_SIZE)
        test.new_data ()    # partition the data into training data, validation data, testing data
        test.train_data()   # train the model
        test.validate_predictor(horizon=1) # test the model on validation data
        test.graph_interface()  # graph results from testing from validation data

