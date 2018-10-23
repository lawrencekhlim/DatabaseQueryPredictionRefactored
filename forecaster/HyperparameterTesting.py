import csv
from models.QLearn import QLearn
from models.NaiveModel import NaiveModel
from models.EarliestModel import EarliestModel
from models.AverageModel import AverageModel
from models.RNNModel import RNNModel

import gc
import numpy as np
import keras.layers as L
#from keras.layers.advanced_activations import LeakyReLU
import keras.models as M
import keras.optimizers as O


DATA_DIR = "../data/timeseriesOnlineRetailCleaned2.csv"

class QOnlineRetail:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.6)
        self.validation = (0.6, 0.8)
        self.testing = (0.8, 0.95)
        #self.data_size = 24
        self.data_size = 7
        
        self.num_queries = 200
        # q = 1, epochs = 100
        # q = 5, epochs = 20
        # q = 100, epochs = 10
        # q = 2000, epochs = 3
        
        with open (DATA_DIR, 'r') as f:
        #with open ('hourlyTimeSeriesOnlineRetailCleaned.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            title_row = True
            for row in reader:
                if title_row:
                    self.products = row
                    title_row = False
                else:
                    integer_data = [int(row[i]) for i in range (self.num_queries)]
                    self.data.append (integer_data)

        self.baseline2 = EarliestModel()
        #self.predictor3 = RNNModel(self.num_queries)
                

    
    def test_hyperparameter (self):
        
        input = []
        output = []
        """
        week_data = []
        for days in range (int (360*self.training[0]), int (360*self.training[0])+self.data_size):
            today = self.data[days]
            week_data = week_data+ today
        
        
        for days in range (int (360*self.training[0])+self.data_size, int(360*self.training[1])):
            today = self.data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            for i in range (len (self.data[0])):
                week_data.pop (0)
        #self.predictor1.set_training_data (input, output)
        """
        
        self.x_data = self.data[:-1]
        self.y_data = self.data[1:]
        


        train_length = int(self.training[1]*len (self.y_data)) - int (self.training[0]*len (self.y_data))
        x_train = np.zeros ((1, train_length, self.num_queries)) # 1 example, number of time steps, number of queries
        for i in range (int(self.training[0] * len (self.x_data)), int(self.training[0] * len (self.x_data)) + train_length):
            for q in range (self.num_queries):
                x_train[0][i][q] = self.x_data[i][q] # sets the query at the time step to either 0 or 1
        
        
        y_train = []
        for q in range (self.num_queries):
            y_train.append(np.zeros ((1, train_length, 2)))
            for i in range (int(self.training[0]*len (self.y_data)), int(self.training[0] * len (self.x_data)) + train_length):
                y_train[q][0][i][self.y_data[i][q]] = 1
        
        
        
        #Validation
        val_length = int (self.validation[1]*len (self.x_data)) - int (self.validation[0]*len (self.x_data))

        x_val = np.zeros ((1, val_length, self.num_queries))
        for i in range (int(self.validation[0] * len (self.x_data)), int(self.validation[0] * len (self.x_data)) + val_length):
            for q in range (self.num_queries):
                x_val[0][i-int (self.validation[0]*len (self.x_data))][q] = self.x_data[i][q]

        y_val = []
        for q in range (self.num_queries):
            y_val.append (np.zeros ((1, val_length, 2)))
            for i in range (int(self.validation[0]*len (self.y_data)), int(self.validation[0] * len (self.x_data)) + val_length):
                y_val[q][0][i-int(self.validation[0]*len (self.y_data))][self.y_data[i][q]] = 1
        
        self.xtrain = x_train
        self.ytrain = y_train
        self.xval = x_val
        self.yval = y_val
        
        
        #Hyperparameters:
        rnn_types = ["GRU", "SimpleRNN", "LSTM"]
        optimizers = ["adam"]
        learning_rates = [0.001]
        hidden_sizes = [32, 64, 16, 8, 4]
        layers = [4, 2, 1]

        for rnn in rnn_types:
            for optimizer in optimizers:
                for learning_rate in learning_rates:
                    for layer in layers:
                        for hs in hidden_sizes:
                            print (rnn)
                            print (optimizer)
                            print ("learning rate = " + str (learning_rate))
                            print ("layers = " + str (layer))
                            print ("hidden states = " + str (hs))
                            model = RNNModel (self.num_queries, rnn_type=rnn, optimizer_type=optimizer, learning_rate=learning_rate, layers=layer, hidden_size=hs)
                            model.train (x_train, y_train, x_val, y_val)
                            model.delete_model()
                            model = None
                            gc.collect()




if __name__== "__main__":
    test = QOnlineRetail ()
    
    #test.clean_data()

    test.test_hyperparameter()
