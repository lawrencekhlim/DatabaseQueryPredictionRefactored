
#Format of data:
The data must be formatted in a CSV such that the each column signifies a query and each row signifies one time period. A value at a specific row and specific column should refer to the popularity of the column's query at row's time. The top row should list query names and the left-most column should list times.

Currently times must be listed in this format:
%Y-%m-%d %H:%M:%S
but by modifying the code in Runner.py, you can make it handle any format.



Database Query Prediction

The data can be placed inside the data folder. It can be referenced by passing its path as a command line argument to the main program.

#Usage:
Usage: python3 forecaster/RunnerPrediction.py <path-to-data-file> <window-size>


The main program requires the user to specify a legal path to a data file and a window size for autoregression to slide over. For instance,

python3 forecaster/RunnerPrediction.py data/words.csv 24
python3 forecaster/RunnerPrediction.py data/words2.csv 24

are legal.



This program serves as a template for how to use the Autoregression class. The flow should generally follow the same path: 
Load data from a source
Partition the data into training sets, validation datasets (skip this step if you are using it to improve a database)
Train a model using the training set
Make predictions or test it.


The main program will also prompt the user for which query he wants to graph at the end.


As this has been skimmed down, it only contains autoregression as its predictor.