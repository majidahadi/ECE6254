import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from lstm import generate_data, lstm_model, generate_stock_data
from readcsv import ReadFromCSVFile


LOG_DIR = './ops_logs/sin'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 500
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 1000

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = "/",title = "Select a stock csv file",filetypes = (("csv files","*.csv"),("all files","*.*")))
#filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),model_dir=LOG_DIR)
#regressor = learn.Estimator(model_fn=None,model_dir=LOG_DIR)

#X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)

#datafromcsv = ReadFromCSVFile('D:\Karnik\Graduate Studies\Statistical Machine Learning\Project\data\ACAS.csv')
datafromcsv = ReadFromCSVFile(file_path)
datafromcsv.reverse()

d3=np.array(datafromcsv)

data=np.zeros((datafromcsv.__len__()-1,2))
data[:,0]= np.arange(0,datafromcsv.__len__()-1,1)
data[:,1]=d3[:-1,1]

X,y = generate_stock_data(data[:,1],TIMESTEPS)#,seperate=True)


# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                     every_n_steps=PRINT_STEPS,
                                                     early_stopping_rounds=1000)
# print(X['train'])
# print(y['train'])
xlabels = np.arange(0,len(X['train']),1)


regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

predictor = regressor.predict(X['test'])


predicted = [predictor.__next__() for i in range(len(X['test']))]
#rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
#score = mean_squared_error(predicted, y['test'])
#print ("MSE: %f" % score)


fig, ax1 = plt.subplots()
plot_predicted,=ax1.plot(predicted,'r-', label='Predicted')#ax1.plot(xlabels,predicted,'r-')
ax1.set_ylabel('Predicted', color='r')

ax2 = ax1.twinx()
plot_test, = ax2.plot(y['test'],'b-', label='Actual')#ax2.plot(xlabels,y['test'], 'b.')
ax2.set_ylabel('Actual', color = 'b')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
