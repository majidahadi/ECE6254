import numpy as np
import pywt as wavelet
from readcsv import ReadFromCSVFile
import time
from keras.models import load_model
from pandas import DataFrame
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from keras.callbacks import TensorBoard
import os

from helper_fns import * #__all__


#LOG_DIR = '.'
TIMESTEPS = 32
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 1000 #1000 for LSTM option 1 and 300 for LSTM option 0 with SINE wave input
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100
TOTAL_OUTPUT_LEN = 4#TIMESTEPS
overlap = 3  # TIMESTEPS - 1
denorm_reference_point = overlap-1#0
shift_ref_frame_for_denor = True
use_norm_on_input = False
use_norm_on_output = True
LSTM_network_selection = 0
training_data_shuffle = True
plot_len=2
plot_step=10

training_size = 0.8
test = 0.2
validation_size = 0.2  # 20% of the training size

root = tk.Tk()
root.withdraw()

file_paths = 'D:/Karnik/Graduate Studies/Statistical Machine Learning/Project/data/a.csv'
images_dir = 'D:/Karnik/Graduate Studies/Statistical Machine Learning/Project/Try6/tmp'


root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames(initialdir="D:\Karnik\Graduate Studies\Statistical Machine Learning\Project\data", title="Select a stock csv file",
                                         filetypes=(("csv files", "*.csv"), ("all files", "*.*")))

root = tk.Tk()
root.withdraw()

images_dir = filedialog.askdirectory(initialdir="D:\Karnik\Graduate Studies\Statistical Machine Learning\Project\Try5\tmp", title="Select a directory to store plot images")


html_start_string = """ <!DOCTYPE html>
                            <html>
                            <body>"""
html_end_string = """</body>
                        </html>"""
#html_RNN_Config_string=''
#html_RNN_Config_string=(["""<h2>""",'TIMESTEPS = ',str(TIMESTEPS)])
html_RNN_Config_string='  '.join(["""<h2>""",'TIMESTEPS = ',str(TIMESTEPS),
                             'TRAINING_STEPS = ' , str(TRAINING_STEPS) ,
                              'Batch Size= ', str(BATCH_SIZE),
                             'TOTAL_OUTPUT_LEN = ' , str(TOTAL_OUTPUT_LEN) ,
                             'Overlap = ' , str(overlap)  ,
                             'Denorm Pos = ', str(denorm_reference_point),
                              'Normalize output= ', str(use_norm_on_output),
                            'Normalize input= ', str(use_norm_on_input),
                              'LSTM Network Selection = ', str(LSTM_network_selection),
                               """</h2>"""])

html_filename = images_dir + '/datadump' + '.html'

f = open(html_filename, 'a')

f.write(html_start_string)
f.write(str(html_RNN_Config_string))
f.close()
model = build_model([TIMESTEPS, 50, 50, TOTAL_OUTPUT_LEN],LSTM_network_selection)

for i in range(0, file_paths.__len__()):
    f = open(html_filename, 'a')
    file_path = file_paths[i]
    # filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    company_name = file_path.split('/')[-1].split('.')[0]
    # LOG_DIR = LOG_DIR + file_path.split('/')[-1].split('.')[0]
    LOG_DIR = images_dir + '/' + company_name + '/log'


    #datafromcsv = ReadFromCSVFile('D:\Karnik\Graduate Studies\Statistical Machine Learning\Project\data\ACAS.csv')
    datafromcsv = ReadFromCSVFile(file_path)
    datafromcsv.reverse()

    d3 = np.array(datafromcsv)

    data = np.zeros((datafromcsv.__len__() - 1, 2))
    data[:, 0] = np.arange(0, datafromcsv.__len__() - 1, 1)
    data[:, 1] = d3[:-1, 1]
    data1 = np.zeros((data.shape[0], 2))
    data1 = data[:, 1]
    data1 = data1/data1.max()
    # data1=np.arange(1,1000,1)
    # data1= data1/data1.max()

    split_data = []
    split_y = []

    for i in range(0, data1.shape[0] - TIMESTEPS - TOTAL_OUTPUT_LEN):
        split_data.append(data1[i:i + TIMESTEPS])
        if (TOTAL_OUTPUT_LEN == 1):
            # split_y.append(data1[i+1: TIMESTEPS+i+1])
            split_y.append(data1[TIMESTEPS + i + 1])
        else:
            # split_y.append(data1[i + 1: TOTAL_OUTPUT_LEN + i + 1])

            starting_pt = i + TIMESTEPS - overlap
            split_y.append(data1[starting_pt: starting_pt + TOTAL_OUTPUT_LEN])

    #train_x, val_x, test_x = prepare_data(np.asarray(split_data), training_size, validation_size, BATCH_SIZE, TIMESTEPS, True)
    #train_y, val_y, test_y = prepare_data(np.asarray(split_y), training_size, validation_size, BATCH_SIZE, TIMESTEPS, True)

    train_x, val_x, test_x = prepare_data(split_data, training_size, validation_size, BATCH_SIZE, TIMESTEPS,
                                          use_norm_on_input)
    train_y, val_y, test_y = prepare_data(split_y, training_size, validation_size, BATCH_SIZE, TIMESTEPS,
                                          use_norm_on_output)

    # x,y = generate_data_XY_with_val(np.asarray(split_data),np.asarray(split_y),TIMESTEPS,TIMESTEPS)

    train_x = conv_to_wavelet(train_x,0,'db1')
    val_x = conv_to_wavelet(val_x, 0, 'db1')
    test_x=conv_to_wavelet(test_x, 0, 'db1')

    #model = build_model([TIMESTEPS, 50, 50, TOTAL_OUTPUT_LEN],LSTM_network_selection)

    tr_y2, va_y2, te_y2 = prepare_data(split_y, training_size, validation_size, BATCH_SIZE, TIMESTEPS, False)

    mypath = LOG_DIR
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    TbCallback = TensorBoard(LOG_DIR,5,True,True)
    callbacklist = [TbCallback]
    History = model.fit(reshape_for_rnn(train_x, model.input_shape),
                        reshape_for_rnn(train_y, model.output_shape),
                        epochs=TRAINING_STEPS,
                        batch_size=BATCH_SIZE,
                        shuffle= training_data_shuffle,
                        verbose=2,
                        callbacks=callbacklist,#TbCallback,
                        validation_data=(
                            reshape_for_rnn(val_x, model.input_shape),
                            reshape_for_rnn(val_y, model.output_shape),
                            )

                        )

    # p=model.predict(test_x)
    modelfilename = images_dir + '/' + company_name + '.hd5'
    model.save(modelfilename)

    p = model.predict(reshape_for_rnn(test_x, model.input_shape))


    if (use_norm_on_output==True):

        #if shift_ref_frame_for_denor== True:
        #    for j in range(0,te_y2.shape[0]):
        #        te_y2[j]= te
        if overlap > 0:
            actual_prediction = denormalise_windows(p, te_y2,denorm_reference_point=denorm_reference_point)
        else:
            actual_prediction = p
    else:
        actual_prediction = p
    img2 = images_dir + '/' + company_name + '_pred_vs_act.png'


    if LSTM_network_selection ==0:
        actual_prediction_for_graphs =actual_prediction[:, 0, -1]
    elif LSTM_network_selection == 1:
        actual_prediction_for_graphs = actual_prediction[:, -1]

    plot_predicted_vs_actual(actual_prediction_for_graphs,
                             te_y2[:, -1],
                             img2)
    img2 = images_dir + '/' + company_name + '_slopes.png'

    plot_results_multiple(p,te_y2,TOTAL_OUTPUT_LEN,plot_len= plot_len,
                          plot_step=plot_step,offset = 0.005,
                          image_filename_with_path=img2)
    pred = DataFrame({'Prediction': actual_prediction_for_graphs})
    ty = DataFrame({'Actual': te_y2[:, -1]})
    combined_data = pred.join(ty, lsuffix='_Predcition', rsuffix='_Actual')
    combined_data.to_csv(images_dir + '/' + company_name + '.csv')
    pred = None
    ty = None


    company_name_text = 'Company = ' + company_name
    html_add_fig_string = """<h2>""" + company_name_text + """</h2>
        <img src=" """ + company_name + '_slopes.png' + """ " alt="AA" style="width:304px;height:228px;">
        <img src=" """ + company_name + '_pred_vs_act.png' + """ " alt="AA" style="width:304px;height:228px;">\n"""
    html_RNN_Result_string = '  '.join(["""<h3>""", 'Training Loss = ', str(History.history['loss'][-1]),
                                        'Training Accuracy = ', str(History.history['acc'][-1]),
                                        'Validation Loss = ', str(History.history['val_loss'][-1]),
                                        'Validation Accuracy = ', str(History.history['val_acc'][-1]),
                                        """</h3>"""])
    f.write(html_add_fig_string)
    f.write(html_RNN_Result_string)
    #del model
    model.reset_states()
    History = None
    print('Done')

    f.write(html_end_string)
    f.close()

print('Done')

'''
plt.close('all')
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Actual')
    # plt.plot(te_y2[0,400:500,-1])
    plt.plot(te_y2[0, :, -1])
    plt.subplot(2, 1, 2)
    plt.title('Prediction')
    # plt.plot(actual_prediction[399:499,-1])
    plt.plot(actual_prediction[:, -1])
    plt.show()
'''
