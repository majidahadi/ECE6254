import numpy as np
import pywt as wavelet
from readcsv import ReadFromCSVFile
import time
from keras.models import load_model
from pandas import DataFrame
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from helper_fns import * #__all__


#LOG_DIR = '.'
TIMESTEPS = 5
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 4500
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 50
PREDICTION_LEN = 3#TIMESTEPS
overlap = 2  # TIMESTEPS - 1
denorm_reference_point = 1

training_size = 0.8
test = 0.2
validation_size = 0.2  # 20% of the training size

root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames(initialdir="D:\Karnik\Graduate Studies\Statistical Machine Learning\Project\data", title="Select a stock csv file",
                                         filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
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
                             'PREDICTION_LEN = ' , str(PREDICTION_LEN) ,
                             'Overlap = ' , str(overlap)  ,
                             'Denorm Pos = ', str(denorm_reference_point),
                                  """</h2>"""])

html_filename = images_dir + '/datadump' + '.html'

f = open(html_filename, 'a')

f.write(html_start_string)
f.write(str(html_RNN_Config_string))

for i in range(0, file_paths.__len__()):
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
    # data1=np.arange(1,1000,1)
    # data1= data1/data1.max()

    split_data = []
    split_y = []

    for i in range(0, data1.shape[0] - TIMESTEPS - PREDICTION_LEN):
        split_data.append(data1[i:i + TIMESTEPS])
        if (PREDICTION_LEN == 1):
            # split_y.append(data1[i+1: TIMESTEPS+i+1])
            split_y.append(data1[TIMESTEPS + i + 1])
        else:
            # split_y.append(data1[i + 1: PREDICTION_LEN + i + 1])

            starting_pt = i + TIMESTEPS - overlap
            split_y.append(data1[starting_pt: starting_pt + PREDICTION_LEN])

    #train_x, val_x, test_x = prepare_data(np.asarray(split_data), training_size, validation_size, BATCH_SIZE, TIMESTEPS, True)
    #train_y, val_y, test_y = prepare_data(np.asarray(split_y), training_size, validation_size, BATCH_SIZE, TIMESTEPS, True)

    train_x, val_x, test_x = prepare_data(split_data, training_size, validation_size, BATCH_SIZE, TIMESTEPS,
                                          True)
    train_y, val_y, test_y = prepare_data(split_y, training_size, validation_size, BATCH_SIZE, TIMESTEPS,
                                          True)

    # x,y = generate_data_XY_with_val(np.asarray(split_data),np.asarray(split_y),TIMESTEPS,TIMESTEPS)

    model = build_model([TIMESTEPS, 50, 50, PREDICTION_LEN])
    # model.add(Dense(TIMESTEPS))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')




    # working one
    # model.fit(train_x, train_y, epochs=TRAINING_STEPS, batch_size=50, verbose=2,validation_data=(val_x, val_y))

    # model=load_model('sp500_csv_ts30_over_29_bs50_tr26_val21.hd5')
    # train_x=reshape_for_rnn(train_x)

    History = model.fit(reshape_for_rnn(train_x, model.input_shape),
                        train_y,
                        epochs=TRAINING_STEPS,
                        batch_size=BATCH_SIZE,
                        verbose=2,
                        validation_data=(
                            reshape_for_rnn(val_x, model.input_shape),
                            val_y,
                            )
                        )
    # p=model.predict(test_x)
    modelfilename = images_dir + '/' + company_name + '.hd5'
    model.save(modelfilename)

    p = model.predict(reshape_for_rnn(test_x, model.input_shape))

    tr_y2, va_y2, te_y2 = prepare_data(split_y, training_size, validation_size, BATCH_SIZE, TIMESTEPS, False)

    actual_prediction = denormalise_windows(p, te_y2,denorm_reference_point=denorm_reference_point)

    pred = DataFrame({'Prediction': actual_prediction[:, -1]})
    ty = DataFrame({'Actual': te_y2[:, -1]})
    combined_data = pred.join(ty, lsuffix='_Predcition', rsuffix='_Actual')
    combined_data.to_csv(images_dir + '/' + company_name + '.csv')

    img2 = images_dir + '/' + company_name + '_pred_vs_act.png'
    plot_predicted_vs_actual(actual_prediction[:,-1],
                             te_y2[:,-1],
                             img2)

    company_name_text = 'Company = ' + company_name
    html_add_fig_string = """<h2>""" + company_name_text + """</h2>
        <img src=" """ + company_name + '_train_validate.png' + """ " alt="AA" style="width:304px;height:228px;">
        <img src=" """ + company_name + '_pred_vs_act.png' + """ " alt="AA" style="width:304px;height:228px;">\n"""
    html_RNN_Result_string = '  '.join(["""<h3>""", 'Training Loss = ', str(History.history['loss'][-1]),
                                        'Training Accuracy = ', str(History.history['acc'][-1]),
                                        'Validation Loss = ', str(History.history['val_loss'][-1]),
                                        'Validation Accuracy = ', str(History.history['val_acc'][-1]),
                                        """</h3>"""])
    f.write(html_add_fig_string)
    f.write(html_RNN_Result_string)
    del model
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
