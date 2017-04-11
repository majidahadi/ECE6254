from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Iterable
import pywt

def conv_to_wavelet(data, axis, filter_to_use_as_string):
    temp_list = []
    for i in range(0,data.shape[axis]):
        wavedec_data = pywt.wavedec(data[i], filter_to_use_as_string)
        temp_list.append(np.hstack(wavedec_data))
    temp_list = np.asarray(temp_list)
    #temp_list[:,0] = 0
    return temp_list

def build_model(layers, option):

    if option == 1:

        model = Sequential()

        model.add(LSTM(layers[0],
            input_dim=layers[0],
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[0],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=layers[3]))

        model.add(Activation("sigmoid"))


    else:
        model = Sequential()
        #model.add(LSTM(TIMESTEPS, input_shape=(BATCH_SIZE, TIMESTEPS),return_sequences=True))
        model.add(LSTM(layers[0], input_dim = layers[0],return_sequences=True,activation='sigmoid'))
        model.add(LSTM(layers[0], return_sequences=True,activation='sigmoid'))
        model.add(LSTM(layers[0], return_sequences=True,activation='sigmoid'))
        model.add(LSTM(layers[0], return_sequences=True,activation='sigmoid'))
        model.add(LSTM(layers[0], return_sequences=True,activation='sigmoid'))
        #model.add(Dense(TIMESTEPS,output_dim = 1))
        model.add(Dense(output_dim=layers[3]))



    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model



def prepare_data_for_batchsize(data,batchsize):
    batchdata = []
    if (batchsize > 1):

        for i in range(0,data.shape[0]-batchsize):
            batchdata.append(data[i:i+batchsize])
    else:
        batchdata.append(data)
        if (len(np.asarray(batchdata).shape)==2):
            tempdata = np.asarray(batchdata)
            t2 = tempdata.reshape(1,tempdata.shape[1],1)
            return t2
    return np.asarray(batchdata)

def denormalise_windows(window_data,reference_data,denorm_reference_point):
    p_de_norm = []
    for i in range(0, reference_data.shape[0]):
        p_de_norm.append(reference_data[i,denorm_reference_point] * (1 + window_data[i]))
    #for window in window_data:

    return np.asarray(p_de_norm)

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return np.asarray(normalised_data)

def prepare_data(data,training_size,validation_size,batchsize, timestep,normalize):
    ntrain = int((round(len(data) * training_size))*(1-validation_size))
    nval = int(ntrain*validation_size)
    ntest = len(data)-nval-ntrain
    tempdata = np.asarray(data)
    if normalize == True:
        tempdata = normalise_windows(tempdata)
    #train = prepare_data_for_batchsize(tempdata[:ntrain], batchsize)
    #val = prepare_data_for_batchsize(tempdata[ntrain:ntrain + nval], batchsize)
    #test = prepare_data_for_batchsize(tempdata[-ntest:], batchsize)

    train = tempdata[:ntrain]
    val = tempdata[ntrain:ntrain + nval]
    test = tempdata[-ntest:]



    return train,val,test

def reshape_for_rnn(data,shape):
    #shape = data.shape
    #shape = [shape[1], shape[0], shape[2]]
    if (len(shape)<len(data.shape)):
        tempdata= np.squeeze(data)
        return tempdata
    elif (len(shape)==len(data.shape)):
        return data

    else:
        newshape=[]
        for i in range(0,len(shape)):
            if shape[i]==None:
                newshape.append(data.shape[i])
            else:
                newshape.append(shape[i])
        shape = np.asarray(newshape)
        #shape = [shape[1], shape[0], shape[2]]
        shape = [shape[0], 1, shape[2]]
        tempdata = data.reshape(shape)
        return tempdata

def plot_predicted_vs_actual(predicted,actual,image_filename_with_path):
    fig, ax1 = plt.subplots()

    plot_predicted,=ax1.plot(predicted,'r-', label='Predicted')#ax1.plot(xlabels,predicted,'r-')
    ax1.set_ylabel('Predicted', color='r')

    ax2 = ax1.twinx()
    plot_test, = ax2.plot(actual,'b-', label='Actual')#ax2.plot(xlabels,y['test'], 'b.')
    ax2.set_ylabel('Actual', color = 'b')
    plt.legend(handles=[plot_predicted, plot_test])
    #plt.show()
    #img2= images_dir + '/' + company_name + '/'+ company_name+ '_pred_vs_act.png'
    #img2 = images_dir + '/' + company_name + '_pred_vs_act.png'
    plt.savefig(image_filename_with_path,dpi=1000)
    plt.close('all')

def animation_predicted_vs_actual(predicted,actual,image_filename_with_path,iteration, fig, ax1):
    anim = FuncAnimation(fig,update_plot(ax1,iteration,predicted,actual),blit=True)

    return anim

def update_plot(ax1,iteration,predicted,actual):
    plt.title('Iteration = %s' %(iteration))
    plot_predicted,=ax1.plot(predicted,'r-', label='Predicted')#ax1.plot(xlabels,predicted,'r-')
    ax1.set_ylabel('Predicted', color='r')

    ax2 = ax1.twinx()
    plot_test, = ax2.plot(actual,'b-', label='Actual')#ax2.plot(xlabels,y['test'], 'b.')
    ax2.set_ylabel('Actual', color = 'b')
    plt.legend(handles=[plot_predicted, plot_test])
    plt.show()
    #img2= images_dir + '/' + company_name + '/'+ company_name+ '_pred_vs_act.png'
    #img2 = images_dir + '/' + company_name + '_pred_vs_act.png'
    return

def plot_results_multiple(predicted_data, true_data, prediction_len,offset,plot_len, plot_step,
                          image_filename_with_path):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data[:,0], label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        if i% plot_step ==0 :
            padding = [None for p in range(i)]# * prediction_len)]

            data1= data[-1*plot_len:]

            '''
            ##### using numpy nd array ### does not work that well
            padding = np.empty((1,prediction_len+i))
            for p in range(i):
                padding[0, p] = None
            padding[0,i:]= data
            '''
            padding += np.ndarray.tolist(data1 - data1.max() + offset + true_data[i, 0])
            #padding = flatten_list(padding)
            #padding = padding +np.ndarray.tolist(true_data[i])# + offset)
            plt.plot(list(flatten_list(padding)), label='Prediction')
            #plt.legend()
    #plt.show()
    plt.savefig(image_filename_with_path,dpi=3000)
    #plt.savefig(image_filename_with_path, format='eps', dpi=1000)
    plt.close('all')
    #plt.show()
    return

def flatten_list(input_list):
    for item in input_list:
        if isinstance(item,Iterable):
            for x in flatten_list(item):
                yield x
        else:
            yield item