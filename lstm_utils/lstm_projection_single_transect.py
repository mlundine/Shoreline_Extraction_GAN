import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import gc
plt.rcParams["figure.figsize"] = (16,6)


def get_shoreline_data(csv_path):
    ##importing data
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame({'datetime':pd.to_datetime(df['datetime']),
                           'value':df['distances']})
    new_df = new_df.drop_duplicates('datetime', 'last')
    new_df = new_df.set_index(['datetime'])
    new_df = new_df.dropna()
    y_rolling = new_df.rolling('365D', min_periods=1).mean()
    y1 = y_rolling.resample('91D').ffill()
    y1 = y1.dropna()
    df=y1
    df['Date'] = pd.to_datetime(df.index)
    df.set_axis(df['Date'], inplace=True)
    return df

def setup_data(df, look_back, batch_size, split_percent=0.80):
    shore_data = df['value'].values
    shore_data = shore_data.reshape((-1,1))

    split = int(split_percent*len(shore_data))

    shore_train = shore_data[:split]
    shore_test = shore_data[split:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    train_generator = TimeseriesGenerator(shore_train, shore_train, length=look_back, batch_size=batch_size)     
    test_generator = TimeseriesGenerator(shore_test, shore_test, length=look_back, batch_size=1)
    return shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator

# Reset Keras Session
def reset_keras():
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()
    sess = keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it does something you should see a number as output

def train_model(train_generator, test_generator, look_back, units=30, num_epochs=60):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential()
    model.add(
        LSTM(units,
             activation='relu',
             input_shape=(look_back,1),
             recurrent_dropout=0.3)
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='auto', restore_best_weights=True)
    history = model.fit_generator(train_generator,
                                  epochs=num_epochs,
                                  callbacks=[early_stopping_callback],
                                  validation_data=test_generator,
                                  verbose=1)


    prediction = model.predict_generator(test_generator)
    return model, prediction

def plot_results(shore_train,
                 shore_test,
                 prediction,
                 date_train,
                 date_test):
    shore_train = shore_train.reshape((-1))
    shore_test = shore_test.reshape((-1))
    prediction = prediction.reshape((-1))

    plt.plot(date_train, shore_train, label='Training Data')
    plt.plot(date_test, shore_test, label='Ground Truth')
    #plt.plot(date_test, prediction, label='Prediction')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.legend()
    plt.minorticks_on()
    plt.xlim(min(date_train), max(date_test))
    plt.tight_layout()
    plt.show()

def project(df,
            shore_data,
            look_back,
            num_prediction,
            model):

    shore_data = shore_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = shore_data[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq='91D').tolist()
        return prediction_dates

    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    
    return forecast, forecast_dates

def run(csv_path,
        site,
        folder,
        bootstrap=30,
        num_prediction=40,
        epochs=35,
        units=20,
        batch_size=32,
        lookback=4):
    look_back=lookback
    num_prediction=num_prediction
    bootstrap=bootstrap
    batch_size = batch_size
    epochs=epochs
    units=units
    df = get_shoreline_data(csv_path)
    forecast_array = np.zeros((bootstrap, num_prediction+1))
    for i in range(bootstrap):
        print('trial: '+str(i))
        shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator = setup_data(df, look_back, batch_size)
        reset_keras()
        model, prediction = train_model(train_generator, test_generator, look_back, units=units, num_epochs=epochs)
        forecast, forecast_dates = project(df,
                                           shore_data,
                                           look_back,
                                           num_prediction,
                                           model)
        forecast_array[i,:] = forecast
        del model 
        del prediction 
        del forecast 
        del shore_data 
        del shore_train 
        del shore_test 
        del date_train 
        del date_test 
        del train_generator 
        del test_generator
        gc.collect()
    forecast_mean = np.mean(forecast_array, axis=0)
    forecast_std_error = np.std(forecast_array, axis=0)/np.sqrt(bootstrap)
    upper_conf_interval = forecast_mean + (forecast_std_error*1.96)
    lower_conf_interval = forecast_mean - (forecast_std_error*1.96)
    plt.plot(df['Date'], df['value'], color='blue',label='Observed Three Month Moving Average')
    plt.plot(forecast_dates,forecast_mean, '--', color='red', label='LSTM Projection Mean')
    plt.fill_between(forecast_dates, lower_conf_interval, upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(df['Date']), max(forecast_dates))
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, site+'project.png'), dpi=300)
    plt.close('all')

    new_df_dict = {'time': forecast_dates,
                   'forecast_mean_position': forecast_mean,
                   'forecast_upper_conf': upper_conf_interval,
                   'forecast_lower_conf': lower_conf_interval}
    new_df = pd.DataFrame(new_df_dict)
    new_df.to_csv(os.path.join(folder, site+'project.csv'),index=False)
    
    del forecast_array 
    del forecast_mean 
    del forecast_std_error 
    del upper_conf_interval 
    del lower_conf_interval 
    del df 
    del new_df_dict 
    del new_df 
    gc.collect()
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str,required=True,help="path to transect csv")
    parser.add_argument("--site", type=str,required=True, help="site name")
    parser.add_argument("--folder", type=str, required=True, help="path to projected folder")
    parser.add_argument("--bootstrap", type=int, required=True, help="number of repeat trials")
    parser.add_argument("--num_prediction",type=int, required=True, help="number of predictions")
    parser.add_argument("--epochs",type=int, required=True, help="number of epochs to train")
    parser.add_argument("--units",type=int, required=True, help="number of LSTM layers")
    parser.add_argument("--batch_size",type=int, required=True, help="training batch size")
    parser.add_argument("--lookback",type=int, required=True, help="look back value")
    args = parser.parse_args()
    run(args.csv_path,
        args.site,
        args.folder,
        bootstrap=args.bootstrap,
        num_prediction=args.num_prediction,
        epochs=args.epochs,
        units=args.units,
        batch_size=args.batch_size,
        lookback=args.lookback)   

   
