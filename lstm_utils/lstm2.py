import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
plt.rcParams["figure.figsize"] = (16,6)


def get_shoreline_data(csv_path):
    ##importing data
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame({'datetime':pd.to_datetime(df['datetime']),
                           'value':df['distances']})
    new_df = new_df.set_index(['datetime'])
    new_df = new_df.dropna()
    y_rolling = new_df.rolling('91D', min_periods=1).mean()
    y1 = y_rolling.resample('91D').ffill()
    y1 = y1.dropna()
    df=y1
    df['Date'] = pd.to_datetime(df.index)
    df.set_axis(df['Date'], inplace=True)
    return df

def setup_data(df, look_back, split_percent=0.90):
    shore_data = df['value'].values
    shore_data = shore_data.reshape((-1,1))

    split = int(split_percent*len(shore_data))

    shore_train = shore_data[:split]
    shore_test = shore_data[split:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    train_generator = TimeseriesGenerator(shore_train, shore_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(shore_test, shore_test, length=look_back, batch_size=1)
    return shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator

def train_model(train_generator, test_generator, look_back, units=30, num_epochs=200):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential()
    model.add(
        LSTM(units,
            activation='relu',
            input_shape=(look_back,1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)


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

    plt.plot(df['Date'], df['value'], color='blue',label='Observed Three Month Moving Average')
    plt.plot(forecast_dates,forecast, '--', color='red', label='LSTM Projection')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(df['Date']), max(forecast_dates))
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()
    return forecast, forecast_dates

def main(csv_path):
    look_back=9
    num_prediction=40
    df = get_shoreline_data(csv_path)

    shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator = setup_data(df, look_back)
    model, prediction = train_model(train_generator, test_generator, look_back)
    project(df,
            shore_data,
            look_back,
            num_prediction,
            model)
