import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import numpy as np



####NEED TO CHANGE TRAINING/TEST SPLIT, LOOKBACK, AND BATCH SIZE, as well as modify the implementation of the LSTM


###keep this
def setup_data(df, look_back, split_percent, batch_size):
    shore_data = df['value'].values
    shore_data = shore_data.reshape((-1,1))

    split = int(split_percent*len(shore_data))

    shore_train = shore_data[:split]
    shore_test = shore_data[split:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    train_generator = TimeseriesGenerator(shore_train, shore_train, length=look_back, batch_size=batch_size)     
    test_generator = TimeseriesGenerator(shore_test, shore_test, length=look_back, batch_size=1)
    return shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator, split

def train_model(train_generator,
                test_generator,
                look_back,
                num_epochs,
                layers):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential()
    model.add(
        LSTM(layers,
            activation='relu',
            input_shape=(look_back,1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history = model.fit_generator(train_generator,
                                  validation_data = test_generator,
                                  epochs=num_epochs,
                                  verbose=0)

    prediction = model.predict_generator(test_generator)
    return model, prediction, history

# run a repeated experiment, need to modify this
def experiment(repeats,
               series,
               epochs,
               look_back,
               split_percent,
               batch_size,
               layers,
               csv_path):
    # transform data to be stationary
    raw_values = series.values
    df = get_shoreline_data(csv_path)
    shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator, split = setup_data(df,
                                                                                                                    look_back,
                                                                                                                    split_percent,
                                                                                                                    batch_size)

    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the model
        batch_size = 1
        model, predictions, history = train_model(train_generator,
                                                  test_generator,
                                                  look_back,
                                                  epochs,
                                                  layers)
        # report performance
        plt.title('Number of Epochs = ' + str(epochs))
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Validation MSE ($m^2$)')
        valid_rmse=np.sqrt(history.history['val_loss'][-1])
        print('%d) Test RMSE: %.3f' % (r+1, valid_rmse))
        error_scores.append(valid_rmse)
    plt.show()
    return error_scores


# load dataset
def get_shoreline_series(csv_path):
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
    return df['value']

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

def main(csv_path, look_back, split_percent, batch_size, layers):
    series = get_shoreline_series(csv_path)
    # experiment
    repeats = 30
    results = pd.DataFrame()
    ###epochs = 
    # vary neurons
    neurons = range(1,50)
    for n in neurons:
        results[str(n)] = experiment(repeats,
                                     series,
                                     epochs,
                                     look_back,
                                     split_percent,
                                     batch_size,
                                     n,
                                     csv_path)
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()
    plt.savefig('boxplot_neurons.png')
