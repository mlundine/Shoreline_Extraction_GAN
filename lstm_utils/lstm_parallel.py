# multivariate output stacked lstm example
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import gc
plt.rcParams["figure.figsize"] = (16,6)
def setup_data(mega_df,
               start_idx,
               end_idx,
               split_percent,
               look_back,
               batch_size):
    ###load mega_df
    dataset_list = []
    for i in range(start_idx, end_idx+1):
        in_seq = np.array(mega_df['value'+str(i)])
        in_seq = in_seq.reshape((len(in_seq),1))
        dataset_list.append(in_seq)

    ###stack timeseries
    dataset = np.hstack(dataset_list)
    n_features = np.shape(dataset)[1]

##    scaler = RobustScaler()
##    scaler = scaler.fit(dataset)
##    normalized_dataset = scaler.transform(dataset)
    
    ### split into training and testing
    split = int(split_percent*len(dataset))
    shore_train = dataset[:split]
    shore_test = dataset[split:]

    train_generator = TimeseriesGenerator(shore_train, shore_train, length=look_back, batch_size=batch_size)     
    test_generator = TimeseriesGenerator(shore_test, shore_test, length=look_back, batch_size=1)
    prediction_generator = TimeseriesGenerator(dataset, dataset, length=look_back, batch_size=1)
 
    return dataset, train_generator, test_generator, n_features, prediction_generator


####to_do
def project(mega_df,
            dataset,
            look_back,
            num_prediction,
            n_features,
            model):


    ##original dimesion is time x transect
    prediction_mat = np.zeros((num_prediction+look_back, n_features))
    prediction_mat[0:look_back,:] = dataset[-look_back:, :]

    
    for i in range(num_prediction):
        update_idx = look_back+i
        x = prediction_mat[i:update_idx, :]
        x = x.reshape((1, look_back, n_features))
        out = model.predict(x, verbose=0)
        prediction_mat[update_idx, :] = out
    prediction = prediction_mat[look_back-1:, :]

    #prediction = scaler.inverse_transform(prediction)

    last_date = mega_df.index.values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq='31D').tolist()
        

    forecast = prediction
    forecast_dates = prediction_dates

    return forecast, forecast_dates

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
    
def train_model(train_generator,
                test_generator,
                num_epochs,
                look_back,
                units,
                n_features):

    ### define model
    model = Sequential()
    
    model.add(Bidirectional(LSTM(units,
                   activation='relu',
                   return_sequences=True,
                   input_shape=(look_back, n_features),
                   recurrent_dropout=0.5
                   ))
              )
    
    model.add(Bidirectional(LSTM(units,
                   activation='relu',
                   recurrent_dropout=0.5
                   ))
              )
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mae')
    ### fit model
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True)
    history = model.fit_generator(train_generator,
                                  epochs=num_epochs,
                                  callbacks=[early_stopping_callback],
                                  validation_data=test_generator,
                                  verbose=1)

    
    
    return model

def predict_data(model, prediction_generator):
    prediction = model.predict_generator(prediction_generator)
    #prediction = scaler.inverse_transform(prediction)
    return prediction

    
def get_csvs(folder,
             basename,
             start_idx,
             end_idx):
    idx = range(start_idx,end_idx+1)
    csvs = [None]*len(idx)
    j=0
    for i in idx:
        csv = os.path.join(folder, basename+'_'+str(i)+'.csv')
        csvs[j] = csv
        j=j+1
    return csvs

def get_mega_df(csv_paths,
                base):
    name = os.path.splitext(os.path.basename(csv_paths[0]))[0]
    idx = name.find('_')
    num = name[idx+1:]

    csv_path = csv_paths[0]
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame({'datetime':pd.to_datetime(df['datetime']),
                           'value'+num:df['distances']})
    new_df = new_df.drop_duplicates('datetime', 'last')
    new_df = new_df.set_index(['datetime'])
    new_df = new_df.dropna()
    y_rolling = new_df.rolling('365D', min_periods=1).mean()
    y1 = y_rolling.resample('31D').ffill()
    y1 = y1.dropna()
    df=y1


    mega_list = [None]*len(csv_paths)
    mega_list[0] = df
    df = None
    new_df = None
    for i in range(1, len(csv_paths)):
        csv = csv_paths[i]
        name = os.path.splitext(os.path.basename(csv))[0]
        idx = name.find('_')
        num = name[idx+1:]
        df = pd.read_csv(csv)
            
        new_df = pd.DataFrame({'datetime':pd.to_datetime(df['datetime']),
                               'value'+num:df['distances']})
        new_df = new_df.drop_duplicates('datetime', 'last')
        new_df = new_df.set_index(['datetime'])
        new_df = new_df.dropna()
        y_rolling = new_df.rolling('365D', min_periods=1).mean()
        y1 = y_rolling.resample('31D').ffill()
        y1 = y1.dropna()
        df=y1

        mega_list[i] = df
    mega_df = pd.concat(mega_list, join='inner', axis=1)

    return mega_df

def make_mega_df(folder,
                 basename,
                 start_idx,
                 end_idx):
    mega_df_save = os.path.join(folder, basename+'_'+str(start_idx)+'to'+str(end_idx)+'.csv')
    csv_list = get_csvs(folder,
                        basename,
                        start_idx,
                        end_idx)
    mega_df = get_mega_df(csv_list,
                          basename)
    mega_df.to_csv(mega_df_save)
    return mega_df


def process_results(sitename,
                    folder,
                    start_idx,
                    end_idx,
                    mega_arr_pred,
                    dataset,
                    observed_dates,
                    date_predict,
                    forecast_dates,
                    mega_arr_forecast):
    n_features = np.shape(mega_arr_pred)[1]
    bootstrap = np.shape(mega_arr_pred)[2]
    idx = start_idx
    for i in range(n_features):
        site = sitename+'_'+str(idx)
        transect_data = mega_arr_pred[:,i,:]
        prediction_mean = np.mean(transect_data, axis=1)
        prediction_std_error = np.std(transect_data, axis=1)/np.sqrt(bootstrap)
        upper_conf_interval = prediction_mean + (prediction_std_error*1.96)
        lower_conf_interval = prediction_mean - (prediction_std_error*1.96)

        gt_date = observed_dates
        gt_vals = dataset[:,i]
        plt.plot(gt_date, gt_vals, color='blue',label='Observed')
        plt.plot(date_predict,prediction_mean, '--', color='red', label='LSTM Projection Mean')
        plt.fill_between(date_predict, lower_conf_interval, upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlim(min(gt_date), max(date_predict))
        plt.minorticks_on()
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, site+'predict.png'), dpi=300)
        plt.close('all')

        new_df_dict = {'time': date_predict,
                       'predicted_mean_position': prediction_mean,
                       'predicted_upper_conf': upper_conf_interval,
                       'predicted_lower_conf': lower_conf_interval}
        new_df = pd.DataFrame(new_df_dict)
        new_df.to_csv(os.path.join(folder, site+'predict.csv'),index=False)

        forecast_array = mega_arr_forecast[:,i,:]
        forecast_mean = np.mean(forecast_array, axis=1)
        forecast_std_error = np.std(forecast_array, axis=1)/np.sqrt(bootstrap)
        upper_conf_interval = forecast_mean + (forecast_std_error*1.96)
        lower_conf_interval = forecast_mean - (forecast_std_error*1.96)
        plt.plot(gt_date, gt_vals, color='blue',label='Observed')
        plt.plot(forecast_dates,forecast_mean, '--', color='red', label='LSTM Projection Mean')
        plt.fill_between(forecast_dates, lower_conf_interval, upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlim(min(gt_date), max(forecast_dates))
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

        idx=idx+1
        
def main(transect_folder,
         output_folder,
         sitename,
         start_idx,
         end_idx,
         bootstrap=30,
         num_prediction=40,
         epochs=2000,
         units=80,
         batch_size=32,
         lookback=3,
         split_percent=0.80):
    
    look_back=lookback
    num_prediction=num_prediction
    bootstrap=bootstrap
    batch_size = batch_size
    num_epochs=epochs
    units=units
    mega_df = make_mega_df(transect_folder,
                           sitename,
                           start_idx,
                           end_idx)
    observed_dates = mega_df.index
    dataset, train_generator, test_generator, n_features, prediction_generator = setup_data(mega_df,
                                                                                                  start_idx,
                                                                                                  end_idx,
                                                                                                  split_percent,
                                                                                                  look_back,
                                                                                                  batch_size)
    date_predict = observed_dates[lookback:]
    mega_arr_pred = np.zeros((len(date_predict), n_features, bootstrap))
    mega_arr_forecast = np.zeros((num_prediction+1, n_features, bootstrap))
    for i in range(bootstrap):
        reset_keras()
        model = train_model(train_generator,
                            test_generator,
                            num_epochs,
                            look_back,
                            units,
                            n_features)
        prediction = predict_data(model, prediction_generator)
        mega_arr_pred[:,:,i] = prediction
        forecast, forecast_dates = project(mega_df,
                                           dataset,
                                           look_back,
                                           num_prediction,
                                           n_features,
                                           model)
        mega_arr_forecast[:,:,i] = forecast

    process_results(sitename,
                    output_folder,
                    start_idx,
                    end_idx,
                    mega_arr_pred,
                    dataset,
                    observed_dates,
                    date_predict,
                    forecast_dates,
                    mega_arr_forecast)
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--transect_folder", type=str,required=True,help="path to transect folder")
    parser.add_argument("--output_folder", type=str,required=True,help="path to output folder folder")
    parser.add_argument("--site", type=str,required=True, help="site name")
    parser.add_argument("--start_idx", type=int, required=True, help="starting transect index")
    parser.add_argument("--end_idx", type=int, required=True, help="ending transect index")
    parser.add_argument("--bootstrap", type=int, required=True, help="number of repeat trials")
    parser.add_argument("--num_prediction",type=int, required=True, help="number of predictions")
    parser.add_argument("--epochs",type=int, required=True, help="number of epochs to train")
    parser.add_argument("--units",type=int, required=True, help="number of LSTM layers")
    parser.add_argument("--batch_size",type=int, required=True, help="training batch size")
    parser.add_argument("--lookback",type=int, required=True, help="look back value")
    parser.add_argument("--split_percent",type=float, required=True, help="training data fraction")
    args = parser.parse_args()
    main(args.transect_folder,
         args.output_folder,
         args.site,
         args.start_idx,
         args.end_idx,
         bootstrap=args.bootstrap,
         num_prediction=args.num_prediction,
         epochs=args.epochs,
         units=args.units,
         batch_size=args.batch_size,
         lookback=args.lookback,
         split_percent=args.split_percent)   



