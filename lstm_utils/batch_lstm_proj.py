import os
import gc
def main(transect_folder,
         projected_folder,
         sitename,
         num1,
         num2,
         bootstrap=30,
         num_prediction=40,
         epochs=40,
         units=30,
         batch_size=20,
         lookback=9):
    """
    Projects timeseries of cross-shore position with a LSTM
    Will save a timeseries figure and a csv for each transect
    inputs:
    transect_folder: path to folder containing extracted data for each transect (str)
    projected_folder: path to save projected data to (str)
    sitename: name for site (str)
    num1: index for starting transect (int)
    num2: index for ending transect, plus one (int)
    """
    root = os.getcwd()
    python_file = os.path.join(root, 'lstm_utils',r'lstm_projection_single_transect.py')
    
    for i in range(num1,num2):
        print('transect:' +str(i))
        site = sitename+'_'+str(i)
        full = os.path.join(transect_folder, site+'.csv')
        cmd1 = r'conda deactivate & conda activate shoreline_prediction & '
        cmd2 = r'python ' + python_file
        cmd3 = ' --csv_path ' + full
        cmd4 = ' --site ' + site
        cmd5 = ' --folder ' + projected_folder
        cmd6 = ' --bootstrap ' + str(bootstrap)
        cmd7 = ' --num_prediction ' + str(num_prediction)
        cmd8 = ' --epochs ' + str(epochs)
        cmd9 = ' --units ' + str(units)
        cmd10 = ' --batch_size ' + str(batch_size)
        cmd11 = ' --lookback ' + str(lookback)
        fullcmd = cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11
        os.system(fullcmd)
        gc.collect()



