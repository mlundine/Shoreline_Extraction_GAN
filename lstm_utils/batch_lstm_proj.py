import os
import gc
def main(transect_folder,
         projected_folder,
         sitename,
         num1,
         num2,
         bootstrap=30,
         num_prediction=40,
         epochs=2000,
         units=80,
         batch_size=32,
         lookback=3,
         split_percent=0.80,
         freq='monthly'):
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
    python_file = os.path.join(root, 'lstm_utils',r'lstm_parallel.py')
    cmd1 = r'conda deactivate & conda activate shoreline_prediction & '
    cmd2 = r'python ' + python_file
    cmd3 = ' --transect_folder ' + transect_folder
    cmd4 = ' --output_folder ' + projected_folder
    cmd5 = ' --site ' + sitename
    cmd6 = ' --start_idx ' + str(num1)
    cmd7 = ' --end_idx ' + str(num2)
    cmd8 = ' --bootstrap ' + str(bootstrap)
    cmd9 = ' --num_prediction ' + str(num_prediction)
    cmd10 = ' --epochs ' + str(epochs)
    cmd11 = ' --units ' + str(units)
    cmd12 = ' --batch_size ' + str(batch_size)
    cmd13 = ' --lookback ' + str(lookback)
    cmd14 = ' --split_percent ' + str(split_percent)
    cmd15 = ' --freq ' + freq
    fullcmd = cmd1+cmd2+cmd3+cmd4+cmd5+cmd6+cmd7+cmd8+cmd9+cmd10+cmd11+cmd12+cmd13+cmd14+cmd15
    os.system(fullcmd)
    gc.collect()



