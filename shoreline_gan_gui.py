import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import cv2

global gdal_modules
global coastsat

wd = os.getcwd()
coastsat = os.path.join(wd, 'utils', 'coastsat')
gdal_modules = os.path.join(wd, 'utils', 'gdal_modules')
lstm_modules = os.path.join(wd, 'lstm_utils')
sys.path.append(lstm_modules)
sys.path.append(coastsat)
sys.path.append(gdal_modules)
from utils import download_utils
from utils import generating_transects_utils
from utils import image_processing_utils
from utils import gan_training_utils
from utils import gan_inference_utils
from utils import shoreline_timeseries_utils
from lstm_utils import lstm_2D_projection as project_shore
from lstm_utils import batch_lstm_proj as project_ts_batch

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

sys.path.append(os.path.join(wd, 'utils'))
sys.path.append(os.path.join(wd, 'utils', 'coastsat'))

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()


        sizeObject = QDesktopWidget().screenGeometry(-1)
        global screenWidth
        screenWidth = sizeObject.width()
        global screenHeight
        screenHeight = sizeObject.height()
        global bw1
        bw1 = int(screenWidth/15)
        global bw2
        bw2 = int(screenWidth/50)
        global bh1
        bh1 = int(screenHeight/15)
        global bh2
        bh2 = int(screenHeight/20)

        self.setWindowTitle("Shoreline Extraction GAN")
        self.home()

    ##downloads satellite imagery
    def start_downloading_button(self, sitename, dates, polygonStuff, satChecks, start_idx, end_idx):
        polygon = []
        for point in polygonStuff:
            try:
                long = float(point[0])
                lat = float(point[1])
                polygon.append([long,lat])
            except:
                pass
        satList = []
        if satChecks[0].isChecked() == True:
            satList.append('S2')
        if satChecks[1].isChecked() == True:
            satList.append('L5')
        if satChecks[2].isChecked() == True:
            satList.append('L7')
        if satChecks[3].isChecked() == True:
            satList.append('L8')

        if satChecks[4].isChecked() == True:
            options = QFileDialog.Options()
            shapefile, _ = QFileDialog.getOpenFileName(self,"Select Study Area Shapefile", "","ESRI Shapefiles (*.shp)", options=options)
            if shapefile:
                download_utils.download_from_shapefile(shapefile, dates, satList, sitename, start_idx, end_idx)
        else:
            download_utils.download_imagery(polygon, dates, satList, sitename)
        
    ## Clicking the exit button hides all of the buttons above it
    def exit_buttons(self, buttons):
        for button in buttons:
            button.hide()

    def download_imagery_window(self):


        start_downloading = QPushButton('Start Download')
        self.vbox.addWidget(start_downloading, 0, 1)

        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)
        
        nameLabel = QLabel('Name')
        name = QLineEdit()
        self.vbox.addWidget(nameLabel, 1,1)
        self.vbox.addWidget(name,2,1)

        shapefile_input_lab = QLabel('Shapefile Input')
        shapefile_input = QCheckBox()
        self.vbox.addWidget(shapefile_input_lab, 1, 2)
        self.vbox.addWidget(shapefile_input, 2, 2)

        start_idx_lab = QLabel('Shapefile Start Index')
        start_idx = QSpinBox()
        start_idx.setValue(0)
        start_idx.setMaximum(999)
        start_idx.setMinimum(1)
        self.vbox.addWidget(start_idx_lab, 1, 3)
        self.vbox.addWidget(start_idx, 2, 3)

        end_idx_lab = QLabel('Shapefile End Index')
        end_idx = QSpinBox()
        end_idx.setValue(10)
        end_idx.setMaximum(9999)
        end_idx.setMinimum(2)
        self.vbox.addWidget(end_idx_lab, 1, 4)
        self.vbox.addWidget(end_idx, 2, 4)
        
        
        
        beginDateLabel = QLabel('Beginning Date (YYYY-MM-DD)')
        beginDate  = QLineEdit()
        self.vbox.addWidget(beginDateLabel,3,1)
        self.vbox.addWidget(beginDate,4,1)

        endDateLabel = QLabel('End Date (YYYY-MM-DD)')
        endDate = QLineEdit()
        self.vbox.addWidget(endDateLabel,3,2)
        self.vbox.addWidget(endDate,4,2)

        longLabel = QLabel('Longitudes')
        latLabel = QLabel('Latitudes')
        self.vbox.addWidget(longLabel,6,1)
        self.vbox.addWidget(latLabel,7,1)

        topLeftLabel = QLabel('Upper Left Coord. (decimal degrees)')
        topLeftLong = QLineEdit()
        topLeftLat = QLineEdit()
        self.vbox.addWidget(topLeftLabel, 5, 2)
        self.vbox.addWidget(topLeftLong,6,2)
        self.vbox.addWidget(topLeftLat,7,2)

        topRightLabel = QLabel('Upper Right Coord. (decimal degrees)')
        topRightLong = QLineEdit()
        topRightLat = QLineEdit()
        self.vbox.addWidget(topRightLabel, 5, 3)
        self.vbox.addWidget(topRightLong, 6, 3)
        self.vbox.addWidget(topRightLat, 7, 3)

        botLeftLabel = QLabel('Lower Right Coord. (decimal degrees)')
        botLeftLong = QLineEdit()
        botLeftLat = QLineEdit()
        self.vbox.addWidget(botLeftLabel, 5, 4)
        self.vbox.addWidget(botLeftLong, 6, 4)
        self.vbox.addWidget(botLeftLat, 7, 4)

        botRightLabel = QLabel('Lower Left Coord. (decimal degrees)')
        botRightLong = QLineEdit()
        botRightLat = QLineEdit()
        self.vbox.addWidget(botRightLabel, 5, 5)
        self.vbox.addWidget(botRightLong, 6, 5)
        self.vbox.addWidget(botRightLat, 7, 5)

        s2checkLabel = QLabel('Sentinel 2')
        s2check = QCheckBox()
        self.vbox.addWidget(s2checkLabel, 9,1)
        self.vbox.addWidget(s2check,10,1)

        L5checkLabel = QLabel('Landsat 5')
        L5check = QCheckBox()
        self.vbox.addWidget(L5checkLabel, 9,2)
        self.vbox.addWidget(L5check,10,2)

        L7checkLabel = QLabel('Landsat 7')
        L7check = QCheckBox()
        self.vbox.addWidget(L7checkLabel, 9,3)
        self.vbox.addWidget(L7check,10,3)

        L8checkLabel = QLabel('Landsat 8')
        L8check = QCheckBox()
        self.vbox.addWidget(L8checkLabel, 9,4)
        self.vbox.addWidget(L8check,10,4)

        buttons = [start_downloading, exit_button,
                   nameLabel, name, beginDateLabel,
                   beginDate, endDate,
                   endDateLabel, longLabel,
                   latLabel, topLeftLabel,
                   topLeftLong, topLeftLat,
                   topRightLabel, topRightLong,
                   topRightLat, botLeftLabel,
                   botLeftLong, botLeftLat,
                   botRightLabel, botRightLong,
                   botRightLat, s2checkLabel,
                   s2check, L5checkLabel,
                   L5check, L7checkLabel,
                   L7check, L8checkLabel,
                   L8check, shapefile_input_lab,
                   shapefile_input, start_idx_lab,
                   start_idx, end_idx_lab,
                   end_idx]
        polygonStuff = [[topRightLong.text(), topRightLat.text()],
                        [topLeftLong.text(), topLeftLat.text()],
                        [botRightLong.text(), botRightLat.text()],
                        [botLeftLong.text(), botLeftLat.text()],
                        [topRightLong.text(),topRightLat.text()]
                        ]
        
        dates = [beginDate.text(), endDate.text()]
        checkboxes = [s2check, L5check, L7check, L8check, shapefile_input]    
        #Actions
        start_downloading.clicked.connect(lambda: self.start_downloading_button(name.text(),
                                                                               [beginDate.text(), endDate.text()],
                                                                               [[topRightLong.text(), topRightLat.text()],
                                                                               [topLeftLong.text(), topLeftLat.text()],
                                                                               [botRightLong.text(), botRightLat.text()],
                                                                               [botLeftLong.text(), botLeftLat.text()],
                                                                               [topRightLong.text(),topRightLat.text()]],
                                                                               checkboxes, start_idx.value(), end_idx.value()))
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))


    def preprocess_button(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_folder = str(QFileDialog.getExistingDirectory(self, "Select Preprocessed JPEG Directory"))
        if image_folder:
            new_image_folder = os.path.join(os.path.dirname(image_folder), 'pix2pix_ready')
            try:
                os.mkdir(new_image_folder)
            except:
                pass
            
            image_processing_utils.split_and_resize(image_folder, new_image_folder)
            
    def select_shoreline_button(self, transect_spacing, transect_length):
        options = QFileDialog.Options()
        shore_shape, _ = QFileDialog.getOpenFileName(self,"Select Shoreline Shapefile", "","ESRI Shapefiles (*.shp)", options=options)
        if shore_shape:
            generating_transects_utils.make_transects(shore_shape, transect_spacing, transect_length)
            
    def make_transects_button(self):

        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)
        
        transect_spacing_label = QLabel('Alongshore Spacing (m)')
        transect_spacing = QSpinBox()
        transect_spacing.setMinimum(1)
        transect_spacing.setMaximum(1000)
        transect_spacing.setValue(200)
        self.vbox.addWidget(transect_spacing_label, 1,1)
        self.vbox.addWidget(transect_spacing, 2, 1)

        transect_length_label = QLabel('Cross-Shore Length (m)')
        transect_length = QSpinBox()
        transect_length.setMinimum(10)
        transect_length.setMaximum(10000)
        transect_length.setValue(500)
        self.vbox.addWidget(transect_length_label, 1,2)
        self.vbox.addWidget(transect_length, 2, 2)

        select_shapefile = QPushButton('Select Reference Shoreline Shapefile')
        self.vbox.addWidget(select_shapefile, 3, 1)
        
        buttons = [transect_spacing_label,
                   transect_spacing,
                   transect_length_label,
                   transect_length,
                   select_shapefile,
                   exit_button]
        #actions
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))
        select_shapefile.clicked.connect(lambda: self.select_shoreline_button(transect_spacing.value(), transect_length.value()))

    def create_timeseries_button(self,
                                 sitename,
                                 start_idx,
                                 switch_dir):
        options = QFileDialog.Options()
        shorelines, _ = QFileDialog.getOpenFileName(self,"Select Shorelines Shapefile", "","ESRI Shapefiles (*.shp)", options=options)
        if shorelines:
            options = QFileDialog.Options()
            transects, _ = QFileDialog.getOpenFileName(self,"Select Transects Shapefile", "","ESRI Shapefiles (*.shp)", options=options)
            if transects:
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog
                output_folder = str(QFileDialog.getExistingDirectory(self, "Select Output Folder for Transects"))
                if output_folder:
                    shoreline_timeseries_utils.batch_transect_timeseries(shorelines,
                                                                         transects,
                                                                         sitename,
                                                                         output_folder,
                                                                         int(start_idx),
                                                                         switch_dir=switch_dir.isChecked())
            
    def timeseries_button(self):
        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)
        
        sitename = QLineEdit()
        sitename_lab = QLabel('Site Name')
        self.vbox.addWidget(sitename_lab, 1, 1)
        self.vbox.addWidget(sitename, 2, 1)

        start_idx = QSpinBox()
        start_idx.setValue(0)
        start_idx.setMaximum(99999)
        start_idx.setMinimum(0)
        start_idx_lab = QLabel('Starting Index')
        self.vbox.addWidget(start_idx_lab, 3,1)
        self.vbox.addWidget(start_idx,4,1)
        
        
        create_timeseries = QPushButton('Create Timeseries')
        self.vbox.addWidget(create_timeseries, 5, 1)

        

        switch_dir_lab = QLabel('Switch Transect Direction')
        switch_dir = QCheckBox()
        self.vbox.addWidget(switch_dir_lab, 1,2)
        self.vbox.addWidget(switch_dir, 2, 2)
        
        buttons = [sitename,
                   sitename_lab,
                   create_timeseries,
                   switch_dir_lab,
                   switch_dir,
                   start_idx,
                   start_idx_lab,
                   exit_button]
        #actions
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))
        create_timeseries.clicked.connect(lambda: self.create_timeseries_button(sitename.text(), start_idx.value(), switch_dir))


    def run_model_button(self,
                         sitename,
                         model_name,
                         epoch,
                         reference_shoreline,
                         reference_region,
                         distance_threshold,
                         clip_length):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dataroot = str(QFileDialog.getExistingDirectory(self, "Select Input Image Folder"))
        if dataroot:
            options = QFileDialog.Options()
            metadata_csv, _ = QFileDialog.getOpenFileName(self,"Select Site Metadata CSV", "","CSVs (*.csv)", options=options)
            if metadata_csv:
                if reference_shoreline.isChecked():
                    options = QFileDialog.Options()
                    reference_shoreline, _ = QFileDialog.getOpenFileName(self,"Select Reference Shoreline", "","Shapefiles (*.shp)", options=options)
                    if reference_region.isChecked():
                        options = QFileDialog.Options()
                        reference_region, _ = QFileDialog.getOpenFileName(self,"Select Reference Region", "","Shapefiles (*.shp)", options=options)
                        if reference_region:
                            gan_inference_utils.run_and_process(sitename,
                                                                dataroot,
                                                                model_name,
                                                                metadata_csv,
                                                                epoch=epoch,
                                                                reference_shoreline=reference_shoreline,
                                                                reference_region=reference_region,
                                                                distance_threshold=distance_threshold,
                                                                clip_length=clip_length)
                    else:
                        gan_inference_utils.run_and_process(sitename,
                                                            dataroot,
                                                            model_name,
                                                            metadata_csv,
                                                            epoch=epoch,
                                                            reference_shoreline=reference_shoreline,
                                                            reference_region=None,
                                                            distance_threshold=distance_threshold,
                                                            clip_length=clip_length)
                        
                elif reference_region.isChecked():
                    options = QFileDialog.Options()
                    reference_region, _ = QFileDialog.getOpenFileName(self,"Select Reference Region", "","Shapefiles (*.shp)", options=options)
                    if reference_region:
                        gan_inference_utils.run_and_process(sitename,
                                                        dataroot,
                                                        model_name,
                                                        metadata_csv,
                                                        epoch=epoch,
                                                        reference_shoreline=None,
                                                        reference_region=reference_region,
                                                        distance_threshold=distance_threshold,
                                                        clip_length=clip_length)
                else:
                    gan_inference_utils.run_and_process(sitename,
                                                        dataroot,
                                                        model_name,
                                                        metadata_csv,
                                                        epoch=epoch,
                                                        reference_shoreline=None,
                                                        reference_region=None,
                                                        distance_threshold=distance_threshold,
                                                        clip_length=clip_length)
                    


        
    def run_and_process_button(self):
        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)
        
        model_name_lab = QLabel('Model Name')
        model_name = QLineEdit()
        self.vbox.addWidget(model_name_lab, 1, 1)
        self.vbox.addWidget(model_name, 2, 1)

        sitename = QLineEdit()
        sitename_lab = QLabel('Site Name')
        self.vbox.addWidget(sitename_lab, 1, 2)
        self.vbox.addWidget(sitename, 2, 2)

        distance_threshold_lab = QLabel('Distance Threshold for Reference Shoreline Filter')
        distance_threshold = QSpinBox()
        distance_threshold.setValue(250)
        distance_threshold.setMaximum(1500)
        distance_threshold.setMinimum(10)
        self.vbox.addWidget(distance_threshold_lab, 5,2)
        self.vbox.addWidget(distance_threshold, 6, 2)
        

        clip_length_lab = QLabel('End Clip Length')
        clip_length = QSpinBox()
        clip_length.setValue(150)
        clip_length.setMaximum(750)
        clip_length.setMinimum(30)
        self.vbox.addWidget(clip_length_lab,3,2)
        self.vbox.addWidget(clip_length,4,2)


        reference_shoreline_lab = QLabel('Use Reference Shoreline')
        reference_shoreline = QCheckBox()
        self.vbox.addWidget(reference_shoreline_lab, 3, 3)
        self.vbox.addWidget(reference_shoreline, 4, 3)

        reference_region_lab = QLabel('Use Reference Region')
        reference_region = QCheckBox()
        self.vbox.addWidget(reference_region_lab, 3, 4)
        self.vbox.addWidget(reference_region, 4, 4)
        
        epoch_lab = QLabel('Epoch')
        epoch = QLineEdit()
        self.vbox.addWidget(epoch_lab, 1, 3)
        self.vbox.addWidget(epoch, 2,3)

        run_model = QPushButton('Run and Process')
        self.vbox.addWidget(run_model, 7, 2)


        buttons = [model_name_lab,
                   model_name,
                   sitename,
                   sitename_lab,
                   epoch_lab,
                   epoch,
                   run_model,
                   distance_threshold_lab,
                   distance_threshold,
                   clip_length_lab,
                   clip_length,
                   reference_shoreline_lab,
                   reference_shoreline,
                   reference_region_lab,
                   reference_region,
                   exit_button]
        #actions
        run_model.clicked.connect(lambda: self.run_model_button(sitename.text(),
                                                                model_name.text(),
                                                                epoch.text(),
                                                                reference_shoreline,
                                                                reference_region,
                                                                distance_threshold.value(),
                                                                clip_length.value()))
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))        


    def run_train_button(self, model_name, epoch, epoch_decay, continue_train, epoch_count): 
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dataroot = str(QFileDialog.getExistingDirectory(self, "Select Dataset Folder"))
        if dataroot:
            if continue_train == False:
                gan_training_utils.train_model(model_name,
                                               'pix2pix',
                                               dataroot,
                                               n_epochs=epoch,
                                               n_epochs_decay=epoch_decay)
            else:
                gan_training_utils.continue_train_model(model_name,
                                                        'pix2pix',
                                                        dataroot,
                                                        epoch_count,
                                                        n_epochs=epoch,
                                                        n_epochs_decay=epoch_decay)
        
    def retrain_button(self):
        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)

        model_name_lab = QLabel('Model Name')
        model_name = QLineEdit()
        self.vbox.addWidget(model_name_lab, 1, 1)
        self.vbox.addWidget(model_name, 2, 1)

        epoch_lab = QLabel('Epochs')
        epoch = QSpinBox()
        epoch.setMinimum(1)
        epoch.setMaximum(99999)
        epoch.setValue(10)
        self.vbox.addWidget(epoch_lab, 3, 1)
        self.vbox.addWidget(epoch, 4, 1)

        epochs_decay_lab = QLabel('Decay Epochs')
        epochs_decay = QSpinBox()
        epochs_decay.setValue(5)
        epochs_decay.setMinimum(1)
        epochs_decay.setMaximum(99999)
        self.vbox.addWidget(epochs_decay_lab, 3, 2)
        self.vbox.addWidget(epochs_decay, 4, 2)
        

        continue_train_lab = QLabel('Continuing Training')
        continue_train = QCheckBox()
        self.vbox.addWidget(continue_train_lab, 5, 1)
        self.vbox.addWidget(continue_train, 6, 1)

        epoch_count_lab = QLabel('Continue Training at Epoch #')
        epoch_count = QLineEdit()
        self.vbox.addWidget(epoch_count_lab, 5, 2)
        self.vbox.addWidget(epoch_count, 6, 2)

        run_train = QPushButton('Run')
        self.vbox.addWidget(run_train, 0, 1)


        buttons = [exit_button,
                   model_name_lab,
                   model_name,
                   epoch_lab,
                   epoch,
                   epochs_decay_lab,
                   epochs_decay,
                   continue_train_lab,
                   continue_train,
                   epoch_count_lab,
                   epoch_count,
                   run_train]
        #actions
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))
        run_train.clicked.connect(lambda: self.run_train_button(model_name.text(),
                                                                epoch.value(),
                                                                epochs_decay.value(),
                                                                continue_train.isChecked(),
                                                                epoch_count.text()))
    def run_project_button(self,
                           sitename,
                           num1,
                           num2,
                           bootstrap,
                           num_prediction,
                           epochs,
                           units,
                           batch_size,
                           lookback,
                           split_percent):
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        transect_folder = str(QFileDialog.getExistingDirectory(self, "Select Extracted Transect Folder"))
        if transect_folder:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            projected_folder = str(QFileDialog.getExistingDirectory(self, "Select Folder to Save Projections To (Make New Folder)"))
            if projected_folder:
                project_ts_batch.main(transect_folder,
                                      projected_folder,
                                      sitename,
                                      num1,
                                      num2,
                                      bootstrap=bootstrap,
                                      num_prediction=num_prediction,
                                      epochs=epochs,
                                      units=units,
                                      batch_size=batch_size,
                                      lookback=lookback,
                                      split_percent=split_percent)
        
    def run_merge_projections_button(self,
                                     sitename,
                                     transect_id_min,
                                     transect_id_max,
                                     epsg,
                                     switch_dir):
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        transect_folder = str(QFileDialog.getExistingDirectory(self, "Select Extracted Transect Folder"))
        if transect_folder:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            projected_folder = str(QFileDialog.getExistingDirectory(self, "Select Projected Transect Folder"))
            if projected_folder:
                options = QFileDialog.Options()
                transect_shp_path, _ = QFileDialog.getOpenFileName(self,"Select Transect Shapefile", "","Shapefiles (*.shp)", options=options)
                if transect_shp_path:
                    project_shore.main(sitename,
                                       transect_id_min,
                                       transect_id_max,
                                       projected_folder,
                                       transect_folder,
                                       projected_folder,
                                       transect_shp_path,
                                       epsg,
                                       switch_dir=switch_dir)
                           
    def project_button(self):
        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)
        
        run_project = QPushButton('Run')
        self.vbox.addWidget(run_project, 0, 1)
        
        site_name_lab = QLabel('Site Name')
        site_name = QLineEdit()
        self.vbox.addWidget(site_name_lab, 1, 1)
        self.vbox.addWidget(site_name, 2, 1)
        
        epoch_lab = QLabel('Epochs')
        epoch = QSpinBox()
        epoch.setMinimum(1)
        epoch.setMaximum(99999)
        epoch.setValue(2000)
        self.vbox.addWidget(epoch_lab, 3, 1)
        self.vbox.addWidget(epoch, 4, 1)

        start_idx_lab = QLabel('Starting Index')
        start_idx = QSpinBox()
        start_idx.setMinimum(0)
        start_idx.setMaximum(99999)
        start_idx.setValue(0)
        self.vbox.addWidget(start_idx_lab, 1, 2)
        self.vbox.addWidget(start_idx, 2, 2)

        end_idx_lab = QLabel('Ending Index')
        end_idx = QSpinBox()
        end_idx.setMinimum(0)
        end_idx.setMaximum(99999)
        end_idx.setValue(200)
        self.vbox.addWidget(end_idx_lab, 1, 3)
        self.vbox.addWidget(end_idx, 2, 3)     

        repeats_lab = QLabel('Number of Repeats')
        repeats = QSpinBox()
        repeats.setMinimum(5)
        repeats.setMaximum(99999)
        repeats.setValue(30)
        self.vbox.addWidget(repeats_lab, 5, 3)
        self.vbox.addWidget(repeats, 6, 3)

        predictions_lab = QLabel('Number of Predictions')
        predictions = QSpinBox()
        predictions.setMinimum(1)
        predictions.setMaximum(99999)
        predictions.setValue(40)
        self.vbox.addWidget(predictions_lab, 5, 2)
        self.vbox.addWidget(predictions, 6, 2)
        
        lstm_units_lab = QLabel('Number of LSTM Layers')
        lstm_units = QSpinBox()
        lstm_units.setMinimum(1)
        lstm_units.setMaximum(99999)
        lstm_units.setValue(20)
        self.vbox.addWidget(lstm_units_lab, 3, 3)
        self.vbox.addWidget(lstm_units, 4, 3)

        batch_size_lab = QLabel('Batch Size')
        batch_size = QSpinBox()
        batch_size.setMinimum(1)
        batch_size.setMaximum(99999)
        batch_size.setValue(32)
        self.vbox.addWidget(batch_size_lab, 3, 2)
        self.vbox.addWidget(batch_size, 4, 2)

        lookback_lab = QLabel('Look-Back Value')
        lookback = QSpinBox()
        lookback.setMinimum(1)
        lookback.setMaximum(99999)
        lookback.setValue(3)
        self.vbox.addWidget(lookback_lab, 5, 1)
        self.vbox.addWidget(lookback, 6, 1)

        split_percent_lab = QLabel('Training Split')
        split_percent = QDoubleSpinBox()
        split_percent.setMinimum(0.40)
        split_percent.setMaximum(0.95)
        split_percent.setValue(0.80)
        self.vbox.addWidget(split_percent_lab, 7, 1)
        self.vbox.addWidget(split_percent, 8, 1)

        buttons = [exit_button,
                   site_name_lab,
                   site_name,
                   epoch_lab,
                   epoch,
                   start_idx_lab,
                   start_idx,
                   end_idx_lab,
                   end_idx,
                   repeats_lab,
                   repeats,
                   predictions_lab,
                   predictions,
                   lstm_units_lab,
                   lstm_units,
                   batch_size_lab,
                   batch_size,
                   lookback_lab,
                   lookback,
                   run_project,
                   split_percent_lab,
                   split_percent
                   ]
        
        #actions
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))
        run_project.clicked.connect(lambda: self.run_project_button(site_name.text(),
                                                                    start_idx.value(),
                                                                    end_idx.value(),
                                                                    repeats.value(),
                                                                    predictions.value(),
                                                                    epoch.value(),
                                                                    lstm_units.value(),
                                                                    batch_size.value(),
                                                                    lookback.value(),
                                                                    split_percent.value()))

    def merge_projections_button(self):
        exit_button = QPushButton('Exit')
        self.vbox.addWidget(exit_button, 0, 2)

        run_merge_projections = QPushButton('Run')
        self.vbox.addWidget(run_merge_projections, 0, 1)
        
        site_name_lab = QLabel('Site Name')
        site_name = QLineEdit()
        self.vbox.addWidget(site_name_lab, 1, 1)
        self.vbox.addWidget(site_name, 2, 1)

        transect_id_start_lab = QLabel('Start Transect ID')
        transect_id_start = QSpinBox()
        transect_id_start.setMinimum(0)
        transect_id_start.setMaximum(99999)
        transect_id_start.setValue(0)
        self.vbox.addWidget(transect_id_start_lab, 3, 1)
        self.vbox.addWidget(transect_id_start, 4, 1)

        transect_id_stop_lab = QLabel('End Transect ID')
        transect_id_stop = QSpinBox()
        transect_id_stop.setMinimum(1)
        transect_id_stop.setMaximum(99999)
        transect_id_stop.setValue(20)
        self.vbox.addWidget(transect_id_stop_lab, 5, 1)
        self.vbox.addWidget(transect_id_stop, 6, 1)

        epsg_code_lab = QLabel('EPSG Code')
        epsg_code = QLineEdit()
        self.vbox.addWidget(epsg_code_lab, 7, 1)
        self.vbox.addWidget(epsg_code, 8, 1)     

        switch_dir_lab = QLabel('Switch Transect Direction')
        switch_dir = QCheckBox()
        self.vbox.addWidget(switch_dir_lab, 1,2)
        self.vbox.addWidget(switch_dir, 2, 2)







        buttons = [exit_button,
                   site_name_lab,
                   site_name,
                   transect_id_start_lab,
                   transect_id_start,
                   transect_id_stop_lab,
                   transect_id_stop,
                   epsg_code_lab,
                   epsg_code,
                   switch_dir_lab,
                   switch_dir,
                   run_merge_projections
                   ]        
        #actions
        exit_button.clicked.connect(lambda: self.exit_buttons(buttons))
        run_merge_projections.clicked.connect(lambda: self.run_merge_projections_button(site_name.text(),
                                                                                        transect_id_start.value(),
                                                                                        transect_id_stop.value(),
                                                                                        int(epsg_code.text()),
                                                                                        switch_dir.isChecked()))
        
    def home(self):
        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QGridLayout()             # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        self.widget.setLayout(self.vbox)

        download_imagery = QPushButton('1. Download Imagery')
        self.vbox.addWidget(download_imagery, 0, 0)

        preprocess = QPushButton('2. Preprocess Images')
        self.vbox.addWidget(preprocess, 1, 0)

        run_and_process = QPushButton('3. Shoreline Extraction')
        self.vbox.addWidget(run_and_process, 2, 0)

        make_transects = QPushButton('4. Make Transects')
        self.vbox.addWidget(make_transects, 3, 0)

        timeseries = QPushButton('5. Make Timeseries')
        self.vbox.addWidget(timeseries, 4, 0)

        project = QPushButton('6. Project Timeseries')
        self.vbox.addWidget(project, 5, 0)

        merge_projections = QPushButton('7. Merge Projections')
        self.vbox.addWidget(merge_projections, 6, 0)

        retrain = QPushButton('8. Retraining Extraction Model')
        self.vbox.addWidget(retrain, 7, 0)
        
        ###Actions
        download_imagery.clicked.connect(lambda: self.download_imagery_window())
        preprocess.clicked.connect(lambda: self.preprocess_button())
        run_and_process.clicked.connect(lambda: self.run_and_process_button())
        make_transects.clicked.connect(lambda: self.make_transects_button())
        timeseries.clicked.connect(lambda: self.timeseries_button())
        project.clicked.connect(lambda: self.project_button())
        merge_projections.clicked.connect(lambda: self.merge_projections_button())
        retrain.clicked.connect(lambda: self.retrain_button())




        
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)



        
## Function outside of the class to run the app   
def run():
    app = QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

## Calling run to run the app
run()
