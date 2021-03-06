import main
from scipy.stats import zscore
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import time

#imports relevant parts of the API package for extacting and manipulating EEG data 
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations, WindowOperations

datalogging = False

board_id = None
params = None
board = None
master_board_id = None
sampling_rate = None
nfft = None
restfulness_params = None
restfulness = None
eeg_channels = None
data = []

def init_bci(_board_type):

    global board_id
    global params
    global board
    global master_board_id
    global sampling_rate
    global nfft
    global restfulness_params
    global restfulness
    global eeg_channels

    print("Initialising BCI parameters")

    # declares which kind of BCI board we are using 

    if _board_type == "Cyton":
        board_id = BoardIds.CYTON_BOARD.value # Use this for real
    else:
        board_id = BoardIds.SYNTHETIC_BOARD.value # Use this for simulated data during dev/debug

    # turns on the loggers for additional debug output during dev
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()

    # this is where to set optional preferences for the BCI (e.g. like which USB port to use etc)
    params = BrainFlowInputParams()
    params.serial_port = "COM4"

    # Parse BCI hardware information to the API
    board = BoardShim(board_id, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    
    #Connect device to BCI and start streaming data
    print("Connecting to BCI")
    board.prepare_session() 
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'starting stream')

    # declare which channels are being used
    # eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    eeg_channels = [1,2,3,4,5,6,7]
    print(eeg_channels)

def filter_signal(_data, _eeg_channels): # this is for cleaning the data 
    for channel in _eeg_channels:
        #0.5hz - 59hz bandpass
        DataFilter.perform_bandpass(_data[channel], BoardShim.get_sampling_rate(board_id), 0.1, 128, 3, FilterTypes.BESSEL.value, 0)
        # 50hz filter
        DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 49, 51, 4, FilterTypes.BUTTERWORTH.value, 0)
        # Denoise
        # DataFilter.perform_wavelet_denoising(_data[channel], 'coif3', 3)

    return _data

def detrend_signal(_data, _eeg_channels): #dont worry about this
    for channel in _eeg_channels:
        DataFilter.detrend(_data[channel], DetrendOperations.LINEAR.value)
    return _data

def update_data():
    
    global data
    data = []
    data = board.get_board_data() # grabs the eeg data currently stored in the boardShim buffer and makes an array called "data
    data = filter_signal(data, eeg_channels) # uses the filter signal function above to clean data

    if datalogging == True:
        _timestamps = []
        data_to_log = data[eeg_channels]
        for count in range(data_to_log.shape[1]-1):
            dt = datetime.now()
            ts = datetime.timestamp(dt)
            _timestamps.append(ts)

        timearray = np.array(_timestamps)
        np.reshape(timearray, [len(timearray), 1])
        timearray = np.transpose(timearray)
        stamped_data = np.vstack((data_to_log,timearray))
        today = str(date.today())
        DataFilter.write_file(stamped_data, today+'-EEG_log.csv', 'a')

# for each 5 second epoch : raw -> z score normalise -> clip between -3 and 3 -> get (alpha/rest of the spectrum) / (theta/rest of the spectrum).
def get_alpha_theta_ratio(_data, _eeg_channels):
    _alpha_theta_array = []

    for channel in _eeg_channels:
        # z-score normalise raw data
        data_zscored = zscore(_data[channel])
        # clip between -3 and 3
        data_zscored = np.clip(data_zscored, -3 , 3)

        _psd = DataFilter.get_psd_welch(data_zscored, nfft, nfft // 2, sampling_rate, WindowOperations.BLACKMAN_HARRIS.value) 

        alpha = DataFilter.get_band_power(_psd, 8, 12)
        theta = DataFilter.get_band_power(_psd, 4, 8)

        _ratio = alpha/theta

        _alpha_theta_array.append(_ratio)

    _relative_alpha_theta_df = pd.DataFrame([_alpha_theta_array])

    return _relative_alpha_theta_df

def get_sleepieness(_sleepy_class):

    _input = get_alpha_theta_ratio(data,eeg_channels)
    _sleepiness = _sleepy_class.predict(_input)

    return _sleepiness



