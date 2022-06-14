from calendar import c
import time

from numpy import true_divide

import bci
import stimulations


def main():

    cooldown = 0

    # initialise BCI parameters
    bci.init_bci("Synthetic") # Use this for synthetically generated test data
    # bci.init_bci("Cyton") # Use this when using the actual bci


    # MAIN LOOP #

    while True:
        
        time.sleep(10) # wait to collect 10 seconds worth of data. It is said that "EEG stationarity" is between 10 and 20 seconds, meaning that timeframes of this length are long enough to see real brain dynamics.
        bci.update_data() # prompts the BCI to pull any new data from the data buffer
        restfulness_val = bci.get_restfulness(bci.data,bci.eeg_channels) *100000000000 # get a measure of restfulness (i multiplied this by 10000000 because the value was originally super tiny )
        print('Restfulness: %s' % str(restfulness_val[0]))
        print('Cooldown: %s' % str(cooldown))

        # update cooldown
        if cooldown > 0:
            cooldown -= 10

        # if restuflness is past threshcold and cooldown is not active, begin stimulation and activate cooldown... i chose the threshold of 5 abitraily, i have no idea what would really be a good value as I havent tested this with real data yet
        if restfulness_val > 5.0 and cooldown == 0:
            stimulations.play_pink_noise()
            cooldown = 600

if __name__ == "__main__":
    main()