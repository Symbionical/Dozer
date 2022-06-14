
from playsound import playsound

def all_stimulations():
    playsound('pink_noise.mp3')

def play_pink_noise():
    print("playing pink noise")
    playsound('pink_noise.mp3', block = False)