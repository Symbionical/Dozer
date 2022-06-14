
from playsound import playsound
# import bluepy.btle as btle #needs to be linux

def stimulate_all():
    stimulate_tES()
    stimulate_pink_noise()

def stimulate_pink_noise():
    print("playing pink noise")
    playsound('pink_noise.mp3', block = False)

def stimulate_tES():
    print("running tES")
    # focus1 = btle.Peripheral("B4:99:4C:4F:88:84") #specific for each focus device,
    # service1 = focus1.getServiceByUUID("0000AAB0-F845-40FA-995D-658A43FEEA4C")
    # characteristic1 = service1.getCharacteristics()[0]
    # characteristic1.write((bytes([2, 7, 5, 0, 0, 0]))) # third number in the sequence is the program number on the focus 
    