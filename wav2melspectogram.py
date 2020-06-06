# 0 = cough, 1 = not_cough
#directories - data -> 0 and 1 | img_data

#code to covert .wav files into mel-spectogram and save them in a folder
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab
import librosa
import librosa.display
import numpy as np
import pathlib

#Extracting the Mel-Spectrogram for every Audio
class_ = '0 1'.split()
for g in class_:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'./data/{g}'):
        audioname = f'./data/{g}/{filename}'
        y, sr = librosa.load(audioname, mono=True, duration=5)

        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        S = librosa.feature.melspectrogram(y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png', bbox_inches=None, pad_inches=0)
        pylab.close()
