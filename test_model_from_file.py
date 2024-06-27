import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from utils.spectrogram_generator import SpectrogramGenerator

DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)

specgen = SpectrogramGenerator()

x = data_dir/'no/01bb6a2a_nohash_0.wav'
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
x = tf.squeeze(x, axis=-1)
waveform = x
x = specgen.get_spectrogram(x)
x = x[tf.newaxis,...]

imported = tf.saved_model.load("saved_model")
print(imported(waveform[tf.newaxis, :]))


