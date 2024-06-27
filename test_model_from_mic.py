from utils.audio_recorder import AudioRecorder
import tensorflow as tf
from utils.spectrogram_generator import SpectrogramGenerator

while True:
    recorder = AudioRecorder()
    specgen = SpectrogramGenerator()

    recorder.record_audio_continuous()

    x = 'recorded_input/recording.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = specgen.get_spectrogram(x)
    x = x[tf.newaxis,...]

    imported = tf.saved_model.load("saved_model")
    results = imported(waveform[tf.newaxis, :])
    command = results['class_names'].numpy()[0].decode('utf-8')
    print(command)