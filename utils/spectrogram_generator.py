import tensorflow as tf

class SpectrogramGenerator:

    def get_spectrogram(self, waveform):
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

