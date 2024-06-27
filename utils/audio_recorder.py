import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import queue
import os

class AudioRecorder:
    # Define a function to record audio continuously and save it as a .wav file after detecting speech
    def record_audio_continuous(self, sample_rate=44100, silence_threshold=0.03):
        # Create a queue to hold the incoming audio chunks
        q = queue.Queue()

        # Initialize variables to track silence
        silent_chunks = 0
        silence_limit = 20  # Convert silence duration to number of samples
        silent_rec = True

        # Specify the directory to save the .wav file
        current_directory = os.getcwd()

        # Callback function to stream audio chunks into the queue
        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())

        # Open a new stream for recording
        with sd.InputStream(samplerate=sample_rate, channels=2, callback=callback):
            print("Recording started. Speak into the microphone.")
            silent_rec = True
            # Create an empty list to hold the audio data
            audio_data = []
            while True:
                # Retrieve audio chunk from the queue
                chunk = q.get()
                # Append chunk to the list
                audio_data.append(chunk)
                # Check if the chunk is silent
                if np.abs(chunk).mean() < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                    silent_rec = False
                # If silence is detected for longer than the silence limit, stop recording
                if silent_chunks > silence_limit:
                    # Concatenate all audio chunks
                    audio_data = np.concatenate(audio_data, axis=0)
                    # Save the recording as a .wav file
                    filename = f"{current_directory}/recorded_input/recording.wav"
                    if(silent_rec == False):
                        audio_data_16bit = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
                        # Save the audio file using scipy.io.wavfile.write
                        write(filename, sample_rate, audio_data_16bit)
                        print(f"Recording saved as {filename}")
                        return
                    silent_chunks = 0
                    audio_data = []
                    silent_rec = True
                    
                    
                # print(silent_chunks)