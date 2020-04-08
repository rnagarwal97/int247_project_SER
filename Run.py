import numpy as np
import librosa
import librosa.display
import soundfile
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
import tkinter.filedialog

# This function is repeated again (from Model.py)
# Extract features from sound file like mfcc, chroma and mel
def extract_feature(file_name, mfcc, chroma, mel):
    
    # Open filename with soundfile
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        # if features are present include it in result ndarray
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs= np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate).T, axis=0)
            result=np.hstack((result, mel))
            
    # return result ndarray with all the features of that file        
    return result


# loading trained model from pickle file
model = pickle.load(open("Saved_model","rb"))


# tkinter file open dialog to let user choose audio file
file_p = tkinter.filedialog.askopenfile(mode="r",  filetypes=[('Audio Files', ['.wav'])])
pred_file = file_p.name

# predictng the output using saved model
ans = model.predict(extract_feature(pred_file, mfcc=True, chroma=True, mel=True).reshape(1,180))
print(ans)

# Graph for the audio waveform
y, sr = librosa.load(pred_file, duration=10)
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title("Waveform Graph")
plt.show()
