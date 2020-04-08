import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



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



# Emotion in the RAVDESS dataset

# dictionary to decode encoded target into verbal emotions
emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']



# Load the data and extract the features for each sound file
def load_data(test_size = 0.2):
    x = []
    y = []
    
    # selecting all filename using same pattern 
    for file in glob.glob(".\\RAVDESS\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        
        # Splitting each emotion and decoding it in emotion using dictionary
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        
        # calling extract feature function created above to get feature of file
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        
        # appending features and target for all the file
        x.append(feature)
        y.append(emotion)
    
    # returning train and test splitted dataset
    return train_test_split(np.array(x), y, test_size = test_size, random_state = 29)



# Split dataset into train and test
x_train, x_test, y_train, y_test = load_data(test_size = 0.25)

# Size of train and test data
print((x_train.shape[0], x_test.shape[0]))

# total features extracted
print("Features extracted:", x_train.shape[1])

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300,), learning_rate = 'adaptive', max_iter = 500)

# Train the model
model.fit(x_train, y_train)

# predict the test set
y_pred = model.predict(x_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy = ", accuracy)

# Predicting emotion for one of audio file
pred_file = r".\RAVDESS\Actor_01\03-01-08-02-02-02-01.wav"
file_feature = extract_feature(pred_file, mfcc=True, chroma=True, mel=True)
file_feature = file_feature.reshape(1,180)
print(model.predict(file_feature))

#Pickle code to save the model as Saved_model
saved_model = open("Saved_model",'ab')
smodel = pickle.dump(model, saved_model)
saved_model.close()
