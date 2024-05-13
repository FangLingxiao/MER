import librosa
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch 
import torchaudio
import torch.nn.functional as F
import pandas as pd
import multiprocessing as mp 
from torch.nn import Sequential

from const import *

# First, all the audio files need to be transform to mel_spectrogram.
class AudioDataset(Dataset):
    def __init__(self, dataframe): 
        super().__init__()
        self.data = self.prepare_wavs(dataframe)
        
    def prepare_wavs(self, df):
        l = []
        for idx, i in enumerate(df["audio_path"]):   # Traversing the entire database
            # Sampling the audio data
            audio, sr = librosa.load(i)   

            # Zero padding if the audio length is less than the target length
            target_samples = DEFAULT_FREQ * DEFAULT_TIME
            if len(audio) < target_samples:
                pad_length = target_samples - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')

            # Truncate it if the audio length is longer than the target length
            elif len(audio) > target_samples:
                audio = audio[:target_samples]

            # Convert it into Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sr,n_fft=2048,hop_length=512,n_mels=64)
            # Execute logarithmic operations
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)   
            # Convert numpy array into tensor
            tensor_log_mel_spectrogram = torch.tensor(log_mel_spectrogram)   
            # Data normalization
            tensor_log_mel_spectrogram  = (tensor_log_mel_spectrogram - torch.mean(tensor_log_mel_spectrogram)) / torch.std(tensor_log_mel_spectrogram)
            # Spectrogram padding
            padding_log_mel_spectrogram= F.pad(tensor_log_mel_spectrogram, (0,1200-tensor_log_mel_spectrogram.shape[1],0,0),"constant",0)

            # Combine the tensors with labels and put it into the list
            row = df.loc[idx]
            valence_level = int(row["valence_level"])
            arousal_level = int(row["arousal_level"])

            # label them
            if valence_level == 2 and arousal_level == 2:
                emotion_label = 0  # happy 
            elif valence_level == 1 and arousal_level == 2:
                emotion_label = 1  # tense /excited
            elif valence_level == 0 and arousal_level == 2:
                emotion_label = 2  # angry
            elif valence_level == 2 and arousal_level == 1:
                emotion_label = 3  # content
            elif valence_level == 1 and arousal_level == 1:
                emotion_label = 4  # neutral
            elif valence_level == 0 and arousal_level == 1:
                emotion_label = 5  # distressed   
            elif valence_level == 2 and arousal_level == 0:
                emotion_label = 6  # relax / calm
            elif valence_level == 1 and arousal_level == 0:
                emotion_label = 7  # peaceful 
            else:
                emotion_label = 8  # sad

            # one-hot encode
            #label = torch.zeros(9)
            #label[emotion_label] = 1
            label = torch.tensor(emotion_label)
            #print(f"Label shape: {label.shape}, Label dtype: {label.dtype}")  # 添加调试信息
            l.append((padding_log_mel_spectrogram, label))
            print(f"{idx+1}/{len(df)} ({100*(idx+1)/len(df):.1f}%)", end = "\r" if idx < len(df) else "\n", flush=True)
        return l

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        return self.data[i]



# The collate function
def custom_collate_fn(batch):
    image_tensors = []
    labels = []
    for image, label in batch:
        image_tensor = image.unsqueeze(0)  # Add 1 dimension at the beginning
        image_tensors.append(image_tensor)
        labels.append(label)

    image_batch_tensor = torch.stack(image_tensors)  # Convert list into tensor
    label_batch_tensor = torch.stack(labels)
    return (image_batch_tensor,label_batch_tensor)

def load_data(data_path, batch_sz=10, train_test_split=[0.7, 0.3]):
    # This is a convenience funtion that returns dataset splits of train, val and test according to the fractions specified in the arguments
    assert sum(train_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    dataset = AudioDataset(data_path)  # Instantiating our previously defined dataset
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_te = []
    for frac in train_test_split:   # Calculate the number of data in train and test part
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_te.append(actual_count)
    
    # split dataset into train, val and test
    train_split, test_split = random_split(dataset, tr_te)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    n_cpus = mp.cpu_count()
    train_dl = DataLoader(train_split, 
                          batch_size=batch_sz, 
                          shuffle=True, 
                          collate_fn=custom_collate_fn,
                          num_workers=n_cpus)            

    test_dl = DataLoader(test_split,
                         batch_size=batch_sz,
                         shuffle=False,
                         collate_fn=custom_collate_fn,
                         num_workers=n_cpus)
    return train_dl, test_dl
