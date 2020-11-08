# -- coding: utf-8 --
import pickle
import numpy as np
import os
import librosa
import time
import itertools
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import tensorflow as tf
# 0 anger 生气； 1 disgust 厌恶；2 fear 恐惧； 3 happy 开心； 4 sad 伤心；5 surprised 惊讶； 6 normal 中性
label_dict_ravdess = {'01': 6, '03': 3, '04': 4, '05': 0, '06': 2, '07': 1,'08': 5}
# 数据增强
def data_aug(signal, snr_low=15, snr_high=30, nb_augmented=2):
    # Signal length
    signal_len = len(signal)

    # Generate White noise
    noise = np.random.normal(size=(nb_augmented, signal_len))

    # Compute signal and noise power
    s_power = np.sum((signal / (2.0 ** 15)) ** 2) / signal_len
    n_power = np.sum((noise / (2.0 ** 15)) ** 2, axis=1) / signal_len

    # Random SNR: Uniform [15, 30]
    snr = np.random.randint(snr_low, snr_high)

    # Compute K coeff for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- snr / 10))
    K = np.ones((signal_len, nb_augmented)) * K

    # Generate noisy signal
    return signal + K.T * noise


# 提取特征

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128,fmax=4000):
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels,fmax=fmax)

    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect


def data_loader():
    print("Import Data: START")

    #audio file path
    file_path=r'G:\deeplearning\FER datasets\RAVDESS\savessData\Audio_speechh_Actors_01-24'

    #audio filename
    filenames=[]
    #audio file label
    labels=[]
    #audio data
    signals=[]

    sample_rate=16000
    max_pad_len = 49100

    keys=list(label_dict_ravdess.keys())
    for dir in os.listdir(file_path):
        for file in os.listdir(os.path.join(file_path,dir)):
            filenames.append(os.path.join(file_path,dir,file))
    #shuffle train data
    np.random.shuffle(filenames)

    #read audio files
    audio_index=0
    for audio_file in filenames[:10]:
        id=audio_file.split('\\')[-1][6:8]
        if id in keys:
            audio_index+=1
            # Read audio file
            y, sr = librosa.load(os.path.join(file_path, dir, file),sr=sample_rate)
            # Z-normalization
            y = zscore(y)
            if len(y) < max_pad_len:
                y_padded = np.zeros(max_pad_len)
                y_padded[:len(y)] = y
                y = y_padded
            elif len(y) > max_pad_len:
                y = np.asarray(y[:max_pad_len])

            # Add to signal list
            signals.append(y)
            # Set label
            labels.append(label_dict_ravdess.get(id))

            # Print running...
            if (audio_index % 100 == 0):
                print("Import Data: RUNNING ... {} files".format(audio_index))
    print("Import Data END\n")
    print("Number of audio files imported: {}".format(len(labels)))

    #Build Train and test dataset
    train_data,test_data,train_label,test_label=train_test_split(signals,labels,test_size=0.2,shuffle=False)

    #Train data augument
    train_auged = np.asarray(list(map(data_aug, train_data)))

    train_auged=np.asarray([train_auged[i][j] for j in range(2) for i in range(len(train_auged))])
    #Build augmented lables
    train_auged_label=np.asarray(train_label*2)
    train_label = np.asarray(train_label)

    #extract train data feature
    mel_spect = np.asarray(list(map(mel_spectrogram, train_data)))
    augmented_mel_spect = np.asarray(list(map(mel_spectrogram, train_auged)))

    #build train  dataset
    x_train=np.concatenate((mel_spect,augmented_mel_spect))
    y_train=np.concatenate((train_label,train_auged_label))

    #build test dataset
    x_test=np.asarray(list(map(mel_spectrogram, test_data)))
    y_test=np.asarray(list(test_label))

    # Split spectrogram into frames
    def frame(x, win_step, win_size):
        nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
        frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
        for t in range(nb_frames):
            frames[:, t, :, :] = np.copy(x[:, :, (t * win_step):(t * win_step + win_size)]).astype(np.float32)
        return frames

    win_ts = 128
    hop_ts = 64
    # Frame for TimeDistributed model
    X_train = frame(x_train, hop_ts, win_ts)
    X_test = frame(x_test, hop_ts, win_ts)
    print(X_train.shape,X_test.shape)
    pickle.dump(X_train.astype(np.float16), open('../cache/X_train.p', 'wb'))
    pickle.dump(y_train, open('../cache/y_train.p', 'wb'))
    pickle.dump(X_test.astype(np.float16), open('../cache/X_test.p', 'wb'))
    pickle.dump(y_test, open('../cache/y_test.p', 'wb'))


data_loader()