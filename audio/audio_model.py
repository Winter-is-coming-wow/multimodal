# -- coding: utf-8 --
import librosa
import librosa.display
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.stats import zscore

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM


class analyser:
    '''
    Voice recording function
    '''

    def __init__(self, subdir_model=None):

        # Load prediction model
        if subdir_model is not None:
            self._model = self.build_model()
            self._model.load_weights(subdir_model)



        # Emotion encoding
        # 0 anger 生气； 1 disgust 厌恶；2 fear 恐惧； 3 happy 开心； 4 sad 伤心；5 surprised 惊讶； 6 normal 中性
        # Emotion encoding
        self._emotion = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprised', 6: 'normal'}

    def build_model(self):

        # Clear Keras session
        K.clear_session()

        # Define input
        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        # First LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(
            input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(
            y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

        # Second LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(
            y)
        y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

        # Third LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(
            y)
        y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

        # Fourth LFLB (local feature learning block)
        y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(
            y)
        y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

        # Flat
        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)

        # LSTM layer
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)

        # Fully connected
        y = Dense(7, activation='softmax', name='FC')(y)

        # Build final model
        model = Model(inputs=input_y, outputs=y)

        return model

    def endpoint_detection(self,y,win_step=64, win_size=128):
        # pad原数据
        len_y = len(y)
        n_frame = int(np.ceil((len_y - win_size) / win_step) + 1)
        pad_length = int((n_frame - 1) * win_step + win_size)  # 所有帧加起来总的铺平后的长度
        zeros = np.zeros((pad_length - len_y,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
        pad_signal = np.concatenate((y, zeros))  # 填补后的信号记为pad_signal

        # 分帧
        indices = np.tile(np.arange(0, win_size), (n_frame, 1)) + np.tile(np.arange(0, n_frame * win_step, win_step),(win_size, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = pad_signal[indices]  # 得到帧信号

        #  计算短时能量
        E = np.sum(np.square(frames), axis=1).tolist()
        # 计算过零率
        Z = librosa.feature.zero_crossing_rate(pad_signal, 128, 64, center=False)[0].tolist()

        E_high = 0.001  # 能量高门限
        E_low = np.sum(E[:10]) / 10  # 能量低门限
        Z_thresh = np.sum(Z[:10]) / 10  # 过零率门限
        # print(E_high,E_low,Z_thresh)
        frame_thresh = 10
        seg = []
        a = []
        # 用能量高门限过滤
        flag = False
        for i in range(len(E)):
            if E[i] >= E_high and not flag:
                a.append(i)
                flag = True
            elif E[i] < E_high and flag:
                a.append(i - 1)
                seg.append(a.copy())
                a.clear()
                flag = False
            if i == len(E) - 1 and flag:
                a.append(i)
                seg.append(a.copy())
                a.clear()
        # 用能量低门限过滤
        for i in range(len(seg)):
            left, right = seg[i]
            l = left - 1
            r = right + 1
            while l >= 0:
                if E[l] < E_low:
                    break
                else:
                    l -= 1
            while r < len(E):
                if E[r] < E_low:
                    break
                else:
                    r += 1
            seg[i] = [l + 1, r - 1]
        # 合并片段
        i = 0
        while (i < len(seg) - 1):
            j = i + 1
            while (j < len(seg)):
                if (seg[i][1] >= seg[j][0] - 1):
                    seg[i][1] = seg[j][1]
                    del seg[j]
                else:
                    i += 1
                    break
        #print(seg)
        # 用短时过零率判断是否是清音
        for i in range(len(seg)):
            left, right = seg[i]
            l = left - 1
            r = right + 1
            while l >= 0:
                if Z[l] < Z_thresh:
                    break
                else:
                    l -= 1
            while r < len(E):
                if Z[r] < Z_thresh:
                    break
                else:
                    r += 1
            seg[i] = [l + 1, r - 1]
        # 合并片段
        i = 0
        while (i < len(seg) - 1):
            j = i + 1
            while (j < len(seg)):
                if (seg[i][1] >= seg[j][0] - 1):
                    seg[i][1] = seg[j][1]
                    del seg[j]
                else:
                    i += 1
                    break

        seg = list(filter(lambda X: X[1] - X[0] > frame_thresh, seg))  # 删除过小片段

        frame_chosen = []
        for index in seg:
            frame_chosen.extend(range(index[0], index[1] + 1))
        signal = []  # 用来存储筛选的帧
        for d in range(len(frame_chosen)):
            i = frame_chosen[d]
            if d == len(frame_chosen) - 1 or i + 1 != frame_chosen[d + 1]:
                signal.extend(frames[i].tolist())
            else:
                signal.extend(frames[i, :win_step].tolist())
        # print(type(signal[1]))
        # signal = np.array(signal, dtype=np.float32)
        return signal

    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128):
        # Compute spectogram
        mel_spect = np.abs(
            librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

        # Compute mel spectrogram
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels)

        # Compute log-mel spectrogram
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        return np.asarray(mel_spect)

    '''
    Audio framing
    '''

    def frame(self, y, win_step=64, win_size=128):

        # Number of frames
        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

        # Framming
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
        for t in range(nb_frames):
            frames[:, t, :, :] = np.copy(y[:, :, (t * win_step):(t * win_step + win_size)]).astype(np.float16)

        return frames

    def predict(self,y,sr=16000):
        chunk_size=sr*2
        y=np.array(y)
        y = zscore(y)

        if len(y)<chunk_size:
            y_padded = np.zeros(chunk_size)
            y_padded[:len(y)] = y
            y = y_padded
        elif len(y) > chunk_size:
            y = np.asarray(y[:chunk_size])

        y = y.reshape((1, -1))

        # Compute mel spectrogram
        mel_spect = np.asarray(list(map(self.mel_spectrogram, y)))

        # Time distributed Framing
        #print(mel_spect.shape)
        mel_spect_ts = self.frame(mel_spect)

        ans=self._model.predict(mel_spect_ts)[0]

        return ans

