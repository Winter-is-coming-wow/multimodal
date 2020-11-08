# -- coding: utf-8 --
# -- coding: utf-8 --
import librosa
import os
import time
import pyaudio
import multiprocessing
from multiprocessing import Process,Pipe
import struct as st
import wave
import matplotlib.pyplot as plt
import soundfile

from audio_model import analyser

def listen(channels,sample_rate,chunk,writer):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk  #pyaudio内置缓存区大小
                    )
    # Determine the timestamp of the start of the response interval
    print('* Start Recording *')
    stream.start_stream()
    # Record audio until timeout
    while True:
        # Record data audio data
        data = stream.read(chunk)
        writer.send(data)

    # Close the audio recording stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('* End Recording * ')

def draw(reader):
    frames=[]
    plt.figure(figsize=(15,5))
    while True:
        if len(frames)>50000:
            frames.clear()
        recv=reader.recv()
        frames.extend(recv)
        plt.plot(range(len(frames)),frames)
        plt.xlim(0,50000)
        plt.ylim(-1.0,1.0)
        plt.pause(0.01)
        plt.clf()

def load_audio(filename,sr,writer):
    y,sr=librosa.load(filename,sr=sr)
    L=0
    start=time.time()
    while L+sr<len(y):
        writer.send(y[L:L+sr].copy())
        L+=sr
        time.sleep(1)
    if L<len(y):
        writer.send(y[L:].copy())
    return


def main(filename=None):
    ser=analyser('../cache/1.h5')

    channels=1
    sample_rate=16000
    chunk=16000

    if(filename==None):
        one,another=Pipe()
        one_,another_=Pipe()
        subprocess=Process(target=listen,args=(channels,sample_rate,chunk,one))
        subprocess2=Process(target=draw,args=(another_,))
        subprocess.start()
        subprocess2.start()

        frame=[]
        while True:
            try:
                recv=another.recv()
            except EOFError:
                break
            slice = st.unpack(str(chunk)+'h', recv)
            slice=[i/32768.0 for i in slice]
            one_.send(slice)
            # if(len(frame)<sample_rate*3):
            #     frame.extend(slice)
            # else:
                #soundfile.write('G:/000.wav',frame,16000)
            signal=ser.endpoint_detection(slice.copy())
            frame.clear()
            if(len(signal)<2000):
                continue
            else:
                ser.predict(signal)

    else:
        one, another = Pipe()
        one_, another_ = Pipe()
        subprocess = Process(target=load_audio, args=(filename, sample_rate, one))
        subprocess2 = Process(target=draw, args=(another_,))
        subprocess.start()
        subprocess2.start()

        frame = []
        while True:
            if not subprocess.is_alive():
                subprocess.close()
                subprocess2.terminate()
                break
            try:
                recv = another.recv()
            except EOFError:
                break
            if (len(frame) < sample_rate * 3):
                frame.extend(recv)
                one_.send(recv)
            else:
                signal = ser.endpoint_detection(frame.copy())
                frame.clear()
                if (len(signal) < 8000):
                    continue
                else:
                    ser.predict(signal)



    print('=======================================END====================================')

if __name__ == "__main__":
    #filename = r'G:\deeplearning\FER datasets\IEMOCAP语料库\Session1\Session1\dialog\wav\Ses01F_impro01.wav'
    filename = r'G:\deeplearning\FER datasets\RAVDESS\savessData\Audio_speechh_Actors_01-24\Actor_01\03-01-01-01-01-01-01.wav'
    main()
