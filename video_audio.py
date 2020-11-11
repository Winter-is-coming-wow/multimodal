# -- coding: utf-8 --
import librosa
import pyaudio
from multiprocessing import Process,Event,Queue
import struct as st
import matplotlib.pyplot as plt
from moviepy.editor import *
import os
import time
import numpy as np
import onnxruntime as ort
from audio_model import analyser
import tensorflow as tf
import cv2 as cv
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

label_dict={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprised', 6: 'normal'}

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def listen(channels,sample_rate,chunk,writer):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk  # pyaudio内置缓存区大小
                    )
    # Determine the timestamp of the start of the response interval
    print('* Start Recording *')
    stream.start_stream()
    # Record audio until timeout
    frames = []
    slices=[]
    plt.figure(figsize=(8, 2))
    while True:
        # Record data audio data
        data = stream.read(chunk)
        slice = st.unpack(str(chunk) + 'h', data)
        slice = [i / 32768.0 for i in slice]
        slices.extend(slice)
        if len(slices)>=chunk*2:
            writer.put(slices.copy())
            slices.clear()
        if len(frames) >= 48000:
            frames.clear()
        frames.extend(slice)
        plt.plot(range(len(frames)), frames)
        plt.xlim(0, 50000)
        plt.ylim(-1.0, 1.0)
        plt.pause(0.01)
        plt.clf()

    # Close the audio recording stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('* End Recording * ')

def load_audio(filename,sr,chunk,writer,ev):
    y,sr=librosa.load(filename,sr=sr)
    L=0
    q = pyaudio.PyAudio()
    out_stream = q.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=16000,
                        output=True
                        )
    slices=[]
    ev.set()
    while L+sr<=len(y):
        slice=y[L:L+sr]
        slices.extend(slice)
        if len(slices)>=chunk*2:
            writer.put(slices.copy())
            slices.clear()
        out_stream.write(st.pack(str(len(slice))+'f',*slice))
        L+=sr
        #time.sleep(1)
    return

def main(filepath=None):
    ser=analyser('cache/1.h5') #语音情感识别模型
    fer=tf.keras.models.load_model('cache/vedio.h5') #人脸表情识别模型
    onnx_path = r'G:\demo\python\practice\Sentiment-Analysis-audio\audio_vedio\cache\ultra_light_320.onnx'
    ort_session = ort.InferenceSession(onnx_path)#人脸检测模型
    input_name = ort_session.get_inputs()[0].name

    if filepath is None:
        cap=cv.VideoCapture(0)
    else:
        video_=VideoFileClip(filepath)
        audio_=video_.audio
        temp_audio='cache/temp.wav'
        audio_.write_audiofile(temp_audio)
        cap=cv.VideoCapture(filepath)

    channels = 1
    sample_rate = 16000
    chunk = 16000

    mQueue = Queue()
    e=Event()
    if filepath is None:
        subprocess = Process(target=listen, args=(channels, sample_rate, chunk, mQueue))
    else:
        subprocess=Process(target=load_audio, args=(temp_audio, sample_rate, chunk, mQueue,e))
    subprocess.start()

    audio_predict=None
    flag_e=True
    while True:
        if not mQueue.empty():
            msg=mQueue.get()
            signal = ser.endpoint_detection(msg)
            if (len(signal) < 12000):
                continue
            else:
                audio_predict=ser.predict(msg)
                print(time.ctime(),' : ',label_dict[np.argmax(audio_predict)])
        ret, frame = cap.read()
        if filepath is None:
            frame=cv.flip(frame,1)
        if frame is not None:
            h, w, _ = frame.shape
            # preprocess img acquired
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert bgr to rgb
            img = cv.resize(img, (320, 240))  # resize
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
            boxes = sorted(boxes, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序
            face_num = min(2, len(boxes))
            faces = []
            if face_num<1:
                cv.imshow('Video', frame)
                continue
            flag=False
            for i in range(face_num):
                box = boxes[i]
                x1, y1, x2, y2 = box
                if x1<0 or y1<0 or x2<0 or y2<0:
                    flag=True
                    break
                cv.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                face_cliped = frame[y1:y2 + 1, x1:x2 + 1, :].copy()
                gray = cv.cvtColor(face_cliped, cv.COLOR_RGB2GRAY)
                resized = cv.resize(gray, (48, 48)).astype(np.float32) / 255.0
                resized = np.expand_dims(resized, axis=-1)
                faces.append(resized)
            if flag:
                continue
            faces = np.asarray(faces)
            emotion = fer.predict(faces)
            if audio_predict is not None:
                emotion[0]+=audio_predict
            emotions=[ label_dict[e] for e in np.argmax(emotion,axis=1) ]
            font = cv.FONT_HERSHEY_DUPLEX
            for i in range(face_num):
                box = boxes[i]
                x1, y1, x2, y2 = box
                text = str(emotions[i])
                cv.putText(frame, text, (x1 - 2, y1 - 2), font, 0.5, (255, 255, 255), 1)
            if flag_e and filepath is not None:
                flag_e=False
                e.wait()
            cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    #filepath = r'G:\demo\python\practice\Sentiment-Analysis-audio\audio_vedio\cache\test.avi'
    filepath = r'G:\deeplearning\FER datasets\Raw\Video\Full\6Egk_28TtTM.mp4'
    main(filepath)