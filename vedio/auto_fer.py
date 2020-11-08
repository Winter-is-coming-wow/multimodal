# -- coding: utf-8 --
# -- coding: utf-8 --
# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-30 11:25:26
import cv2
import numpy as np
import onnx
import imutils
import onnxruntime as ort
from onnx_tf.backend import prepare
import tensorflow as tf
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from PyQt5 import QtGui, QtWidgets
from load_data import preprocess_input
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings(action='ignore')


from sys import argv, exit
from PyQt5.QtWidgets import QApplication,QMainWindow

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

class Emotion_Rec():
    def __init__(self, model_path=None):

        # 载入数据和图片的参数
        detection_model_path = r'G:\demo\python\practice\Sentiment-Analysis-audio\audio_vedio\cache\ultra_light_640.onnx'

        if model_path == None:  # 若未指定路径，则使用默认模型
            self.emotion_model_path = '../cache/vedio.h5'
        else:
            self.emotion_model_path = model_path

        # 载入人脸检测模型
        self.face_detection = ort.InferenceSession(detection_model_path)  # 级联分类器
        self.input_name=self.face_detection.get_inputs()[0].name
        # 载入人脸表情识别模型
        self.emotion_classifier = tf.keras.models.load_model(self.emotion_model_path)
        # 表情类别
        self.EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised",
                         "neutral"]

    def run(self, frame_in, canvas, label_face, label_result):
        frame_in = imutils.resize(frame_in, width=300)  # 缩放画面
        frameclone=frame_in.copy()
        h, w, _ = frame_in.shape
        img = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (640, 480))  # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = self.face_detection.run(None, {self.input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
        # frame_in 摄像画面或图像
        # canvas 用于显示的背景图
        # label_face 用于人脸显示画面的label对象
        # label_result 用于显示结果的label对象

        # 调节画面大小
        # frame = imutils.resize(frame_in, width=300)  # 缩放画面
        # # frame = cv2.resize(frame, (300,300))  # 缩放画面
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        #
        # # 检测人脸
        # faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
        #                                              minNeighbors=5, minSize=(30, 30),
        #                                              flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # 预测的结果
        label = None  # 预测的标签
        (fX, fY, fx2, fy2) = None, None, None, None  # 人脸位置

        if len(boxes) > 0:
            # 根据ROI大小将检测到的人脸排序
            faces = sorted(boxes, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序
            face_num=min(2,len(faces))
            for i in range(face_num):  # 遍历每张检测到的人脸，默认识别全部人脸
                # 如果只希望识别和显示最大的那张人脸，可取消注释此处if...else的代码段
                # if i == 0:
                #     i = -1
                # else:
                #     break

                (fX, fY, fx2, fy2) = faces[i]
                # 从灰度图中提取感兴趣区域（ROI），将其大小转换为与模型输入相同的尺寸，并为通过CNN的分类器准备ROI
                roi = gray[fY:fy2, fX:fx2]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = preprocess_input(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # 用模型预测各分类的概率
                preds = self.emotion_classifier.predict(roi)[0]
                # emotion_probability = np.max(preds)  # 最大的概率
                label = self.EMOTIONS[preds.argmax()]  # 选取最大概率的表情类

                # 圈出人脸区域并显示识别结果
                cv2.putText(frameclone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)
                cv2.rectangle(frameclone, (fX, fY), (fx2, fy2), (255, 255, 0), 1)

        # canvas = 255* np.ones((250, 300, 3), dtype="uint8")
        # canvas = cv2.imread('slice.png', flags=cv2.IMREAD_UNCHANGED)

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
            # 用于显示各类别概率
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # 绘制表情类和对应概率的条形图
            w = int(prob * 300) + 7
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # 调整画面大小与界面相适应
        frameClone = cv2.resize(frameclone, (420, 280))

        # 在Qt界面中显示人脸
        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)

        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()
        # 在显示结果的label中显示结果
        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return (label)