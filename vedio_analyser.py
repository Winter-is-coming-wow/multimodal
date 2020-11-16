# -- coding: utf-8 --
import os
import time
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import cv2 as cv
from PyQt5 import QtGui
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

label_dict={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprised', 6: 'normal'}

class frame_analyser():
    def __init__(self):
        self.model=tf.keras.models.load_model('cache/vedio.h5')
        self.onnx_path = r'cache\ultra_light_320.onnx'
        self.ort_session = ort.InferenceSession(self.onnx_path)  # 人脸检测模型
        self.input_name = self.ort_session.get_inputs()[0].name

    def area_of(self,left_top, right_bottom):
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

    def iou_of(self,boxes0, boxes1, eps=1e-5):
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

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hard_nms(self,box_scores, iou_threshold, top_k=-1, candidate_size=200):
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
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def predict(self,width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
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
            box_probs = self.hard_nms(box_probs,
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

    def predictor(self,frame,label_face=None,type=2):
        h, w, _ = frame.shape
        # preprocess img acquired
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert bgr to rgb
        img = cv.resize(img, (320, 240))  # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: img})
        boxes, labels, probs = self.predict(w, h, confidences, boxes, 0.7)
        boxes = sorted(boxes, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序
        if type==2:
            face_num = min(1, len(boxes))
            faces = []
            if face_num >0:
                for i in range(face_num):

                    box = boxes[i]
                    x1, y1, x2, y2 = box
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                        continue
                    cv.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                    face_cliped = frame[y1:y2 + 1, x1:x2 + 1, :].copy()
                    gray = cv.cvtColor(face_cliped, cv.COLOR_RGB2GRAY)
                    resized = cv.resize(gray, (48, 48)).astype(np.float32) / 255.0
                    resized = np.expand_dims(resized, axis=-1)
                    faces.append(resized)
                if len(faces)>0:
                    faces = np.asarray(faces)
                    emotion = self.model.predict(faces)
                    return boxes,emotion
                else:
                    return None,None

                # 调整画面大小与界面相适应
                # frameClone = cv.resize(frame, (600, int(600/w*h)))
                # # 在Qt界面中显示人脸
                # show = cv.cvtColor(frameClone, cv.COLOR_BGR2RGB)
                # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                # label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
            else:
                return None,None

        else:
            face_num = len(boxes)
            faces = []
            if face_num > 0:
                for i in range(face_num):
                    box = boxes[i]
                    x1, y1, x2, y2 = box
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                        continue
                    cv.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                    face_cliped = frame[y1:y2 + 1, x1:x2 + 1, :].copy()
                    gray = cv.cvtColor(face_cliped, cv.COLOR_RGB2GRAY)
                    resized = cv.resize(gray, (48, 48)).astype(np.float32) / 255.0
                    resized = np.expand_dims(resized, axis=-1)
                    faces.append(resized)
                if len(faces) > 0:
                    faces = np.asarray(faces)
                    emotion = self.model.predict(faces)
                    emotions = [label_dict[e] for e in np.argmax(emotion, axis=1)]
                    font = cv.FONT_HERSHEY_DUPLEX
                    for i in range(face_num):
                        box = boxes[i]
                        x1, y1, x2, y2 = box
                        text = str(emotions[i])
                        cv.putText(frame, text, (x1 - 2, y1 - 2), font, 0.5, (255, 255, 255), 1)
            frameClone = cv.resize(frame, (600, int(600 / w * h)))
            # 在Qt界面中显示人脸
            show = cv.cvtColor(frameClone, cv.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))


