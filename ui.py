# -- coding: utf-8 --
from showface import Ui_MainWindow
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow
import cv2 as cv
import os
import time
import librosa
from vedio_analyser import frame_analyser
from moviepy.editor import VideoFileClip
import pyaudio
from audio_model import analyser
import struct as st
import numpy as np
label_dict={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprised', 6: 'normal'}

class Mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self,mainwindow):
        super().__init__()

        self.path=os.getcwd() #获取当前路径

        self.cap=cv.VideoCapture()  #摄像头
        self.cap2=None              #视频
        self.CAM_NUM=0

        self.timer=QtCore.QTimer()  #用于显示时间的定时器
        self.timer.start()

        self.setupUi(mainwindow)
        self.retranslateUi(mainwindow)

        self.thread_video=None
        self.thread_camera=Mythread(self.cap)
        self.thread_listen=listen()
        self.thread_load=load_audio()

        self.image_analyser=frame_analyser()#预测图片

        self.slots_init()

    def reset_ui(self):
        self.label_show.setText('请打开文件或摄像头')
        self.label_video.setText('None')
        self.label_audio.setText('None')
        self.label_multimodal.setText('None')
        self.label_text.setText('None')
        self.lineEdit_image.setText('选择图片')
        self.lineEdit_video.setText('选择视频')

    def slots_init(self):
        self.pushButton_camera.clicked.connect(self.on_pushbutton_carema_click)
        self.toolButton_picture.clicked.connect(self.on_toolbutton_picture_click)
        self.toolButton_video.clicked.connect(self.on_toolbutton_video_click)
        self.timer.timeout.connect(self.showtime)
        self.thread_camera.hasans.connect(self.show_ans)
        self.thread_listen.sinsignal.connect(self.thread_camera.handle_audio)



    def on_pushbutton_carema_click(self):
        if self.thread_video!=None:
            self.cap2.release()
            if self.thread_video.isRunning():
                self.thread_video.terminate()
                self.thread_load.terminate()

        if self.cap.isOpened() or self.thread_camera.isRunning():
            self.thread_camera.terminate()
            self.thread_listen.terminate()

        flag=self.cap.open(self.CAM_NUM)
        if not flag:
            msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                u"请检测相机与电脑是否连接正确！ ",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.reset_ui()
            #准备运行识别程序
            QtWidgets.QApplication.processEvents()
            # 对于执行很耗时的程序来说，由于PyQt需要等待程序执行完毕才能进行下一步，这个过程表现在界面上就是卡顿，
            # 而如果需要执行这个耗时程序时不断的刷新界面。那么就可以使用QApplication.processEvents()，
            # 那么就可以一边执行耗时程序，一边刷新界面的功能，给人的感觉就是程序运行很流畅，因此QApplicationEvents（）的使用方法就是，
            # 在主函数执行耗时操作的地方，加入QApplication.processEvents()
            self.label_show.setText('实时摄像已开启')
            self.thread_camera.start()
            self.thread_listen.start()

    def on_toolbutton_picture_click(self):
        if self.cap2!=None:
            self.cap2.release()
            if self.thread_video.isRunning():
                self.thread_video.terminate()
                self.thread_load.terminate()
        if self.thread_camera.isRunning():
            self.cap.release()
            self.thread_camera.terminate()
            self.thread_listen.terminate()
        self.reset_ui()
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if fileName_choose != '':
            self.lineEdit_image.setText(fileName_choose + '文件已选中')
            image = cv.imread(fileName_choose)  # 读取选择的图片
            # 计时并开始模型预测
            QtWidgets.QApplication.processEvents()
            # image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            self.image_analyser.predictor(image.copy(),self.label_show,type=1)

    def on_toolbutton_video_click(self):
        if self.thread_camera.isRunning():
            self.cap.release()
            self.thread_camera.terminate()
            self.thread_listen.terminate()
        # 界面处理
        if self.cap2!=None:
            self.cap2.release()
            self.thread_video.terminate()
            self.thread_load.terminate()

        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取视频文件",
            self.path,  # 起始路径
            "视频(*.mp4;*.avi)")  # 文件类型
        self.cap2 = cv.VideoCapture(fileName_choose)
        if fileName_choose != '':
            self.reset_ui()
            self.lineEdit_video.setText(fileName_choose + '文件已选中')
            self.thread_video=Mythread(self.cap2,fileName_choose)
            self.thread_video.hasans.connect(self.show_ans)
            self.thread_load.loadsignal.connect(self.thread_video.handle_audio)
            self.thread_video.start()
            self.thread_load.start()

    def showtime(self):
        #获取当前时间
        t=QtCore.QDateTime.currentDateTime()
        #设置时间显示格式
        time_display=t.toString('yyyy-MM-dd hh:mm:ss')
        #显示当前时间
        self.label_time.setText(time_display)

    def show_ans(self,answer):
        QtWidgets.QApplication.processEvents()
        ans=answer[0]
        frame=answer[1]
        if ans !=None:
            self.label_video.setText(ans[0])
            self.label_audio.setText(ans[1])
            self.label_text.setText(ans[2])
            self.label_multimodal.setText(ans[3])
        h,w=frame.shape[:2]
        frameClone = cv.resize(frame, (600, int(600 / w * h)))
        # 在Qt界面中显示人脸
        show = cv.cvtColor(frameClone, cv.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show.setPixmap(QtGui.QPixmap.fromImage(showImage))






class Mythread(QtCore.QThread):
    hasans = QtCore.pyqtSignal(list)  # 将结果传到主线程
    def __init__(self,cap,filepath=None):
        super().__init__()
        self.cap=cap
        self.analyzer_frame=frame_analyser()  #预测图片
        self.filepath=filepath
        self.ser=analyser('cache/1.h5')  #预测语音
        self.audio_predict=[0]*7
    # def return_ans(self,pre):
    #     boxes=pre[0]
    #     frame=pre[2]
    #     pre=pre[1]
    #     pre=pre[0]
    #     multimodal=pre+self.audio_predict
    #     if self.audio_predict==[0]*7:
    #         audio_ans=None
    #     else:
    #         audio_ans=np.argmax(self.audio_predict)
    #         self.audio_predict=[0]*7
    #     frame_ans=np.argmax(pre)
    #     multimodal_ans=np.argmax(multimodal)
    #     ans=[frame_ans,audio_ans,None,multimodal_ans]
    #
    #     self.hasans.emit([ans,boxes,frame])


    def handle_audio(self,audio_slice):
        signal = self.ser.endpoint_detection(audio_slice)
        if len(signal) > 12000:
            self.audio_predict = self.ser.predict(audio_slice).tolist()


    def run(self):
        if self.filepath!=None:
            video_ = VideoFileClip(self.filepath)
            audio_ = video_.audio
            temp_audio = 'cache/temp.wav'
            audio_.write_audiofile(temp_audio)
        while True:
            ret,frame=self.cap.read()
            if ret:
                if self.filepath == None:
                    frame = cv.flip(frame, 1)
                boxes,emotion=self.analyzer_frame.predictor(frame.copy())
                if boxes!=None:
                    multimodal = emotion[0] + self.audio_predict
                    if self.audio_predict == [0] * 7:
                        audio_ans = 'None'
                    else:
                        audio_ans = label_dict[np.argmax(self.audio_predict)]
                        #self.audio_predict = [0] * 7
                    frame_ans = label_dict[np.argmax(emotion[0])]
                    multimodal_ans = label_dict[np.argmax(multimodal)]
                    ans = [frame_ans, audio_ans, 'None', multimodal_ans]

                    face_num = min(1, len(boxes))

                    if face_num > 0:
                        font = cv.FONT_HERSHEY_DUPLEX
                        for i in range(face_num):
                            box = boxes[i]
                            x1, y1, x2, y2 = box
                            cv.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                            text = str(multimodal_ans)
                            cv.putText(frame, text, (x1 - 2, y1 - 2), font, 0.5, (255, 255, 255), 1)
                    self.hasans.emit([ans,frame])

                else:
                    self.hasans.emit([None,frame])





class listen(QtCore.QThread):
    sinsignal=QtCore.pyqtSignal(list)
    def run(self):
        channels=1
        sample_rate=16000
        chunk=sample_rate
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
        slices = []
        #plt.figure(figsize=(8, 2))
        while True:
            # Record data audio data
            data = stream.read(chunk)
            slice = st.unpack(str(chunk) + 'h', data)
            slice = [i / 32768.0 for i in slice]
            slices.extend(slice)
            if len(slices) >= chunk * 2:
                self.sinsignal.emit(slices.copy())
                slices.clear()
            #     writer.put(slices.copy())
            #     slices.clear()
            # if len(frames) >= 48000:
            #     frames.clear()
            # frames.extend(slice)
            # plt.plot(range(len(frames)), frames)
            # plt.xlim(0, 50000)
            # plt.ylim(-1.0, 1.0)
            # plt.pause(0.01)
            # plt.clf()

        # Close the audio recording stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording * ')


class load_audio(QtCore.QThread):
    loadsignal = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.filename='cache/temp.wav'

    def run(self):
        sample_rate = 16000
        chunk=sample_rate
        y, sr = librosa.load(self.filename, sr=sample_rate)
        L = 0
        q = pyaudio.PyAudio()
        out_stream = q.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=16000,
                            output=True
                            )
        # Determine the timestamp of the start of the response interval
        slices = []
        while L + sr <= len(y):
            slice = y[L:L + sr]
            slices.extend(slice)
            L += sr
            if len(slices) >= chunk * 2:
                self.loadsignal.emit(slices.copy())
                slices.clear()
            out_stream.write(st.pack(str(len(slice)) + 'f', *slice))

