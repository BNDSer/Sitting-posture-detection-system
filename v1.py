# encodin: utf-8

import threading
import cv2
from ultralytics import YOLO
import time
import winsound
import pyautogui
import numpy
import torch


sensitivity = float(input('Input the sensitivity(1.2~1.5, Recommend 1.3):'))

class RTSCapture(cv2.VideoCapture):
    """Real Time Streaming Capture.
    这个类必须使用 RTSCapture.create 方法创建，不要直接实例化
    """

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"] #用于识别实时流

    @staticmethod
    def create(url, *schemes):
        """实例化&初始化
        rtscap = RTSCapture.create("rtsp://example.com/live/1")
        or
        rtscap = RTSCapture.create("http://example.com/live/1.m3u8", "http://")
        """
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 这里可能是本机设备
            rtscap._reading = True

        return rtscap

    def isStarted(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        """启动子线程读取视频帧"""
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()



#model = YOLO('D:/ZichenFeng/Python/runs/detect/train13/weights/best.pt')
model = YOLO('yolov8n-pose.pt')


rtscap = RTSCapture.create(1)
#rtscap = RTSCapture.create("rtsp://admin:admin@192.168.0.100:8554/live")
#rtscap = RTSCapture.create("http://192.168.0.100:4747/video")
#rtscap = RTSCapture.create("https://192.168.0.100:8080")


rtscap.start_read() #启动子线程并改变 read_latest_frame 的指向


keypoints = torch.tensor([])
keypoints.xyn = torch.tensor([])
eyeDistant = 0
standardDistant = -1


def check_box():
    while True:
        if eyeDistant < sensitivity*standardDistant:
            winsound.Beep(2000,120)
            time.sleep(0.4)

        time.sleep(0.05)
    
t = threading.Thread(target=check_box, args = ())
t.start()



while rtscap.isStarted():
    ok, frame = rtscap.read_latest_frame() #read_latest_frame() 替代 read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not ok:
        continue


    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    keypoints = results[0].keypoints
    keypoints = keypoints.cpu()
    keypoints = keypoints.numpy()
    keypoints.xyn
    
    if keypoints.xyn[0].size > 0:

        x1 = keypoints.xyn[0][1][0]
        y1 = keypoints.xyn[0][1][1]
        x6 = keypoints.xyn[0][6][0]
        y6 = keypoints.xyn[0][6][1]
        x5 = keypoints.xyn[0][5][0]
        y5 = keypoints.xyn[0][5][1]
        x7 = keypoints.xyn[0][7][0]
        y7 = keypoints.xyn[0][7][1]

        eyeDistant = ((x7-x1)**2 + (y7-y1)**2)**0.5
        standardDistant = ((x5-x6)**2 + (y5-y6)**2)**0.5

        print(eyeDistant)
        print(standardDistant)



rtscap.stop_read()
rtscap.release()
cv2.destroyAllWindows()