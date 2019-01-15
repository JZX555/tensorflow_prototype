# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:16:11 2019

@author: 50568
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import cv2

def Label_Read(Path):
    labels = []
    
    return labels

def Video_Read(Path, Shape, Size):
    """
        Video_Read is used to extract the image of mouth from a video;\n
        parameter:\n
        Path: the string path of video\n
        Shape: the (min, max) size tuple of the mouth you extract from the video\n
        Size: the (high, weight) size tuple of the mouth image you save
    """
    cap = cv2.VideoCapture(Path)
    images = []
    mouth = None
    
    if(cap.isOpened() == False):
        print("Read video failed!")
        return None
    
    #检测是否摄像头正常打开:成功打开时，isOpened返回ture
    classifier_face = cv2.CascadeClassifier("D:/Code/opencv_contrib/modules/face/data/cascades/haarcascade_frontalface_alt.xml")
    #定义分类器（人脸识别）
    classifier_eye = cv2.CascadeClassifier("D:/Code/opencv_contrib/modules/face/data/cascades/haarcascade_eye.xml")
    #定义分类器（人眼识别）
    classifier_mouth=cv2.CascadeClassifier("D:/Code/opencv_contrib/modules/face/data/cascades/haarcascade_mcs_mouth.xml")
    #定义分类器（嘴巴识别）
    
    while (True):
        # 取得cap.isOpened()返回状态为True,即检测到人脸       
        ret, img = cap.read()
        img = cv2.imread("C:/Users/50568/Desktop/qqq.png")     
        '''
            第一个参数ret的值为True或False，代表有没有读到图片
            第二个参数是frame，是当前截取一帧的图片
        '''
        
        if ret == False:
            break
        
        faceRects_face = classifier_face.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, Shape)
        #检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
        key = cv2.waitKey(1)
        #键盘等待
        if len(faceRects_face) > 0:
            #检测到人脸
            for faceRect_face in faceRects_face:
                x, y, w, h = faceRect_face
                #获取图像x起点,y起点,宽，高
                h1=int(float(h / 1.8))
                #截取人脸区域高度的一半位置，以精确识别眼睛的位置
                intx = int(x)
                inty = int(y)
                intw = int(w)
                inth = int(h)
                #转换类型为int，方便之后图像截取
                my = int(float(y + 0.6 * h))
                #截取人脸区域下半部分左上角的y起点，以精确识别嘴巴的位置
                mh = int(0.5 * h)
                #截取人脸区域下半部分高度，以精确识别嘴巴的位置
                img_facehalf_bottom = img[my : (my + mh), intx : intx + intw]
                '''img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
                   img_facehalf是截取人脸识别到区域上半部分
                   img_facehalf_bottom是截取人脸识别到区域下半部分
                '''
                cv2.rectangle(img, (int(x), my), (int(x) + int(w), my + mh), (0, 255, 0), 2, 0)
                '''矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）
                    画出人脸识别下部分区域，方便定位
                '''
                faceRects_mouth = classifier_mouth.detectMultiScale(img_facehalf_bottom, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (5, 20))
                #嘴巴检测器
                if len(faceRects_mouth) > 0:
                    for faceRect_mouth in faceRects_mouth:
                        xm1, ym1, wm1, hm2 = faceRect_mouth
                        cv2.rectangle(img_facehalf_bottom, (int(xm1), int(ym1)), (int(xm1) + int(wm1), int(ym1) + int(hm2)), (0,0, 255), 2, 0)
                        #调整覆盖图片大小 resize参数（图像，检测到的（宽，高），缩放类型）
                        
                        mouth = img_facehalf_bottom[ym1 : (ym1 + hm2), xm1 : (xm1 + wm1)]
                        mouth = cv2.resize(mouth, Size, interpolation = cv2.INTER_CUBIC)
                        
                        images.append(mouth)
                        
        cv2.imshow('video', mouth)

               
        if(key == ord('q')):
            break
                
    return images

if __name__ == '__main__':   
    xxx = Video_Read(0, (30, 50), (50, 100))
    cv2.imwrite("111.jpg", xxx[0])