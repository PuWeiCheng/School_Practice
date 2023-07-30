# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import neurolab as nl
cv2.namedWindow("home1",cv2.WINDOW_NORMAL)
cv2.namedWindow("home2",cv2.WINDOW_NORMAL)
cv2.namedWindow("home3",cv2.WINDOW_NORMAL)
inputnum = []
target = []
for i in range(0,10):
    for j in range(1,6):
        image = cv2.imread('number '+str(i)+'-'+str(j)+'.png')
        image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,bin_image = cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
        resimage = cv2.resize(bin_image,(10, 10))
        reshimage = resimage.reshape(1,100)
        for k in range(len(reshimage[0])):
            reshimage[0][k] = reshimage[0][k]/255
        inputnum.append(reshimage[0])
        if i == 0:
            des_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif i == 1:
            des_y = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif i == 2:
            des_y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif i == 3:
            des_y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif i == 4:
            des_y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif i == 5:
            des_y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif i == 6:
            des_y = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif i == 7:
            des_y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif i == 8:
            des_y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif i == 9:
            des_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                                    
        target.append(des_y)
input_layer = []
for i in range(100):
    input_layer.append([0,1])
net = nl.net.newff(input_layer, [10, 10])
error_progress = net.train(inputnum, target, epochs=500, show=100, goal=0.02)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.show()

testnum = []

pic = input('輸入辨識照片編號：')
image = cv2.imread('test '+pic+'.png')
image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,bin_image = cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
resimage = cv2.resize(bin_image,(10, 10))
cv2.imshow("home1",image)
cv2.imshow("home2",image1)
cv2.imshow("home3",resimage)
cv2.waitKey(10000)
cv2.destroyAllWindows()
reshimage = resimage.reshape(1,100)
testnum.append(reshimage[0])
for i in range(len(testnum)):
    testnum[i] = testnum[i]/255
y = net.sim(list(testnum))
print(y)
print('輸入的圖片數字為：', np.argmax(y))