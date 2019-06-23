import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def line_removal(image,to_show=False):
    #crop image first
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #sobel = cv2.Sobel(gray,cv2.CV_8U, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
    sobel = gray
    _ , threshold = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    threshold = cv2.bitwise_not(threshold)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((3,3)))
    
    lines = cv2.HoughLinesP(threshold, 1, 2*math.pi/180, 50, 300, 40 )

    bigx = lines[0][0]
    bigy = lines[0][0]
    for line in lines:
        line = line[0]
        if abs(line[0] - line[2]) > abs(bigx[0] - bigx[2]):
            bigx = line
        if abs(line[1] - line[3]) > abs(bigy[1] - bigy[3]):
            bigy = line
        #cv2.line(image, (line[0],line[1]), (line[2],line[3]), (0,255,0),1)
    
    factor=2
    thres_minus = []
    for line in (bigx,bigy):
        x1,y1,x2,y2 = line
        min_x = min(x1,x2)
        min_y = min(y1,y2)
        x1,x2 = int((x1-min_x)/factor),int((x2-min_x)/factor)
        y1,y2 = int((y1-min_y)/factor),int((y2-min_y)/factor)

        max_max = max(abs(x1-x2), abs(y1-y2))
        diff_x = math.floor((max_max - abs(x1-x2))/2)
        diff_y = math.floor((max_max - abs(y1-y2))/2)
        x1,x2 = x1+diff_x, x2+diff_x
        y1,y2 = y1+diff_y, y2+diff_y

        #print(x1,y1,x2,y2)
        matrix = np.zeros((max_max,max_max),np.uint8)
        cv2.line(matrix, (x1,y1), (x2,y2), 1 ,1)
        thres_minus.append(cv2.morphologyEx(threshold, cv2.MORPH_OPEN, matrix))
        #cv2.line(image, (line[0],line[1]), (line[2],line[3]), (255,0,0),3)

    threshold = cv2.subtract(threshold,thres_minus[0])
    threshold = cv2.subtract(threshold,thres_minus[1])

    if to_show:
        plt.matshow(threshold)
        plt.show()
    return threshold
