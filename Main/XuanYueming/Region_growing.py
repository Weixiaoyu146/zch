import numpy as np
import cv2

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),\
                    Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-2, 0), Point(-2, 1),\
                    Point(-2, 2), Point(-2, -1), Point(-2, -2), Point(2, 0), Point(2, 1),\
                    Point(2, 2), Point(2, -1), Point(2, -2), Point(1, -2), Point(1, 2),\
                    Point(0, -2), Point(0, 2), Point(-1, -2), Point(-1, 2)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img,seeds,max,p = 1):
    height, weight = img.shape
    seedMark = np.ones(img.shape)
    seedList = []
    regionGrow_area = []
    for seed in seeds:
        seedList.append(seed)
    label = 0
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x,currentPoint.y] = label
        regionGrow_area.append((currentPoint.y, currentPoint.x))
        for i in range(24):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            gray_Value = int(img[tmpX,tmpY])
            #grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if gray_Value > max*0.6 and seedMark[tmpX,tmpY] == 1:
                seedMark[tmpX,tmpY] = label
                regionGrow_area.append((tmpY, tmpX))
                seedList.append(Point(tmpX,tmpY))
    return regionGrow_area





