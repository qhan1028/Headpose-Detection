#
#   Draw Headpose Annotations
#   Written by Qhan
#

import numpy as np
import cv2


class Color():
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


class Draw():
    
    def __init__(self, im, angles, bbox, lm, rvec, tvec, cm, dc, b=10.0):
        self.im = im

        self.angles = angles
        self.bbox = bbox
        self.lm = lm
        self.rvec = rvec
        self.tvec = tvec
        self.cm = cm
        self.dc = dc
        self.nose = tuple(lm[0].astype(int))
        self.box = np.array([
            ( b,  b,  b), ( b,  b, -b), ( b, -b, -b), ( b, -b,  b),
            (-b,  b,  b), (-b,  b, -b), (-b, -b, -b), (-b, -b,  b)
        ])
        self.b = b

        h, w, c = im.shape
        self.fs = ((h + w) / 2) / 500
        self.ls = round(self.fs * 2)
        self.ps = self.ls


    def drawAll(self):
        self.drawBound()
        self.drawLandmark()
        self.drawAxis()
        self.drawDirection()
        self.drawInfo()
        return self.im


    def drawBound(self):
        x1, y1, x2, y2 = self.bbox.left(), self.bbox.top(), self.bbox.right(), self.bbox.bottom()
        cv2.rectangle(self.im, (x1, y1), (x2, y2), Color.green, self.ls)


    def drawLandmark(self):
        for p in self.lm:
            point = tuple(p.astype(int))
            cv2.circle(self.im, point, self.ps, Color.red, -1)


    # axis lines index
    box_lines = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ])
    def drawAxis(self):
        (projected_box, _) = cv2.projectPoints(self.box, self.rvec, self.tvec, self.cm, self.dc)
        pbox = projected_box[:, 0]
        for p in self.box_lines:
            p1 = tuple(pbox[p[0]].astype(int))
            p2 = tuple(pbox[p[1]].astype(int))
            cv2.line(self.im, p1, p2, Color.blue, self.ls)


    def drawDirection(self):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm, self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        cv2.line(self.im, p1, p2, Color.yellow, self.ls)


    def drawInfo(self, fontColor=Color.yellow):
        x, y, z = self.angles
        px, py, dy = int(5 * self.fs), int(25 * self.fs), int(30 * self.fs)
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = self.fs
        cv2.putText(self.im, "X: %+06.2f" % x, (px, py), font, fontScale=fs, color=fontColor)
        cv2.putText(self.im, "Y: %+06.2f" % y, (px, py + dy), font, fontScale=fs, color=fontColor)
        cv2.putText(self.im, "Z: %+06.2f" % z, (px, py + 2 * dy), font, fontScale=fs, color=fontColor)
