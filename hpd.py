#
#   Headpose Detection
#   Referenced code:
#       https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib
#       https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python
#       https://github.com/lincolnhard/head-pose-estimation
#   Modified by Qhan
#

import os
import os.path as osp
import argparse
import cv2
import numpy as np
import dlib
from timer import Timer


t = Timer()

class HPD():

    # 3D facial model coordinates
    landmarks_3d_list = [
        np.array([
            [   0.0,    0.0,    0.0],    # Nose tip
            [   0.0, -330.0,  -65.0],    # Chin
            [-225.0,  170.0, -135.0],    # Left eye left corner
            [ 225.0,  170.0, -135.0],    # Right eye right corner
            [-150.0, -150.0, -125.0],    # Left Mouth corner
            [ 150.0, -150.0, -125.0]     # Right mouth corner 
        ], dtype=np.double),
        np.array([
            [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
            [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
            [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
            [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
            [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
            [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
            [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
            [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
            [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
        ], dtype=np.double),
        np.array([
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654]    # 21 right eye right corner
        ], dtype=np.double)
    ]

    # 2d facial landmark list
    lm_2d_index_list = [
        [30, 8, 36, 45, 48, 54],
        [17, 21, 22, 26, 36, 39, 42, 45, 31, 33, 35, 48, 54, 57, 8], # 14 points
        [36, 39, 33, 42, 45] # 5 points
    ]

    nose_index_list = [0, 9, 2]

    def __init__(self, lm_type=1, predictor="model/shape_predictor_68_face_landmarks.dat", box_length=10.0):
        self.bbox_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(predictor)

        b = box_length
        self.box = np.array([
            ( b,  b,  b), ( b,  b, -b), ( b, -b, -b), ( b, -b,  b),
            (-b,  b,  b), (-b,  b, -b), (-b, -b, -b), (-b, -b,  b)
        ])
        self.b = box_length
        self.n = self.nose_index_list[lm_type]

        self.lm_2d_index = self.lm_2d_index_list[lm_type]
        self.landmarks_3d = self.landmarks_3d_list[lm_type]


    def class2np(self, landmarks):
        coords = []
        for i in self.lm_2d_index:
            coords += [[landmarks.part(i).x, landmarks.part(i).y]]
        return np.array(coords).astype(np.int)


    def getLandmark(self, im):
        # Detect bounding boxes of faces
        t.tic()
        if im is not None:
            rects = self.bbox_detector(im, 1)
        else:
            rects = []
        print(', bb: %.2f' % t.toc(), end='ms')

        if len(rects) > 0:
            # Detect landmark of first face
            t.tic()
            landmarks_2d = self.landmark_predictor(im, rects[0])

            # Choose specific landmarks corresponding to 3D facial model
            landmarks_2d = self.class2np(landmarks_2d)
            print(', lm: %.2f' % t.toc(), end='ms')

            return landmarks_2d.astype(np.double), rects[0]

        else:
            return None, None


    def getHeadpose(self, im, landmarks_2d, verbose=False):
        h, w, c = im.shape
        f = w # column size = x axis length (focal length)
        u0, v0 = w / 2, h / 2 # center of image plane
        camera_matrix = np.array(
            [[f, 0, u0],
             [0, f, v0],
             [0, 0, 1]], dtype = np.double
         )
         
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1)) 

        # Find rotation, translation
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs)
        
        if verbose:
            print("Camera Matrix:\n {0}".format(camera_matrix))
            print("Distortion Coefficients:\n {0}".format(dist_coeffs))
            print("Rotation Vector:\n {0}".format(rotation_vector))
            print("Translation Vector:\n {0}".format(translation_vector))

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs


    # rotation vector to euler angles
    def getAngles(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec)) # projection matrix [R | t]
        degrees = -cv2.decomposeProjectionMatrix(P)[6]
        rx, ry, rz = degrees[:, 0]
        return [rx, ry, rz]


    def drawBound(self, im, rect):
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)


    def drawLandmark(self, im, landmarks_2d):
        for p in landmarks_2d:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1) 


    # axis lines index
    box_lines = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ])
    def drawAxis(self, im, rvec, tvec, cm, dc):
        (projected_box, _) = cv2.projectPoints(self.box, rvec, tvec, cm, dc)
        pbox = projected_box[:, 0]
        for p in self.box_lines:
            p1, p2 = tuple(pbox[p[0]].astype(int)), tuple(pbox[p[1]].astype(int))
            cv2.line(im, p1, p2, (255, 0, 0), 2)
        

    def drawDirection(self, im, rvec, tvec, cm, dc, landmarks_2d):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), rvec, tvec, cm, dc)
        p1 = ( int(landmarks_2d[self.n, 0]), int(landmarks_2d[self.n, 1]))
        p2 = ( int(nose_end_point2D[0, 0, 0]), int(nose_end_point2D[0, 0, 1]))
        cv2.line(im, p1, p2, (0, 255, 255), 2)


    def drawInfo(self, im, angles, fontColor=(0, 255, 255)):
        x, y, z = angles
        cv2.putText(im, "X: %+06.2f" % x, (50, 40), cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=fontColor)
        cv2.putText(im, "Y: %+06.2f" % y, (50, 70), cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=fontColor)
        cv2.putText(im, "Z: %+06.2f" % z, (50, 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=fontColor)


    # return image and angles
    def processImage(self, im, draw=True):
        # landmark Detection
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        landmarks_2d, rect = self.getLandmark(im_gray)

        # if no face deteced, return original image
        if landmarks_2d is None:
            return im, None

        # Headpose Detection
        t.tic()
        rvec, tvec, cm, dc = self.getHeadpose(im, landmarks_2d)
        print(', hp: %.2f' % t.toc(), end='ms')

        t.tic()
        angles = self.getAngles(rvec, tvec)
        print(', ga: %.2f' % t.toc(), end='ms')

        if draw:
            t.tic()

            # draw Bounding Box
            self.drawBound(im, rect)
            
            # draw Landmark
            self.drawLandmark(im, landmarks_2d)

            # draw Axis
            self.drawAxis(im, rvec, tvec, cm, dc)

            # Project a 3D point (0, 0, self.b) onto the image plane.
            # We use this to draw a line sticking out of the nose
            self.drawDirection(im, rvec, tvec, cm, dc, landmarks_2d)

            # draw Rotation Angle Text
            self.drawInfo(im, angles) 
            print(', draw: %.2f' % t.toc(), end='ms' + ' ' * 10)
         
        return im, angles


def main(args):
    in_dir = args["input_dir"]
    out_dir = args["output_dir"]

    # Initialize head pose detection
    hpd = HPD(args["landmark_type"], args["landmark_predictor"], args["box_length"])

    for filename in os.listdir(in_dir):
        name, ext = osp.splitext(filename)
        if ext in ['.jpg', '.png', '.gif']: 
            print("> image:", filename, end='')
            image = cv2.imread(in_dir + filename)
            res, angles = hpd.processImage(image)
            cv2.imwrite(out_dir + name + '_out.png', res)
        else:
            print("> skip:", filename, end='')
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='DIR', dest='input_dir', default='images/')
    parser.add_argument('-o', metavar='DIR', dest='output_dir', default='res/')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    parser.add_argument('-b', metavar='N', dest='box_length', type=float, default=10.0)
    args = vars(parser.parse_args())

    if not osp.exists(args["output_dir"]): os.mkdir(args["output_dir"])
    if args["output_dir"][-1] != '/': args["output_dir"] += '/'
    if args["input_dir"][-1] != '/': args["input_dir"] += '/'
    main(args)
